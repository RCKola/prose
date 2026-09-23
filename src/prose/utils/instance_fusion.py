"""Per-instance fusion of 3D points across frames using SAM3 masks.

Shared primitives backing the Stage 4 (fusion) scene graph: per-instance
point aggregation, intra-side dedup, oriented bounding boxes, and k-NN
proximity edges.

All operations are pure numpy. No torch, no I/O.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-(frame, iid) 3D point grouping
# ---------------------------------------------------------------------------

_FUSION_THREADS = max(1, int(os.environ.get("RECON_UA_FUSION_THREADS", "8")))


def _group_p2p_by_frame(
    point_to_pixels: Sequence[Sequence[dict]],
    n_points: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Invert per-point projections into per-frame (us, vs, pt_idx) arrays."""
    buckets: Dict[int, List[Tuple[int, int, int]]] = {}
    limit = min(n_points, len(point_to_pixels))
    for pt_idx in range(limit):
        entries = point_to_pixels[pt_idx]
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            continue
        for entry in entries:
            if not entry:
                continue
            fid = entry.get("frame_id")
            if fid is None:
                continue
            u = entry.get("pixel_u")
            v = entry.get("pixel_v")
            if u is None or v is None:
                continue
            buckets.setdefault(int(fid), []).append((int(u), int(v), pt_idx))
    out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fid, rows in buckets.items():
        arr = np.asarray(rows, dtype=np.int64)
        out[fid] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def _rescale_uv(
    us: np.ndarray,
    vs: np.ndarray,
    mask_shape: Tuple[int, int],
    src_res: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel-center-correct rescale of (u, v) from src_res to mask_shape.

    Matches `eval.metrics._make_pixel_rescaler`. When `src_res` is None or
    already equals `mask_shape`, returns the inputs unchanged (with boundary
    snap and bounds clip).
    """
    mh, mw = int(mask_shape[0]), int(mask_shape[1])
    if src_res is None:
        qu = us.copy()
        qv = vs.copy()
    else:
        sh, sw = int(src_res[0]), int(src_res[1])
        if sh == mh and sw == mw:
            qu = us.copy()
            qv = vs.copy()
        else:
            qu = np.rint((us.astype(np.float64) + 0.5) * mw / sw - 0.5).astype(np.int64)
            qv = np.rint((vs.astype(np.float64) + 0.5) * mh / sh - 0.5).astype(np.int64)
    np.clip(qu, 0, mw - 1, out=qu)
    np.clip(qv, 0, mh - 1, out=qv)
    return qu, qv


def _assign_frame(
    fid: int,
    us: np.ndarray,
    vs: np.ndarray,
    pt_idxs: np.ndarray,
    masks_by_iid: Dict[int, np.ndarray],
    src_res: Optional[Tuple[int, int]],
) -> Tuple[int, Dict[int, np.ndarray]]:
    """Assign each point in one frame to its smallest covering iid mask.

    Returns (fid, {iid: pt_idx array}). Empty dict when no point hits any mask.
    """
    if not masks_by_iid or us.size == 0:
        return fid, {}
    iids = sorted(int(i) for i in masks_by_iid)
    # Stack masks. Use bool view; cost is small (K x H x W) and lets us index
    # all (K, N) hits in one numpy op.
    stack = np.stack([np.asarray(masks_by_iid[i], dtype=bool) for i in iids], axis=0)
    H, W = stack.shape[1], stack.shape[2]
    qu, qv = _rescale_uv(us, vs, (H, W), src_res)
    hits = stack[:, qv, qu]                                # (K, N)
    valid = hits.any(axis=0)                                # (N,)
    if not valid.any():
        return fid, {}
    areas = stack.reshape(len(iids), -1).sum(axis=1)        # (K,)
    # Smallest-area-wins: mask out non-hits with +inf, then argmin.
    cost = np.where(hits, areas[:, None].astype(np.float64), np.inf)
    winners_idx = np.argmin(cost, axis=0)                   # (N,) into iids
    winner_iid = np.asarray(iids, dtype=np.int64)[winners_idx]
    # Group surviving points by winner iid.
    out: Dict[int, np.ndarray] = {}
    survivors_mask = valid
    surv_iid = winner_iid[survivors_mask]
    surv_pt = pt_idxs[survivors_mask]
    # Sort by iid for grouped slicing (fast on already small arrays).
    order = np.argsort(surv_iid, kind="stable")
    surv_iid = surv_iid[order]
    surv_pt = surv_pt[order]
    # Find segment boundaries per iid.
    if surv_iid.size:
        change = np.concatenate(([True], surv_iid[1:] != surv_iid[:-1]))
        starts = np.flatnonzero(change)
        ends = np.concatenate((starts[1:], [surv_iid.size]))
        for s, e in zip(starts, ends):
            out[int(surv_iid[s])] = surv_pt[s:e]
    return fid, out


def instance_points_per_frame(
    points: np.ndarray,
    point_to_pixels: Sequence[Sequence[dict]],
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
    *,
    stage1_pixel_resolution: Optional[Tuple[int, int]] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Group 3D points by (frame_id, iid) using each point's recorded pixels.

    `point_to_pixels[pt_idx]` is a list of (frame_id, pixel_u, pixel_v) records
    for each frame the point projected into. Every record is considered — a
    point visible in 30 frames contributes to up to 30 (frame, iid) buckets.
    Within a single frame the **smallest** matching mask wins (favours the
    more specific instance when nested masks overlap; replaces the legacy
    first-match-wins logic, 2026-05-17). Boundary pixels are snapped inside.

    `stage1_pixel_resolution` is the (H, W) the (u, v) coords in
    `point_to_pixels` live at. SAM3 masks live at the source RGB resolution
    while DA3 records pixel coords at its processed resolution (e.g. 280×504
    vs 540×960 on 3RScan), so passing this is required to avoid attributing
    points to the wrong region of the mask. None = (u, v) already match mask
    coordinates (the GT path).
    """
    pts = np.asarray(points)
    per_frame = _group_p2p_by_frame(point_to_pixels, pts.shape[0])
    fids = [fid for fid in per_frame if per_frame_masks.get(fid)]
    if not fids:
        return {}

    out: Dict[int, Dict[int, np.ndarray]] = {}

    def _run(fid: int):
        us, vs, idxs = per_frame[fid]
        return _assign_frame(fid, us, vs, idxs, per_frame_masks[fid], stage1_pixel_resolution)

    if _FUSION_THREADS > 1 and len(fids) > 1:
        with ThreadPoolExecutor(max_workers=min(_FUSION_THREADS, len(fids))) as ex:
            for fid, by_iid in ex.map(_run, fids):
                if by_iid:
                    out[fid] = {iid: pts[idx_arr].astype(np.float64) for iid, idx_arr in by_iid.items()}
    else:
        for fid in fids:
            fid_out, by_iid = _run(fid)
            if by_iid:
                out[fid_out] = {iid: pts[idx_arr].astype(np.float64) for iid, idx_arr in by_iid.items()}

    return out


def instance_pixels_per_frame(
    points: np.ndarray,
    point_to_pixels: Sequence[Sequence[dict]],
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
    *,
    stage1_pixel_resolution: Optional[Tuple[int, int]] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Like instance_points_per_frame but stores pixel (u, v) instead of 3D pts.

    Per-frame tiebreak: smallest mask wins (matches
    `instance_points_per_frame`, 2026-05-17).
    """
    pts = np.asarray(points)
    per_frame = _group_p2p_by_frame(point_to_pixels, pts.shape[0])
    fids = [fid for fid in per_frame if per_frame_masks.get(fid)]
    if not fids:
        return {}

    out: Dict[int, Dict[int, np.ndarray]] = {}

    def _run(fid: int):
        us, vs, idxs = per_frame[fid]
        fid_out, by_iid = _assign_frame(fid, us, vs, idxs, per_frame_masks[fid], stage1_pixel_resolution)
        # Replace point-index payload with (u, v) at the *original* Stage 1
        # resolution (callers expect un-rescaled pixel coords).
        pixel_out: Dict[int, np.ndarray] = {}
        for iid, idx_arr in by_iid.items():
            uv = np.stack([us[np.isin(idxs, idx_arr)], vs[np.isin(idxs, idx_arr)]], axis=1).astype(np.float64)
            pixel_out[iid] = uv
        return fid_out, pixel_out

    if _FUSION_THREADS > 1 and len(fids) > 1:
        with ThreadPoolExecutor(max_workers=min(_FUSION_THREADS, len(fids))) as ex:
            for fid, by_iid in ex.map(_run, fids):
                if by_iid:
                    out[fid] = by_iid
    else:
        for fid in fids:
            fid_out, by_iid = _run(fid)
            if by_iid:
                out[fid_out] = by_iid

    return out


# ---------------------------------------------------------------------------
# Aggregation across frames
# ---------------------------------------------------------------------------

def instance_points(
    per_frame_inst_points: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, np.ndarray]:
    """Concatenate each instance's points across frames."""
    by_iid: Dict[int, list] = {}
    for inst in per_frame_inst_points.values():
        for iid, pts in inst.items():
            by_iid.setdefault(iid, []).append(pts)
    return {iid: np.concatenate(arrs, axis=0) for iid, arrs in by_iid.items()}


def instance_centroids(inst_points: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {iid: pts.mean(axis=0) for iid, pts in inst_points.items()}


def per_frame_centroids(
    per_frame_inst_points: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, Dict[int, np.ndarray]]:
    return {
        f: {iid: pts.mean(axis=0) for iid, pts in inst.items()}
        for f, inst in per_frame_inst_points.items()
    }


def instance_centroids_robust(
    per_frame_inst_points: Dict[int, Dict[int, np.ndarray]],
    *,
    method: str = "median_of_frame_centroids",
) -> Dict[int, np.ndarray]:
    """Robust per-instance centroid: median of per-frame centroids.

    Each frame gets equal weight (vs point-count-weighted mean). Component-wise
    median rejects depth-error outlier frames that shift one axis.
    """
    if method == "mean":
        return instance_centroids(instance_points(per_frame_inst_points))

    pf_cents = per_frame_centroids(per_frame_inst_points)
    # Collect per-frame centroids for each iid across all frames.
    by_iid: Dict[int, list] = {}
    for _fid, inst in pf_cents.items():
        for iid, c in inst.items():
            by_iid.setdefault(iid, []).append(c)
    out: Dict[int, np.ndarray] = {}
    for iid, cs in by_iid.items():
        stacked = np.stack(cs, axis=0)  # (F_i, 3)
        out[iid] = np.median(stacked, axis=0)
    return out


def outlier_frame_filter(
    per_frame_inst_points: Dict[int, Dict[int, np.ndarray]],
    robust_centroids: Dict[int, np.ndarray],
    *,
    max_deviation_m: float = 0.15,
    min_frames: int = 5,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, List[int]]]:
    """Drop (frame, iid) entries whose per-frame centroid deviates from robust centroid.

    Returns (cleaned pf_inst_pts, dropped: {iid: [frame_ids]}).
    IIDs appearing in fewer than min_frames are not filtered.
    """
    pf_cents = per_frame_centroids(per_frame_inst_points)

    # Count frames per iid.
    iid_frame_count: Dict[int, int] = {}
    for _fid, inst in pf_cents.items():
        for iid in inst:
            iid_frame_count[iid] = iid_frame_count.get(iid, 0) + 1

    dropped: Dict[int, List[int]] = {}
    out: Dict[int, Dict[int, np.ndarray]] = {}

    for fid, by_iid in per_frame_inst_points.items():
        kept: Dict[int, np.ndarray] = {}
        for iid, pts in by_iid.items():
            rc = robust_centroids.get(iid)
            fc = pf_cents.get(fid, {}).get(iid)
            if (
                rc is not None
                and fc is not None
                and iid_frame_count.get(iid, 0) >= min_frames
                and np.linalg.norm(fc - rc) > max_deviation_m
            ):
                dropped.setdefault(iid, []).append(fid)
            else:
                kept[iid] = pts
        if kept:
            out[fid] = kept

    return out, dropped


def depth_inlier_filter(
    per_frame_inst_points: Dict[int, Dict[int, np.ndarray]],
    *,
    mad_k: float = 2.0,
    min_points: int = 10,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], int]:
    """Per-(frame, iid) depth inlier filter using median absolute deviation.

    For each (frame, iid), keeps only points whose depth (z-coordinate) is
    within mad_k * MAD of the median depth.  Groups with fewer than min_points
    are left unfiltered.

    Returns (filtered pf_inst_pts, total_removed_points).
    """
    out: Dict[int, Dict[int, np.ndarray]] = {}
    total_removed = 0
    for fid, by_iid in per_frame_inst_points.items():
        kept: Dict[int, np.ndarray] = {}
        for iid, pts in by_iid.items():
            if pts.shape[0] < min_points:
                kept[iid] = pts
                continue
            z = pts[:, 2]
            med = np.median(z)
            mad = np.median(np.abs(z - med))
            if mad < 1e-9:
                kept[iid] = pts
                continue
            inlier = np.abs(z - med) <= mad_k * mad
            n_removed = int((~inlier).sum())
            total_removed += n_removed
            kept[iid] = pts[inlier]
        if kept:
            out[fid] = kept
    return out, total_removed


def instance_centroids_2d(
    points: np.ndarray,
    point_to_pixels: Sequence[Sequence[dict]],
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, np.ndarray]:
    """Mean pixel (u, v) for each instance across all frames."""
    pf_pixels = instance_pixels_per_frame(points, point_to_pixels, per_frame_masks)
    by_iid: Dict[int, list] = {}
    for inst in pf_pixels.values():
        for iid, uvs in inst.items():
            by_iid.setdefault(iid, []).append(uvs)
    return {
        iid: np.concatenate(arrs, axis=0).mean(axis=0)
        for iid, arrs in by_iid.items()
    }


# ---------------------------------------------------------------------------
# Dynamic-instance filter via centroid jitter
# ---------------------------------------------------------------------------

def centroid_jitter(
    pf_centroids: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, float]:
    """Mean radial distance of an instance's per-frame centroids from their mean.

    A static instance with consistent SAM3 tracking should have low jitter.
    Returns 0.0 for instances appearing in only one frame.
    """
    by_iid: Dict[int, list] = {}
    for inst in pf_centroids.values():
        for iid, c in inst.items():
            by_iid.setdefault(iid, []).append(np.asarray(c, dtype=np.float64))
    out: Dict[int, float] = {}
    for iid, cs in by_iid.items():
        if len(cs) <= 1:
            out[iid] = 0.0
            continue
        arr = np.stack(cs, axis=0)
        mean = arr.mean(axis=0)
        out[iid] = float(np.linalg.norm(arr - mean, axis=1).mean())
    return out


def filter_dynamic(jitter: Dict[int, float], threshold_m: float) -> set:
    return {iid for iid, j in jitter.items() if j <= threshold_m}


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def instance_max_size(
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, int]:
    """Largest mask area (in pixels) attained by each instance across frames."""
    out: Dict[int, int] = {}
    for inst in per_frame_masks.values():
        for iid, m in inst.items():
            a = int(m.sum())
            if a > out.get(iid, 0):
                out[iid] = a
    return out


def instance_persistence(
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, int]:
    """Number of frames in which each instance has a non-empty mask."""
    out: Dict[int, int] = {}
    for inst in per_frame_masks.values():
        for iid in inst:
            out[iid] = out.get(iid, 0) + 1
    return out


# ---------------------------------------------------------------------------
# New for Stage 4: per-frame fractional visibility + bbox-diag normalization
# ---------------------------------------------------------------------------

def per_frame_visibility(
    per_frame_inst_points: Dict[int, Dict[int, np.ndarray]],
    instance_ids: List[int],
    frame_ids: List[int],
) -> np.ndarray:
    """Build (n_instances, n_frames) fractional visibility matrix.

    visibility[i, f] = (#points hitting frame f's mask for instance i)
                       / (total points fused for instance i)

    Rows are 0 for instances absent from `per_frame_inst_points` (e.g. dropped
    by dynamic filter); columns are 0 for frames in which an instance has no
    mask hits. Both `instance_ids` and `frame_ids` set the row/col ordering.
    """
    n_iids = len(instance_ids)
    n_frames = len(frame_ids)
    iid_to_row = {int(iid): i for i, iid in enumerate(instance_ids)}
    fid_to_col = {int(fid): j for j, fid in enumerate(frame_ids)}

    counts = np.zeros((n_iids, n_frames), dtype=np.float32)
    for fid, inst in per_frame_inst_points.items():
        col = fid_to_col.get(int(fid))
        if col is None:
            continue
        for iid, pts in inst.items():
            row = iid_to_row.get(int(iid))
            if row is None:
                continue
            counts[row, col] = float(len(pts))

    totals = counts.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return (counts / totals).astype(np.float32)


def bbox_diag_normalize(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Center on centroid, scale by axis-aligned bbox diagonal so it equals 1.

    Returns (points_normalized, original_bbox_diag). For degenerate clouds
    (single point or zero extent), returns the centered cloud with diag=1.0
    so callers don't divide by zero.
    """
    pts = np.asarray(points, dtype=np.float32)
    centroid = pts.mean(axis=0, keepdims=True)
    centered = pts - centroid
    extent = centered.max(axis=0) - centered.min(axis=0)
    diag = float(np.linalg.norm(extent))
    if diag < 1e-9:
        return centered, 1.0
    return (centered / diag).astype(np.float32), diag


def oriented_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Oriented bounding box via PCA.

    Computes the principal axes of the centered point cloud (eigenvectors of
    the covariance matrix) and returns the OBB axes and edge lengths.

    Returns:
        rotation: (3, 3) float32 — columns are OBB principal axes in the
          original coordinate frame, sorted by **descending** variance (column
          0 = longest extent axis, column 2 = shortest). The rotation is a
          proper rotation (det = +1 after sign-normalisation).
        extents:  (3,) float32 — full edge lengths along each OBB axis
          (extents[i] = max − min projection on rotation[:, i]).

    For degenerate clouds (< 3 points or zero variance), returns identity
    rotation and zero extents.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    centered = pts - pts.mean(axis=0)
    n = max(len(pts) - 1, 1)
    cov = (centered.T @ centered) / n

    # eigh returns eigenvalues ascending; flip to descending.
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, idx]  # (3, 3), columns = principal axes

    # Enforce a consistent orientation: make each axis point in the direction
    # of greater positive projection (keeps the rotation determinant positive
    # and reproducible across calls on the same cloud).
    for k in range(3):
        proj = centered @ rotation[:, k]
        if proj.mean() < 0:
            rotation[:, k] = -rotation[:, k]

    # Project points onto OBB axes and measure extents.
    projected = centered @ rotation  # (N, 3)
    extents = (projected.max(axis=0) - projected.min(axis=0)).astype(np.float32)

    return rotation.astype(np.float32), extents


__all__ = [
    "instance_points_per_frame",
    "instance_pixels_per_frame",
    "voxel_revote_iids",
    "instance_points",
    "instance_centroids",
    "per_frame_centroids",
    "instance_centroids_2d",
    "centroid_jitter",
    "filter_dynamic",
    "instance_max_size",
    "instance_persistence",
    "per_frame_visibility",
    "bbox_diag_normalize",
    "oriented_bbox",
    "aggregate_per_frame_scores",
    "dedup_and_fuse_instances",
    "apply_alias_to_per_frame_masks",
    "apply_alias_to_per_frame_scores",
    "apply_alias_to_instance_map",
]


def _size_within_frac(extents_a: np.ndarray, extents_b: np.ndarray, frac: float) -> bool:
    ea = np.abs(np.asarray(extents_a, dtype=np.float64))
    eb = np.abs(np.asarray(extents_b, dtype=np.float64))
    if ea.size != 3 or eb.size != 3:
        return False
    ratio_lo = np.minimum(ea, eb) / np.maximum(np.maximum(ea, eb), 1e-6)
    return bool(np.all(ratio_lo >= (1.0 - frac)))


# Bit shifts that pack a signed-21-bit (x, y, z) voxel triple into one int64.
# 2^20 voxels ≈ 50 km at 5 cm; well past any indoor scene.
_VOXEL_OFFSET = 1 << 20
_VOXEL_BITS = 21


def _voxel_keys(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Sorted int64-packed unique voxel keys for a cloud.

    Encodes each voxel's (x, y, z) integer key as one int64 so set ops can
    use numpy primitives (`np.intersect1d`) instead of Python tuple-sets.
    Returns a 1-D int64 array in ascending order with no duplicates.
    """
    if pts.size == 0:
        return np.empty(0, dtype=np.int64)
    k = np.floor(np.asarray(pts, dtype=np.float64) / float(voxel)).astype(np.int64)
    packed = (
        ((k[:, 0] + _VOXEL_OFFSET) << (2 * _VOXEL_BITS))
        | ((k[:, 1] + _VOXEL_OFFSET) << _VOXEL_BITS)
        | (k[:, 2] + _VOXEL_OFFSET)
    )
    return np.unique(packed)


def _voxel_containment(va: np.ndarray, vb: np.ndarray) -> Tuple[float, float]:
    """(IoU, containment = inter / min(|va|, |vb|)) for two packed-int64 sets."""
    if va.size == 0 or vb.size == 0:
        return 0.0, 0.0
    inter = int(np.intersect1d(va, vb, assume_unique=True).size)
    if inter == 0:
        return 0.0, 0.0
    iou = inter / (va.size + vb.size - inter)
    cont = inter / min(va.size, vb.size)
    return iou, cont


def _subsample(pts: np.ndarray, max_pts: int, seed: int = 0) -> np.ndarray:
    if len(pts) <= max_pts:
        return pts
    return pts[np.random.default_rng(seed).choice(len(pts), size=max_pts, replace=False)]


def voxel_revote_iids(
    pf_inst_pts: Dict[int, Dict[int, np.ndarray]],
    *,
    voxel_size: float = 0.02,
    absorb_majority_threshold: float = 0.5,
    level1_min_majority_frac: float = 0.6,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, int]]:
    """Per-voxel majority-vote relabel of (frame, iid) per-point assignments.

    The idea: each Stage 4 3D point currently carries the iid of its birth
    frame's mask. When two iids both claim a region (e.g. SAM3 re-detected
    the same cushion as a new iid mid-video), the per-point assignment is
    arbitrary. Voxel-vote relabels every point inside a voxel to the
    iid with the most points there — a hard mode filter on the 3D label
    grid that resolves SAM3 per-frame ambiguity into one label per region.

    Args:
        pf_inst_pts: per-(frame, iid) 3D point lists from
            `instance_points_per_frame`.
        voxel_size: voxel grid spacing in meters.
        absorb_majority_threshold: (Level 2) if at least this fraction of an
            iid's original points get revoted to a single other iid, alias the
            losing iid into that absorber (and move any surviving points
            from the losing iid over too). 0 or 1 = disable.
        level1_min_majority_frac: (Level 1) per-point confidence gate. A
            point is only reassigned if the winning iid holds at least this
            fraction of the voxel's total points. 0.0 = no gate (always
            reassign). 0.6 = require 60% majority.

    Returns:
        (new pf_inst_pts, iid_alias). iid_alias maps any iid that lost all
        its voxels to the iid that absorbed its largest voxel-cluster. iids
        with surviving voxels do not appear in iid_alias.
    """
    if not pf_inst_pts:
        return {}, {}

    # 1) Flatten per-(frame, iid) blocks into parallel arrays.
    all_pts_list: List[np.ndarray] = []
    all_iid_list: List[np.ndarray] = []
    all_fid_list: List[np.ndarray] = []
    for fid, by_iid in pf_inst_pts.items():
        for iid, arr in by_iid.items():
            if arr.shape[0] == 0:
                continue
            all_pts_list.append(np.asarray(arr, dtype=np.float64))
            all_iid_list.append(np.full(arr.shape[0], int(iid), dtype=np.int64))
            all_fid_list.append(np.full(arr.shape[0], int(fid), dtype=np.int64))
    if not all_pts_list:
        return {}, {}
    pts = np.concatenate(all_pts_list, axis=0)
    iid_per_pt = np.concatenate(all_iid_list, axis=0)
    fid_per_pt = np.concatenate(all_fid_list, axis=0)

    unique_iids = np.unique(iid_per_pt)
    n_iids = len(unique_iids)
    iid_to_idx = {int(i): k for k, i in enumerate(unique_iids.tolist())}
    idx_per_pt = np.fromiter((iid_to_idx[int(i)] for i in iid_per_pt), dtype=np.int64, count=len(iid_per_pt))

    # 2) Voxel hash; unique-and-inverse to map each point to a voxel id.
    keys = np.floor(pts / float(voxel_size)).astype(np.int64)
    voxel_ids, voxel_inv = np.unique(keys, axis=0, return_inverse=True)
    n_voxels = voxel_ids.shape[0]

    # 3) Vectorized per-voxel iid count: (n_voxels, n_iids) via np.add.at.
    counts = np.zeros((n_voxels, n_iids), dtype=np.int64)
    np.add.at(counts, (voxel_inv, idx_per_pt), 1)
    voxel_iid_idx = counts.argmax(axis=1)              # (n_voxels,) winner index
    voxel_iid = unique_iids[voxel_iid_idx]              # (n_voxels,) winner iid

    # 3b) Level 1 confidence gate: only reassign points in voxels where the
    # winner holds >= level1_min_majority_frac of the total count.
    if level1_min_majority_frac > 0.0:
        voxel_totals = counts.sum(axis=1).clip(1)
        voxel_max = counts.max(axis=1)
        confident = (voxel_max.astype(np.float64) / voxel_totals) >= level1_min_majority_frac
        # For non-confident voxels, each point keeps its original iid.
        # We mark these voxels so the per-point relabel below skips them.
        voxel_confident = confident  # (n_voxels,) bool

    # 5) Iid alias for dead iids: an original iid is "dead" if it has zero
    #    surviving voxels. Map it to the iid that absorbed its largest
    #    voxel-block (= iid that wins the most voxels where this iid had
    #    any votes).
    surviving_voxels_per_iid_idx = np.zeros(n_iids, dtype=np.int64)
    np.add.at(surviving_voxels_per_iid_idx, voxel_iid_idx, 1)

    iid_alias: Dict[int, int] = {}
    for k, iid in enumerate(unique_iids.tolist()):
        if surviving_voxels_per_iid_idx[k] > 0:
            continue
        had_presence = counts[:, k] > 0
        if not had_presence.any():
            continue
        winner_idxs = voxel_iid_idx[had_presence]
        vals, cnts = np.unique(winner_idxs, return_counts=True)
        absorber_idx = vals[cnts.argmax()]
        iid_alias[int(iid)] = int(unique_iids[absorber_idx])

    # 5b) Absorb-on-majority pass. For each original iid X whose original
    #     points were revoted away to a single absorber Y in at least
    #     `absorb_majority_threshold` proportion, alias X -> Y. This
    #     captures iids that survived as fringe-only voxels after the
    #     winner-take-all vote — they describe the same physical region as
    #     their dominant absorber.
    # Per-point new iid index, respecting Level 1 confidence gate.
    new_iid_idx_per_pt = voxel_iid_idx[voxel_inv].copy()  # (N,) winner iid idx per point
    if level1_min_majority_frac > 0.0:
        not_confident = ~voxel_confident[voxel_inv]
        new_iid_idx_per_pt[not_confident] = idx_per_pt[not_confident]
    if absorb_majority_threshold > 0.0 and absorb_majority_threshold < 1.0:
        # Transition counts: T[orig_idx, new_idx] = #points.
        T = np.zeros((n_iids, n_iids), dtype=np.int64)
        np.add.at(T, (idx_per_pt, new_iid_idx_per_pt), 1)
        row_sums = T.sum(axis=1)
        for k in range(n_iids):
            if int(unique_iids[k]) in iid_alias:
                continue  # already absorbed
            total = int(row_sums[k])
            if total == 0:
                continue
            # Best off-diagonal absorber.
            row = T[k].copy()
            row[k] = 0
            best = int(row.argmax())
            if row[best] / float(total) < absorb_majority_threshold:
                continue
            # Chase the alias chain in case the absorber itself is being
            # absorbed (rare on real data but handle it).
            target = int(unique_iids[best])
            seen = {int(unique_iids[k])}
            while target in iid_alias and target not in seen:
                seen.add(target)
                target = iid_alias[target]
            if target == int(unique_iids[k]):
                continue
            iid_alias[int(unique_iids[k])] = target

    # 6) Re-emit per-(frame, iid) per-point arrays using new labels.
    # Start from the confidence-gated per-point labels.
    new_iid_per_pt = unique_iids[new_iid_idx_per_pt]
    # Apply alias to per-point labels so absorbed iids' surviving points
    # move to their absorber.
    if iid_alias:
        remap = unique_iids.copy()
        for src, dst in iid_alias.items():
            if int(src) in iid_to_idx:
                remap[iid_to_idx[int(src)]] = int(dst)
        new_iid_per_pt = remap[new_iid_idx_per_pt]
    out: Dict[int, Dict[int, np.ndarray]] = {}
    # Group by (fid, new_iid) — vectorized via lexsort.
    if len(new_iid_per_pt) > 0:
        fids_sel = fid_per_pt
        iids_sel = new_iid_per_pt
        pts_sel = pts
        order = np.lexsort((iids_sel, fids_sel))
        fids_sel = fids_sel[order]; iids_sel = iids_sel[order]; pts_sel = pts_sel[order]
        # Find runs of equal (fid, iid).
        keys2 = np.stack([fids_sel, iids_sel], axis=1)
        changes = np.any(keys2[1:] != keys2[:-1], axis=1)
        starts = np.concatenate(([0], np.where(changes)[0] + 1, [len(keys2)]))
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            fid = int(fids_sel[s]); iid = int(iids_sel[s])
            out.setdefault(fid, {})[iid] = pts_sel[s:e].astype(np.float64)

    return out, iid_alias


def dedup_and_fuse_instances(
    *,
    inst_pts: Dict[int, np.ndarray],
    pf_inst_pts: Dict[int, Dict[int, np.ndarray]],
    centroids: Dict[int, np.ndarray],
    obb_extents: Dict[int, np.ndarray],
    categories: Optional[Dict[int, str]] = None,
    # Legacy gate params (used when gate == "centroid_obb").
    centroid_thresh_m: float = 0.15,
    obb_size_gate_frac: float = 0.40,
    require_category_match: bool = False,
    min_points_per_cluster: int = 0,
    # New gate (default).
    gate: str = "voxel_chamfer",
    voxel_size_m: float = 0.05,
    voxel_containment_thresh: float = 0.6,
    chamfer_thresh_m: float = 0.03,
    chamfer_max_pts: int = 10000,
) -> Tuple[
    Dict[int, np.ndarray],                 # fused inst_pts (keyed by canonical iid)
    Dict[int, Dict[int, np.ndarray]],      # fused pf_inst_pts
    Dict[int, int],                        # iid_alias: original iid -> canonical iid
    List[Tuple[int, List[int]]],           # merges: (canonical_iid, [merged member iids])
]:
    """Scene-wide intra-side dedup + 3D fusion.

    Two gate flavours (`gate`):
      * "centroid_obb" — legacy. Centroid distance + OBB extent similarity.
      * "voxel_chamfer" — current default. Pair merges when:
          - voxel containment @ voxel_size_m  ≥ voxel_containment_thresh, AND
          - min(chamfer_AB, chamfer_BA)       ≤ chamfer_thresh_m,
          where containment = |Va ∩ Vb| / min(|Va|, |Vb|). vCont catches the
          "small object entirely inside big object" case (true duplicates);
          min-chamfer rejects "two unrelated objects whose bboxes happen to
          overlap" by requiring at least one of the two clouds to lie close
          to the other on average. Chamfer queries run in parallel via a
          thread pool (numpy/scipy KDTree releases the GIL).

    Qualifying pairs are union-found into clusters; the lowest iid in each
    cluster is canonical. `inst_pts` and `pf_inst_pts` are concatenated under
    the canonical iid. Clusters with fewer than `min_points_per_cluster`
    total points are dropped entirely (and absent from the returned maps).
    """
    iids = sorted(int(i) for i in inst_pts)
    if not iids:
        return {}, {}, {}, []

    cat = categories or {}

    parent = {i: i for i in iids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if gate == "centroid_obb":
        for i, a in enumerate(iids):
            ca = np.asarray(centroids.get(a))
            ea = obb_extents.get(a)
            if ca is None or ea is None:
                continue
            cat_a = str(cat.get(a, "")).strip().lower()
            for b in iids[i + 1 :]:
                cb = np.asarray(centroids.get(b))
                eb = obb_extents.get(b)
                if cb is None or eb is None:
                    continue
                if float(np.linalg.norm(ca - cb)) >= centroid_thresh_m:
                    continue
                if not _size_within_frac(ea, eb, obb_size_gate_frac):
                    continue
                if require_category_match:
                    cat_b = str(cat.get(b, "")).strip().lower()
                    if cat_a and cat_b and cat_a != cat_b:
                        continue
                union(a, b)
    elif gate == "voxel_chamfer":
        # Cheap O(N) voxel hash per iid, then O(K²) packed-int set intersections.
        voxels: Dict[int, np.ndarray] = {
            iid: _voxel_keys(inst_pts[iid], voxel_size_m) for iid in iids
        }
        # Cheap pass: collect pair candidates that pass vCont.
        candidates: List[Tuple[int, int]] = []
        for i, a in enumerate(iids):
            for b in iids[i + 1 :]:
                if require_category_match:
                    ca = str(cat.get(a, "")).strip().lower()
                    cb = str(cat.get(b, "")).strip().lower()
                    if ca and cb and ca != cb:
                        continue
                _, cont = _voxel_containment(voxels[a], voxels[b])
                if cont >= voxel_containment_thresh:
                    candidates.append((a, b))
        # Build one KDTree per iid that participates in a candidate pair.
        from scipy.spatial import cKDTree  # local import: heavy
        needed = sorted({iid for pair in candidates for iid in pair})
        trees: Dict[int, "cKDTree"] = {}
        pts_for: Dict[int, np.ndarray] = {}
        for iid in needed:
            sub = _subsample(np.asarray(inst_pts[iid], dtype=np.float64), chamfer_max_pts)
            pts_for[iid] = sub
            trees[iid] = cKDTree(sub)

        def _min_cham(pair: Tuple[int, int]) -> Tuple[Tuple[int, int], float]:
            a, b = pair
            d_ab, _ = trees[b].query(pts_for[a], k=1)
            d_ba, _ = trees[a].query(pts_for[b], k=1)
            return pair, float(min(d_ab.mean(), d_ba.mean()))

        if candidates:
            workers = min(_FUSION_THREADS, max(1, len(candidates)))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    cham_results = list(ex.map(_min_cham, candidates))
            else:
                cham_results = [_min_cham(p) for p in candidates]
            for (a, b), cham in cham_results:
                if cham <= chamfer_thresh_m:
                    union(a, b)
    else:
        raise ValueError(f"unknown dedup gate: {gate!r}")

    clusters: Dict[int, List[int]] = {}
    for iid in iids:
        clusters.setdefault(find(iid), []).append(int(iid))

    fused_inst_pts: Dict[int, np.ndarray] = {}
    fused_pf: Dict[int, Dict[int, np.ndarray]] = {}
    alias: Dict[int, int] = {}
    merges: List[Tuple[int, List[int]]] = []

    for canonical, members in clusters.items():
        # Fuse 3D points (simple concat — duplicates across SAM3 instances
        # are rare since each 3D point belongs to one mask per frame).
        chunks = [inst_pts[m] for m in members if m in inst_pts]
        if not chunks:
            continue
        pts = np.concatenate(chunks, axis=0)
        if pts.shape[0] < int(min_points_per_cluster):
            continue
        fused_inst_pts[int(canonical)] = pts
        for m in members:
            alias[int(m)] = int(canonical)
        if len(members) > 1:
            merges.append((int(canonical), sorted(int(m) for m in members)))

    for fid, inst_dict in pf_inst_pts.items():
        out: Dict[int, np.ndarray] = {}
        for iid, pts in inst_dict.items():
            canon = alias.get(int(iid))
            if canon is None:
                continue
            if canon in out:
                out[canon] = np.concatenate([out[canon], pts], axis=0)
            else:
                out[canon] = pts
        if out:
            fused_pf[int(fid)] = out

    return fused_inst_pts, fused_pf, alias, merges


def apply_alias_to_per_frame_masks(
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
    alias: Dict[int, int],
) -> Dict[int, Dict[int, np.ndarray]]:
    """Rewrite per-frame mask iids using `alias`; OR the masks when multiple
    raw iids collapse onto the same canonical id. Raw iids absent from the
    alias are dropped (matches the dedup semantics where unsurvived iids are
    removed entirely)."""
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for fid, inst in per_frame_masks.items():
        merged: Dict[int, np.ndarray] = {}
        for iid, m in inst.items():
            canon = alias.get(int(iid))
            if canon is None:
                continue
            if canon in merged:
                merged[canon] = np.logical_or(merged[canon], m.astype(bool))
            else:
                merged[canon] = m.astype(bool).copy()
        if merged:
            out[int(fid)] = merged
    return out


def apply_alias_to_per_frame_scores(
    per_frame_scores: Dict[int, Dict[int, float]],
    alias: Dict[int, int],
    *,
    reducer: str = "max",
) -> Dict[int, Dict[int, float]]:
    """Rewrite per-frame score iids using `alias`. When multiple raw iids
    collapse onto the same canonical id, combine via `reducer` ('max' or
    'mean').
    """
    out: Dict[int, Dict[int, float]] = {}
    for fid, inst in per_frame_scores.items():
        merged: Dict[int, List[float]] = {}
        for iid, v in inst.items():
            canon = alias.get(int(iid))
            if canon is None:
                continue
            merged.setdefault(canon, []).append(float(v))
        if merged:
            row: Dict[int, float] = {}
            for canon, vals in merged.items():
                if reducer == "mean":
                    row[int(canon)] = float(np.mean(vals))
                else:
                    row[int(canon)] = float(np.max(vals))
            out[int(fid)] = row
    return out


def apply_alias_to_instance_map(
    instance_map: Dict[int, "object"],
    alias: Dict[int, int],
    *,
    on_conflict: str = "first",
) -> Dict[int, "object"]:
    """Rewrite an iid-keyed flat map using `alias`. On collision keep either
    the first value seen (`first`) or the value whose original iid equals the
    canonical id (`canonical`)."""
    out: Dict[int, object] = {}
    for iid, v in instance_map.items():
        canon = alias.get(int(iid))
        if canon is None:
            continue
        if canon not in out:
            out[int(canon)] = v
        elif on_conflict == "canonical" and int(iid) == int(canon):
            out[int(canon)] = v
    return out


def aggregate_per_frame_scores(
    per_frame_scores: Dict[int, Dict[int, float]],
    *,
    instance_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """Aggregate a per-(frame, iid) score map into per-iid summary stats.

    Returned dict maps `iid -> {"mean": ..., "max": ..., "min": ..., "n": ...}`.
    Used by the mosaic-matcher backend to score frame-selection candidates by
    per-instance `predicted_iou` quality and to break Hungarian ties.

    If `instance_ids` is given, iids absent from the per-frame map are emitted
    with stats=NaN (callers can decide how to handle missing data).
    """
    bucket: Dict[int, list] = {}
    for inst in per_frame_scores.values():
        for iid, v in inst.items():
            bucket.setdefault(int(iid), []).append(float(v))

    out: Dict[int, Dict[str, float]] = {}
    if instance_ids is not None:
        for iid in instance_ids:
            iid_int = int(iid)
            vals = bucket.get(iid_int, [])
            out[iid_int] = _stats(vals)
    else:
        for iid, vals in bucket.items():
            out[int(iid)] = _stats(vals)
    return out


def _stats(vals: Sequence[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "max": float("nan"), "min": float("nan"), "n": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "n": float(len(arr)),
    }


# ---------------------------------------------------------------------------
# Scene-graph edges (paper Sec. 3.1: "edges connecting instances with nearby
# centroids"). Connects each instance node to its k nearest neighbours within a
# radius and labels each edge with a coarse spatial relation from the gravity
# axis. Pure numpy.
# ---------------------------------------------------------------------------
def build_scene_graph_edges(
    centroids: Dict[int, np.ndarray],
    instance_ids: Sequence[int],
    *,
    k: int = 5,
    radius: float = 1.0,
    up_axis: str = "y",
    vertical_threshold: float = 0.25,
) -> List[Dict[str, object]]:
    """Build the scene graph's edge set E from instance centroids.

    Each instance is connected to its up-to-``k`` nearest neighbours whose
    centroid lies within ``radius`` metres. Edges are directed (``src`` -> ``dst``)
    and carry a coarse spatial ``relation`` derived from the gravity (``up_axis``)
    direction: ``"above"`` / ``"below"`` when the vertical centroid gap dominates
    and exceeds ``vertical_threshold`` metres, otherwise ``"near"``.

    Returns a list of ``{"src", "dst", "dist", "relation"}`` dicts.
    """
    axis = {"x": 0, "y": 1, "z": 2}.get(str(up_axis).lower(), 1)
    ids = [int(i) for i in instance_ids if int(i) in centroids]
    edges: List[Dict[str, object]] = []
    if len(ids) < 2:
        return edges
    C = np.stack([np.asarray(centroids[i], dtype=np.float64) for i in ids], axis=0)
    for a in range(len(ids)):
        d = np.linalg.norm(C - C[a], axis=1)
        d[a] = np.inf
        order = np.argsort(d)
        n = 0
        for b in order:
            if n >= k or not np.isfinite(d[b]) or d[b] > radius:
                break
            dv = float(C[a, axis] - C[b, axis])              # +ve => a is higher
            horiz = float(np.sqrt(max(d[b] ** 2 - dv ** 2, 0.0)))
            if abs(dv) >= vertical_threshold and abs(dv) >= horiz:
                rel = "above" if dv > 0 else "below"
            else:
                rel = "near"
            edges.append({
                "src": ids[a], "dst": ids[b],
                "dist": round(float(d[b]), 4), "relation": rel,
            })
            n += 1
    return edges
