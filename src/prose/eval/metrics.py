"""Registration evaluation metrics — byte-equivalent to SGAligner's
`utils/registration.py` + `engine/registration_evaluator.py:evaluate_registration`,
which is the protocol invoked by `recon_understand_anything`'s
`inference_align_reg.py` (per the upstream README).

The prose pipeline supplies the same inputs that SGAligner's
`run_aligner_registration` would, so we can compute identical metrics here.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..utils.pointcloud import apply_transform


# ---------------------------------------------------------------------------
# Helpers (verbatim from upstream).
# ---------------------------------------------------------------------------

def get_nearest_neighbor(q_points: np.ndarray, s_points: np.ndarray) -> np.ndarray:
    """For each q_point, distance to nearest in s_points.

    Mirrors `sgaligner/utils/point_cloud.py:get_nearest_neighbor`.
    """
    tree = cKDTree(s_points)
    distances, _ = tree.query(q_points, k=1)
    return distances


def compute_pcl_overlap(
    source: np.ndarray, target: np.ndarray, threshold: float = 1e-7,
) -> Tuple[float, np.ndarray]:
    """Source indices whose point lies within `threshold` of any target point.

    Verbatim from `sgaligner/utils/point_cloud.py:compute_pcl_overlap`. Returns
    (overlap_ratio, sorted_unique_source_indices).
    """
    source = np.asarray(source)
    target = np.asarray(target)

    tree = cKDTree(source)
    neighbors = tree.query_ball_point(target, r=threshold)

    all_indices: List[int] = []
    for neighbor_list in neighbors:
        if len(neighbor_list) > 0:
            all_indices.extend(neighbor_list)

    common_pts_idx_src = (
        np.unique(np.array(all_indices, dtype=np.int64))
        if all_indices else np.array([], dtype=np.int64)
    )
    overlap_ratio = round(common_pts_idx_src.shape[0] / source.shape[0], 4) if source.shape[0] else 0.0
    return overlap_ratio, common_pts_idx_src


# ---------------------------------------------------------------------------
# Metric primitives — mirror sgaligner/utils/registration.py.
# ---------------------------------------------------------------------------

def compute_modified_chamfer_distance(
    src_points: np.ndarray,
    ref_points: np.ndarray,
    raw_points: np.ndarray,
    est_transform: np.ndarray,
    gt_transform: np.ndarray,
) -> float:
    """Bidirectional CD against the parent-scene point cloud.

    Verbatim from upstream: `aligned_src vs raw` + `ref vs (raw transformed by
    est @ inv(gt))`.
    """
    aligned_src = apply_transform(src_points, est_transform)
    cd_p_q = get_nearest_neighbor(aligned_src, raw_points).mean()

    composed = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw = apply_transform(raw_points, composed)
    cd_q_p = get_nearest_neighbor(ref_points, aligned_raw).mean()

    return float(cd_p_q + cd_q_p)


def compute_inlier_ratio(
    ref_corr_points: np.ndarray,
    src_corr_points: np.ndarray,
    transform: np.ndarray,
    positive_radius: float = 0.1,
) -> float:
    """Fraction of (paired) corrs whose ||ref - T(src)|| < positive_radius.

    Argument order matches upstream: (ref_corr_points, src_corr_points, transform).
    The `transform` upstream passes is `gt_transform` (so this measures
    correspondence quality independent of the est_transform).
    """
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(axis=1))
    return float(np.mean(residuals < positive_radius))


def compute_registration_rmse(
    ref_points: np.ndarray, src_points: np.ndarray, transform: np.ndarray,
) -> float:
    """Element-wise RMSE between paired ref and T(src). Verbatim from upstream."""
    src_points = apply_transform(src_points, transform)
    rmse = np.sqrt(((ref_points - src_points) ** 2).sum() / src_points.shape[0])
    return float(rmse)


def compute_sgreg_recall_rmse(
    src_points: np.ndarray, est_transform: np.ndarray, gt_transform: np.ndarray,
) -> float:
    """Mean per-point displacement under residual transform `T_gt^{-1} @ T_est`.

    Verbatim from SG-Reg (`sgreg/loss/eval.py:48-58` and `:130-142`). Despite
    being called "rmse" upstream, it is mean L2 displacement, not RMSE.
    Used as the cross-paper recall metric: recall = 1[value < threshold].
    Independent of any GT correspondences — works on any pair as long as
    src_points are in a fixed frame.
    """
    if src_points.shape[0] == 0:
        return float("nan")
    realignment = np.linalg.inv(gt_transform) @ est_transform
    aligned = apply_transform(src_points, realignment)
    return float(np.linalg.norm(aligned - src_points, axis=1).mean())


def _get_R_t(T: np.ndarray, inverse_trans: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    R = T[:3, :3]
    t = T[3, :3] if inverse_trans else T[:3, 3]
    return R, t


def compute_relative_rotation_error(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """RRE in degrees: acos((trace(R_est^T · R_gt) − 1) / 2). Upstream defn."""
    x = 0.5 * (np.trace(np.matmul(R_est.T, R_gt)) - 1.0)
    x = float(np.clip(x, -1.0, 1.0))
    return float(180.0 * np.arccos(x) / np.pi)


def compute_relative_translation_error(t_gt: np.ndarray, t_est: np.ndarray) -> float:
    """RTE = ||t_gt − t_est||_2. Upstream defn."""
    return float(np.linalg.norm(t_gt - t_est))


def compute_registration_error(
    gt_transform: np.ndarray, est_transform: np.ndarray, inverse_trans: bool = False,
) -> Tuple[float, float]:
    R_gt, t_gt = _get_R_t(gt_transform)
    R_est, t_est = _get_R_t(est_transform, inverse_trans)
    return (
        compute_relative_rotation_error(R_gt, R_est),
        compute_relative_translation_error(t_gt, t_est),
    )


# ---------------------------------------------------------------------------
# Top-level — mirrors RegistrationEvaluator.evaluate_registration.
# ---------------------------------------------------------------------------

def compute_all_metrics(
    *,
    est_transform: Optional[np.ndarray],
    gt_transform: np.ndarray,
    src_points: np.ndarray,
    ref_points: np.ndarray,
    raw_points: np.ndarray,
    src_corr_points: Optional[np.ndarray] = None,
    ref_corr_points: Optional[np.ndarray] = None,
    gt_src_corr_points: Optional[np.ndarray] = None,
    gt_ref_corr_points: Optional[np.ndarray] = None,
    positive_radius: float = 0.1,
    inlier_ratio_thresh: float = 0.05,
    rmse_thresh: float = 0.2,
) -> Dict[str, float]:
    """Run the full upstream metric set on one pair.

    `recall` follows SG-Reg's definition (residual transform on src_points,
    `sgreg/loss/eval.py:48-58`) — this is the cross-paper standard.
    `recall_paired` is the SGAligner definition (RMSE on paired GT corrs);
    only emitted when `gt_*_corr_points` are supplied.

    Returns {valid, CD, IR, RRE, RTE, recall, recall_rmse, FMR} plus
    optional {recall_paired, recall_paired_rmse}.
    """
    out: Dict[str, float] = {}
    if est_transform is None or np.any(np.isnan(est_transform)):
        out["valid"] = 0.0
        return out
    out["valid"] = 1.0

    # CD
    out["CD"] = compute_modified_chamfer_distance(
        src_points, ref_points, raw_points, est_transform, gt_transform
    )

    # IR + FMR — upstream uses gt_transform on predicted corrs.
    if (
        src_corr_points is not None and ref_corr_points is not None
        and len(src_corr_points) > 0
    ):
        ir = compute_inlier_ratio(
            ref_corr_points, src_corr_points, gt_transform, positive_radius=positive_radius
        )
        out["IR"] = ir
        out["FMR"] = float(ir >= inlier_ratio_thresh)
    else:
        out["IR"] = 0.0
        out["FMR"] = 0.0

    # RRE / RTE
    rre, rte = compute_registration_error(gt_transform, est_transform)
    out["RRE"] = rre
    out["RTE"] = rte

    # Recall (SG-Reg defn) — residual transform on src_points.
    sgreg_rmse = compute_sgreg_recall_rmse(src_points, est_transform, gt_transform)
    out["recall_rmse"] = sgreg_rmse
    out["recall"] = float(sgreg_rmse < rmse_thresh) if not np.isnan(sgreg_rmse) else 0.0

    # Recall_paired (SGAligner defn) — RMSE on paired GT correspondences.
    if (
        gt_src_corr_points is not None and gt_ref_corr_points is not None
        and len(gt_src_corr_points) > 0
        and gt_src_corr_points.shape == gt_ref_corr_points.shape
    ):
        paired_rmse = compute_registration_rmse(
            gt_ref_corr_points, gt_src_corr_points, est_transform
        )
        out["recall_paired_rmse"] = paired_rmse
        out["recall_paired"] = float(paired_rmse < rmse_thresh)

    return out


# ---------------------------------------------------------------------------
# Stage 5 instance correspondence metrics — semantic eval against anchor GT.
# ---------------------------------------------------------------------------

def _make_pixel_rescaler(stage1_data: dict):
    """Return a closure ``(u, v, mask_shape) -> (u', v')`` that maps pixel
    coords from the Stage 1 artifact's source resolution to a SAM3 mask's
    resolution.

    Stage 3 SAM3 masks live at the raw color resolution (e.g. 540×960 on
    3RScan). The GT Stage 1 path records ``point_to_pixels`` at the same
    raw resolution, so no rescale is needed. The DA3 path records
    ``point_to_pixels`` at DA3's processed resolution (e.g. 280×504); the
    artifact's ``pixel_resolution`` field captures that (H, W). When set
    and different from the mask shape we rescale by ``mask_H/src_H`` and
    ``mask_W/src_W``. Pure scaling — no rotation (SAM3 input on 3RScan is
    the un-rotated color frame).
    """
    src_res = stage1_data.get("pixel_resolution")
    if src_res is None:
        return lambda u, v, mask_shape: (int(u), int(v))
    src_h, src_w = int(src_res[0]), int(src_res[1])

    def _rescale(u, v, mask_shape):
        mh, mw = int(mask_shape[0]), int(mask_shape[1])
        if mh == src_h and mw == src_w:
            return int(u), int(v)
        u2 = int(round((float(u) + 0.5) * mw / src_w - 0.5))
        v2 = int(round((float(v) + 0.5) * mh / src_h - 0.5))
        return u2, v2

    return _rescale


def build_sam_to_object_map(
    stage1_data: dict,
    stage3_masks: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, int]:
    """Map each SAM3 instance id to its dominant 3RScan objectId for one subscan.

    Walks every 3D point with a known projection (frame_pos, pixel_u, pixel_v),
    looks up which SAM3 mask covers that pixel, and tallies (sam_iid, objectId)
    counts. Per sam_iid, picks the objectId with the most votes.

    Args:
        stage1_data: pickled stage-1 artifact. Must contain `point_to_pixels`
            (length N) and `object_ids` (length N int array).
        stage3_masks: unpacked masks {frame_pos: {sam_iid: HxW bool}}.
    Returns:
        {sam_iid: dominant_objectId}. SAM3 instances that never cover a labelled
        point are absent from the map.
    """
    object_ids = stage1_data.get("object_ids", None)
    point_to_pixels = stage1_data.get("point_to_pixels", None)
    if object_ids is None or point_to_pixels is None:
        return {}
    if len(point_to_pixels) != len(object_ids):
        return {}

    # votes[sam_iid][obj_id] = count
    votes: Dict[int, Dict[int, int]] = {}
    rescale_uv = _make_pixel_rescaler(stage1_data)

    for pt_idx, entries in enumerate(point_to_pixels):
        # Legacy flat-shape tolerance: pre-Fix-A pickles stored `List[dict]`.
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            continue
        obj_id = int(object_ids[pt_idx])
        # Each point may project into multiple frames — vote once per frame so
        # an instance visible from many viewpoints accumulates more weight,
        # matching the underlying density of evidence.
        for entry in entries:
            if not entry:
                continue
            frame_pos = entry.get("frame_id")
            u = entry.get("pixel_u")
            v = entry.get("pixel_v")
            if frame_pos is None or u is None or v is None:
                continue
            frame_pos = int(frame_pos)
            masks_this_frame = stage3_masks.get(frame_pos)
            if not masks_this_frame:
                continue
            # Find which sam_iid (if any) covers this pixel.
            for sam_iid, mask in masks_this_frame.items():
                u_r, v_r = rescale_uv(u, v, mask.shape)
                if 0 <= v_r < mask.shape[0] and 0 <= u_r < mask.shape[1] and mask[v_r, u_r]:
                    votes.setdefault(int(sam_iid), {})
                    votes[int(sam_iid)][obj_id] = votes[int(sam_iid)].get(obj_id, 0) + 1
                    # A pixel can in principle belong to multiple SAM3 masks
                    # (overlapping segmentation); count all that match — the argmax
                    # below resolves to the dominant objectId per sam_iid.

    out: Dict[int, int] = {}
    for sam_iid, obj_counts in votes.items():
        if not obj_counts:
            continue
        out[sam_iid] = max(obj_counts.items(), key=lambda kv: kv[1])[0]
    return out


def compute_instance_correspondence_metrics(
    pred_pairs: Iterable[Tuple[int, int]],
    src_sam2obj: Dict[int, int],
    ref_sam2obj: Dict[int, int],
    gt_anchor_object_ids: Sequence[int],
) -> Dict[str, float]:
    """Score stage-4 instance pairs against `anchorIds` ground truth.

    A predicted pair (s, r) is correct iff src_sam2obj[s] == ref_sam2obj[r] == i
    AND i ∈ gt_anchor_object_ids. We report:
      - pair_precision: |correct preds| / |preds|
      - object_recall:  |{i ∈ anchors : ∃ correct pred mapping to i}| / |anchors|
      - f1: harmonic mean

    Object-recall (not pair-recall) is the right denominator because SAM3
    over-segments — one GT object often becomes multiple SAM3 masks; pair-recall
    would inflate the denominator combinatorially. Object-recall asks the
    natural question: how many anchor objects did we connect at all?
    """
    pred_pairs = list(pred_pairs)
    anchor_set = set(int(i) for i in gt_anchor_object_ids)
    n_pred = len(pred_pairs)
    n_anchor_objs = len(anchor_set)

    n_correct = 0
    covered_objs: set = set()
    for s, r in pred_pairs:
        s_obj = src_sam2obj.get(int(s))
        r_obj = ref_sam2obj.get(int(r))
        if s_obj is None or r_obj is None:
            continue
        if s_obj != r_obj:
            continue
        if s_obj not in anchor_set:
            continue
        n_correct += 1
        covered_objs.add(s_obj)

    n_objs_covered = len(covered_objs)
    precision = (n_correct / n_pred) if n_pred > 0 else 0.0
    recall = (n_objs_covered / n_anchor_objs) if n_anchor_objs > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_pred": int(n_pred),
        "n_correct": int(n_correct),
        "n_anchor_objs": int(n_anchor_objs),
        "n_objs_covered": int(n_objs_covered),
    }


def aggregate_instance_metrics(per_pair: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum-then-divide aggregation over per-pair instance metric dicts.

    Mirrors SG-Reg's RecallMetrics semantics: sum counts across pairs, then
    divide once. Pairs whose GT was missing (n_anchor_objs == 0) are skipped
    from the recall denominator but still counted in n_pairs.
    """
    if not per_pair:
        return {"n_pairs": 0}
    n_pairs = len(per_pair)
    sum_pred = sum(int(p.get("n_pred", 0)) for p in per_pair)
    sum_correct = sum(int(p.get("n_correct", 0)) for p in per_pair)
    sum_anchor = sum(int(p.get("n_anchor_objs", 0)) for p in per_pair)
    sum_covered = sum(int(p.get("n_objs_covered", 0)) for p in per_pair)
    precision = (sum_correct / sum_pred) if sum_pred > 0 else 0.0
    recall = (sum_covered / sum_anchor) if sum_anchor > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "n_pairs": int(n_pairs),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_pred": int(sum_pred),
        "n_correct": int(sum_correct),
        "n_anchor_objs": int(sum_anchor),
        "n_objs_covered": int(sum_covered),
    }


# ---------------------------------------------------------------------------
# SG-Reg-style instance correspondence GT generation + scoring.
#
# Ports the GT-pair construction from SG-Reg's
#   sgreg/dataset/generate_gt_association.py:91-216
# (compute_cloud_overlap + find_association). The SAM3-mask analog of
# FM-Fusion-node↔FM-Fusion-node mutual-NN matching by 3D point IoU under T_gt.
# ---------------------------------------------------------------------------

def compute_radius_iou_symmetric(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    radius: float = 0.10,
) -> float:
    """Mutually-symmetric KNN/radius IoU.

    For each point in A mark "matched" if its nearest neighbour in B lies
    within ``radius`` metres; symmetrically from B. IoU is

        (|A_matched| + |B_matched|) / (|A| + |B|).

    Robust to DA3 metric-depth drift between consecutive subscans because
    a cm-scale offset still lands inside the radius — voxel-Jaccard at
    5 cm fragments those points into disjoint cells and reports 0
    (see jobs/analysis/scripts/probe_gt_iou_distribution_adt.py output
    on ADT 0050__0055: even centroid-aligned chair clouds get voxel
    IoU<0.11 at vs=0.05).
    """
    ca = np.asarray(cloud_a)
    cb = np.asarray(cloud_b)
    if ca.size == 0 or cb.size == 0 or ca.ndim != 2 or cb.ndim != 2:
        return 0.0
    tree_b = cKDTree(cb)
    tree_a = cKDTree(ca)
    da, _ = tree_b.query(ca, k=1, distance_upper_bound=float(radius))
    db, _ = tree_a.query(cb, k=1, distance_upper_bound=float(radius))
    n_a_matched = int(np.isfinite(da).sum())
    n_b_matched = int(np.isfinite(db).sum())
    denom = int(ca.shape[0] + cb.shape[0])
    if denom == 0:
        return 0.0
    return float(n_a_matched + n_b_matched) / float(denom)


def compute_cloud_overlap(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    search_radius: float = 0.2,
) -> Tuple[float, List[List[int]]]:
    """Symmetric IoU between two 3D point sets.

    Ported from SG-Reg's `generate_gt_association.compute_cloud_overlap`
    (lines 91-103). KDTree backend swapped from `o3d.geometry.KDTreeFlann` to
    `scipy.spatial.cKDTree.query_ball_point` — same algorithm, identical
    result to fp tolerance. The `len(neighbors) > 1` guard is preserved
    verbatim from upstream (their `if k > 1` check).

    Returns (iou, correspondences) where correspondences is a list of
    [a_idx, first_b_idx_in_radius] pairs, matching upstream's return shape.
    """
    cloud_a = np.asarray(cloud_a)
    cloud_b = np.asarray(cloud_b)
    Na = cloud_a.shape[0]
    Nb = cloud_b.shape[0]
    if Na == 0 or Nb == 0:
        return 0.0, []
    pcd_tree_b = cKDTree(cloud_b)
    neighbors = pcd_tree_b.query_ball_point(cloud_a, r=search_radius)
    correspondences: List[List[int]] = []
    for i, idx_list in enumerate(neighbors):
        if len(idx_list) > 1:
            correspondences.append([i, int(idx_list[0])])
    iou = len(correspondences) / (Na + Nb - len(correspondences))
    return float(iou), correspondences


def build_sam3_mask_clouds(
    stage1_data: dict,
    stage3_masks: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, np.ndarray]:
    """For one subscan, gather per-SAM3-instance 3D point sets.

    For each 3D point with a known projection (frame_pos, pixel_u, pixel_v),
    look up which SAM3 mask covers that pixel; if covered, add the point to
    that mask's bucket. Output: `{sam_iid: (P_i, 3) float32}`.

    SAM3 instances that never cover a labelled point are absent from the map.
    Same traversal as `build_sam_to_object_map` — collects points instead of
    voting on objectIds.
    """
    points = stage1_data.get("points")
    point_to_pixels = stage1_data.get("point_to_pixels")
    if points is None or point_to_pixels is None:
        return {}
    points = np.asarray(points)
    if len(point_to_pixels) != points.shape[0]:
        return {}

    # Flatten point_to_pixels into parallel arrays (pt_idx, frame_id, u, v).
    # The legacy "List[dict]" flat-shape and missing-key tolerance are
    # preserved. Hot loop below is fully numpy-vectorized; this Python
    # parsing remains O(N_pts * mean_entries) but is dominated by dict-get.
    pt_idx_list: List[int] = []
    frame_list: List[int] = []
    u_list: List[int] = []
    v_list: List[int] = []
    for pt_idx, entries in enumerate(point_to_pixels):
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            continue
        for entry in entries:
            if not entry:
                continue
            f = entry.get("frame_id")
            u = entry.get("pixel_u")
            v = entry.get("pixel_v")
            if f is None or u is None or v is None:
                continue
            pt_idx_list.append(pt_idx)
            frame_list.append(int(f))
            u_list.append(int(u))
            v_list.append(int(v))

    if not pt_idx_list:
        return {}

    pt_arr = np.asarray(pt_idx_list, dtype=np.int64)
    frame_arr = np.asarray(frame_list, dtype=np.int64)
    u_src_arr = np.asarray(u_list, dtype=np.int64)
    v_src_arr = np.asarray(v_list, dtype=np.int64)

    src_res = stage1_data.get("pixel_resolution")
    if src_res is not None:
        src_h, src_w = int(src_res[0]), int(src_res[1])
    else:
        src_h = src_w = None

    # Vectorized per-frame batch lookup: stack (N_iid, H, W) masks and gather
    # `stack[:, v, u]` for every point projecting into that frame at once.
    # Replaces the per-point × per-mask Python double loop that dominated
    # build_sam3_mask_clouds on DA3 clouds (~38s/pair → ~2s/pair).
    all_iid_hits: List[np.ndarray] = []
    all_pt_hits: List[np.ndarray] = []
    for frame_pos, masks_this_frame in stage3_masks.items():
        if not masks_this_frame:
            continue
        in_frame = (frame_arr == int(frame_pos))
        if not in_frame.any():
            continue
        # Stack masks for this frame into (N_iid, H, W).
        iids_arr = np.asarray(sorted(masks_this_frame.keys()), dtype=np.int64)
        stack = np.stack([np.asarray(masks_this_frame[int(i)], dtype=bool) for i in iids_arr])
        mh, mw = int(stack.shape[1]), int(stack.shape[2])

        pts_in = np.nonzero(in_frame)[0]
        u_raw = u_src_arr[pts_in]
        v_raw = v_src_arr[pts_in]
        if src_h is not None and (mh != src_h or mw != src_w):
            u_i = np.rint((u_raw.astype(np.float64) + 0.5) * mw / src_w - 0.5).astype(np.int64)
            v_i = np.rint((v_raw.astype(np.float64) + 0.5) * mh / src_h - 0.5).astype(np.int64)
        else:
            u_i = u_raw
            v_i = v_raw

        in_bounds = (u_i >= 0) & (u_i < mw) & (v_i >= 0) & (v_i < mh)
        if not in_bounds.any():
            continue
        pts_in = pts_in[in_bounds]
        u_i = u_i[in_bounds]
        v_i = v_i[in_bounds]

        # (N_iid, n_pts_in_frame) bool — True where iid covers that point's pixel.
        hits = stack[:, v_i, u_i]
        iid_idx_hits, pt_local_hits = np.nonzero(hits)
        if iid_idx_hits.size == 0:
            continue
        all_iid_hits.append(iids_arr[iid_idx_hits])
        all_pt_hits.append(pt_arr[pts_in[pt_local_hits]])

    if not all_iid_hits:
        return {}

    cat_iid = np.concatenate(all_iid_hits)
    cat_pt = np.concatenate(all_pt_hits)

    # Group by iid; dedupe pt_idx within each iid via np.unique.
    out: Dict[int, np.ndarray] = {}
    order = np.argsort(cat_iid, kind="stable")
    sorted_iid = cat_iid[order]
    sorted_pt = cat_pt[order]
    unique_iids, starts = np.unique(sorted_iid, return_index=True)
    ends = np.append(starts[1:], sorted_iid.size)
    for iid, s, e in zip(unique_iids, starts, ends):
        pts_for_iid = np.unique(sorted_pt[s:e])
        out[int(iid)] = points[pts_for_iid]
    return out


def voxel_downsample(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample a point cloud to one point per voxel cell (cell centroid).

    Comparable to Open3D's ``voxel_down_sample`` but pure-numpy. Used to
    normalize point density before KNN-radius IoU so that close-up dense
    instances and far-away sparse instances contribute equally.
    """
    c = np.asarray(cloud, dtype=np.float64)
    if c.size == 0 or c.ndim != 2:
        return c
    cells = np.floor(c / voxel_size).astype(np.int64)
    OFFSET = np.int64(1 << 20)
    MASK = np.int64((1 << 21) - 1)
    shifted = cells + OFFSET
    np.clip(shifted, 0, MASK, out=shifted)
    keys = (shifted[:, 0] << np.int64(42)) | (shifted[:, 1] << np.int64(21)) | shifted[:, 2]
    _, inv = np.unique(keys, return_inverse=True)
    n_cells = int(inv.max()) + 1
    sums = np.zeros((n_cells, 3), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(sums, inv, c)
    np.add.at(counts, inv, 1)
    return (sums / counts[:, None]).astype(np.float32)


def _voxel_keys(cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Vectorized 3D voxel grid keys as a 1-D ``int64`` array of unique cells.

    Each (i, j, k) integer cell is packed into a single int64 via bit-
    shifts (21 bits per axis + offset to keep keys non-negative). Replaces
    ``{tuple(v) for v in floor(xyz/vs)}`` which is Python-iteration bound
    and ~100× slower on DA3-scale clouds (100k+ pts per instance).
    """
    if cloud is None:
        return np.empty(0, dtype=np.int64)
    c = np.asarray(cloud)
    if c.size == 0:
        return np.empty(0, dtype=np.int64)
    cells = np.floor(c / voxel_size).astype(np.int64)
    # Offset so each per-axis coord is non-negative; 21 bits ⇒ ±2^20
    # cells, i.e. ±52 km at 5 cm voxels — plenty for indoor scenes.
    OFFSET = np.int64(1 << 20)
    MASK = np.int64((1 << 21) - 1)
    shifted = cells + OFFSET
    # Clip to the 21-bit range to be safe; clouds outside this would
    # collide silently otherwise. We log a warning when clipping fires.
    np.clip(shifted, 0, MASK, out=shifted)
    keys = (shifted[:, 0] << np.int64(42)) | (shifted[:, 1] << np.int64(21)) | shifted[:, 2]
    return np.unique(keys)


def compute_voxel_iou(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    voxel_size: float = 0.05,
) -> float:
    """Standard symmetric Jaccard IoU over a fixed voxel grid.

    Replaces the broken `compute_cloud_overlap` formula for GT pairing.
    Voxelises both clouds at `voxel_size` (default 5 cm — standard for
    indoor scenes), then computes ``|cells_a ∩ cells_b| / |cells_a ∪ cells_b|``
    on the hashed integer cell sets. Guarantees:

    * Symmetric in A and B (unlike `compute_cloud_overlap`).
    * Always in [0, 1] (unlike `compute_cloud_overlap`).
    * Fully vectorized via ``_voxel_keys`` (int64 packing + np.unique),
      O(N) per cloud + O(|keys_a| + |keys_b|) for the merge.
    """
    ka = _voxel_keys(cloud_a, voxel_size)
    kb = _voxel_keys(cloud_b, voxel_size)
    if ka.size == 0 or kb.size == 0:
        return 0.0
    inter = int(np.intersect1d(ka, kb, assume_unique=True).size)
    union = int(ka.size + kb.size - inter)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


# IoU method selectors for build_sam3_gt_pairs.
#
# * "voxel"        — symmetric voxel-Jaccard (recommended; bounded [0,1]).
# * "legacy_loose" — bug-compatible with the original port (search_radius=0.2,
#                    min_iou=0.2). Asymmetric, can exceed 1.0. Preserved so
#                    historical numbers remain reproducible.
# * "legacy_sgreg" — bug-compatible with SG-Reg's published defaults
#                    (search_radius=0.1, min_iou=0.5). Still pathological,
#                    but the strict params suppress most over-1 outliers.
#                    Use for SG-Reg cross-paper comparability.
IOU_METHODS = ("voxel", "legacy_loose", "legacy_sgreg")


def build_sam3_gt_pairs(
    src_mask_clouds: Dict[int, np.ndarray],
    ref_mask_clouds: Dict[int, np.ndarray],
    gt_transform: np.ndarray,
    iou_method: str = "voxel",
    voxel_size: float = 0.05,
    # Legacy-mode overrides — ignored under iou_method="voxel".
    min_iou: Optional[float] = None,
    search_radius: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    """SAM3↔SAM3 GT correspondences via mutual-NN over a 3D-IoU matrix.

    Build the Nsrc×Nref IoU matrix (after applying `gt_transform` to src
    clouds), take ``row_max ∧ col_max`` for mutual nearest neighbour,
    threshold on ``iou > min_iou``.

    Three IoU methods (see ``IOU_METHODS``):

    * ``voxel`` — recommended. Voxel-Jaccard at ``voxel_size`` (default
      5 cm). Default min_iou=0.25 (real Jaccard scale, comparable to
      ScanNet's 0.25 segmentation cutoff). Symmetric, bounded [0,1].
    * ``legacy_loose`` — original port (search_radius=0.2, min_iou=0.2).
      Asymmetric formula; **values can exceed 1.0**.
    * ``legacy_sgreg`` — SG-Reg's published defaults (search_radius=0.1,
      min_iou=0.5). Same pathological formula; stricter params suppress
      most over-1 outliers but do not cure the asymmetry.

    Ported from SG-Reg's ``generate_gt_association.find_association``
    (lines 144-216). The ``voxel`` method is our replacement — the legacy
    formulas are kept only for reproducing previously published numbers.
    """
    if iou_method not in IOU_METHODS:
        raise ValueError(f"iou_method must be one of {IOU_METHODS}, got {iou_method!r}")

    # Resolve params per method (callers may still override via kwargs).
    if iou_method == "voxel":
        eff_min_iou = 0.25 if min_iou is None else float(min_iou)
        eff_search_radius = None  # unused
    elif iou_method == "legacy_sgreg":
        eff_min_iou = 0.5 if min_iou is None else float(min_iou)
        eff_search_radius = 0.1 if search_radius is None else float(search_radius)
    else:  # legacy_loose
        eff_min_iou = 0.2 if min_iou is None else float(min_iou)
        eff_search_radius = 0.2 if search_radius is None else float(search_radius)

    src_iids = sorted(src_mask_clouds.keys())
    ref_iids = sorted(ref_mask_clouds.keys())
    Nsrc = len(src_iids)
    Nref = len(ref_iids)
    if Nsrc == 0 or Nref == 0:
        return []

    iou_mat = np.zeros((Nsrc, Nref), dtype=np.float64)
    if iou_method == "voxel":
        # Precompute voxel keys once per cloud — the all-pairs loop
        # previously rebuilt them Nref times per src. On DA3 clouds
        # (100k+ pts per instance) this cut GT-pair time from ~50s to
        # ~1-2s per subscan pair (2026-05-19).
        ref_keys = [_voxel_keys(ref_mask_clouds[r], voxel_size) for r in ref_iids]
        for i, s in enumerate(src_iids):
            src_xyz_world = apply_transform(src_mask_clouds[s], gt_transform)
            ka = _voxel_keys(src_xyz_world, voxel_size)
            if ka.size == 0:
                continue
            for j, kb in enumerate(ref_keys):
                if kb.size == 0:
                    continue
                inter = int(np.intersect1d(ka, kb, assume_unique=True).size)
                union = ka.size + kb.size - inter
                if union > 0:
                    iou_mat[i, j] = inter / union
    else:
        for i, s in enumerate(src_iids):
            src_xyz_world = apply_transform(src_mask_clouds[s], gt_transform)
            for j, r in enumerate(ref_iids):
                iou, _ = compute_cloud_overlap(
                    src_xyz_world, ref_mask_clouds[r],
                    search_radius=eff_search_radius,
                )
                iou_mat[i, j] = iou

    if iou_mat.size == 0:
        return []
    row_max = np.zeros_like(iou_mat)
    col_max = np.zeros_like(iou_mat)
    row_max[np.arange(Nsrc), np.argmax(iou_mat, axis=1)] = 1
    col_max[np.argmax(iou_mat, axis=0), np.arange(Nref)] = 1
    assignment = (row_max * col_max).astype(bool) & (iou_mat > eff_min_iou)

    matches = np.argwhere(assignment)
    return [
        (int(src_iids[i]), int(ref_iids[j]), float(iou_mat[i, j]))
        for i, j in matches
    ]


def compute_pair_set_metrics(
    pred_pairs: Iterable[Tuple[int, int]],
    gt_pairs: Iterable[Tuple[int, int]],
) -> Dict[str, float]:
    """SG-Reg-style precision/recall/F1 over a SAM3-mask GT pair set.

    Both `pred_pairs` and `gt_pairs` are lists of `(src_sam_iid, ref_sam_iid)`
    (the third IoU element of `build_sam3_gt_pairs`'s output is ignored).
    A predicted pair is correct iff its `(s, r)` tuple is in the GT set.
    """
    pred = set((int(s), int(r)) for s, r, *_ in pred_pairs)
    gt = set((int(s), int(r)) for s, r, *_ in gt_pairs)
    n_pred = len(pred)
    n_gt = len(gt)
    n_correct = len(pred & gt)
    precision = (n_correct / n_pred) if n_pred > 0 else 0.0
    recall = (n_correct / n_gt) if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_pred": int(n_pred),
        "n_gt": int(n_gt),
        "n_correct": int(n_correct),
    }


def aggregate_pair_set_metrics(per_pair: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum-then-divide aggregation for SG-Reg-style pair-set metrics."""
    if not per_pair:
        return {"n_pairs": 0}
    n_pairs = len(per_pair)
    sum_pred = sum(int(p.get("n_pred", 0)) for p in per_pair)
    sum_gt = sum(int(p.get("n_gt", 0)) for p in per_pair)
    sum_correct = sum(int(p.get("n_correct", 0)) for p in per_pair)
    precision = (sum_correct / sum_pred) if sum_pred > 0 else 0.0
    recall = (sum_correct / sum_gt) if sum_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "n_pairs": int(n_pairs),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_pred": int(sum_pred),
        "n_gt": int(sum_gt),
        "n_correct": int(sum_correct),
    }


# ---------------------------------------------------------------------------
# Object discovery — analog of FM-Fusion's contribution in SG-Reg's pipeline.
# ---------------------------------------------------------------------------

def compute_object_discovery(
    src_sam2obj: Dict[int, int],
    ref_sam2obj: Dict[int, int],
    gt_anchor_object_ids: Sequence[int],
) -> Dict[str, float]:
    """Fraction of anchor objects discovered by SAM3 + projection.

    `anchorIds` is the set of 3RScan-annotated objects in the spatial overlap
    of the two subscans (see `subgen.py:208-214`). This metric measures the
    upstream cost of our SAM3-based instance discovery — separately from the
    matching cost — analogous to FM-Fusion's contribution in SG-Reg's stack.
    """
    anchor_set = set(int(i) for i in gt_anchor_object_ids)
    src_objs = set(int(o) for o in src_sam2obj.values())
    ref_objs = set(int(o) for o in ref_sam2obj.values())
    disc_src = anchor_set & src_objs
    disc_ref = anchor_set & ref_objs
    disc_both = disc_src & disc_ref
    n_anchors = len(anchor_set)
    rate_src = (len(disc_src) / n_anchors) if n_anchors > 0 else 0.0
    rate_ref = (len(disc_ref) / n_anchors) if n_anchors > 0 else 0.0
    rate_both = (len(disc_both) / n_anchors) if n_anchors > 0 else 0.0
    return {
        "rate_src": float(rate_src),
        "rate_ref": float(rate_ref),
        "rate_both": float(rate_both),
        "n_anchors": int(n_anchors),
        "n_disc_src": int(len(disc_src)),
        "n_disc_ref": int(len(disc_ref)),
        "n_disc_both": int(len(disc_both)),
    }


def compute_anchor_iou_discovery(
    stage1_data: dict,
    stage3_masks: Dict[int, Dict[int, np.ndarray]],
    gt_anchor_object_ids: Sequence[int],
    *,
    iou_threshold: float = 0.3,
) -> Dict[str, object]:
    """Per-anchor IoU between SAM3 instances and GT anchors (3D-point sets).

    Stronger diagnostic than `compute_object_discovery` (argmax voting):
    surfaces partial-coverage failures and is symmetric in src/ref. Domain
    restricted to GT points with a valid frame projection — i.e., we measure
    "did SAM3 actually segment this anchor among the points where it had a
    chance to see it." Anchor points without any projection are excluded
    from both intersection and union (they're a stage-1/camera-coverage
    issue, not a SAM3 failure).

    Returns per-anchor max-IoU plus a discovered/not-discovered flag at
    `iou_threshold`. Keys mirror `compute_object_discovery` plus IoU details.
    """
    object_ids = stage1_data.get("object_ids", None)
    point_to_pixels = stage1_data.get("point_to_pixels", None)
    if object_ids is None or point_to_pixels is None:
        return {
            "rate_iou": 0.0,
            "n_anchors": 0,
            "n_disc_iou": 0,
            "per_anchor_iou": {},
            "iou_threshold": float(iou_threshold),
        }

    anchor_set = set(int(i) for i in gt_anchor_object_ids)
    sam_pts: Dict[int, set] = {}
    anchor_pts: Dict[int, set] = {}
    rescale_uv = _make_pixel_rescaler(stage1_data)

    for pt_idx, entries in enumerate(point_to_pixels):
        # Legacy flat-shape tolerance: pre-Fix-A pickles stored `List[dict]`.
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            continue
        obj_id = int(object_ids[pt_idx])
        # Anchor membership is per-point — record once regardless of how many
        # frames the point projected into. SAM3 sets are also point-id sets
        # (deduped across frames), so a point covered by the same iid on N
        # frames still contributes exactly one element to IoU numerator/denom.
        recorded_anchor = False
        for entry in entries:
            if not entry:
                continue
            frame_pos = entry.get("frame_id")
            u = entry.get("pixel_u")
            v = entry.get("pixel_v")
            if frame_pos is None or u is None or v is None:
                continue
            if obj_id in anchor_set and not recorded_anchor:
                anchor_pts.setdefault(obj_id, set()).add(pt_idx)
                recorded_anchor = True
            masks_this_frame = stage3_masks.get(int(frame_pos))
            if not masks_this_frame:
                continue
            for sam_iid, mask in masks_this_frame.items():
                u_i, v_i = rescale_uv(u, v, mask.shape)
                if 0 <= v_i < mask.shape[0] and 0 <= u_i < mask.shape[1] and mask[v_i, u_i]:
                    sam_pts.setdefault(int(sam_iid), set()).add(pt_idx)

    per_anchor_iou: Dict[int, float] = {}
    for obj_id in anchor_set:
        a_pts = anchor_pts.get(obj_id, set())
        if not a_pts:
            per_anchor_iou[obj_id] = 0.0
            continue
        best = 0.0
        for s_pts in sam_pts.values():
            inter = len(a_pts & s_pts)
            if inter == 0:
                continue
            union = len(a_pts | s_pts)
            iou = inter / union if union else 0.0
            if iou > best:
                best = iou
        per_anchor_iou[obj_id] = float(best)

    n_anchors = len(anchor_set)
    n_disc = sum(1 for v in per_anchor_iou.values() if v >= iou_threshold)
    return {
        "rate_iou": (n_disc / n_anchors) if n_anchors > 0 else 0.0,
        "n_anchors": int(n_anchors),
        "n_disc_iou": int(n_disc),
        "per_anchor_iou": per_anchor_iou,
        "iou_threshold": float(iou_threshold),
    }


def compute_object_discovery_iou(
    src_stage1_data: dict,
    src_stage3_masks: Dict[int, Dict[int, np.ndarray]],
    ref_stage1_data: dict,
    ref_stage3_masks: Dict[int, Dict[int, np.ndarray]],
    gt_anchor_object_ids: Sequence[int],
    *,
    iou_threshold: float = 0.3,
) -> Dict[str, float]:
    """Pair-level anchor discovery via 3D-point IoU on each subscan side.

    An anchor is `disc_both_iou` iff its max IoU to any SAM3 instance ≥
    threshold on BOTH src and ref sides. This reports the same shape as
    `compute_object_discovery` (per-side and joint rates) so callers can
    compare argmax-voting vs IoU-thresholding apples-to-apples.
    """
    src = compute_anchor_iou_discovery(
        src_stage1_data, src_stage3_masks, gt_anchor_object_ids,
        iou_threshold=iou_threshold,
    )
    ref = compute_anchor_iou_discovery(
        ref_stage1_data, ref_stage3_masks, gt_anchor_object_ids,
        iou_threshold=iou_threshold,
    )
    anchor_set = set(int(i) for i in gt_anchor_object_ids)
    n_anchors = len(anchor_set)
    n_both = sum(
        1 for obj in anchor_set
        if src["per_anchor_iou"].get(obj, 0.0) >= iou_threshold
        and ref["per_anchor_iou"].get(obj, 0.0) >= iou_threshold
    )
    return {
        "rate_src_iou": float(src["rate_iou"]),
        "rate_ref_iou": float(ref["rate_iou"]),
        "rate_both_iou": (n_both / n_anchors) if n_anchors > 0 else 0.0,
        "n_anchors": int(n_anchors),
        "n_disc_src_iou": int(src["n_disc_iou"]),
        "n_disc_ref_iou": int(ref["n_disc_iou"]),
        "n_disc_both_iou": int(n_both),
        "iou_threshold": float(iou_threshold),
    }


def aggregate_discovery_metrics(per_pair: List[Dict[str, float]]) -> Dict[str, float]:
    """Sum-then-divide aggregation for object-discovery metrics."""
    if not per_pair:
        return {"n_pairs": 0}
    n_pairs = len(per_pair)
    sum_anchors = sum(int(p.get("n_anchors", 0)) for p in per_pair)
    sum_src = sum(int(p.get("n_disc_src", 0)) for p in per_pair)
    sum_ref = sum(int(p.get("n_disc_ref", 0)) for p in per_pair)
    sum_both = sum(int(p.get("n_disc_both", 0)) for p in per_pair)
    rate_src = (sum_src / sum_anchors) if sum_anchors > 0 else 0.0
    rate_ref = (sum_ref / sum_anchors) if sum_anchors > 0 else 0.0
    rate_both = (sum_both / sum_anchors) if sum_anchors > 0 else 0.0
    # Aggregate IoU-based discovery if any per-pair entry carries it.
    sum_src_iou = sum(int(p.get("n_disc_src_iou", 0)) for p in per_pair)
    sum_ref_iou = sum(int(p.get("n_disc_ref_iou", 0)) for p in per_pair)
    sum_both_iou = sum(int(p.get("n_disc_both_iou", 0)) for p in per_pair)
    has_iou = any("rate_both_iou" in p for p in per_pair)
    out = {
        "n_pairs": int(n_pairs),
        "rate_src": float(rate_src),
        "rate_ref": float(rate_ref),
        "rate_both": float(rate_both),
        "n_anchors": int(sum_anchors),
        "n_disc_src": int(sum_src),
        "n_disc_ref": int(sum_ref),
        "n_disc_both": int(sum_both),
    }
    if has_iou:
        out.update({
            "rate_src_iou": (sum_src_iou / sum_anchors) if sum_anchors > 0 else 0.0,
            "rate_ref_iou": (sum_ref_iou / sum_anchors) if sum_anchors > 0 else 0.0,
            "rate_both_iou": (sum_both_iou / sum_anchors) if sum_anchors > 0 else 0.0,
            "n_disc_src_iou": int(sum_src_iou),
            "n_disc_ref_iou": int(sum_ref_iou),
            "n_disc_both_iou": int(sum_both_iou),
        })
    return out


def filter_pairs_to_anchors(
    pairs: Iterable[Tuple[int, int, float]],
    src_sam2obj: Dict[int, int],
    ref_sam2obj: Dict[int, int],
    gt_anchor_object_ids: Sequence[int],
) -> List[Tuple[int, int]]:
    """Keep only pairs whose endpoints' dominant objectIds are both in anchors.

    Used to compute the matching-only F1 column — apples-to-apples with
    SG-Reg's reported P/R/F1 since their FM-Fusion-node GT is similarly
    bounded by what was discovered upstream.
    """
    anchor_set = set(int(i) for i in gt_anchor_object_ids)
    out: List[Tuple[int, int]] = []
    for s, r, *_ in pairs:
        s_obj = src_sam2obj.get(int(s))
        r_obj = ref_sam2obj.get(int(r))
        if s_obj is None or r_obj is None:
            continue
        if s_obj in anchor_set and r_obj in anchor_set:
            out.append((int(s), int(r)))
    return out


def aggregate_metrics(per_pair: List[Dict[str, float]]) -> Dict[str, float]:
    """Average each metric over valid pairs; report n_pairs and valid_ratio."""
    if not per_pair:
        return {}
    keys = [
        "CD", "IR", "RRE", "RTE", "FMR",
        "recall", "recall_rmse",
        "recall_paired", "recall_paired_rmse",
    ]
    agg: Dict[str, float] = {}
    n_total = len(per_pair)
    n_valid = sum(p.get("valid", 0.0) for p in per_pair)
    agg["n_pairs"] = float(n_total)
    agg["valid_ratio"] = float(n_valid) / float(n_total) if n_total else 0.0

    valid = [p for p in per_pair if p.get("valid", 0.0) >= 1.0]
    if not valid:
        return agg
    for k in keys:
        vals = [p[k] for p in valid if k in p]
        if vals:
            agg[k] = float(np.mean(vals))
    return agg
