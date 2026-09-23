"""Point-cloud helpers (subset extraction, overlap, transform)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N,3) point cloud."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    assert transform.shape == (4, 4), transform.shape
    R = transform[:3, :3]
    t = transform[:3, 3]
    return points @ R.T + t


def random_downsample(points: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    rng = rng or np.random.default_rng()
    idx = rng.choice(points.shape[0], size=n, replace=False)
    return points[idx]


def voxel_downsample(points: np.ndarray, voxel_size: float,
                     max_points: int | None = None) -> np.ndarray:
    """Voxel-grid downsample at a fixed `voxel_size`, one point per voxel.

    Unlike random_downsample, this yields a *uniform-density* cloud — what
    GeoTransformer's 4-stage grid subsampling expects. A randomly subsampled
    room-scale cloud is non-uniform, so the coarse level never pools and the
    O(N^2) superpoint matching blows up (40+ GB OOM). A voxel grid pools
    predictably -> bounded coarse level -> no OOM. Spatial extent is
    preserved (no cropping); only density is regularised.

    If `max_points` is given and the fixed-voxel result still exceeds it,
    the voxel is grown until the count fits — an OOM safety net for
    pathologically large clouds.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    lo = pts.min(axis=0)
    v = float(voxel_size)
    keys = np.floor((pts - lo) / v).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    if max_points is not None:
        for _ in range(40):
            if idx.shape[0] <= max_points:
                break
            v *= 1.3
            keys = np.floor((pts - lo) / v).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


def nearest_neighbor_distance(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """For each row in src, return distance to nearest point in dst."""
    from scipy.spatial import cKDTree

    tree = cKDTree(dst)
    dists, _ = tree.query(src, k=1)
    return dists


def compute_overlap(src: np.ndarray, dst: np.ndarray, radius: float = 0.05) -> Tuple[float, np.ndarray]:
    """Overlap = fraction of src points within `radius` of any dst point."""
    dists = nearest_neighbor_distance(src, dst)
    mask = dists < radius
    return float(mask.mean()), np.where(mask)[0]


def extract_instance_points(
    points: np.ndarray,
    point_to_instance: np.ndarray,
    instance_id: int,
) -> np.ndarray:
    """Select points belonging to a given instance id."""
    return points[point_to_instance == instance_id]


def build_point_to_instance(
    points_xyz: np.ndarray,
    point_to_pixels: dict,
    masks_per_frame: dict,
) -> np.ndarray:
    """Assign each 3D point the iid whose 2D mask most consistently covers it.

    Vote across all (frame, iid) where the point's 2D reprojection falls
    inside `masks_per_frame[frame][iid]`. Tiebreak by smaller mask area
    (more specific instance — a small foreground object inside a large
    background mask should win).

    Replaces the legacy first-hit-wins logic (2026-05-17): under
    insertion-order iteration, a small instance could steal 3D points
    from a larger overlapping instance just because its mask was added
    to the per-frame dict first — caused 7 instances in v8 to land at
    exactly 0 points and 29 below the 50-point Stage 6 threshold.

    Points whose reprojection lands on no mask → iid = -1.
    """
    n = points_xyz.shape[0]
    out = np.full(n, -1, dtype=np.int64)

    # Precompute mask total area per (frame, iid) for the tiebreak.
    mask_area: dict = {}
    for fid, by_iid in masks_per_frame.items():
        for iid, m in by_iid.items():
            mask_area[(fid, int(iid))] = int(np.asarray(m, dtype=bool).sum())

    for pt_idx, pixels in point_to_pixels.items():
        if pt_idx >= n:
            continue
        votes: dict = {}  # iid -> vote count
        smallest_area: dict = {}  # iid -> smallest mask area seen for this iid
        for px in pixels:
            frame_id = px["frame_id"]
            u = px["pixel_u"]
            v = px["pixel_v"]
            if frame_id not in masks_per_frame:
                continue
            for iid, mask in masks_per_frame[frame_id].items():
                # Boundary handling: snap pixels landing exactly on the
                # mask edge inside instead of dropping (matches upstream
                # 0223_assign_instance_ids.py:109-113).
                qu, qv = u, v
                if qv == mask.shape[0]:
                    qv -= 1
                if qu == mask.shape[1]:
                    qu -= 1
                if 0 <= qv < mask.shape[0] and 0 <= qu < mask.shape[1] and mask[qv, qu]:
                    iid_i = int(iid)
                    votes[iid_i] = votes.get(iid_i, 0) + 1
                    a = mask_area.get((frame_id, iid_i), 0)
                    if iid_i not in smallest_area or a < smallest_area[iid_i]:
                        smallest_area[iid_i] = a
        if not votes:
            continue
        # argmax votes, tiebreak by smallest mask area; final tiebreak
        # by iid for determinism.
        best_iid = max(
            votes.keys(),
            key=lambda i: (votes[i], -smallest_area.get(i, 0), -i),
        )
        out[pt_idx] = best_iid
    return out


# ---------------------------------------------------------------------------
# Back-projection + pose alignment (used by the geometry stage).
# Relocated here so the geometry stage has no dependency on any external
# depth-prediction package beyond the model wrapper it actually runs.
# ---------------------------------------------------------------------------
from typing import List, Optional  # noqa: E402


def backproject_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    conf: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
    stride: int = 1,
    *,
    return_resolution: bool = False,
):
    """Back-project a stack of depth maps into a single world-frame point cloud.

    Also returns per-point (frame_id, pixel_u, pixel_v) correspondence entries
    required downstream (the correspondence stage uses these to extract
    per-instance sub-clouds).

    Args:
        depth: (N, H, W)
        intrinsics: (N, 3, 3)
        extrinsics: (N, 3, 4) world-to-camera.
        conf: (N, H, W) or None. Points with conf < threshold are discarded.
        stride: subsample every `stride`-th pixel in each dim.
    Returns:
        points_world: (M, 3) float32
        point_to_pixels: list of {'frame_id', 'pixel_u', 'pixel_v'} per point
    """
    assert depth.ndim == 3
    n, h, w = depth.shape
    points_world_list: List[np.ndarray] = []
    per_point_meta: List[dict] = []

    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    us = us.flatten()
    vs = vs.flatten()

    for i in range(n):
        d = depth[i][::stride, ::stride].flatten()
        K = intrinsics[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        valid = d > 0
        if conf is not None:
            c = conf[i][::stride, ::stride].flatten()
            valid &= c >= conf_threshold
        if not np.any(valid):
            continue

        xs = (us[valid] - cx) / fx * d[valid]
        ys = (vs[valid] - cy) / fy * d[valid]
        zs = d[valid]
        cam_pts = np.stack([xs, ys, zs], axis=1)  # (K, 3)

        # extrinsics is world→camera; invert to camera→world.
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        world_pts = (cam_pts - t) @ R  # inv(R)=R.T and R.T @ (p-t)

        points_world_list.append(world_pts.astype(np.float32))
        for (u, v) in zip(us[valid], vs[valid]):
            per_point_meta.append({"frame_id": int(i), "pixel_u": int(u), "pixel_v": int(v)})

    if not points_world_list:
        result = (np.zeros((0, 3), dtype=np.float32), [])
    else:
        result = (np.concatenate(points_world_list, axis=0), per_point_meta)
    if return_resolution:
        return (*result, (int(h), int(w)))
    return result


def _camera_centers_from_w2c(ext_4x4: np.ndarray) -> np.ndarray:
    """Camera centers (N, 3) from a stack of world-to-camera 4x4 matrices."""
    R = ext_4x4[:, :3, :3]
    t = ext_4x4[:, :3, 3:4]
    # center = -R^T t
    C = -np.transpose(R, (0, 2, 1)) @ t
    return C[:, :, 0]


def align_poses_umeyama(target_w2c: np.ndarray, source_w2c: np.ndarray):
    """Umeyama Sim(3) alignment of two camera trajectories.

    Estimates the similarity transform (R, t, s) that best maps the *source*
    camera centers onto the *target* camera centers in the least-squares sense
    (Umeyama, 1991). Returns ``(R, t, s)`` with ``s * R @ C_src + t ≈ C_tgt``.

    The geometry stage uses only the scale ``s`` to rescale predicted depth to
    the (metric) ground-truth scale, mirroring the convention
    ``align_poses_umeyama(gt, pred)``.
    """
    tgt = _camera_centers_from_w2c(np.asarray(target_w2c, dtype=np.float64))
    src = _camera_centers_from_w2c(np.asarray(source_w2c, dtype=np.float64))
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt
    var_src = (src_c ** 2).sum() / n
    cov = (tgt_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_src) if var_src > 1e-12 else 1.0
    t = mu_tgt - s * (R @ mu_src)
    return R, t, s
