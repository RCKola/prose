"""FPFH (Open3D) — handcrafted descriptor baseline for Stage 6 corr extraction.

Drop-in sibling of `_GeoTransformerRunner.run(sp, rp) -> {src, ref, scores}`.
Pipeline: voxel downsample -> normals -> FPFH (33-dim) -> mutual nearest-
neighbour in feature space + ratio test -> point correspondences.

`scores` is the negative L2 feature distance (larger = more similar) so the
existing `top-K by scores` cap in `run_stage5` keeps the best matches.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class _FPFHRunner:
    def __init__(
        self,
        max_points: int = 10000,
        voxel_size: float = 0.025,
        normal_radius_mult: float = 2.0,
        feature_radius_mult: float = 5.0,
        ratio_test: float = 0.95,
    ):
        import open3d as o3d  # noqa: F401 — sanity import

        self.max_points = int(max_points)
        self.voxel_size = float(voxel_size)
        self.normal_radius = float(normal_radius_mult) * self.voxel_size
        self.feature_radius = float(feature_radius_mult) * self.voxel_size
        self.ratio_test = float(ratio_test)

    def _to_o3d(self, points: np.ndarray):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.voxel_size)
        if len(pcd.points) > self.max_points:
            stride = max(1, len(pcd.points) // self.max_points)
            pcd = pcd.uniform_down_sample(stride)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30)
        )
        return pcd

    def run(self, src_pts: np.ndarray, ref_pts: np.ndarray) -> Optional[dict]:
        import open3d as o3d

        if src_pts.shape[0] < 4 or ref_pts.shape[0] < 4:
            return None

        try:
            src_pcd = self._to_o3d(src_pts)
            ref_pcd = self._to_o3d(ref_pts)
        except Exception:  # noqa: BLE001
            return None

        if len(src_pcd.points) < 4 or len(ref_pcd.points) < 4:
            return None

        try:
            src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                src_pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius, max_nn=100),
            )
            ref_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                ref_pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius, max_nn=100),
            )
        except Exception:  # noqa: BLE001
            return None

        # (33, N) → (N, 33) and run mutual NN via scipy KDTree.
        src_feat = np.asarray(src_fpfh.data).T.astype(np.float32)  # (Ns, 33)
        ref_feat = np.asarray(ref_fpfh.data).T.astype(np.float32)  # (Nr, 33)
        src_xyz = np.asarray(src_pcd.points).astype(np.float32)
        ref_xyz = np.asarray(ref_pcd.points).astype(np.float32)

        from scipy.spatial import cKDTree

        ref_tree = cKDTree(ref_feat)
        src_tree = cKDTree(src_feat)

        # src -> ref (top-2 for ratio test)
        d_sr, idx_sr = ref_tree.query(src_feat, k=2, workers=-1)
        # Mutual NN: ref -> src for the chosen ref index
        nearest_ref = idx_sr[:, 0]
        d_rs, idx_rs = src_tree.query(ref_feat[nearest_ref], k=1, workers=-1)
        mutual = idx_rs == np.arange(len(src_feat))

        # Lowe-style ratio test (when k=2 available)
        ratio_ok = (
            d_sr[:, 0] < self.ratio_test * np.maximum(d_sr[:, 1], 1e-9)
            if d_sr.shape[1] >= 2
            else np.ones(len(src_feat), dtype=bool)
        )

        keep = mutual & ratio_ok
        if not keep.any():
            return None

        src_corr = src_xyz[keep]
        ref_corr = ref_xyz[nearest_ref[keep]]
        # Score: similarity = -distance (larger = better) for downstream top-K.
        scores = (-d_sr[keep, 0]).astype(np.float32)

        return {
            "src": src_corr.astype(np.float32),
            "ref": ref_corr.astype(np.float32),
            "scores": scores,
        }
