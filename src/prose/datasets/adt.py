"""Aria Digital Twin (ADT) loader.

Expected layout after running scripts/preprocess_adt.py:

    <root_dir>/<sequence_name>/
      rectified/frame_%06d.png              <-- undistorted RGB frames
      depth/frame_%06d.npy                  <-- GT depth per frame (optional)
      poses.npy                              <-- (N, 4, 4) world→camera per frame
      intrinsics.npy                         <-- (3, 3) rectified intrinsic
      anchors_val.json                       <-- sliding-window subscan pairs

The anchors JSON looks like:
  [
    {"src":"000", "ref":"001",
     "src_frames":[..],"ref_frames":[..],
     "transform":[[...]], "overlap":0.37},
    ...
  ]
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..utils.io import load_json
from ..utils.logging import get_logger
from .base import BaseDataset, RGBFrames, SubscanPair

log = get_logger(__name__)


class ADTDataset(BaseDataset):
    name = "adt"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        self.sequence_dir = Path(cfg.sequence_dir)
        self.rectified_dir = Path(cfg.rectified_dir)
        self.anchors_file = Path(cfg.anchors_file)
        # Optional overrides — let adt.yaml swap in the rotated artifacts
        # (depth_rot, poses_rot, intrinsics_rot) without touching code.
        self.depth_dir = Path(getattr(cfg, "depth_dir", str(self.sequence_dir / "depth")))
        self.poses_file = Path(getattr(cfg, "poses_file", str(self.sequence_dir / "poses.npy")))
        self.intrinsics_file = Path(getattr(cfg, "intrinsics_file", str(self.sequence_dir / "intrinsics.npy")))

    def list_subscan_pairs(self) -> List[SubscanPair]:
        if not self.anchors_file.exists():
            log.warning("ADT anchors file missing: %s", self.anchors_file)
            return []
        records = load_json(self.anchors_file)
        pairs: List[SubscanPair] = []
        for rec in records:
            src = str(rec["src"])
            ref = str(rec["ref"])
            T = np.asarray(rec.get("transform", np.eye(4).tolist()), dtype=np.float64)
            if T.shape != (4, 4):
                T = np.eye(4, dtype=np.float64)
            pairs.append(
                SubscanPair(
                    pair_id=f"{src}__{ref}",
                    src_id=src,
                    ref_id=ref,
                    gt_transform=T,
                    overlap=float(rec.get("overlap", 0.0)),
                )
            )
        return pairs

    def load_frame_poses(self, subscan_id: str, frame_ids: List[int]) -> np.ndarray:
        """Return per-frame camera-to-world poses (N, 4, 4).

        `poses.npy` stores world→camera (see back-projection in
        `load_gt_pointcloud`). DA3's GT-pose alignment in Stage 1 wants
        camera-to-world, so we invert each frame here.
        """
        poses_wc = np.load(self.poses_file)
        out = np.zeros((len(frame_ids), 4, 4), dtype=np.float32)
        for i, fid in enumerate(frame_ids):
            T_wc = poses_wc[int(fid)]
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3:4]
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            out[i, :3, :3] = R_cw
            out[i, :3, 3:4] = t_cw
            out[i, 3, 3] = 1.0
        return out

    def load_frame_intrinsics(self, subscan_id: str, n_frames: int) -> np.ndarray:
        """Return per-frame rectified intrinsics (N, 3, 3).

        ADT preprocessing produces a single shared K per sequence; tile it.
        """
        K = np.load(self.intrinsics_file).astype(np.float32)
        return np.broadcast_to(K, (n_frames, 3, 3)).copy()

    def load_rgb_frames(self, subscan_id: str) -> RGBFrames:
        """A subscan is a window of frame indices. The anchors file stores
        per-subscan frame lists; we look them up by id."""
        records = load_json(self.anchors_file)
        entry = next((r for r in records if str(r["src"]) == subscan_id or str(r["ref"]) == subscan_id), None)
        if entry is None:
            raise KeyError(f"Subscan {subscan_id} not found in {self.anchors_file}")
        frame_ids = (
            entry["src_frames"] if str(entry["src"]) == subscan_id else entry["ref_frames"]
        )
        paths = [self.rectified_dir / f"frame_{int(i):06d}.png" for i in frame_ids]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} frames missing for subscan {subscan_id}; first: {missing[0]}"
            )
        return RGBFrames(paths=paths, frame_ids=list(map(int, frame_ids)))

    def load_gt_pointcloud(
        self, subscan_id: str
    ) -> Tuple[np.ndarray, List[List[dict]], List[int]]:
        """Back-project GT depth from each frame in the subscan into a world PC."""
        rgb = self.load_rgb_frames(subscan_id)

        depth_dir = self.depth_dir
        poses = np.load(self.poses_file)  # (N, 4, 4), world→camera
        K = np.load(self.intrinsics_file)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        points_world: List[np.ndarray] = []
        # Each ADT point is back-projected from one frame's depth, so its
        # per-point list has exactly one entry. The list-of-lists shape is
        # preserved for schema parity with scan3r (see datasets/scan3r.py).
        point_to_pixels: List[List[dict]] = []
        for i, frame_id in enumerate(rgb.frame_ids):
            d_path = depth_dir / f"frame_{frame_id:06d}.npy"
            if not d_path.exists():
                continue
            depth = np.load(d_path).astype(np.float32)
            H, W = depth.shape
            stride = max(1, H // 128)
            us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
            us_f, vs_f = us.flatten(), vs.flatten()
            d = depth[vs_f, us_f]
            valid = d > 0
            if not np.any(valid):
                continue

            xs = (us_f[valid] - cx) / fx * d[valid]
            ys = (vs_f[valid] - cy) / fy * d[valid]
            zs = d[valid]
            cam = np.stack([xs, ys, zs], axis=1)

            # poses stored as world→camera; invert to camera→world.
            T_wc = poses[frame_id]
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            world = (cam - t_wc) @ R_wc  # R_wc^T @ (p - t) with row vectors

            points_world.append(world.astype(np.float32))
            # Store POSITIONAL frame index (0..N-1 within the subscan) — this
            # matches Stage 3 mask keys and Stage 4 frame_paths dict.
            for u, v in zip(us_f[valid], vs_f[valid]):
                point_to_pixels.append([
                    {"frame_id": int(i), "pixel_u": int(u), "pixel_v": int(v)}
                ])

        if not points_world:
            return np.zeros((0, 3), dtype=np.float32), [], list(rgb.frame_ids)
        pts = np.concatenate(points_world, axis=0)
        return pts, point_to_pixels, list(rgb.frame_ids)
