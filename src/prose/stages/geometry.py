"""Scene parsing — per-frame geometry (paper Sec. 3.1, "Per-frame geometry").

Lifts an egocentric RGB sequence into a single per-subscan point cloud with a
2D-to-3D index mapping each pixel to its 3D point. Two modes, selected by the
top-level ``use_gt_pointclouds`` flag:

  (a) Ground-truth point clouds — back-projected from the dataset's GT depth
      (``use_gt_pointclouds=true``). Used for the GT-cloud benchmark setting.
  (b) VGGT-Omega — Meta's feed-forward depth + camera estimator predicts the
      geometry directly from RGB (``use_gt_pointclouds=false``). This is the
      sensor-free setting PROSE targets.

If the VGGT-Omega package or checkpoint is unavailable, the stage falls back to
GT point clouds with a warning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.io import dump_pickle, ensure_dir
from ..utils.logging import get_logger
from ..utils.pointcloud import backproject_to_points

log = get_logger(__name__)


@dataclass
class PointCloudArtifact:
    """Per-subscan output of the geometry stage."""
    subscan_id: str
    points: np.ndarray              # (M, 3) float32 — world coords
    # One inner list per 3D point: each entry is a (frame_id, pixel_u, pixel_v)
    # record of a frame the point projected into. Multiple entries per point
    # encode multi-frame visibility — preserved so per-frame mask fusion sees
    # every (frame, mask) pair the same physical point hits.
    point_to_pixels: List[List[dict]]
    frame_ids: List[int]            # frame indices in the subscan
    source: str                     # 'gt' or 'vggt_omega'
    object_ids: Optional[np.ndarray] = None  # (M,) int64 GT instance ids; None if dataset has no GT labels
    # Pixel-coord resolution of `point_to_pixels` entries, as (H, W). Masks may
    # live at a different (raw) resolution; downstream rescales (u, v) when this
    # differs from the mask shape. None = coords already at mask resolution.
    pixel_resolution: Optional[Tuple[int, int]] = None
    # Per-frame depth / cameras from the predictor (None for the GT path).
    depth_maps: Optional[np.ndarray] = None      # (N, H, W) float16
    conf_maps: Optional[np.ndarray] = None       # (N, H, W) float16
    extrinsics: Optional[np.ndarray] = None      # (N, 3, 4) float32 world-to-camera
    intrinsics: Optional[np.ndarray] = None      # (N, 3, 3) float32

    def save(self, out_dir: Path) -> Path:
        out_dir = ensure_dir(out_dir)
        path = out_dir / f"{self.subscan_id}.pkl"
        d: Dict = {
            "points": self.points,
            "point_to_pixels": self.point_to_pixels,
            "frame_ids": self.frame_ids,
            "source": self.source,
            "object_ids": self.object_ids,
            "pixel_resolution": self.pixel_resolution,
        }
        if self.depth_maps is not None:
            d["depth_maps"] = self.depth_maps
        if self.conf_maps is not None:
            d["conf_maps"] = self.conf_maps
        if self.extrinsics is not None:
            d["extrinsics"] = self.extrinsics
        if self.intrinsics is not None:
            d["intrinsics"] = self.intrinsics
        dump_pickle(d, path)
        return path


def run_geometry_gt(dataset, subscan_id: str) -> PointCloudArtifact:
    """Load ground-truth points from the dataset."""
    pts, p2p, frame_ids = dataset.load_gt_pointcloud(subscan_id)
    object_ids = dataset.load_gt_object_ids(subscan_id)
    if object_ids is not None and object_ids.shape[0] != pts.shape[0]:
        log.warning(
            "Geometry [%s]: objectIds length (%d) != points length (%d); dropping",
            subscan_id, object_ids.shape[0], pts.shape[0],
        )
        object_ids = None
    return PointCloudArtifact(
        subscan_id=subscan_id,
        points=pts.astype(np.float32),
        point_to_pixels=list(p2p),
        frame_ids=list(frame_ids),
        source="gt",
        object_ids=object_ids,
    )


def run_geometry_vggt(
    dataset,
    subscan_id: str,
    *,
    cfg_geometry,
    cfg_pipeline,
    wrapper=None,
) -> PointCloudArtifact:
    """Run VGGT-Omega on the subscan's frames and back-project to world points.

    Raises ImportError if the `vggt_omega` package is not importable — the
    caller should fall back to `run_geometry_gt`.
    """
    from ..models.vggt_omega import VGGTOmegaWrapper

    frames = dataset.load_rgb_frames(subscan_id)
    frame_paths, frame_ids = frames.paths, frames.frame_ids

    use_predicted_cam = bool(getattr(cfg_geometry, "use_predicted_camera", False))
    align_to_gt = bool(getattr(cfg_geometry, "align_to_gt_poses", True))

    gt_w2c_4x4: Optional[np.ndarray] = None
    if not use_predicted_cam and align_to_gt and hasattr(dataset, "load_frame_poses"):
        try:
            gt_c2w = dataset.load_frame_poses(subscan_id, frame_ids)
            R_c2w = gt_c2w[:, :3, :3]
            t_c2w = gt_c2w[:, :3, 3:4]
            R_w2c = np.transpose(R_c2w, (0, 2, 1))
            t_w2c = -(R_w2c @ t_c2w)
            w2c_3x4 = np.concatenate([R_w2c, t_w2c], axis=2)
            N = w2c_3x4.shape[0]
            gt_w2c_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
            gt_w2c_4x4[:, :3, :] = w2c_3x4
            gt_w2c_4x4[:, 3, 3] = 1.0
        except FileNotFoundError:
            gt_w2c_4x4 = None

    # use_predicted_camera / align_to_gt_poses both require a single batch
    # (VGGT sees the full sequence context; Umeyama needs all frames at once).
    max_bs = int(getattr(cfg_geometry, "max_frames_per_call", 8))
    if use_predicted_cam or gt_w2c_4x4 is not None:
        max_bs = len(frame_paths)

    all_points: List[np.ndarray] = []
    all_meta: List[List[dict]] = []
    all_depth: List[np.ndarray] = []
    all_conf: List[np.ndarray] = []
    all_ext: List[np.ndarray] = []
    all_intr: List[np.ndarray] = []

    owned = wrapper is None
    if owned:
        dev = cfg_pipeline.get("device_map", None) or "cuda"
        if dev == "auto":
            dev = "cuda"
        wrapper = VGGTOmegaWrapper(
            checkpoint_path=cfg_geometry.checkpoint_path,
            device=dev,
            image_resolution=int(getattr(cfg_geometry, "image_resolution", 512)),
        )
    pix_res = None
    try:
        for start in range(0, len(frame_paths), max_bs):
            batch = frame_paths[start : start + max_bs]
            if use_predicted_cam:
                pred = wrapper.infer(batch, align_to_gt_poses=False)
            else:
                batch_ext = gt_w2c_4x4[start : start + max_bs] if gt_w2c_4x4 is not None else None
                pred = wrapper.infer(
                    batch, extrinsics=batch_ext, align_to_gt_poses=align_to_gt,
                )
            if start == 0:
                log.info(
                    "Geometry [%s]: VGGT-Omega use_predicted_camera=%s is_metric=%s scale_factor=%s",
                    subscan_id, use_predicted_cam, pred.is_metric, pred.scale_factor,
                )
            all_depth.append(pred.depth.astype(np.float16))
            all_conf.append(pred.conf.astype(np.float16))
            all_ext.append(pred.extrinsics.astype(np.float32))
            all_intr.append(pred.intrinsics.astype(np.float32))
            bp_stride = int(getattr(cfg_geometry, "backproject_stride", 1))
            pts, meta, pix_res = backproject_to_points(
                pred.depth, pred.intrinsics, pred.extrinsics, pred.conf,
                conf_threshold=float(getattr(cfg_geometry, "conf_threshold", 1.5)),
                stride=bp_stride,
                return_resolution=True,
            )
            for m in meta:
                m["frame_id"] = int(start + m["frame_id"])
            all_points.append(pts)
            all_meta.extend([[m] for m in meta])
    finally:
        if owned:
            wrapper.close()

    if all_points:
        pts_cat = np.concatenate(all_points, axis=0)
    else:
        pts_cat = np.zeros((0, 3), dtype=np.float32)

    # VGGT's predicted frame is OpenCV (Y-down, Z-forward); flip Y and Z so the
    # cloud is Y-up for standard viewers / downstream stages.
    if use_predicted_cam and len(pts_cat) > 0:
        pts_cat[:, 1] *= -1
        pts_cat[:, 2] *= -1

    vox = getattr(cfg_geometry, "voxel_downsample", None)
    if vox is not None and len(pts_cat) > 0:
        v = float(vox)
        keys = np.floor(pts_cat.astype(np.float64) / v).astype(np.int64)
        SHIFT = 21
        packed = (keys[:, 0] & ((1 << SHIFT) - 1)) \
                 | ((keys[:, 1] & ((1 << SHIFT) - 1)) << SHIFT) \
                 | ((keys[:, 2] & ((1 << SHIFT) - 1)) << (2 * SHIFT))
        _, first_idx = np.unique(packed, return_index=True)
        first_idx.sort()
        n_before = len(pts_cat)
        pts_cat = pts_cat[first_idx]
        all_meta = [all_meta[i] for i in first_idx]
        log.info(
            "Geometry [%s]: voxel_downsample=%.3fm: %d -> %d points",
            subscan_id, v, n_before, len(pts_cat),
        )

    return PointCloudArtifact(
        subscan_id=subscan_id,
        points=pts_cat,
        point_to_pixels=all_meta,
        frame_ids=list(frame_ids),
        source="vggt_omega",
        pixel_resolution=pix_res,
        depth_maps=np.concatenate(all_depth, axis=0) if all_depth else None,
        conf_maps=np.concatenate(all_conf, axis=0) if all_conf else None,
        extrinsics=np.concatenate(all_ext, axis=0) if all_ext else None,
        intrinsics=np.concatenate(all_intr, axis=0) if all_intr else None,
    )


def run_geometry(
    dataset,
    subscan_id: str,
    *,
    use_gt: bool,
    cfg_geometry,
    cfg_pipeline,
    wrapper=None,
) -> PointCloudArtifact:
    """Orchestrate the geometry stage: GT point clouds or VGGT-Omega."""
    if use_gt:
        log.info("Geometry [%s]: using GT point cloud", subscan_id)
        return run_geometry_gt(dataset, subscan_id)

    backend = getattr(cfg_geometry, "name", "vggt_omega")
    try:
        if backend == "vggt_omega":
            log.info("Geometry [%s]: running VGGT-Omega", subscan_id)
            return run_geometry_vggt(
                dataset, subscan_id,
                cfg_geometry=cfg_geometry, cfg_pipeline=cfg_pipeline,
                wrapper=wrapper,
            )
        raise ValueError(f"Unknown geometry backend: {backend!r} (expected 'vggt_omega')")
    except ImportError as e:
        log.warning(
            "Geometry backend %s unavailable (%s); falling back to GT point cloud.",
            backend, e,
        )
        return run_geometry_gt(dataset, subscan_id)
    except Exception as e:  # noqa: BLE001
        # GT-pose Umeyama alignment fails on subscans whose camera centres are
        # rank-deficient (e.g. all frames from one viewpoint). Fall back to GT
        # cloud rather than aborting the whole run.
        name = type(e).__name__
        if "Geometry" in name or "Umeyama" in name or "rank" in str(e).lower():
            log.warning(
                "Geometry [%s]: %s GT-pose alignment degenerate (%s: %s); "
                "falling back to GT point cloud.",
                subscan_id, backend, name, e,
            )
            return run_geometry_gt(dataset, subscan_id)
        raise
