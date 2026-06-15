"""Fusion stage — per-instance 3D fusion into a scene graph (paper Sec. 3.1).

Consumes the geometry stage (`PointCloudArtifact.points`, `point_to_pixels`)
and the segmentation stage (`SegmentationArtifact.per_frame_masks`), and
produces a `PriorArtifact` per subscan: the scene graph G = (V, E) whose nodes
V are temporally consistent 3D instances (fused point set + PCA oriented
bounding box + descriptor) and whose edges E connect instances with nearby
centroids (k-NN proximity, with coarse above/below/near relations).

Pure CPU / numpy. Open3D is imported lazily and only when the optional global
floor-plane scale correction is enabled — the default config keeps it off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.instance_fusion import (
    apply_alias_to_instance_map,
    apply_alias_to_per_frame_masks,
    apply_alias_to_per_frame_scores,
    bbox_diag_normalize,
    build_scene_graph_edges,
    centroid_jitter,
    dedup_and_fuse_instances,
    instance_centroids,
    instance_centroids_robust,
    instance_points,
    instance_points_per_frame,
    oriented_bbox,
    depth_inlier_filter,
    outlier_frame_filter,
    per_frame_centroids,
    per_frame_visibility,
    voxel_revote_iids,
)
from ..utils.io import dump_pickle, ensure_dir, load_pickle
from ..utils.logging import get_logger
from ..utils.shape_descriptors import instance_descriptor

log = get_logger(__name__)


@dataclass
class PriorArtifact:
    """Per-subscan output of Stage 3.5."""
    subscan_id: str
    instance_ids: List[int] = field(default_factory=list)
    instance_points: Dict[int, np.ndarray] = field(default_factory=dict)        # iid -> (Ki, 3) f32
    instance_points_norm: Dict[int, np.ndarray] = field(default_factory=dict)   # iid -> (Ki, 3) f32, unit-bbox
    instance_centroids: Dict[int, np.ndarray] = field(default_factory=dict)     # iid -> (3,) f32
    bbox_diag: Dict[int, float] = field(default_factory=dict)                   # iid -> scalar (axis-aligned)
    bbox_orientation: Dict[int, np.ndarray] = field(default_factory=dict)      # iid -> (3,3) f32 OBB axes
    bbox_extents: Dict[int, np.ndarray] = field(default_factory=dict)          # iid -> (3,) f32 OBB edge lengths
    # raw Stage 3 iid -> canonical (post-dedup) iid. Identity when dedup off
    # or no clusters formed. Downstream stages apply this to per_frame_masks /
    # per_frame_iou / instance_to_prompt before consuming Stage 3.5 instances.
    iid_alias: Dict[int, int] = field(default_factory=dict)
    # Audit fields for Stage 3.5 intra-side dedup. Both empty when dedup off
    # or no clusters formed.
    # dedup_summary keys: "n_raw_iids", "n_canonical_iids", "n_merged_clusters",
    # "n_iids_merged_away" (n_raw - n_canonical). All ints.
    dedup_summary: Dict[str, int] = field(default_factory=dict)
    # list of {"canonical": int, "members": [int, ...]} dicts, one per cluster
    # with ≥2 members. Members are the raw Stage 3 iids; canonical is the
    # surviving iid (= min of members).
    dedup_merges: List[Dict[str, object]] = field(default_factory=list)
    visibility: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    frame_ids: List[int] = field(default_factory=list)
    is_dynamic: Dict[int, bool] = field(default_factory=dict)
    is_unstable: Dict[int, bool] = field(default_factory=dict)
    descriptors: Dict[int, np.ndarray] = field(default_factory=dict)            # iid -> (3 + n_bins,) f32
    categories: Dict[int, str] = field(default_factory=dict)
    floor_scale: Optional[float] = None
    # Scene-graph edges E (paper Sec. 3.1): directed proximity edges between
    # instance nodes, each {"src", "dst", "dist", "relation"} where relation in
    # {"above", "below", "near"}. Empty when scene_graph is disabled or <2 nodes.
    edges: List[Dict[str, object]] = field(default_factory=list)

    def save(self, out_dir: Path) -> Path:
        out_dir = ensure_dir(out_dir)
        path = out_dir / f"{self.subscan_id}.pkl"
        dump_pickle(
            {
                "subscan_id": self.subscan_id,
                "instance_ids": self.instance_ids,
                "instance_points": self.instance_points,
                "instance_points_norm": self.instance_points_norm,
                "instance_centroids": self.instance_centroids,
                "bbox_diag": self.bbox_diag,
                "bbox_orientation": self.bbox_orientation,
                "bbox_extents": self.bbox_extents,
                "iid_alias": self.iid_alias,
                "dedup_summary": self.dedup_summary,
                "dedup_merges": self.dedup_merges,
                "visibility": self.visibility,
                "frame_ids": self.frame_ids,
                "is_dynamic": self.is_dynamic,
                "is_unstable": self.is_unstable,
                "descriptors": self.descriptors,
                "categories": self.categories,
                "floor_scale": self.floor_scale,
                "edges": self.edges,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "PriorArtifact":
        d = load_pickle(path)
        return cls(
            subscan_id=d["subscan_id"],
            instance_ids=d.get("instance_ids", []),
            instance_points=d.get("instance_points", {}),
            instance_points_norm=d.get("instance_points_norm", {}),
            instance_centroids=d.get("instance_centroids", {}),
            bbox_diag=d.get("bbox_diag", {}),
            bbox_orientation=d.get("bbox_orientation", {}),
            bbox_extents=d.get("bbox_extents", {}),
            iid_alias=d.get("iid_alias", {}),
            dedup_summary=d.get("dedup_summary", {}),
            dedup_merges=d.get("dedup_merges", []),
            visibility=d.get("visibility", np.zeros((0, 0), dtype=np.float32)),
            frame_ids=d.get("frame_ids", []),
            is_dynamic=d.get("is_dynamic", {}),
            is_unstable=d.get("is_unstable", {}),
            descriptors=d.get("descriptors", {}),
            categories=d.get("categories", {}),
            floor_scale=d.get("floor_scale"),
            edges=d.get("edges", []),
        )


def _floor_plane_scale(
    points: np.ndarray,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    nominal_scene_height_m: float,
) -> Optional[float]:
    """Optional: fit a floor plane via RANSAC and return a scene scale factor.

    `nominal_scene_height_m` is the expected vertical extent of a normal room
    (~2.5m). The returned scale is `nominal / observed_height`. None if Open3D
    is unavailable or the cloud is too small.
    """
    try:
        import open3d as o3d  # noqa: WPS433 (lazy import is intentional)
    except ImportError:
        log.warning("Stage 3.5: open3d not available; skipping floor-plane scale")
        return None
    if points.shape[0] < max(ransac_n, 100):
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=float(distance_threshold),
        ransac_n=int(ransac_n),
        num_iterations=int(num_iterations),
    )
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-9)
    proj = points @ n
    observed_height = float(proj.max() - proj.min())
    if observed_height < 1e-3:
        return None
    return float(nominal_scene_height_m) / observed_height


def run_fusion(
    *,
    subscan_id: str,
    points_xyz: np.ndarray,
    point_to_pixels: Sequence[Sequence[dict]],
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
    categories: Optional[Dict[int, str]] = None,
    cfg,
    stage1_pixel_resolution: Optional[Tuple[int, int]] = None,
) -> PriorArtifact:
    """Build the per-subscan 3D prior.

    Args:
        subscan_id: identifier; used for the pickle filename downstream.
        points_xyz: (M, 3) world-space points from Stage 1.
        point_to_pixels: per-point list of frame projection records
            `{frame_id, pixel_u, pixel_v}` with positional frame_ids. A point
            visible in multiple frames carries multiple entries — they're all
            used for per-frame fusion downstream.
        per_frame_masks: SAM3 masks indexed by positional frame_id.
        categories: optional iid -> SAM3 prompt-label map (for debugging /
            future category-aware affinity). May be None or empty.
        cfg: a Hydra cfg node with the keys defined in
            `configs/stage3p5_prior/default.yaml`.

    Empty inputs produce an empty `PriorArtifact` with `instance_ids=[]` —
    safe for downstream `.get(...)` and cache hits.
    """
    categories = dict(categories or {})

    # Empty inputs — emit a structurally valid empty artifact.
    if points_xyz is None or points_xyz.shape[0] == 0 or not per_frame_masks:
        log.info("Stage 3.5 [%s]: empty inputs; writing empty PriorArtifact", subscan_id)
        return PriorArtifact(subscan_id=subscan_id)

    # Step 1: per-(frame, iid) 3D points.
    pf_inst_pts = instance_points_per_frame(
        points_xyz, point_to_pixels, per_frame_masks,
        stage1_pixel_resolution=stage1_pixel_resolution,
    )

    # Step 1.2: per-(frame, iid) depth inlier filter. Discard points whose
    # depth deviates from the per-mask median (MAD gate). Cleans within-frame
    # depth distortion from predicted cameras before any aggregation.
    depth_cfg = getattr(cfg, "depth_inlier_filter", None)
    if depth_cfg is not None and bool(getattr(depth_cfg, "enabled", False)):
        pf_inst_pts, n_removed = depth_inlier_filter(
            pf_inst_pts,
            mad_k=float(getattr(depth_cfg, "mad_k", 2.0)),
            min_points=int(getattr(depth_cfg, "min_points", 10)),
        )
        if n_removed:
            log.info(
                "Stage 3.5 [%s]: depth inlier filter removed %d points",
                subscan_id, n_removed,
            )

    # Step 1.5: per-voxel iid majority-vote relabel. Resolves SAM3
    # re-detections and per-frame iid switching by assigning each 3D voxel
    # to the iid with the most points there.
    revote_cfg = getattr(cfg, "voxel_revote", None)
    revote_alias: Dict[int, int] = {}
    if revote_cfg is not None and bool(getattr(revote_cfg, "enabled", True)):
        pf_inst_pts, revote_alias = voxel_revote_iids(
            pf_inst_pts,
            voxel_size=float(getattr(revote_cfg, "voxel_size_m", 0.02)),
            absorb_majority_threshold=float(getattr(revote_cfg, "absorb_majority_threshold", 0.5)),
            level1_min_majority_frac=float(getattr(revote_cfg, "level1_min_majority_frac", 0.6)),
        )
        if revote_alias:
            log.info(
                "Stage 3.5 [%s]: voxel revote absorbed %d iid(s): %s",
                subscan_id, len(revote_alias), revote_alias,
            )

    if not pf_inst_pts:
        log.info(
            "Stage 3.5 [%s]: no point/mask intersections; writing empty PriorArtifact",
            subscan_id,
        )
        return PriorArtifact(subscan_id=subscan_id)

    # Step 2: aggregate across frames.
    inst_pts = instance_points(pf_inst_pts)
    centroid_method = str(getattr(cfg, "centroid_method", "median_of_frame_centroids"))
    if centroid_method == "median_of_frame_centroids":
        centroids = instance_centroids_robust(pf_inst_pts, method=centroid_method)
    else:
        centroids = instance_centroids(inst_pts)

    # Step 2.3: outlier frame filter — drop (frame, iid) pairs whose per-frame
    # centroid deviates from the robust centroid. Produces a cleaned cloud for OBB.
    outlier_cfg = getattr(cfg, "outlier_frame_filter", None)
    if outlier_cfg is not None and bool(getattr(outlier_cfg, "enabled", False)):
        pf_inst_pts_clean, dropped = outlier_frame_filter(
            pf_inst_pts, centroids,
            max_deviation_m=float(getattr(outlier_cfg, "max_deviation_m", 0.15)),
            min_frames=int(getattr(outlier_cfg, "min_frames", 5)),
        )
        n_dropped = sum(len(v) for v in dropped.values())
        if n_dropped:
            log.info(
                "Stage 3.5 [%s]: outlier filter dropped %d (frame, iid) pairs",
                subscan_id, n_dropped,
            )
        inst_pts_for_obb = instance_points(pf_inst_pts_clean)
    else:
        inst_pts_for_obb = inst_pts

    # Step 2.5: scene-wide intra-side dedup + ID/geometry fusion.
    # Uses proper OBB extents (PCA-aligned edge lengths) as the size gate;
    # axis-aligned extents would conflate orientation with size. Final OBB is
    # recomputed post-merge in step 6. Outputs `iid_alias` so Stage 4 can
    # rewrite its per-frame mask / score / category maps to canonical iids.
    iid_alias: Dict[int, int] = {iid: iid for iid in inst_pts}
    merges: List = []
    dedup_cfg = getattr(cfg, "dedup", None)
    if dedup_cfg is not None and bool(getattr(dedup_cfg, "enabled", False)):
        pre_extents: Dict[int, np.ndarray] = {}
        for iid, pts in inst_pts.items():
            if pts.shape[0] == 0:
                continue
            _, ext = oriented_bbox(pts)
            pre_extents[iid] = np.asarray(ext, dtype=np.float64)
        inst_pts, pf_inst_pts, iid_alias, merges = dedup_and_fuse_instances(
            inst_pts=inst_pts,
            pf_inst_pts=pf_inst_pts,
            centroids=centroids,
            obb_extents=pre_extents,
            categories=categories,
            centroid_thresh_m=float(getattr(dedup_cfg, "centroid_thresh_m", 0.15)),
            obb_size_gate_frac=float(getattr(dedup_cfg, "obb_size_gate_frac", 0.40)),
            require_category_match=bool(getattr(dedup_cfg, "require_category_match", False)),
            min_points_per_cluster=int(getattr(dedup_cfg, "min_points_per_cluster", 0)),
            gate=str(getattr(dedup_cfg, "gate", "voxel_chamfer")),
            voxel_size_m=float(getattr(dedup_cfg, "voxel_size_m", 0.05)),
            voxel_containment_thresh=float(getattr(dedup_cfg, "voxel_containment_thresh", 0.6)),
            chamfer_thresh_m=float(getattr(dedup_cfg, "chamfer_thresh_m", 0.03)),
            chamfer_max_pts=int(getattr(dedup_cfg, "chamfer_max_pts", 10000)),
        )
        # Recompute centroids on fused clouds.
        if centroid_method == "median_of_frame_centroids":
            centroids = instance_centroids_robust(pf_inst_pts, method=centroid_method)
        else:
            centroids = instance_centroids(inst_pts)
        # Categories: keep canonical iid's label; fall back to any merged
        # member's non-empty label.
        if categories:
            new_cats: Dict[int, str] = {}
            for canon in inst_pts:
                lbl = str(categories.get(canon, "")).strip()
                if not lbl:
                    for raw_iid, dst in iid_alias.items():
                        if dst == canon and str(categories.get(raw_iid, "")).strip():
                            lbl = str(categories[raw_iid]).strip()
                            break
                if lbl:
                    new_cats[int(canon)] = lbl
            categories = new_cats
        log.info(
            "Stage 3.5 [%s]: dedup fused %d → %d iids (%d clusters merged)",
            subscan_id, len(iid_alias), len(inst_pts), len(merges),
        )

    # Step 3: dynamic-instance flag (centroid drift across frames).
    pf_cents = per_frame_centroids(pf_inst_pts)
    jitter = centroid_jitter(pf_cents)
    dyn_thresh_m = float(getattr(cfg, "dynamic_centroid_var_threshold_m", 0.10))
    is_dynamic = {
        int(iid): bool(jitter.get(int(iid), 0.0) > dyn_thresh_m)
        for iid in inst_pts
    }

    # Step 4: unstable flag (too few points to descriptor-match reliably).
    min_pts = int(getattr(cfg, "min_instance_points", 50))
    is_unstable = {int(iid): bool(p.shape[0] < min_pts) for iid, p in inst_pts.items()}

    # Step 5: visibility matrix (n_instances, n_frames).
    instance_ids = sorted(int(iid) for iid in inst_pts)
    frame_ids = sorted(int(f) for f in per_frame_masks)
    vis = per_frame_visibility(pf_inst_pts, instance_ids, frame_ids)

    # Step 6: per-instance normalized cloud + descriptor + oriented bbox.
    n_bins = int(getattr(cfg, "esf_n_bins", 64))
    n_samples = int(getattr(cfg, "esf_n_samples", 4096))
    rng = np.random.default_rng(0)
    inst_pts_norm: Dict[int, np.ndarray] = {}
    bbox_diag: Dict[int, float] = {}
    bbox_orientation_map: Dict[int, np.ndarray] = {}
    bbox_extents_map: Dict[int, np.ndarray] = {}
    descriptors: Dict[int, np.ndarray] = {}
    for iid in instance_ids:
        pts_norm, diag = bbox_diag_normalize(inst_pts[iid])
        inst_pts_norm[iid] = pts_norm
        bbox_diag[iid] = float(diag)
        rotation, extents = oriented_bbox(inst_pts_for_obb.get(iid, inst_pts[iid]))
        bbox_orientation_map[iid] = rotation
        bbox_extents_map[iid] = extents
        descriptors[iid] = instance_descriptor(
            pts_norm, n_bins=n_bins, n_samples=n_samples, rng=rng,
        )

    # Step 7 (optional): global floor-plane scale.
    floor_scale: Optional[float] = None
    fp_cfg = getattr(cfg, "floor_plane_scale", None)
    if fp_cfg is not None and bool(getattr(fp_cfg, "enabled", False)):
        all_pts = np.concatenate(list(inst_pts.values()), axis=0).astype(np.float32)
        floor_scale = _floor_plane_scale(
            all_pts,
            distance_threshold=float(fp_cfg.distance_threshold),
            ransac_n=int(fp_cfg.ransac_n),
            num_iterations=int(fp_cfg.num_iterations),
            nominal_scene_height_m=float(getattr(fp_cfg, "nominal_scene_height_m", 2.5)),
        )

    # Step 8: scene-graph edges (k-NN proximity over instance centroids).
    sg_cfg = getattr(cfg, "scene_graph", None)
    edges: List[Dict[str, object]] = []
    if sg_cfg is None or bool(getattr(sg_cfg, "enabled", True)):
        edges = build_scene_graph_edges(
            centroids, instance_ids,
            k=int(getattr(sg_cfg, "k_neighbors", 5)) if sg_cfg is not None else 5,
            radius=float(getattr(sg_cfg, "radius_m", 1.0)) if sg_cfg is not None else 1.0,
            up_axis=str(getattr(sg_cfg, "up_axis", "y")) if sg_cfg is not None else "y",
            vertical_threshold=(
                float(getattr(sg_cfg, "vertical_relation_threshold_m", 0.25))
                if sg_cfg is not None else 0.25
            ),
        )

    n_dyn = sum(1 for v in is_dynamic.values() if v)
    n_unstable = sum(1 for v in is_unstable.values() if v)
    log.info(
        "Stage 3.5 [%s]: %d instances (%d dyn, %d unstable) across %d frames; "
        "%d scene-graph edges, desc_dim=%d, floor_scale=%s",
        subscan_id, len(instance_ids), n_dyn, n_unstable, len(frame_ids),
        len(edges), 3 + n_bins, "%.3f" % floor_scale if floor_scale is not None else "None",
    )

    return PriorArtifact(
        subscan_id=subscan_id,
        instance_ids=instance_ids,
        instance_points={iid: inst_pts[iid].astype(np.float32) for iid in instance_ids},
        instance_points_norm=inst_pts_norm,
        instance_centroids={iid: centroids[iid].astype(np.float32) for iid in instance_ids},
        bbox_diag=bbox_diag,
        bbox_orientation=bbox_orientation_map,
        bbox_extents=bbox_extents_map,
        iid_alias=iid_alias,
        dedup_summary={
            "n_raw_iids": int(len(iid_alias)),
            "n_canonical_iids": int(len(set(iid_alias.values()))),
            "n_merged_clusters": int(len(merges)),
            "n_iids_merged_away": int(len(iid_alias) - len(set(iid_alias.values()))),
        },
        dedup_merges=[{"canonical": int(c), "members": [int(m) for m in members]}
                      for c, members in merges],
        visibility=vis,
        frame_ids=frame_ids,
        is_dynamic=is_dynamic,
        is_unstable=is_unstable,
        descriptors=descriptors,
        categories={int(iid): str(categories.get(iid, "")) for iid in instance_ids},
        floor_scale=floor_scale,
        edges=edges,
    )


__all__ = ["PriorArtifact", "run_fusion"]
