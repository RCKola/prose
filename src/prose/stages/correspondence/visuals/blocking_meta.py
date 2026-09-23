"""Blocking meta-composer: per-bin loop that owns phases C–F.

Replaces a clean per-phase chain because blocking touches every phase
in a way that doesn't fit the canonical skeleton.

Pure-function pieces (bin schedule, partition, active-bin enumeration,
intra-bin coalesce) are self-contained copies of the legacy v2 logic.
The per-pair runner ``run_blocking_pipeline(pipeline, ctx)`` does:

    1. extract lightweight per-instance features from ctx (centroid + OBB + h_center)
    2. derive bin schedule (uniform or quantile) on joint ref+src heights
    3. partition each side; intra-bin coalesce; enumerate active bins
    4. Stage-A per-bin frame ranking (when per-bin selection is on)
    5. for each active bin:
         - Stage-B frame selection
         - assign markers (this bin's iids only)
         - render frame mosaic + (optional) ref/src BEVs + (optional) crops mosaic
         - delegate to pipeline.prompt.build(...) with bin_ctx
         - call pipeline.vlm.call(...)
         - delegate to pipeline.parser.parse(...) → list of (src_iid, ref_iid)
    6. pass aggregated proposals + features to pipeline.resolver.resolve(...)

The prompt/parser/resolver are pluggable; their concrete implementations
(thinking-mode blocking prompt, tuple-JSON parser, geo_correction
resolver) are added in tasks #4 and #5.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

log = logging.getLogger(__name__)

import numpy as np
from PIL import Image

from ..context import (
    CorrespondenceResult,
    InstanceSet,
    PairContext,
    PromptBundle,
    VLMResult,
)
from ..shared.frame_io import load_rgb
from ..views.per_bin import (
    PerBinFrameSelectionConfig,
    build_iid_frame_ranking,
    select_bin_frames,
)
from .mosaic import assign_marker_ids


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    enabled: bool = True
    n_bins: int = 5
    overlap_frac: float = 0.2
    bin_mode: str = "uniform"          # "uniform" | "quantile"

    # Derived per-pair; kept on the config for diagnostics.
    h_bw_m: float = 0.5
    h_step_m: float = 0.4

    coalesce_pop_threshold: int = 12
    coalesce_radius_frac: float = 0.5
    coalesce_obb_extent_frac: float = 0.40

    # Frame composition knobs.
    use_bev: bool = True
    subpanel_longside_px: int = 512
    bev_resolution_px: int = 1024
    bev_point_size_px: int = 6
    bev_bg_rgb: Tuple[int, int, int] = (245, 245, 245)
    bev_colormap_when_no_color: str = "viridis"
    bev_projections: Tuple[str, ...] = ("bev",)
    marker_namespace: str = "shared_distinct"

    # Mosaic render knobs.
    draw_circles: bool = False
    mosaic_font_scale: float = 0.9
    mosaic_mask_outline_thickness: int = 1
    mosaic_max_cols: int = 0    # 0 = strip (all frames in one row); 3 = 3-col grid

    # Per-bin pairwise crops.
    crops_enabled: bool = False
    crops_top_k: int = 6
    crops_pad_frac: float = 0.4
    crops_min_side_px: int = 256
    crops_outline_thickness: int = 1
    crops_panel_px: int = 384

    # Per-bin visual composer selector. "legacy" = frame mosaic (+BEV
    # +crops, as wired below); "crops_context" = outlined crops of every
    # bin instance + one max-coverage context frame per scene.
    bin_visuals: str = "legacy"
    context_frame_enabled: bool = True

    # Thinking-mode toggle (consumed by the prompt builder; runner just
    # passes ``enable_thinking`` to the VLM invoker).
    enable_thinking_mode: bool = False
    thinking_max_new_tokens: int = 4096

    per_bin_frame_selection: PerBinFrameSelectionConfig = field(
        default_factory=PerBinFrameSelectionConfig
    )

    # Pairwise mode: expand each bin into N_ref VLM calls, each showing
    # 1 REF crop vs all SRC candidates in that bin. More calls, simpler task.
    pairwise: bool = False

    # Spatial-only separate VLM call per bin (sv3 path).
    # Renders BEV/axis views only, no crops. Merged with appearance proposals.
    spatial_match_enabled: bool = False
    spatial_merge_strategy: str = "intersection"  # "intersection" | "union"
    spatial_bin_window: int = 1  # 1 = per-bin; 2 = sliding window of 2 adjacent bins

    # Cross-bin stitching: re-query IIDs that never co-occurred in any bin.
    cross_bin_stitch: bool = False
    cross_bin_max_per_side: int = 15
    cross_bin_window: int = 0  # 0 = single catchall group; 3 = sliding window of 3 adjacent bins


# ---------------------------------------------------------------------------
# Lightweight per-instance features (geometric only)
# ---------------------------------------------------------------------------

def _height_axis_index(up_axis: str) -> int:
    ax = (up_axis or "z").strip().lower()
    if ax == "y":
        return 1
    if ax in ("z", ""):
        return 2
    raise ValueError(f"up_axis must be 'y' or 'z' (got {up_axis!r})")


@dataclass
class _InstanceFeat:
    iid: int
    centroid: np.ndarray              # (3,) f64
    obb_extents: np.ndarray           # (3,) f64
    h_center: float                   # centroid[h_axis]
    z_min: float                      # points[:, h_axis].min()


def extract_features(side: InstanceSet, *, up_axis: str) -> Dict[int, _InstanceFeat]:
    h_axis = _height_axis_index(up_axis)
    out: Dict[int, _InstanceFeat] = {}
    for iid in side.iids:
        c = side.centroids.get(int(iid))
        e = side.obbs.get(int(iid))
        p = side.points.get(int(iid))
        if c is None or e is None or p is None or len(p) == 0:
            continue
        c64 = np.asarray(c, dtype=np.float64).reshape(-1)
        e64 = np.asarray(e, dtype=np.float64).reshape(-1)
        p64 = np.asarray(p, dtype=np.float64)
        if c64.shape != (3,) or e64.shape != (3,) or p64.shape[1] != 3:
            continue
        out[int(iid)] = _InstanceFeat(
            iid=int(iid),
            centroid=c64,
            obb_extents=e64,
            h_center=float(c64[h_axis]),
            z_min=float(p64[:, h_axis].min()),
        )
    return out


# ---------------------------------------------------------------------------
# Bin schedule
# ---------------------------------------------------------------------------

BinKey = Tuple[int, int]


def derive_h_bin_geometry(
    h_min: float, h_max: float, *, n_bins: int, overlap_frac: float,
    min_bw_m: float = 0.05,
) -> Tuple[float, float]:
    N = max(int(n_bins), 1)
    f = float(overlap_frac)
    if not (math.isfinite(h_min) and math.isfinite(h_max)):
        return max(float(min_bw_m), 1e-3), max(float(min_bw_m), 1e-3)
    H = float(h_max) - float(h_min)
    if N <= 1 or H <= float(min_bw_m):
        bw = max(H, float(min_bw_m))
        return bw, bw
    f = min(max(f, 0.0), 0.95)
    denom = (N - 1) * (1.0 - f) + 1.0
    w = H / denom
    s = (1.0 - f) * w
    return float(w), float(s)


def compute_h_schedule(
    h_min: float, h_max: float, *, h_bw_m: float, h_step_m: float,
) -> List[Tuple[float, float]]:
    h_bw_m = float(h_bw_m)
    h_step_m = float(h_step_m)
    if not math.isfinite(h_min) or not math.isfinite(h_max) or h_bw_m <= 0.0:
        return [(0.0, h_bw_m if h_bw_m > 0 else 1.0)]
    if h_max <= h_min:
        return [(h_min, h_min + h_bw_m)]
    if (h_max - h_min) <= h_bw_m:
        return [(h_max - h_bw_m, h_max)]
    step = max(h_step_m, 1e-6)
    bins: List[Tuple[float, float]] = []
    lo = float(h_min)
    while lo + h_bw_m < h_max:
        bins.append((lo, lo + h_bw_m))
        lo += step
    last_lo = float(h_max) - h_bw_m
    if not bins or last_lo > bins[-1][0] + 1e-6:
        bins.append((last_lo, float(h_max)))
    return bins


def compute_h_schedule_quantile(
    heights: Sequence[float], *, n_bins: int, overlap_frac: float,
) -> List[Tuple[float, float]]:
    h = np.asarray(list(heights), dtype=np.float64)
    if h.size == 0 or n_bins <= 0:
        return [(0.0, 1.0)]
    h_lo = float(h.min()); h_hi = float(h.max())
    if h_hi <= h_lo:
        return [(h_lo, h_lo + 1.0)]
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(h, qs)
    native: List[Tuple[float, float]] = [
        (float(edges[i]), float(edges[i + 1])) for i in range(int(n_bins))
    ]
    widths = [max(hi - lo, 1e-6) for lo, hi in native]
    f = float(overlap_frac)
    out: List[Tuple[float, float]] = []
    for i, (lo, hi) in enumerate(native):
        ext_lo = f * (widths[i] if i == 0 else min(widths[i], widths[i - 1]))
        ext_hi = f * (widths[i] if i == len(native) - 1
                      else min(widths[i], widths[i + 1]))
        out.append((max(h_lo, lo - ext_lo), min(h_hi, hi + ext_hi)))
    return out


# ---------------------------------------------------------------------------
# Partition + coalesce + active-bin enumeration
# ---------------------------------------------------------------------------

@dataclass
class BinPartition:
    by_bin: Dict[BinKey, List[int]] = field(default_factory=dict)
    tag_for_iid: Dict[int, BinKey] = field(default_factory=dict)


@dataclass
class BinJob:
    key: BinKey
    ref_iids: List[int]
    src_iids: List[int]
    z_range_m: Tuple[float, float]


@dataclass
class BinCoalesceReport:
    remap: Dict[int, int] = field(default_factory=dict)
    groups: List[List[int]] = field(default_factory=list)
    n_dropped: int = 0
    bins_coalesced: List[BinKey] = field(default_factory=list)


def partition_side(
    features: Dict[int, _InstanceFeat],
    *, schedule: Sequence[Tuple[float, float]],
) -> BinPartition:
    part = BinPartition()
    for iid, f in features.items():
        h = float(f.h_center)
        primary_set = False
        for idx, (lo, hi) in enumerate(schedule):
            in_bin = (h >= lo) and (h < hi or idx == len(schedule) - 1)
            if not in_bin:
                continue
            key: BinKey = (int(idx), 0)
            if not primary_set:
                part.tag_for_iid[int(iid)] = key
                primary_set = True
            part.by_bin.setdefault(key, []).append(int(iid))
    return part


def intra_bin_coalesce_side(
    partition: BinPartition,
    features: Dict[int, _InstanceFeat],
    *, cfg: BlockingConfig,
) -> BinCoalesceReport:
    report = BinCoalesceReport()
    pop_thresh = max(1, int(cfg.coalesce_pop_threshold))
    obb_frac = float(cfg.coalesce_obb_extent_frac)
    rad_frac = float(cfg.coalesce_radius_frac)
    radius = rad_frac * float(cfg.h_bw_m)
    rad2 = radius * radius
    for key, members in partition.by_bin.items():
        if len(members) <= pop_thresh:
            continue
        N = len(members)
        parent = {m: m for m in members}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        centroids = {m: features[m].centroid for m in members if m in features}
        extents = {m: features[m].obb_extents for m in members if m in features}
        for i in range(N):
            mi = members[i]
            if mi not in centroids:
                continue
            ci = centroids[mi]; ei = extents[mi]
            for j in range(i + 1, N):
                mj = members[j]
                if mj not in centroids:
                    continue
                d2 = float(np.sum((ci - centroids[mj]) ** 2))
                if d2 > rad2:
                    continue
                si = np.sort(ei)[::-1]
                sj = np.sort(extents[mj])[::-1]
                ratios = np.minimum(si, sj) / np.maximum(np.maximum(si, sj), 1e-9)
                if float(ratios.min()) < (1.0 - obb_frac):
                    continue
                union(mi, mj)
        clusters: Dict[int, List[int]] = {}
        for m in members:
            if m in centroids:
                clusters.setdefault(find(m), []).append(m)
        any_merge = False
        for _root, group in clusters.items():
            if len(group) < 2:
                continue
            group_sorted = sorted(group)
            canonical = group_sorted[0]
            report.groups.append(group_sorted)
            for m in group_sorted[1:]:
                report.remap[int(m)] = int(canonical)
                report.n_dropped += 1
            any_merge = True
        if any_merge:
            report.bins_coalesced.append(key)
    return report


def apply_coalesce_to_partition(
    partition: BinPartition, remap: Dict[int, int],
) -> BinPartition:
    if not remap:
        return partition
    out = BinPartition()
    drop = set(int(k) for k in remap.keys())
    for key, members in partition.by_bin.items():
        kept = [m for m in members if int(m) not in drop]
        if kept:
            out.by_bin[key] = kept
    out.tag_for_iid = {
        int(iid): tag for iid, tag in partition.tag_for_iid.items() if int(iid) not in drop
    }
    return out


def enumerate_active_bins(
    ref_part: BinPartition, src_part: BinPartition,
    *, schedule: Sequence[Tuple[float, float]],
) -> List[BinJob]:
    active_keys = sorted(set(ref_part.by_bin.keys()) & set(src_part.by_bin.keys()))
    jobs: List[BinJob] = []
    for key in active_keys:
        bin_idx = int(key[0])
        if 0 <= bin_idx < len(schedule):
            z_lo, z_hi = float(schedule[bin_idx][0]), float(schedule[bin_idx][1])
        else:
            z_lo, z_hi = 0.0, 0.0
        jobs.append(BinJob(
            key=key,
            ref_iids=sorted(int(i) for i in ref_part.by_bin[key]),
            src_iids=sorted(int(i) for i in src_part.by_bin[key]),
            z_range_m=(z_lo, z_hi),
        ))
    return jobs


# ---------------------------------------------------------------------------
# Per-pair runner
# ---------------------------------------------------------------------------

@dataclass
class BinContext:
    """Bin-scoped info handed to the prompt builder + parser."""
    key: BinKey
    z_range_m: Tuple[float, float]
    ref_iids: List[int]
    src_iids: List[int]
    ref_marker_ids: List[int]
    src_marker_ids: List[int]
    ref_marker_to_iid: Dict[int, int]
    src_marker_to_iid: Dict[int, int]
    asymmetric: bool
    use_bev: bool


@dataclass
class _PendingBin:
    """One bin's fully-built request, awaiting the batched VLM call."""
    job: "BinJob"
    bin_ctx: "BinContext"
    images: List[Image.Image]
    prompt: "PromptBundle"
    n_ref: int
    n_src: int
    asymmetric: bool


# ---------------------------------------------------------------------------
# Pluggable per-bin visual composer
# ---------------------------------------------------------------------------

@dataclass
class BinVisualInputs:
    """Everything a per-bin visual composer needs for one active bin.

    Carries the bin-restricted instance/mask views plus enough pair-level
    state (3D cloud, partition, frame loaders, shared stat caches) for any
    composer to render. Composers read the knobs they need off ``cfg``.
    """
    cfg: "BlockingConfig"
    job: "BinJob"
    up_axis: str
    image_rotation_k: int
    depth_dir: Optional[Path]
    # bin instances
    ref_iids_bin: List[int]
    src_iids_bin: List[int]
    ref_markers_bin: Dict[int, int]          # iid -> marker id
    src_markers_bin: Dict[int, int]
    # frames
    bin_ref_fids: List[int]                  # Stage-B selection (legacy mosaic)
    bin_src_fids: List[int]
    ref_color_paths: Dict[int, Path]
    src_color_paths: Dict[int, Path]
    get_ref_bgr: Callable[[int], np.ndarray]
    get_src_bgr: Callable[[int], np.ndarray]
    # masks
    ref_masks_bin: Dict[int, Dict[int, np.ndarray]]   # bin-restricted
    src_masks_bin: Dict[int, Dict[int, np.ndarray]]
    ref_per_frame_masks: Dict[int, Dict[int, np.ndarray]]  # full
    src_per_frame_masks: Dict[int, Dict[int, np.ndarray]]
    # features / partition (BEV context)
    ref_feats: Dict[int, "_InstanceFeat"]
    src_feats: Dict[int, "_InstanceFeat"]
    ref_part: "BinPartition"
    src_part: "BinPartition"
    # fused 3D cloud
    ref_points: Optional[np.ndarray]
    ref_colors: Optional[np.ndarray]
    src_points: Optional[np.ndarray]
    src_colors: Optional[np.ndarray]
    # shared mutable per-pair caches (crop frame stats)
    ref_stats_cache: Dict[int, dict]
    src_stats_cache: Dict[int, dict]
    # per-side Stage-A frame candidates (None when PBFS disabled)
    ref_cands: Optional[Any] = None
    src_cands: Optional[Any] = None


class BinVisualComposer(Protocol):
    """Renders the image list handed to the VLM for one active bin."""

    def compose(self, inp: BinVisualInputs) -> List[Image.Image]: ...


def _abs_ids(paths: Dict[int, Path]) -> List[int]:
    out = [0] * (max(paths.keys()) + 1)
    for k, p in paths.items():
        stem = Path(p).stem
        try:
            out[int(k)] = int(stem.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            out[int(k)] = int(k)
    return out


def run_blocking_pipeline(pipeline, ctx: PairContext) -> CorrespondenceResult:
    """Per-pair blocking runner.

    ``pipeline`` is the ``CorrespondencePipeline``; this function reads
    ``pipeline.prompt`` / ``parser`` / ``vlm`` / ``resolver`` and
    the ``BlockingConfig`` attached as ``pipeline.blocking_cfg``.
    """
    cfg: BlockingConfig = getattr(pipeline, "blocking_cfg", None) or BlockingConfig()
    up_axis = str(getattr(ctx, "up_axis", "z"))

    # (1) features
    ref_feats = extract_features(ctx.ref, up_axis=up_axis)
    src_feats = extract_features(ctx.src, up_axis=up_axis)

    # (2) schedule
    all_h = [f.h_center for f in ref_feats.values()] + [f.h_center for f in src_feats.values()]
    if all_h:
        h_min, h_max = min(all_h), max(all_h)
    else:
        h_min, h_max = 0.0, float(cfg.h_bw_m)
    h_bw, h_step = derive_h_bin_geometry(
        h_min, h_max, n_bins=cfg.n_bins, overlap_frac=cfg.overlap_frac,
    )
    cfg.h_bw_m = float(h_bw); cfg.h_step_m = float(h_step)
    if str(cfg.bin_mode).lower() == "quantile":
        schedule = compute_h_schedule_quantile(
            all_h, n_bins=cfg.n_bins, overlap_frac=cfg.overlap_frac,
        )
    else:
        schedule = compute_h_schedule(h_min, h_max, h_bw_m=h_bw, h_step_m=h_step)

    # (3) partition + coalesce
    ref_part = partition_side(ref_feats, schedule=schedule)
    src_part = partition_side(src_feats, schedule=schedule)
    ref_per_frame_masks = dict(ctx.ref.per_frame_masks)
    src_per_frame_masks = dict(ctx.src.per_frame_masks)
    ref_report = intra_bin_coalesce_side(ref_part, ref_feats, cfg=cfg)
    src_report = intra_bin_coalesce_side(src_part, src_feats, cfg=cfg)
    if ref_report.remap:
        drop = set(ref_report.remap.keys())
        ref_feats = {i: f for i, f in ref_feats.items() if i not in drop}
        ref_per_frame_masks = {
            fid: {i: m for i, m in inst.items() if i not in drop}
            for fid, inst in ref_per_frame_masks.items()
        }
        ref_part = apply_coalesce_to_partition(ref_part, ref_report.remap)
    if src_report.remap:
        drop = set(src_report.remap.keys())
        src_feats = {i: f for i, f in src_feats.items() if i not in drop}
        src_per_frame_masks = {
            fid: {i: m for i, m in inst.items() if i not in drop}
            for fid, inst in src_per_frame_masks.items()
        }
        src_part = apply_coalesce_to_partition(src_part, src_report.remap)
    jobs = enumerate_active_bins(ref_part, src_part, schedule=schedule)

    # (4) marker assignment (global; bin loop subsets per bin)
    ref_marker_map, src_marker_map = assign_marker_ids(
        list(ref_feats.keys()), list(src_feats.keys()),
        namespace=str(cfg.marker_namespace),
    )
    ref_marker_to_iid = {int(m): int(i) for i, m in ref_marker_map.items()}
    src_marker_to_iid = {int(m): int(i) for i, m in src_marker_map.items()}

    # (5) Stage-A per-bin frame ranking (if PBFS enabled)
    pbfs = cfg.per_bin_frame_selection
    ref_cands = src_cands = None
    if bool(pbfs.enabled):
        ref_cands = build_iid_frame_ranking(
            ref_per_frame_masks,
            top_n=int(pbfs.top_n_per_iid),
            min_pixel_area=int(pbfs.min_pixel_area),
        )
        src_cands = build_iid_frame_ranking(
            src_per_frame_masks,
            top_n=int(pbfs.top_n_per_iid),
            min_pixel_area=int(pbfs.min_pixel_area),
        )

    # Lazy BGR cache.
    ref_bgr: Dict[int, np.ndarray] = {}
    src_bgr: Dict[int, np.ndarray] = {}

    def _get_ref_bgr(fid: int) -> np.ndarray:
        if fid not in ref_bgr:
            ref_bgr[fid] = load_rgb(ctx.ref_frames.color_paths[fid])
        return ref_bgr[fid]

    def _get_src_bgr(fid: int) -> np.ndarray:
        if fid not in src_bgr:
            src_bgr[fid] = load_rgb(ctx.src_frames.color_paths[fid])
        return src_bgr[fid]

    # Lazy crops stats.
    ref_stats_cache: Dict[int, dict] = {}
    src_stats_cache: Dict[int, dict] = {}

    # Per-bin visual composer (phase D) — pluggable; defaults to legacy.
    bin_visuals: BinVisualComposer = getattr(pipeline, "bin_visuals", None)
    if bin_visuals is None:
        from .bin_visuals import LegacyBinVisuals
        bin_visuals = LegacyBinVisuals()

    # Bin loop.
    all_proposals: List[Tuple[int, int]] = []      # (src_iid, ref_iid)
    raw_chunks: List[str] = []
    per_bin_audit: List[Dict[str, Any]] = []
    pending: List[_PendingBin] = []                # built bins, awaiting batch

    # Points + colors for the BEV passes (fused 3D cloud lives in InstanceSet.extra).
    ref_points = ctx.ref.extra.get("points")
    ref_colors = ctx.ref.extra.get("colors")
    src_points = ctx.src.extra.get("points")
    src_colors = ctx.src.extra.get("colors")

    for job in jobs:
        ref_iids_bin = [i for i in job.ref_iids if i in ref_feats]
        src_iids_bin = [i for i in job.src_iids if i in src_feats]
        if not ref_iids_bin or not src_iids_bin:
            continue

        # Stage-B per-bin frame selection.
        ref_low_vis: List[int] = []
        src_low_vis: List[int] = []
        bin_ref_fids: Sequence[int] = ctx.ref_frames.frame_indices
        bin_src_fids: Sequence[int] = ctx.src_frames.frame_indices
        if ref_cands is not None and src_cands is not None:
            bin_ref_fids, ref_low_vis = select_bin_frames(
                ref_iids_bin, ref_cands, k=int(pbfs.k_per_bin),
            )
            bin_src_fids, src_low_vis = select_bin_frames(
                src_iids_bin, src_cands, k=int(pbfs.k_per_bin),
            )
            if ref_low_vis:
                ref_iids_bin = [i for i in ref_iids_bin if i not in set(ref_low_vis)]
            if src_low_vis:
                src_iids_bin = [i for i in src_iids_bin if i not in set(src_low_vis)]
            if not ref_iids_bin or not src_iids_bin or not bin_ref_fids or not bin_src_fids:
                continue

        ref_markers_bin = {int(i): int(ref_marker_map[int(i)]) for i in ref_iids_bin
                           if int(i) in ref_marker_map}
        src_markers_bin = {int(i): int(src_marker_map[int(i)]) for i in src_iids_bin
                           if int(i) in src_marker_map}
        if not ref_markers_bin or not src_markers_bin:
            continue

        # Restricted mask dicts (this bin only).
        ref_keep = set(ref_iids_bin)
        src_keep = set(src_iids_bin)
        ref_masks_bin = {
            fid: {iid: m for iid, m in inst.items() if int(iid) in ref_keep}
            for fid, inst in ref_per_frame_masks.items()
        }
        src_masks_bin = {
            fid: {iid: m for iid, m in inst.items() if int(iid) in src_keep}
            for fid, inst in src_per_frame_masks.items()
        }

        # Pairwise expansion: one VLM call per REF in this bin.
        if cfg.pairwise:
            for ref_iid in ref_iids_bin:
                pw_ref_markers = {int(ref_iid): int(ref_marker_map[int(ref_iid)])}
                pw_ref_masks = {
                    fid: {iid: m for iid, m in inst.items() if int(iid) == int(ref_iid)}
                    for fid, inst in ref_per_frame_masks.items()
                }
                vis_inp = BinVisualInputs(
                    cfg=cfg, job=job, up_axis=up_axis,
                    image_rotation_k=int(ctx.image_rotation_k),
                    depth_dir=ctx.depth_dir,
                    ref_iids_bin=[int(ref_iid)], src_iids_bin=list(src_iids_bin),
                    ref_markers_bin=dict(pw_ref_markers),
                    src_markers_bin=dict(src_markers_bin),
                    bin_ref_fids=[int(f) for f in bin_ref_fids],
                    bin_src_fids=[int(f) for f in bin_src_fids],
                    ref_color_paths=dict(ctx.ref_frames.color_paths),
                    src_color_paths=dict(ctx.src_frames.color_paths),
                    get_ref_bgr=_get_ref_bgr, get_src_bgr=_get_src_bgr,
                    ref_masks_bin=pw_ref_masks, src_masks_bin=src_masks_bin,
                    ref_per_frame_masks=ref_per_frame_masks,
                    src_per_frame_masks=src_per_frame_masks,
                    ref_feats=ref_feats, src_feats=src_feats,
                    ref_part=ref_part, src_part=src_part,
                    ref_points=ref_points, ref_colors=ref_colors,
                    src_points=src_points, src_colors=src_colors,
                    ref_stats_cache=ref_stats_cache, src_stats_cache=src_stats_cache,
                    ref_cands=ref_cands, src_cands=src_cands,
                )
                images_pil = bin_visuals.compose(vis_inp)
                if not images_pil:
                    continue
                pw_bin_ctx = BinContext(
                    key=job.key, z_range_m=job.z_range_m,
                    ref_iids=[int(ref_iid)], src_iids=list(src_iids_bin),
                    ref_marker_ids=sorted(pw_ref_markers.values()),
                    src_marker_ids=sorted(src_markers_bin.values()),
                    ref_marker_to_iid={int(m): int(i) for i, m in pw_ref_markers.items()},
                    src_marker_to_iid={int(m): int(i) for i, m in src_markers_bin.items()},
                    asymmetric=True, use_bev=bool(cfg.use_bev),
                )
                prompt: PromptBundle = pipeline.prompt.build(
                    ctx, images_pil, bin_ctx=pw_bin_ctx)
                pending.append(_PendingBin(
                    job=job, bin_ctx=pw_bin_ctx, images=images_pil, prompt=prompt,
                    n_ref=1, n_src=len(src_markers_bin), asymmetric=True,
                ))
            continue

        # Visuals via the pluggable per-bin composer (phase D).
        vis_inp = BinVisualInputs(
            cfg=cfg, job=job, up_axis=up_axis,
            image_rotation_k=int(ctx.image_rotation_k),
            depth_dir=ctx.depth_dir,
            ref_iids_bin=list(ref_iids_bin), src_iids_bin=list(src_iids_bin),
            ref_markers_bin=dict(ref_markers_bin),
            src_markers_bin=dict(src_markers_bin),
            bin_ref_fids=[int(f) for f in bin_ref_fids],
            bin_src_fids=[int(f) for f in bin_src_fids],
            ref_color_paths=dict(ctx.ref_frames.color_paths),
            src_color_paths=dict(ctx.src_frames.color_paths),
            get_ref_bgr=_get_ref_bgr, get_src_bgr=_get_src_bgr,
            ref_masks_bin=ref_masks_bin, src_masks_bin=src_masks_bin,
            ref_per_frame_masks=ref_per_frame_masks,
            src_per_frame_masks=src_per_frame_masks,
            ref_feats=ref_feats, src_feats=src_feats,
            ref_part=ref_part, src_part=src_part,
            ref_points=ref_points, ref_colors=ref_colors,
            src_points=src_points, src_colors=src_colors,
            ref_stats_cache=ref_stats_cache, src_stats_cache=src_stats_cache,
            ref_cands=ref_cands, src_cands=src_cands,
        )
        images_pil = bin_visuals.compose(vis_inp)
        if not images_pil:
            continue

        # Prompt build (phase E). The VLM call is deferred so every bin of
        # this pair goes out in one batched LLM.chat below.
        asym = (len(ref_markers_bin) == 1) or (len(src_markers_bin) == 1)
        bin_ctx = BinContext(
            key=job.key,
            z_range_m=job.z_range_m,
            ref_iids=list(ref_iids_bin),
            src_iids=list(src_iids_bin),
            ref_marker_ids=sorted(ref_markers_bin.values()),
            src_marker_ids=sorted(src_markers_bin.values()),
            ref_marker_to_iid={int(m): int(i) for i, m in ref_markers_bin.items()},
            src_marker_to_iid={int(m): int(i) for i, m in src_markers_bin.items()},
            asymmetric=bool(asym),
            use_bev=bool(cfg.use_bev),
        )
        prompt: PromptBundle = pipeline.prompt.build(ctx, images_pil, bin_ctx=bin_ctx)
        pending.append(_PendingBin(
            job=job, bin_ctx=bin_ctx, images=images_pil, prompt=prompt,
            n_ref=len(ref_markers_bin), n_src=len(src_markers_bin),
            asymmetric=bool(asym),
        ))

    # (6) Batched VLM call — all bins of this pair in one LLM.chat so the
    # engine applies continuous batching across them (phase F).
    mnt = int(cfg.thinking_max_new_tokens) if cfg.enable_thinking_mode else None
    raws: List[str] = []
    if pending:
        raws = pipeline.vlm.call_batch(
            [(p.images, p.prompt.system, p.prompt.user) for p in pending],
            max_new_tokens=mnt,
            enable_thinking=cfg.enable_thinking_mode or None,
        )

    # (7) Parse + debug-dump per bin.
    sample_dir = getattr(pipeline, "vlm_sample_dir", None)
    for p, raw in zip(pending, raws):
        try:
            vlm_result: VLMResult = pipeline.parser.parse(
                raw, ctx, bin_ctx=p.bin_ctx)
        except Exception as e:  # noqa: BLE001
            vlm_result = VLMResult(raw_text=raw or "", proposals=[],
                                   audit={"parse_error": str(e)})

        if sample_dir is not None:
            try:
                bin_label = f"bin_{int(p.job.key[0])}_{int(p.job.key[1])}"
                if p.n_ref == 1 and cfg.pairwise:
                    bin_label += f"_ref{p.bin_ctx.ref_iids[0]}"
                bdir = Path(sample_dir) / str(ctx.pair_id) / bin_label
                bdir.mkdir(parents=True, exist_ok=True)
                for ii, im in enumerate(p.images):
                    im.convert("RGB").save(bdir / f"img_{ii:02d}.jpg")
                (bdir / "prompt.txt").write_text(
                    f"=== SYSTEM ===\n{p.prompt.system}\n\n"
                    f"=== USER ===\n{p.prompt.user}\n")
                (bdir / "vlm_raw.txt").write_text(raw or "")
            except Exception:  # noqa: BLE001
                pass

        all_proposals.extend(vlm_result.proposals)
        raw_chunks.append(f"--- bin {p.job.key} ---\n{raw}")
        per_bin_audit.append({
            "bin_key": [int(p.job.key[0]), int(p.job.key[1])],
            "n_ref": p.n_ref,
            "n_src": p.n_src,
            "z_range_m": [float(p.job.z_range_m[0]), float(p.job.z_range_m[1])],
            "asymmetric": p.asymmetric,
            "n_proposals": len(vlm_result.proposals),
        })

    # (7b) Spatial-only separate VLM call (sv3 path).
    # Renders BEV for each bin (or sliding window of adjacent bins), asks a
    # spatial-only prompt, then merges with appearance proposals.
    if cfg.spatial_match_enabled and ref_points is not None and src_points is not None:
        from .bev import render_bev
        from ..prompts.spatial_bin import SpatialBinPrompt
        spatial_prompt_builder = SpatialBinPrompt()
        spatial_pending: List[_PendingBin] = []
        spatial_labels: List[str] = []
        sw = max(1, int(cfg.spatial_bin_window))
        projections = getattr(cfg, "bev_projections", ("bev",)) or ("bev",)

        for start_idx in range(max(1, len(jobs) - sw + 1)):
            end_idx = min(start_idx + sw, len(jobs))
            window_jobs = jobs[start_idx:end_idx]
            win_ref_iids: List[int] = []
            win_src_iids: List[int] = []
            for wj in window_jobs:
                win_ref_iids.extend(i for i in wj.ref_iids if i in ref_feats
                                    and i not in win_ref_iids)
                win_src_iids.extend(i for i in wj.src_iids if i in src_feats
                                    and i not in win_src_iids)
            if not win_ref_iids or not win_src_iids:
                continue

            win_ref_markers = {int(i): int(ref_marker_map[int(i)])
                               for i in win_ref_iids if int(i) in ref_marker_map}
            win_src_markers = {int(i): int(src_marker_map[int(i)])
                               for i in win_src_iids if int(i) in src_marker_map}
            if not win_ref_markers or not win_src_markers:
                continue

            h_bin = int(window_jobs[0].key[0])
            win_ref_keep = set(win_ref_iids)
            win_src_keep = set(win_src_iids)
            ref_ctx_iids = {iid for (hb, _vb), members in ref_part.by_bin.items()
                            if hb == h_bin
                            for iid in members
                            if int(iid) in ref_feats and int(iid) not in win_ref_keep}
            src_ctx_iids = {iid for (hb, _vb), members in src_part.by_bin.items()
                            if hb == h_bin
                            for iid in members
                            if int(iid) in src_feats and int(iid) not in win_src_keep}

            sp_images: List[Image.Image] = []
            for proj in projections:
                proj_ref = render_bev(
                    points=ref_points, colors=ref_colors,
                    instance_centroids_3d={iid: ref_feats[iid].centroid
                                           for iid in win_ref_iids},
                    marker_ids=win_ref_markers,
                    up_axis=up_axis,
                    projection=str(proj),
                    resolution_px=int(cfg.bev_resolution_px),
                    point_size_px=int(cfg.bev_point_size_px),
                    bg_color=tuple(cfg.bev_bg_rgb),
                    height_colormap=str(cfg.bev_colormap_when_no_color),
                    marker_fill_color=(220, 0, 0),
                    context_centroids_3d={iid: ref_feats[iid].centroid
                                          for iid in ref_ctx_iids},
                )
                proj_src = render_bev(
                    points=src_points, colors=src_colors,
                    instance_centroids_3d={iid: src_feats[iid].centroid
                                           for iid in win_src_iids},
                    marker_ids=win_src_markers,
                    up_axis=up_axis,
                    projection=str(proj),
                    resolution_px=int(cfg.bev_resolution_px),
                    point_size_px=int(cfg.bev_point_size_px),
                    bg_color=tuple(cfg.bev_bg_rgb),
                    height_colormap=str(cfg.bev_colormap_when_no_color),
                    marker_fill_color=(0, 60, 220),
                    context_centroids_3d={iid: src_feats[iid].centroid
                                          for iid in src_ctx_iids},
                )
                sp_images.extend([proj_ref.image, proj_src.image])

            z_lo = float(window_jobs[0].z_range_m[0])
            z_hi = float(window_jobs[-1].z_range_m[1])
            sp_key: BinKey = (70 + start_idx, 0)
            sp_job = BinJob(key=sp_key, ref_iids=win_ref_iids,
                            src_iids=win_src_iids, z_range_m=(z_lo, z_hi))
            sp_bctx = BinContext(
                key=sp_key, z_range_m=(z_lo, z_hi),
                ref_iids=win_ref_iids, src_iids=win_src_iids,
                ref_marker_ids=sorted(win_ref_markers.values()),
                src_marker_ids=sorted(win_src_markers.values()),
                ref_marker_to_iid={int(m): int(i) for i, m in win_ref_markers.items()},
                src_marker_to_iid={int(m): int(i) for i, m in win_src_markers.items()},
                asymmetric=False, use_bev=True,
            )
            sp_prompt = spatial_prompt_builder.build(ctx, sp_images, bin_ctx=sp_bctx)
            spatial_pending.append(_PendingBin(
                job=sp_job, bin_ctx=sp_bctx, images=sp_images, prompt=sp_prompt,
                n_ref=len(win_ref_markers), n_src=len(win_src_markers),
                asymmetric=False,
            ))
            label = f"spatial_{start_idx}" if sw == 1 else f"spatial_{start_idx}_{end_idx}"
            spatial_labels.append(label)

        if spatial_pending:
            sp_raws = pipeline.vlm.call_batch(
                [(p.images, p.prompt.system, p.prompt.user) for p in spatial_pending],
                max_new_tokens=mnt,
                enable_thinking=cfg.enable_thinking_mode or None,
            )
            spatial_proposals: List[Tuple[int, int]] = []
            for sp, sp_raw, sp_label in zip(spatial_pending, sp_raws, spatial_labels):
                try:
                    sp_vlm: VLMResult = pipeline.parser.parse(
                        sp_raw, ctx, bin_ctx=sp.bin_ctx)
                except Exception as e:  # noqa: BLE001
                    sp_vlm = VLMResult(raw_text=sp_raw or "", proposals=[],
                                       audit={"parse_error": str(e)})
                if sample_dir is not None:
                    try:
                        sp_dir = Path(sample_dir) / str(ctx.pair_id) / sp_label
                        sp_dir.mkdir(parents=True, exist_ok=True)
                        for ii, im in enumerate(sp.images):
                            im.convert("RGB").save(sp_dir / f"img_{ii:02d}.jpg")
                        (sp_dir / "prompt.txt").write_text(
                            f"=== SYSTEM ===\n{sp.prompt.system}\n\n"
                            f"=== USER ===\n{sp.prompt.user}\n")
                        (sp_dir / "vlm_raw.txt").write_text(sp_raw or "")
                    except Exception:  # noqa: BLE001
                        pass
                spatial_proposals.extend(sp_vlm.proposals)
                per_bin_audit.append({
                    "bin_key": list(sp.job.key),
                    "n_ref": sp.n_ref, "n_src": sp.n_src,
                    "z_range_m": [float(sp.job.z_range_m[0]),
                                  float(sp.job.z_range_m[1])],
                    "spatial": True, "spatial_label": sp_label,
                    "n_proposals": len(sp_vlm.proposals),
                })
                log.info(
                    "Spatial match [%s/%s]: %d proposals from %d ref × %d src",
                    ctx.pair_id, sp_label, len(sp_vlm.proposals),
                    sp.n_ref, sp.n_src,
                )

            # Merge appearance + spatial proposals.
            appearance_set = set(all_proposals)
            spatial_set = set(spatial_proposals)
            strategy = str(cfg.spatial_merge_strategy).lower()
            if strategy == "intersection":
                merged = appearance_set & spatial_set
            elif strategy == "union":
                merged = appearance_set | spatial_set
            else:
                log.warning("Unknown spatial_merge_strategy %r, using intersection",
                            strategy)
                merged = appearance_set & spatial_set
            log.info(
                "Spatial merge [%s] strategy=%s: appearance=%d, spatial=%d → merged=%d",
                ctx.pair_id, strategy, len(appearance_set), len(spatial_set),
                len(merged),
            )
            all_proposals = list(merged)

    # (8) Cross-bin stitching: re-query IIDs that never co-occurred in any bin.
    n_cross_bin = 0
    if cfg.cross_bin_stitch:
        # IIDs that were binned but never proposed by the VLM + true orphans.
        proposed_ref: Set[int] = set()
        proposed_src: Set[int] = set()
        for s_iid, r_iid in all_proposals:
            proposed_ref.add(int(r_iid))
            proposed_src.add(int(s_iid))
        unmatched_ref_all = sorted(i for i in ref_feats if i not in proposed_ref)
        unmatched_src_all = sorted(i for i in src_feats if i not in proposed_src)

        window = int(cfg.cross_bin_window)
        max_ps = int(cfg.cross_bin_max_per_side)

        if window > 0 and len(schedule) > 1:
            # Sliding window: groups of `window` adjacent bins.
            stitch_groups: List[Tuple[str, List[int], List[int], Tuple[float, float]]] = []
            for start in range(max(1, len(schedule) - window + 1)):
                end = min(start + window, len(schedule))
                z_lo = float(schedule[start][0])
                z_hi = float(schedule[end - 1][1])
                # IIDs whose height centroid falls in this window range.
                grp_ref = [i for i in unmatched_ref_all
                           if z_lo <= ref_feats[i].h_center <= z_hi]
                grp_src = [i for i in unmatched_src_all
                           if z_lo <= src_feats[i].h_center <= z_hi]
                if len(grp_ref) > max_ps:
                    grp_ref = grp_ref[:max_ps]
                if len(grp_src) > max_ps:
                    grp_src = grp_src[:max_ps]
                if grp_ref and grp_src:
                    label = f"xbin_{start}_{end}"
                    stitch_groups.append((label, grp_ref, grp_src, (z_lo, z_hi)))
        else:
            # Single catchall group (window=0).
            grp_ref = unmatched_ref_all[:max_ps]
            grp_src = unmatched_src_all[:max_ps]
            stitch_groups = []
            if grp_ref and grp_src:
                h_vals = ([ref_feats[i].h_center for i in grp_ref]
                          + [src_feats[i].h_center for i in grp_src])
                stitch_groups.append(("xbin_all", grp_ref, grp_src,
                                      (min(h_vals), max(h_vals))))

        # Build all stitch groups as pending bins, then batch them.
        xbin_pending: List[_PendingBin] = []
        xbin_labels: List[str] = []
        for label, grp_ref, grp_src, z_range in stitch_groups:
            log.info(
                "Cross-bin stitch [%s/%s]: %d ref + %d src unmatched IIDs",
                ctx.pair_id, label, len(grp_ref), len(grp_src),
            )
            xbin_key: BinKey = (90 + len(xbin_pending), 0)
            xbin_job = BinJob(key=xbin_key, ref_iids=grp_ref,
                              src_iids=grp_src, z_range_m=z_range)
            xref_markers = {int(i): int(ref_marker_map[int(i)]) for i in grp_ref
                            if int(i) in ref_marker_map}
            xsrc_markers = {int(i): int(src_marker_map[int(i)]) for i in grp_src
                            if int(i) in src_marker_map}
            if not xref_markers or not xsrc_markers:
                continue
            xref_keep = set(grp_ref)
            xsrc_keep = set(grp_src)
            xref_masks = {
                fid: {iid: m for iid, m in inst.items() if int(iid) in xref_keep}
                for fid, inst in ref_per_frame_masks.items()
            }
            xsrc_masks = {
                fid: {iid: m for iid, m in inst.items() if int(iid) in xsrc_keep}
                for fid, inst in src_per_frame_masks.items()
            }
            xbin_ref_fids: Sequence[int] = ctx.ref_frames.frame_indices
            xbin_src_fids: Sequence[int] = ctx.src_frames.frame_indices
            if ref_cands is not None and src_cands is not None:
                xbin_ref_fids, _ = select_bin_frames(
                    grp_ref, ref_cands, k=int(pbfs.k_per_bin))
                xbin_src_fids, _ = select_bin_frames(
                    grp_src, src_cands, k=int(pbfs.k_per_bin))

            # Pairwise expansion for cross-bin stitch groups.
            if cfg.pairwise:
                for xr_iid in grp_ref:
                    pw_xr_markers = {int(xr_iid): int(ref_marker_map[int(xr_iid)])}
                    pw_xr_masks = {
                        fid: {iid: m for iid, m in inst.items() if int(iid) == int(xr_iid)}
                        for fid, inst in ref_per_frame_masks.items()
                    }
                    vis_inp = BinVisualInputs(
                        cfg=cfg, job=xbin_job, up_axis=up_axis,
                        image_rotation_k=int(ctx.image_rotation_k),
                        depth_dir=ctx.depth_dir,
                        ref_iids_bin=[int(xr_iid)], src_iids_bin=list(grp_src),
                        ref_markers_bin=dict(pw_xr_markers),
                        src_markers_bin=dict(xsrc_markers),
                        bin_ref_fids=[int(f) for f in xbin_ref_fids],
                        bin_src_fids=[int(f) for f in xbin_src_fids],
                        ref_color_paths=dict(ctx.ref_frames.color_paths),
                        src_color_paths=dict(ctx.src_frames.color_paths),
                        get_ref_bgr=_get_ref_bgr, get_src_bgr=_get_src_bgr,
                        ref_masks_bin=pw_xr_masks, src_masks_bin=xsrc_masks,
                        ref_per_frame_masks=ref_per_frame_masks,
                        src_per_frame_masks=src_per_frame_masks,
                        ref_feats=ref_feats, src_feats=src_feats,
                        ref_part=ref_part, src_part=src_part,
                        ref_points=ref_points, ref_colors=ref_colors,
                        src_points=src_points, src_colors=src_colors,
                        ref_stats_cache=ref_stats_cache, src_stats_cache=src_stats_cache,
                        ref_cands=ref_cands, src_cands=src_cands,
                    )
                    images_pil = bin_visuals.compose(vis_inp)
                    if not images_pil:
                        continue
                    pw_xbctx = BinContext(
                        key=xbin_key, z_range_m=z_range,
                        ref_iids=[int(xr_iid)], src_iids=list(grp_src),
                        ref_marker_ids=sorted(pw_xr_markers.values()),
                        src_marker_ids=sorted(xsrc_markers.values()),
                        ref_marker_to_iid={int(m): int(i) for i, m in pw_xr_markers.items()},
                        src_marker_to_iid={int(m): int(i) for i, m in xsrc_markers.items()},
                        asymmetric=True, use_bev=bool(cfg.use_bev),
                    )
                    xprompt: PromptBundle = pipeline.prompt.build(
                        ctx, images_pil, bin_ctx=pw_xbctx)
                    xbin_pending.append(_PendingBin(
                        job=xbin_job, bin_ctx=pw_xbctx, images=images_pil,
                        prompt=xprompt, n_ref=1, n_src=len(xsrc_markers),
                        asymmetric=True,
                    ))
                    xbin_labels.append(f"{label}_ref{xr_iid}")
            else:
                vis_inp = BinVisualInputs(
                    cfg=cfg, job=xbin_job, up_axis=up_axis,
                    image_rotation_k=int(ctx.image_rotation_k),
                    depth_dir=ctx.depth_dir,
                    ref_iids_bin=list(grp_ref), src_iids_bin=list(grp_src),
                    ref_markers_bin=dict(xref_markers),
                    src_markers_bin=dict(xsrc_markers),
                    bin_ref_fids=[int(f) for f in xbin_ref_fids],
                    bin_src_fids=[int(f) for f in xbin_src_fids],
                    ref_color_paths=dict(ctx.ref_frames.color_paths),
                    src_color_paths=dict(ctx.src_frames.color_paths),
                    get_ref_bgr=_get_ref_bgr, get_src_bgr=_get_src_bgr,
                    ref_masks_bin=xref_masks, src_masks_bin=xsrc_masks,
                    ref_per_frame_masks=ref_per_frame_masks,
                    src_per_frame_masks=src_per_frame_masks,
                    ref_feats=ref_feats, src_feats=src_feats,
                    ref_part=ref_part, src_part=src_part,
                    ref_points=ref_points, ref_colors=ref_colors,
                    src_points=src_points, src_colors=src_colors,
                    ref_stats_cache=ref_stats_cache, src_stats_cache=src_stats_cache,
                    ref_cands=ref_cands, src_cands=src_cands,
                )
                images_pil = bin_visuals.compose(vis_inp)
                if not images_pil:
                    continue
                asym = (len(xref_markers) == 1) or (len(xsrc_markers) == 1)
                xbin_bctx = BinContext(
                    key=xbin_key, z_range_m=z_range,
                    ref_iids=list(grp_ref), src_iids=list(grp_src),
                    ref_marker_ids=sorted(xref_markers.values()),
                    src_marker_ids=sorted(xsrc_markers.values()),
                    ref_marker_to_iid={int(m): int(i) for i, m in xref_markers.items()},
                    src_marker_to_iid={int(m): int(i) for i, m in xsrc_markers.items()},
                    asymmetric=bool(asym), use_bev=bool(cfg.use_bev),
                )
                xprompt: PromptBundle = pipeline.prompt.build(
                    ctx, images_pil, bin_ctx=xbin_bctx)
                xbin_pending.append(_PendingBin(
                    job=xbin_job, bin_ctx=xbin_bctx, images=images_pil,
                    prompt=xprompt, n_ref=len(xref_markers), n_src=len(xsrc_markers),
                    asymmetric=bool(asym),
                ))
                xbin_labels.append(label)

        # Batched VLM call for all stitch groups at once.
        if xbin_pending:
            xraws = pipeline.vlm.call_batch(
                [(p.images, p.prompt.system, p.prompt.user) for p in xbin_pending],
                max_new_tokens=mnt,
                enable_thinking=cfg.enable_thinking_mode or None,
            )
            for xp, xraw, xlabel in zip(xbin_pending, xraws, xbin_labels):
                try:
                    xvlm: VLMResult = pipeline.parser.parse(
                        xraw, ctx, bin_ctx=xp.bin_ctx)
                except Exception as e:  # noqa: BLE001
                    xvlm = VLMResult(raw_text=xraw or "", proposals=[],
                                     audit={"parse_error": str(e)})
                if sample_dir is not None:
                    try:
                        xdir = (Path(sample_dir) / str(ctx.pair_id) / xlabel)
                        xdir.mkdir(parents=True, exist_ok=True)
                        for ii, im in enumerate(xp.images):
                            im.convert("RGB").save(xdir / f"img_{ii:02d}.jpg")
                        (xdir / "prompt.txt").write_text(
                            f"=== SYSTEM ===\n{xp.prompt.system}\n\n"
                            f"=== USER ===\n{xp.prompt.user}\n")
                        (xdir / "vlm_raw.txt").write_text(xraw or "")
                    except Exception:  # noqa: BLE001
                        pass
                n_xp = len(xvlm.proposals)
                n_cross_bin += n_xp
                all_proposals.extend(xvlm.proposals)
                raw_chunks.append(f"--- {xlabel} ---\n{xraw}")
                per_bin_audit.append({
                    "bin_key": list(xp.job.key),
                    "n_ref": xp.n_ref, "n_src": xp.n_src,
                    "z_range_m": [float(xp.job.z_range_m[0]),
                                  float(xp.job.z_range_m[1])],
                    "asymmetric": xp.asymmetric,
                    "n_proposals": n_xp,
                    "cross_bin": True, "cross_bin_label": xlabel,
                })
                log.info(
                    "Cross-bin stitch [%s/%s]: %d proposals from %d ref × %d src",
                    ctx.pair_id, xlabel, n_xp, xp.n_ref, xp.n_src,
                )

    aggregated = VLMResult(
        raw_text="\n\n".join(raw_chunks) if raw_chunks else "(blocking: no active bins)",
        proposals=all_proposals,
        audit={
            "per_bin": per_bin_audit,
            "n_active_bins": len(jobs),
            "n_cross_bin_proposals": n_cross_bin,
            "ref_features": ref_feats,
            "src_features": src_feats,
            "ref_marker_to_iid": ref_marker_to_iid,
            "src_marker_to_iid": src_marker_to_iid,
        },
    )
    return pipeline.resolver.resolve(aggregated, ctx)
