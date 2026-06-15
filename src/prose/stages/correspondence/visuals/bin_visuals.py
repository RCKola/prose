"""Pluggable per-bin visual composers (phase D for the blocking path).

The blocking runner (``blocking_meta.run_blocking_pipeline``) used to
hard-wire frame-mosaic + BEV + crops. That block is now extracted behind
the ``BinVisualComposer`` protocol so the visual stack can be swapped per
experiment without touching the runner.

Three implementations:

* ``LegacyBinVisuals`` — exact reproduction of the old hard-wired path
  (frame mosaic, optional BEVs, optional crops mosaic). Default; keeps
  prior blocking runs reproducible.
* ``CropsContextBinVisuals`` — per-bin design: an outlined min-sized
  crop of *every* instance in the bin, plus one max-coverage context
  frame per scene. Decomposes the match into a small same-height set.
* ``GomBinVisuals`` — GoM (Graph-of-Marks) per-bin rendering: each
  selected frame is annotated with mask overlays, circle badges, and
  optional k-NN topology edges via ``render_gom_frame``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from ..views.per_bin import (
    build_iid_frame_ranking,
    select_bin_frames,
)
from .bev import render_bev
from .blocking_meta import BinVisualInputs, _abs_ids
from .crops import (
    build_bin_crop_grids,
    build_bin_crops_mosaic,
    compute_per_frame_stats,
)
from .mosaic import compose_frame_mosaic


# ---------------------------------------------------------------------------
# Legacy composer — frame mosaic (+ optional BEV / crops)
# ---------------------------------------------------------------------------

@dataclass
class LegacyBinVisuals:
    """Frame mosaic + optional BEVs + optional crops mosaic.

    Byte-for-byte the behaviour the blocking runner had inline before the
    composer was extracted — selected by ``blocking.bin_visuals=legacy``.
    """

    def compose(self, inp: BinVisualInputs) -> List[Image.Image]:
        cfg = inp.cfg

        # Frame mosaic.
        ref_frames_bgr_bin = [inp.get_ref_bgr(int(f)) for f in inp.bin_ref_fids]
        src_frames_bgr_bin = [inp.get_src_bgr(int(f)) for f in inp.bin_src_fids]
        mosaic = compose_frame_mosaic(
            ref_frames_bgr=ref_frames_bgr_bin,
            src_frames_bgr=src_frames_bgr_bin,
            ref_frame_ids=list(inp.bin_ref_fids),
            src_frame_ids=list(inp.bin_src_fids),
            ref_masks_per_frame=inp.ref_masks_bin,
            src_masks_per_frame=inp.src_masks_bin,
            ref_marker_ids=inp.ref_markers_bin,
            src_marker_ids=inp.src_markers_bin,
            subpanel_longside_px=int(cfg.subpanel_longside_px),
            image_rotation_k=int(inp.image_rotation_k),
            draw_circles=bool(cfg.draw_circles),
            font_scale=float(cfg.mosaic_font_scale),
            mask_outline_thickness=int(cfg.mosaic_mask_outline_thickness),
            max_cols=int(cfg.mosaic_max_cols),
        )
        images_pil: List[Image.Image] = [mosaic.image]

        # BEVs (optional).
        if cfg.use_bev and inp.ref_points is not None and inp.src_points is not None:
            h_bin = int(inp.job.key[0])
            ref_keep = set(inp.ref_iids_bin)
            src_keep = set(inp.src_iids_bin)
            ref_ctx_iids = {iid for (hb, _vb), members in inp.ref_part.by_bin.items()
                            if hb == h_bin
                            for iid in members
                            if int(iid) in inp.ref_feats and int(iid) not in ref_keep}
            src_ctx_iids = {iid for (hb, _vb), members in inp.src_part.by_bin.items()
                            if hb == h_bin
                            for iid in members
                            if int(iid) in inp.src_feats and int(iid) not in src_keep}
            bev_ref = render_bev(
                points=inp.ref_points, colors=inp.ref_colors,
                instance_centroids_3d={iid: inp.ref_feats[iid].centroid
                                       for iid in inp.ref_iids_bin},
                marker_ids=inp.ref_markers_bin,
                up_axis=inp.up_axis,
                resolution_px=int(cfg.bev_resolution_px),
                point_size_px=int(cfg.bev_point_size_px),
                bg_color=tuple(cfg.bev_bg_rgb),
                height_colormap=str(cfg.bev_colormap_when_no_color),
                marker_fill_color=(220, 0, 0),
                context_centroids_3d={iid: inp.ref_feats[iid].centroid
                                      for iid in ref_ctx_iids},
            )
            bev_src = render_bev(
                points=inp.src_points, colors=inp.src_colors,
                instance_centroids_3d={iid: inp.src_feats[iid].centroid
                                       for iid in inp.src_iids_bin},
                marker_ids=inp.src_markers_bin,
                up_axis=inp.up_axis,
                resolution_px=int(cfg.bev_resolution_px),
                point_size_px=int(cfg.bev_point_size_px),
                bg_color=tuple(cfg.bev_bg_rgb),
                height_colormap=str(cfg.bev_colormap_when_no_color),
                marker_fill_color=(0, 60, 220),
                context_centroids_3d={iid: inp.src_feats[iid].centroid
                                      for iid in src_ctx_iids},
            )
            images_pil.extend([bev_ref.image, bev_src.image])

        # Crops mosaic (optional).
        if cfg.crops_enabled:
            _ensure_stats(inp)
            crops_img, _, _ = build_bin_crops_mosaic(
                ref_iids=list(inp.ref_iids_bin), src_iids=list(inp.src_iids_bin),
                ref_i2m=inp.ref_markers_bin, src_i2m=inp.src_markers_bin,
                ref_masks_all=inp.ref_per_frame_masks,
                src_masks_all=inp.src_per_frame_masks,
                ref_frame_bgr_loader=inp.get_ref_bgr,
                src_frame_bgr_loader=inp.get_src_bgr,
                ref_stats=inp.ref_stats_cache, src_stats=inp.src_stats_cache,
                top_k=int(cfg.crops_top_k),
                pad_frac=float(cfg.crops_pad_frac),
                min_side_px=int(cfg.crops_min_side_px),
                outline_thickness=int(cfg.crops_outline_thickness),
                panel_px=int(cfg.crops_panel_px),
            )
            if crops_img is not None:
                images_pil.append(crops_img)

        return images_pil


# ---------------------------------------------------------------------------
# Crops + context composer — the per-bin design
# ---------------------------------------------------------------------------

@dataclass
class CropsContextBinVisuals:
    """Outlined min-sized crops of every bin instance + one context frame.

    Images 1-2: per-side compact crop grids — image 1 REF, image 2 SRC —
    one zoomed, green-outlined, ID-stamped crop per instance in the bin.
    Each side is its own near-square grid so the VLM's image downscale
    barely shrinks the crops (a shared 2×K strip got squashed flat).

    Image 3 (optional): one wide RGB frame per scene chosen to cover the
    most bin instances, annotated with the same IDs — gives the VLM
    spatial layout without the 24-frame mosaic that caused one-shot
    matching to degenerate into sequential pairing.
    """

    def compose(self, inp: BinVisualInputs) -> List[Image.Image]:
        cfg = inp.cfg
        images: List[Image.Image] = []
        _ensure_stats(inp)

        # --- crops: every instance in the bin, one compact grid per side ---
        if cfg.crops_enabled:
            k_all = max(len(inp.ref_iids_bin), len(inp.src_iids_bin), 1)
            ref_grid, src_grid, _, _ = build_bin_crop_grids(
                ref_iids=list(inp.ref_iids_bin), src_iids=list(inp.src_iids_bin),
                ref_i2m=inp.ref_markers_bin, src_i2m=inp.src_markers_bin,
                ref_masks_all=inp.ref_per_frame_masks,
                src_masks_all=inp.src_per_frame_masks,
                ref_frame_bgr_loader=inp.get_ref_bgr,
                src_frame_bgr_loader=inp.get_src_bgr,
                ref_stats=inp.ref_stats_cache, src_stats=inp.src_stats_cache,
                top_k=k_all,
                pad_frac=float(cfg.crops_pad_frac),
                min_side_px=int(cfg.crops_min_side_px),
                outline_thickness=int(cfg.crops_outline_thickness),
                panel_px=int(cfg.crops_panel_px),
            )
            images.append(ref_grid)
            images.append(src_grid)

        # --- BEV / multi-axis projections (optional, between crops and context frame) ---
        if cfg.use_bev and inp.ref_points is not None and inp.src_points is not None:
            h_bin = int(inp.job.key[0])
            ref_keep = set(inp.ref_iids_bin)
            src_keep = set(inp.src_iids_bin)
            ref_ctx_iids = {iid for (hb, _vb), members in inp.ref_part.by_bin.items()
                            if hb == h_bin
                            for iid in members
                            if int(iid) in inp.ref_feats and int(iid) not in ref_keep}
            src_ctx_iids = {iid for (hb, _vb), members in inp.src_part.by_bin.items()
                            if hb == h_bin
                            for iid in members
                            if int(iid) in inp.src_feats and int(iid) not in src_keep}
            projections = getattr(cfg, "bev_projections", ("bev",)) or ("bev",)
            for proj in projections:
                proj_ref = render_bev(
                    points=inp.ref_points, colors=inp.ref_colors,
                    instance_centroids_3d={iid: inp.ref_feats[iid].centroid
                                           for iid in inp.ref_iids_bin},
                    marker_ids=inp.ref_markers_bin,
                    up_axis=inp.up_axis,
                    projection=str(proj),
                    resolution_px=int(cfg.bev_resolution_px),
                    point_size_px=int(cfg.bev_point_size_px),
                    bg_color=tuple(cfg.bev_bg_rgb),
                    height_colormap=str(cfg.bev_colormap_when_no_color),
                    marker_fill_color=(220, 0, 0),
                    context_centroids_3d={iid: inp.ref_feats[iid].centroid
                                          for iid in ref_ctx_iids},
                )
                proj_src = render_bev(
                    points=inp.src_points, colors=inp.src_colors,
                    instance_centroids_3d={iid: inp.src_feats[iid].centroid
                                           for iid in inp.src_iids_bin},
                    marker_ids=inp.src_markers_bin,
                    up_axis=inp.up_axis,
                    projection=str(proj),
                    resolution_px=int(cfg.bev_resolution_px),
                    point_size_px=int(cfg.bev_point_size_px),
                    bg_color=tuple(cfg.bev_bg_rgb),
                    height_colormap=str(cfg.bev_colormap_when_no_color),
                    marker_fill_color=(0, 60, 220),
                    context_centroids_3d={iid: inp.src_feats[iid].centroid
                                          for iid in src_ctx_iids},
                )
                images.extend([proj_ref.image, proj_src.image])

        # --- one max-coverage context frame per scene ---
        if cfg.context_frame_enabled:
            ref_cands = inp.ref_cands or build_iid_frame_ranking(
                inp.ref_per_frame_masks, top_n=8, min_pixel_area=200)
            src_cands = inp.src_cands or build_iid_frame_ranking(
                inp.src_per_frame_masks, top_n=8, min_pixel_area=200)
            ref_fids, _ = select_bin_frames(inp.ref_iids_bin, ref_cands, k=1)
            src_fids, _ = select_bin_frames(inp.src_iids_bin, src_cands, k=1)
            if ref_fids and src_fids:
                rf, sf = int(ref_fids[0]), int(src_fids[0])
                ctx_mosaic = compose_frame_mosaic(
                    ref_frames_bgr=[inp.get_ref_bgr(rf)],
                    src_frames_bgr=[inp.get_src_bgr(sf)],
                    ref_frame_ids=[rf], src_frame_ids=[sf],
                    ref_masks_per_frame={rf: inp.ref_masks_bin.get(rf, {})},
                    src_masks_per_frame={sf: inp.src_masks_bin.get(sf, {})},
                    ref_marker_ids=inp.ref_markers_bin,
                    src_marker_ids=inp.src_markers_bin,
                    subpanel_longside_px=int(cfg.subpanel_longside_px),
                    image_rotation_k=int(inp.image_rotation_k),
                    draw_circles=bool(cfg.draw_circles),
                    font_scale=float(cfg.mosaic_font_scale),
                    mask_outline_thickness=int(cfg.mosaic_mask_outline_thickness),
                    max_cols=int(cfg.mosaic_max_cols),
                )
                images.append(ctx_mosaic.image)

        return images


def _ensure_stats(inp: BinVisualInputs) -> None:
    """Populate the shared per-pair crop-frame stat caches once."""
    if not inp.ref_stats_cache:
        inp.ref_stats_cache.update(compute_per_frame_stats(
            inp.ref_per_frame_masks,
            _abs_ids(inp.ref_color_paths),
            inp.depth_dir,
        ))
    if not inp.src_stats_cache:
        inp.src_stats_cache.update(compute_per_frame_stats(
            inp.src_per_frame_masks,
            _abs_ids(inp.src_color_paths),
            inp.depth_dir,
        ))
