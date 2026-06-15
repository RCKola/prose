"""Top-down BEV renderer (self-contained copy).

Orthographic projection of the fused colored point cloud with numeric
marker IDs at projected 3D centroids. Gravity axis configurable via
``up_axis`` ("z" for 3RScan, "y" for ADT).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class BevRender:
    image: "PIL.Image.Image"  # type: ignore[name-defined]
    resolution_px: int
    extent_min_xy: Tuple[float, float]
    extent_max_xy: Tuple[float, float]
    instance_marker_positions: Dict[int, Tuple[int, int]] = field(default_factory=dict)


def _plane_axes(up_axis: str, projection: str = "bev") -> Tuple[int, int, int]:
    """Return (col_axis, row_axis, height_axis) for a given projection.

    ``projection``:
      - ``"bev"``   — top-down, collapse the gravity axis (default)
      - ``"front"`` — frontal, collapse the depth axis (Y for z-up, Z for y-up)
      - ``"side"``  — lateral, collapse the lateral axis (X always)
    """
    ax = (up_axis or "z").strip().lower()
    proj = (projection or "bev").strip().lower()
    if ax == "z":
        if proj == "bev":
            return 0, 1, 2
        if proj == "front":
            return 0, 2, 1
        if proj == "side":
            return 1, 2, 0
    elif ax == "y":
        if proj == "bev":
            return 0, 2, 1
        if proj == "front":
            return 0, 1, 2
        if proj == "side":
            return 2, 1, 0
    else:
        raise ValueError(f"up_axis must be 'y' or 'z' (got {up_axis!r})")
    raise ValueError(f"projection must be 'bev', 'front', or 'side' (got {projection!r})")


def _project_plane_to_pixel(
    xy: np.ndarray,
    extent_min: Tuple[float, float],
    extent_max: Tuple[float, float],
    resolution_px: int,
) -> np.ndarray:
    xmin, ymin = extent_min
    xmax, ymax = extent_max
    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    cols = ((xy[:, 0] - xmin) / span_x) * (resolution_px - 1)
    rows = (1.0 - (xy[:, 1] - ymin) / span_y) * (resolution_px - 1)
    return np.stack([cols, rows], axis=-1)


def _compute_scene_extent(
    points_xy: np.ndarray, margin_frac: float = 0.0
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if points_xy.size == 0:
        return (-1.0, -1.0), (1.0, 1.0)
    p_lo = points_xy.min(axis=0).astype(np.float64)
    p_hi = points_xy.max(axis=0).astype(np.float64)
    if margin_frac > 0:
        span = p_hi - p_lo
        p_lo -= span * margin_frac
        p_hi += span * margin_frac
    cx, cy = (p_lo + p_hi) / 2.0
    half = float(max(p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]) / 2.0)
    return (cx - half, cy - half), (cx + half, cy + half)


_VIRIDIS_LUT = np.array([
    [68, 1, 84], [70, 39, 117], [60, 78, 138], [45, 112, 142],
    [35, 144, 140], [42, 175, 127], [104, 203, 89], [186, 222, 39],
    [253, 231, 36],
], dtype=np.uint8)


def _height_colormap(z: np.ndarray, name: str) -> np.ndarray:
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if name == "viridis":
        lut = _VIRIDIS_LUT
    else:
        lut = np.repeat(np.linspace(40, 235, 9).astype(np.uint8)[:, None], 3, axis=1)
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    idx = np.clip((z_norm * (len(lut) - 1)).round().astype(np.int64), 0, len(lut) - 1)
    return lut[idx]


def render_bev(
    *,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    instance_centroids_3d: Dict[int, np.ndarray],
    marker_ids: Dict[int, int],
    up_axis: str = "z",
    projection: str = "bev",
    resolution_px: int = 1024,
    point_size_px: int = 2,
    marker_radius_px: int = 16,
    marker_text_px: int = 20,
    marker_outline_thickness: int = 2,
    marker_text_color: Tuple[int, int, int] = (255, 255, 255),
    marker_fill_color: Tuple[int, int, int] = (220, 30, 30),
    marker_outline_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Tuple[int, int, int] = (245, 245, 245),
    height_colormap: str = "viridis",
    context_centroids_3d: Optional[Dict[int, np.ndarray]] = None,
    context_dot_radius_px: int = 5,
    context_dot_color: Tuple[int, int, int] = (70, 70, 70),
    context_dot_outline_color: Tuple[int, int, int] = (0, 0, 0),
    context_dot_outline_thickness: int = 1,
) -> BevRender:
    import cv2
    from PIL import Image

    col_ax, row_ax, h_ax = _plane_axes(up_axis, projection)
    plane_idx = np.array([col_ax, row_ax], dtype=np.int64)
    plane_pts = points[:, plane_idx] if points.size > 0 else points
    extent_min, extent_max = _compute_scene_extent(plane_pts)

    canvas = np.zeros((resolution_px, resolution_px, 3), np.uint8)
    canvas[:] = bg_color

    if points.size > 0:
        pix = _project_plane_to_pixel(plane_pts, extent_min, extent_max, resolution_px)
        cols = np.clip(pix[:, 0].astype(np.int64), 0, resolution_px - 1)
        rows = np.clip(pix[:, 1].astype(np.int64), 0, resolution_px - 1)
        use_rgb = (
            colors is not None
            and getattr(colors, "ndim", 0) == 2
            and colors.shape[0] == points.shape[0]
            and colors.shape[1] >= 3
        )
        if use_rgb:
            rgb = np.asarray(colors[:, :3], dtype=np.uint8)
        else:
            rgb = _height_colormap(points[:, h_ax], height_colormap)

        canvas[rows, cols] = rgb
        if point_size_px > 1:
            r = point_size_px // 2
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    rr = np.clip(rows + dr, 0, resolution_px - 1)
                    cc = np.clip(cols + dc, 0, resolution_px - 1)
                    canvas[rr, cc] = rgb

    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font_scale = max(0.5, float(marker_text_px) / 22.0)
    text_thickness = max(2, int(round(font_scale * 2)))
    fill_bgr = (int(marker_fill_color[2]), int(marker_fill_color[1]), int(marker_fill_color[0]))
    outline_bgr = (int(marker_outline_color[2]), int(marker_outline_color[1]), int(marker_outline_color[0]))
    text_bgr = (int(marker_text_color[2]), int(marker_text_color[1]), int(marker_text_color[0]))

    if context_centroids_3d:
        ctx_color_bgr = (int(context_dot_color[2]),
                         int(context_dot_color[1]),
                         int(context_dot_color[0]))
        ctx_outline_bgr = (int(context_dot_outline_color[2]),
                           int(context_dot_outline_color[1]),
                           int(context_dot_outline_color[0]))
        named = set(int(i) for i in instance_centroids_3d.keys())
        for ctx_iid, ctx_centroid in context_centroids_3d.items():
            if int(ctx_iid) in named:
                continue
            ctx_plane = np.asarray([ctx_centroid[col_ax], ctx_centroid[row_ax]],
                                   dtype=np.float64).reshape(1, 2)
            cpix = _project_plane_to_pixel(ctx_plane, extent_min, extent_max, resolution_px)[0]
            cc = int(np.clip(cpix[0], context_dot_radius_px,
                             resolution_px - context_dot_radius_px - 1))
            cr = int(np.clip(cpix[1], context_dot_radius_px,
                             resolution_px - context_dot_radius_px - 1))
            cv2.circle(bgr, (cc, cr), int(context_dot_radius_px),
                       ctx_color_bgr, -1)
            if context_dot_outline_thickness > 0:
                cv2.circle(bgr, (cc, cr), int(context_dot_radius_px),
                           ctx_outline_bgr, int(context_dot_outline_thickness))

    marker_positions: Dict[int, Tuple[int, int]] = {}
    for iid, centroid_3d in instance_centroids_3d.items():
        if iid not in marker_ids:
            continue
        marker_id = int(marker_ids[iid])
        plane = np.asarray([centroid_3d[col_ax], centroid_3d[row_ax]],
                           dtype=np.float64).reshape(1, 2)
        pix = _project_plane_to_pixel(plane, extent_min, extent_max, resolution_px)[0]
        col = int(np.clip(pix[0], marker_radius_px, resolution_px - marker_radius_px - 1))
        row = int(np.clip(pix[1], marker_radius_px, resolution_px - marker_radius_px - 1))

        cv2.circle(bgr, (col, row), int(marker_radius_px), fill_bgr, -1)
        if marker_outline_thickness > 0:
            cv2.circle(bgr, (col, row), int(marker_radius_px), outline_bgr,
                       int(marker_outline_thickness))

        text = str(marker_id)
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness,
        )
        cv2.putText(
            bgr,
            text,
            (col - tw // 2, row + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_bgr,
            text_thickness,
            cv2.LINE_AA,
        )
        marker_positions[int(iid)] = (col, row)

    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return BevRender(
        image=img,
        resolution_px=resolution_px,
        extent_min_xy=extent_min,
        extent_max_xy=extent_max,
        instance_marker_positions=marker_positions,
    )
