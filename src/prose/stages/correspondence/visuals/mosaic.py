"""Frame mosaic composer (self-contained copy).

Grid layout (top rows=ref, bottom rows=src) with SoM-style circle +
numeric ID overlays. ``max_cols`` controls how many frames per row
(0 = all in one strip). ``image_rotation_k`` is applied to both frame
and masks before annotation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ....utils.annotate import annotate_image_with_instances


@dataclass
class FrameMosaic:
    image: "PIL.Image.Image"  # type: ignore[name-defined]
    subpanel_longside_px: int
    overlay_metadata: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    panel_layout: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)


def _resize_longside_np(img_bgr: np.ndarray, longside_px: int) -> np.ndarray:
    import cv2
    h, w = img_bgr.shape[:2]
    if max(h, w) == longside_px:
        return img_bgr
    if w >= h:
        new_w = longside_px
        new_h = max(1, int(round(h * longside_px / w)))
    else:
        new_h = longside_px
        new_w = max(1, int(round(w * longside_px / h)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _rotate_frame_and_masks(
    frame_bgr: np.ndarray,
    masks: Dict[int, np.ndarray],
    rot_k: int,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    if not rot_k:
        return frame_bgr, masks
    rf = np.rot90(frame_bgr, k=rot_k).copy()
    rm = {
        int(iid): np.rot90(m.astype(np.uint8), k=rot_k).astype(bool).copy()
        for iid, m in masks.items()
    }
    return rf, rm


def compose_frame_mosaic(
    *,
    ref_frames_bgr: Sequence[np.ndarray],
    src_frames_bgr: Sequence[np.ndarray],
    ref_frame_ids: Sequence[int],
    src_frame_ids: Sequence[int],
    ref_masks_per_frame: Dict[int, Dict[int, np.ndarray]],
    src_masks_per_frame: Dict[int, Dict[int, np.ndarray]],
    ref_marker_ids: Dict[int, int],
    src_marker_ids: Dict[int, int],
    subpanel_longside_px: int = 512,
    image_rotation_k: int = 0,
    circle_color_bgr_ref: Sequence[int] = (0, 0, 220),
    circle_color_bgr_src: Sequence[int] = (220, 60, 0),
    text_color_bgr: Sequence[int] = (255, 255, 255),
    circle_radius: int = 28,
    font_scale: float = 1.0,
    font_thickness: int = 3,
    outline_thickness: int = 4,
    draw_circles: bool = True,
    mask_outline_thickness: int = 2,
    bg_bgr: Sequence[int] = (10, 10, 10),
    max_cols: int = 0,
) -> FrameMosaic:
    import cv2
    from PIL import Image

    if len(ref_frames_bgr) != len(ref_frame_ids):
        raise ValueError("ref_frames_bgr / ref_frame_ids length mismatch")
    if len(src_frames_bgr) != len(src_frame_ids):
        raise ValueError("src_frames_bgr / src_frame_ids length mismatch")

    def _process_row(
        frames_bgr: Sequence[np.ndarray],
        frame_ids: Sequence[int],
        masks_per_frame: Dict[int, Dict[int, np.ndarray]],
        marker_ids: Dict[int, int],
        circle_color_bgr: Sequence[int],
    ) -> List[Tuple[np.ndarray, Dict[int, Tuple[int, int]]]]:
        out: List[Tuple[np.ndarray, Dict[int, Tuple[int, int]]]] = []
        for arr, fid in zip(frames_bgr, frame_ids):
            if arr is None:
                placeholder = np.zeros((subpanel_longside_px, subpanel_longside_px, 3), np.uint8)
                placeholder[:, :] = bg_bgr
                out.append((placeholder, {}))
                continue
            masks = masks_per_frame.get(int(fid), {})
            visible_masks = {int(iid): m for iid, m in masks.items() if int(iid) in marker_ids}
            rot_frame, rot_masks = _rotate_frame_and_masks(arr, visible_masks, image_rotation_k)

            annotated = annotate_image_with_instances(
                rot_frame,
                rot_masks,
                circle_color_bgr=circle_color_bgr,
                text_color_bgr=text_color_bgr,
                radius=int(circle_radius),
                font_scale=float(font_scale),
                font_thickness=int(font_thickness),
                outline_thickness=int(outline_thickness),
                label_remap=marker_ids,
                draw_masks=True,
                mask_alpha=0.03,
                mask_outline_thickness=int(mask_outline_thickness),
                draw_circles=bool(draw_circles),
                mask_outline_color_bgr=(255, 255, 255),
                per_marker_color=True,
                use_largest_component_centroid=True,
            )
            h0, w0 = annotated.shape[:2]
            resized = _resize_longside_np(annotated, subpanel_longside_px)
            h1, w1 = resized.shape[:2]
            sx = w1 / max(w0, 1)
            sy = h1 / max(h0, 1)
            overlay: Dict[int, Tuple[int, int]] = {}
            from ....utils.mask_ops import mask_centroid_largest_component
            for iid, m in rot_masks.items():
                c = mask_centroid_largest_component(m)
                if c is None:
                    continue
                cx0, cy0 = c
                overlay[int(marker_ids[int(iid)])] = (int(cx0 * sx), int(cy0 * sy))
            out.append((resized, overlay))
        return out

    top_panels = _process_row(
        ref_frames_bgr, ref_frame_ids, ref_masks_per_frame, ref_marker_ids,
        circle_color_bgr_ref,
    )
    bot_panels = _process_row(
        src_frames_bgr, src_frame_ids, src_masks_per_frame, src_marker_ids,
        circle_color_bgr_src,
    )

    def _layout_grid(
        panels: List[Tuple[np.ndarray, Dict[int, Tuple[int, int]]]],
        frame_ids: Sequence[int],
        mc: int,
    ) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int]], Dict[int, Tuple[int, int]]]:
        """Arrange panels in a near-square grid, return (canvas, panel_layout, overlay)."""
        n = len(panels)
        cols = min(n, mc) if mc > 0 else n
        rows = math.ceil(n / cols) if cols > 0 else 1
        cell_h = max((p.shape[0] for p, _ in panels), default=subpanel_longside_px)
        cell_w = max((p.shape[1] for p, _ in panels), default=subpanel_longside_px)
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), np.uint8)
        grid[:, :] = bg_bgr
        p_layout: Dict[int, Tuple[int, int, int, int]] = {}
        o_meta: Dict[int, Tuple[int, int]] = {}
        for idx, ((img, overlay), fid) in enumerate(zip(panels, frame_ids)):
            r, c = divmod(idx, cols)
            ih, iw = img.shape[:2]
            px = c * cell_w + (cell_w - iw) // 2
            py = r * cell_h + (cell_h - ih) // 2
            grid[py : py + ih, px : px + iw] = img
            p_layout[int(fid)] = (px, py, iw, ih)
            for marker, (cx, cy) in overlay.items():
                o_meta[int(marker)] = (px + int(cx), py + int(cy))
        return grid, p_layout, o_meta

    mc = int(max_cols) if max_cols else 0
    top_grid, top_playout, top_overlay = _layout_grid(top_panels, ref_frame_ids, mc)
    bot_grid, bot_playout, bot_overlay = _layout_grid(bot_panels, src_frame_ids, mc)

    mosaic_w = max(top_grid.shape[1], bot_grid.shape[1])
    mosaic_h = top_grid.shape[0] + bot_grid.shape[0]
    canvas = np.zeros((mosaic_h, mosaic_w, 3), np.uint8)
    canvas[:, :] = bg_bgr
    canvas[: top_grid.shape[0], : top_grid.shape[1]] = top_grid
    y_off = top_grid.shape[0]
    canvas[y_off : y_off + bot_grid.shape[0], : bot_grid.shape[1]] = bot_grid

    overlay_metadata: Dict[int, Tuple[int, int]] = {}
    panel_layout: Dict[int, Tuple[int, int, int, int]] = {}
    overlay_metadata.update(top_overlay)
    panel_layout.update(top_playout)
    for fid, (px, py, w, h) in bot_playout.items():
        panel_layout[fid] = (px, py + y_off, w, h)
    for marker, (cx, cy) in bot_overlay.items():
        overlay_metadata[marker] = (cx, cy + y_off)

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    return FrameMosaic(
        image=pil_img,
        subpanel_longside_px=subpanel_longside_px,
        overlay_metadata=overlay_metadata,
        panel_layout=panel_layout,
    )


def assign_marker_ids(
    ref_iids: Sequence[int],
    src_iids: Sequence[int],
    *,
    namespace: str = "shared_distinct",
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Assign marker integers to ref / src instance IDs.

    ``shared_distinct``    — ref = 1..N, src = N+1..N+M (default).
    ``color_same_number``  — ref and src both = 1..max; caller renders
                             different colors per side (ablation).
    """
    ref_sorted = sorted({int(i) for i in ref_iids})
    src_sorted = sorted({int(i) for i in src_iids})
    if namespace == "shared_distinct":
        ref_map = {iid: idx + 1 for idx, iid in enumerate(ref_sorted)}
        offset = len(ref_sorted)
        src_map = {iid: offset + idx + 1 for idx, iid in enumerate(src_sorted)}
    elif namespace == "color_same_number":
        ref_map = {iid: idx + 1 for idx, iid in enumerate(ref_sorted)}
        src_map = {iid: idx + 1 for idx, iid in enumerate(src_sorted)}
    else:
        raise ValueError(f"unknown marker namespace: {namespace}")
    return ref_map, src_map
