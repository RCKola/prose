"""Red/blue circle annotation used for VLM instance-correspondence prediction (§3.5)."""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def color_for_marker(marker_id: int) -> Tuple[int, int, int]:
    """Deterministic visually-distinct BGR color seeded by marker integer.

    Uses golden-ratio hue spacing so consecutive marker IDs are maximally
    separated in hue. Saturation/value fixed for legibility.
    """
    h = (int(marker_id) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def annotate_image_with_instances(
    image: np.ndarray,
    masks_by_instance: Dict[int, np.ndarray],
    *,
    circle_color_bgr: Sequence[int],
    text_color_bgr: Sequence[int] = (255, 255, 255),
    radius: int = 26,
    font_scale: float = 0.95,
    font_thickness: int = 3,
    outline_thickness: int = 4,
    label_remap: Dict[int, int] | None = None,
    draw_masks: bool = False,
    mask_alpha: float = 0.30,
    mask_outline_thickness: int = 2,
    mask_outline_color_bgr: Sequence[int] | None = None,
    per_marker_color: bool = False,
    use_largest_component_centroid: bool = False,
    draw_circles: bool = True,
) -> np.ndarray:
    """Draw a filled circle + instance-id text at each mask's centroid.

    Args:
        image: (H, W, 3) BGR uint8.
        masks_by_instance: {iid: (H, W) bool mask}.
        circle_color_bgr: BGR tuple, e.g. (0,0,255) for red. Used for circles
            (and masks when `per_marker_color=False`).
        label_remap: Optional {actual_iid -> display_id}. When provided, the
            text rendered on each mark uses display_id instead of actual_iid.
            Also seeds per-instance colors when `per_marker_color=True`.
        draw_masks: when True, alpha-blend a colored mask region under each
            circle. Used by the mosaic-matcher backend to give the VLM the
            full SoM picture (mask + label).
        mask_alpha: blend factor for `draw_masks` (0=no overlay, 1=opaque).
        per_marker_color: when True, color the mask + circle outline by a
            deterministic hash of the *displayed marker id* (so the same
            marker across panels shares a hue). When False, all masks use
            `circle_color_bgr`.
        use_largest_component_centroid: when True, place the circle at the
            largest connected component's centroid (robust for disjoint
            SAM3 masks). When False, use cv2.moments (cheaper, can land
            off-mask).
    Returns:
        Annotated copy of `image`.
    """
    from .mask_ops import mask_centroid, mask_centroid_largest_component

    annotated = image.copy()

    # Pre-compute display ids and per-iid colors.
    def _display(iid: int) -> int:
        return label_remap.get(int(iid), int(iid)) if label_remap is not None else int(iid)

    def _mask_color(iid: int) -> Tuple[int, int, int]:
        if per_marker_color:
            return color_for_marker(_display(iid))
        return tuple(int(c) for c in circle_color_bgr)

    # 1) Per-instance alpha-blended mask + thin outline (mirrors the GoM
    # `_draw_translucent_overlay` + `_draw_outline` pattern). Doing this
    # per-instance instead of one batched overlay so each mask gets its
    # own outline contour in the same per-marker color.
    if draw_masks and masks_by_instance:
        overlay = annotated.astype(np.float32)
        for iid, mask in masks_by_instance.items():
            if mask is None or not mask.any():
                continue
            color = np.asarray(_mask_color(int(iid)), dtype=np.float32)
            m_bool = mask.astype(bool)
            region = overlay[m_bool]
            blended = (1.0 - mask_alpha) * region + mask_alpha * color[None, :]
            overlay[m_bool] = blended
        annotated = np.clip(overlay, 0, 255).astype(np.uint8)
        if mask_outline_thickness > 0:
            for iid, mask in masks_by_instance.items():
                if mask is None or not mask.any():
                    continue
                # Outline color: uniform when `mask_outline_color_bgr` is
                # set (used by mosaic-blocking to break the
                # "VLM-cites-the-color" leak — see prompt.py); else per-
                # marker so it matches the filled-blend hue.
                color_bgr = (tuple(int(c) for c in mask_outline_color_bgr)
                             if mask_outline_color_bgr is not None
                             else _mask_color(int(iid)))
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(
                    annotated, contours, -1, tuple(int(c) for c in color_bgr),
                    int(mask_outline_thickness),
                )

    # 2) Draw the circle + numeric id.
    centroid_fn = (
        mask_centroid_largest_component
        if use_largest_component_centroid
        else mask_centroid
    )
    for iid, mask in masks_by_instance.items():
        c = centroid_fn(mask)
        if c is None:
            continue
        cx, cy = c
        if draw_circles:
            ring_color = _mask_color(int(iid))
            cv2.circle(annotated, (cx, cy), radius, ring_color, -1)
            cv2.circle(annotated, (cx, cy), radius, (0, 0, 0), int(outline_thickness))

        text = str(_display(int(iid)))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        org = (cx - tw // 2, cy + th // 2)
        cv2.putText(
            annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (0, 0, 0), int(font_thickness) + 2, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            tuple(int(c) for c in text_color_bgr), int(font_thickness), cv2.LINE_AA,
        )
    return annotated


def annotate_frame_sequence(
    frames: List[np.ndarray],
    frame_masks: List[Dict[int, np.ndarray]],
    **kwargs,
) -> List[np.ndarray]:
    """Apply the same annotation to a list of frames."""
    assert len(frames) == len(frame_masks), "frames and frame_masks must be the same length"
    return [
        annotate_image_with_instances(img, masks, **kwargs)
        for img, masks in zip(frames, frame_masks)
    ]


def save_annotated_frames(
    frames: List[np.ndarray],
    out_dir: Path,
    prefix: str = "annot",
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        p = out_dir / f"{prefix}_{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def select_keyframes_by_instance_coverage(
    per_frame_masks: Dict[int, Dict[int, np.ndarray]],
    max_frames: int,
) -> List[int]:
    """Greedy set-cover: pick frames that together cover the largest set of instances.

    Starts from all frames, picks the one covering the most uncovered instance IDs,
    repeats until `max_frames` frames are picked or all instances are covered.
    """
    uncovered: set = set()
    for inst_masks in per_frame_masks.values():
        uncovered.update(inst_masks.keys())

    chosen: List[int] = []
    while uncovered and len(chosen) < max_frames:
        best_frame = None
        best_coverage: set = set()
        for frame_idx, inst_masks in per_frame_masks.items():
            if frame_idx in chosen:
                continue
            coverage = set(inst_masks.keys()) & uncovered
            if len(coverage) > len(best_coverage):
                best_coverage = coverage
                best_frame = frame_idx
        if best_frame is None:
            break
        chosen.append(best_frame)
        uncovered -= best_coverage

    # If we have budget remaining, fill with evenly-spaced frames.
    if len(chosen) < max_frames:
        all_frames = sorted(per_frame_masks.keys())
        remaining = [f for f in all_frames if f not in chosen]
        stride = max(1, len(remaining) // (max_frames - len(chosen) + 1))
        for i in range(0, len(remaining), stride):
            if len(chosen) >= max_frames:
                break
            chosen.append(remaining[i])

    return sorted(chosen)
