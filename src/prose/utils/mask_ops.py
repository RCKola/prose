"""Binary-mask operations: dedup (containment + IoU) and centroid finding."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of A that lies inside B."""
    a_area = _mask_area(a)
    if a_area == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(a_area)


def dedup_masks_per_frame(
    frame_masks: Dict[int, np.ndarray],
    *,
    containment: bool = True,
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.95,
) -> Dict[int, np.ndarray]:
    """Apply paper Algorithms 1 & 2 to one frame.

    Args:
        frame_masks: {instance_id: (H, W) bool mask}
        containment: apply Algorithm 1 (drop A if A fully inside B).
        iou_threshold: drop A if IoU(A,B) > threshold (Algorithm 2).

    Returns:
        Subset of frame_masks with duplicates removed. Ties are broken by
        keeping the larger mask.
    """
    # Sort instance_ids by area ascending — smaller ones are more likely to be dropped.
    items = sorted(frame_masks.items(), key=lambda kv: _mask_area(kv[1]))
    keep = {iid: mask for iid, mask in items}

    ids = list(keep.keys())
    to_drop: set = set()

    for i, iid_a in enumerate(ids):
        if iid_a in to_drop:
            continue
        a = keep[iid_a]
        for iid_b in ids[i + 1 :]:
            if iid_b in to_drop:
                continue
            b = keep[iid_b]
            # Algorithm 1: A fully inside B → drop A
            if containment and _containment(a, b) >= containment_threshold:
                to_drop.add(iid_a)
                break
            # Algorithm 2: IoU > threshold → drop the smaller one
            if _iou(a, b) > iou_threshold:
                if _mask_area(a) <= _mask_area(b):
                    to_drop.add(iid_a)
                    break
                else:
                    to_drop.add(iid_b)
    return {iid: m for iid, m in keep.items() if iid not in to_drop}


def dedup_masks_across_frames(
    per_frame: Dict[int, Dict[int, np.ndarray]],
    **kwargs,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Apply dedup to each frame independently."""
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for frame_idx, inst_masks in per_frame.items():
        out[frame_idx] = dedup_masks_per_frame(inst_masks, **kwargs)
    return out


def mask_centroid(mask: np.ndarray) -> Tuple[int, int] | None:
    """Return (cx, cy) centroid of a binary mask, or None if empty.

    Uses cv2.moments — for multi-modal (disjoint) masks the centroid can
    fall outside the mask. Prefer `mask_centroid_largest_component` when
    placing visible markers.
    """
    import cv2

    # cv2.moments wants uint8
    m = cv2.moments(mask.astype(np.uint8))
    if m["m00"] == 0:
        return None
    cx = int(round(m["m10"] / m["m00"]))
    cy = int(round(m["m01"] / m["m00"]))
    return cx, cy


def mask_centroid_largest_component(mask: np.ndarray) -> Tuple[int, int] | None:
    """Centroid of the LARGEST connected component of a binary mask.

    Robust for SAM3 instances that span multiple disjoint regions (the
    moments centroid would otherwise land between blobs). Returns the
    integer pixel coordinates of the largest blob's centroid, or None
    if the mask is empty.
    """
    import cv2

    arr = mask.astype(np.uint8)
    if arr.sum() == 0:
        return None
    n, _, stats, centroids = cv2.connectedComponentsWithStats(arr, connectivity=8)
    if n <= 1:
        return None
    # Skip background (label 0). Pick largest by area (stats[:, 4]).
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    cx, cy = centroids[best]
    return int(round(cx)), int(round(cy))


def largest_mask_per_instance(
    per_frame: Dict[int, Dict[int, np.ndarray]],
) -> Dict[int, Tuple[int, np.ndarray]]:
    """Pick the frame where each instance has largest mask area.

    Returns {instance_id: (frame_idx, mask)}.
    """
    best: Dict[int, Tuple[int, np.ndarray, int]] = {}
    for frame_idx, inst_masks in per_frame.items():
        for iid, mask in inst_masks.items():
            area = _mask_area(mask)
            if iid not in best or area > best[iid][2]:
                best[iid] = (frame_idx, mask, area)
    return {iid: (f, m) for iid, (f, m, _) in best.items()}


def instance_id_set(per_frame: Dict[int, Dict[int, np.ndarray]]) -> List[int]:
    """Sorted list of all distinct instance IDs appearing in any frame."""
    ids: set = set()
    for inst_masks in per_frame.values():
        ids.update(inst_masks.keys())
    return sorted(ids)
