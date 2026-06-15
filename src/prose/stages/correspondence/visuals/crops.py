"""Per-bin pairwise square-crop mosaic (self-contained copy).

For each bin: pick top-K iids per side by mean depth across frames,
select a wide-context frame per iid (scene_avg_depth ≥ median, then
max instance area), square-crop with padding + min-side floor, stamp
marker label, compose a 2×K mosaic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Per-frame stats (depth + area)
# ---------------------------------------------------------------------------

def compute_per_frame_stats(
    masks_all: Dict[int, Dict[int, np.ndarray]],
    abs_frame_ids: Sequence[int],
    depth_dir: Optional[Path],
) -> Dict[int, dict]:
    stats: Dict[int, dict] = {}
    for fid, inst in masks_all.items():
        abs_id = int(abs_frame_ids[int(fid)]) if int(fid) < len(abs_frame_ids) else None
        depth = _load_depth(depth_dir, abs_id) if (depth_dir is not None and abs_id is not None) else None
        scene_avg = float(depth[depth > 0].mean()) if depth is not None and (depth > 0).any() else None
        per_inst: Dict[int, dict] = {}
        for iid, m in inst.items():
            if m is None or not m.any():
                continue
            area = int(m.sum())
            d_mean = d_std = None
            if depth is not None and depth.shape == m.shape:
                dvals = depth[m & (depth > 0)]
                if dvals.size:
                    d_mean = float(dvals.mean()); d_std = float(dvals.std())
            per_inst[int(iid)] = {"area": area, "depth_mean": d_mean, "depth_std": d_std}
        stats[int(fid)] = {"scene_avg_depth": scene_avg,
                           "abs_frame_id": abs_id,
                           "per_instance": per_inst}
    return stats


def _load_depth(depth_dir: Path, abs_frame_id: int) -> Optional[np.ndarray]:
    p = depth_dir / f"frame_{int(abs_frame_id):06d}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def instance_mean_depth_across_frames(iid: int, stats: Dict[int, dict]) -> Optional[float]:
    vals = []
    for s in stats.values():
        rec = s["per_instance"].get(int(iid))
        if rec is None:
            continue
        d = rec.get("depth_mean")
        if d is not None:
            vals.append(float(d))
    if not vals:
        return None
    return float(np.mean(vals))


def top_k_by_mean_depth(iids: Sequence[int], stats: Dict[int, dict], k: int) -> List[int]:
    by_depth = []
    for iid in iids:
        d = instance_mean_depth_across_frames(int(iid), stats)
        if d is not None:
            by_depth.append((d, int(iid)))
    if by_depth:
        by_depth.sort(reverse=True)
        return [iid for _, iid in by_depth[:k]]
    by_area: Dict[int, int] = {}
    for s in stats.values():
        for iid, rec in s["per_instance"].items():
            by_area[int(iid)] = by_area.get(int(iid), 0) + int(rec.get("area", 0))
    scored = [(by_area.get(int(iid), 0), int(iid)) for iid in iids]
    scored.sort(reverse=True)
    return [iid for _, iid in scored[:k]]


def select_crop_frame(
    iid: int, candidate_frame_ids: Sequence[int], stats: Dict[int, dict],
) -> Optional[int]:
    candidates = [int(f) for f in candidate_frame_ids
                  if int(f) in stats and iid in stats[int(f)]["per_instance"]]
    if not candidates:
        return None
    scene_depths = [stats[f]["scene_avg_depth"] for f in candidates
                    if stats[f]["scene_avg_depth"] is not None]
    if scene_depths:
        med = float(np.median(scene_depths))
        kept = [f for f in candidates
                if (stats[f]["scene_avg_depth"] or -1) >= med]
        if kept:
            candidates = kept
    return max(candidates, key=lambda f: stats[f]["per_instance"][iid]["area"])


# ---------------------------------------------------------------------------
# Crop + compose
# ---------------------------------------------------------------------------

def square_crop(
    frame_bgr: np.ndarray, mask: np.ndarray, *,
    pad_frac: float, min_side_px: int, outline_thickness: int,
    outline_color_bgr: Tuple[int, int, int] = (0, 255, 0),
) -> Optional[np.ndarray]:
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    H, W = frame_bgr.shape[:2]
    bh = y1 - y0 + 1; bw = x1 - x0 + 1
    side = max(bh, bw)
    pad = int(round(side * pad_frac))
    side_padded = max(side + 2 * pad, int(min_side_px))
    side_padded = min(side_padded, min(H, W))
    cx = (x0 + x1) // 2; cy = (y0 + y1) // 2
    half = side_padded // 2
    sx0 = max(0, cx - half); sy0 = max(0, cy - half)
    sx1 = min(W, sx0 + side_padded); sy1 = min(H, sy0 + side_padded)
    sx0 = max(0, sx1 - side_padded); sy0 = max(0, sy1 - side_padded)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    crop = frame_bgr[sy0:sy1, sx0:sx1].copy()
    mcrop = mask[sy0:sy1, sx0:sx1].astype(np.uint8)
    contours, _ = cv2.findContours(mcrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(crop, contours, -1, outline_color_bgr,
                         int(max(1, outline_thickness)))
    return crop


def _resize_longside(img: np.ndarray, longside: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) == longside:
        return img
    if w >= h:
        nw, nh = longside, max(1, int(round(h * longside / w)))
    else:
        nh, nw = longside, max(1, int(round(w * longside / h)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def compose_crops_2xk(
    ref_crops: List[Optional[np.ndarray]],
    src_crops: List[Optional[np.ndarray]],
    *, panel_px: int, bg_bgr: Tuple[int, int, int] = (10, 10, 10),
):
    from PIL import Image as _Image
    k = max(len(ref_crops), len(src_crops))
    def _prep(x: Optional[np.ndarray]) -> np.ndarray:
        if x is None:
            p = np.zeros((panel_px, panel_px, 3), np.uint8); p[:] = bg_bgr; return p
        return _resize_longside(x, panel_px)
    t = [_prep(ref_crops[i] if i < len(ref_crops) else None) for i in range(k)]
    b = [_prep(src_crops[i] if i < len(src_crops) else None) for i in range(k)]
    col_w = [max(t[i].shape[1], b[i].shape[1], 1) for i in range(k)]
    col_x = [0]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)
    th = max((p.shape[0] for p in t), default=panel_px)
    bh = max((p.shape[0] for p in b), default=panel_px)
    W = sum(col_w); H = th + bh
    canvas = np.zeros((H, W, 3), np.uint8); canvas[:] = bg_bgr
    for i, img in enumerate(t):
        ih, iw = img.shape[:2]
        canvas[(th - ih) // 2:(th - ih) // 2 + ih,
               col_x[i] + (col_w[i] - iw) // 2:col_x[i] + (col_w[i] - iw) // 2 + iw] = img
    for i, img in enumerate(b):
        ih, iw = img.shape[:2]
        canvas[th + (bh - ih) // 2:th + (bh - ih) // 2 + ih,
               col_x[i] + (col_w[i] - iw) // 2:col_x[i] + (col_w[i] - iw) // 2 + iw] = img
    return _Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def compose_crops_grid(
    crops: List[Optional[np.ndarray]],
    *, panel_px: int, bg_bgr: Tuple[int, int, int] = (10, 10, 10),
):
    """Lay N crops out in a compact near-square grid.

    Replaces the 2xK strip for one side: ``cols = ceil(sqrt(N))`` keeps
    the canvas roughly square so the VLM's image downscale barely shrinks
    each crop (an ultra-wide strip gets squashed hard, losing detail).
    """
    import math as _math

    from PIL import Image as _Image

    items = [c for c in crops if c is not None]
    n = len(items)
    if n == 0:
        canvas = np.zeros((panel_px, panel_px, 3), np.uint8)
        canvas[:] = bg_bgr
        return _Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    cols = int(_math.ceil(_math.sqrt(n)))
    rows = int(_math.ceil(n / cols))
    cell = int(panel_px)
    canvas = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    canvas[:] = bg_bgr
    for idx, crop in enumerate(items):
        r, c = idx // cols, idx % cols
        img = _resize_longside(crop, cell)
        ih, iw = img.shape[:2]
        y0 = r * cell + (cell - ih) // 2
        x0 = c * cell + (cell - iw) // 2
        canvas[y0:y0 + ih, x0:x0 + iw] = img
    return _Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def stamp_label_top_left(crop: np.ndarray, marker_id: int) -> None:
    text = str(int(marker_id))
    fs = 0.9
    ft = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    org = (8, th + 8)
    cv2.putText(crop, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 0, 0), ft + 3, cv2.LINE_AA)
    cv2.putText(crop, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), ft, cv2.LINE_AA)


def build_bin_crops_mosaic(
    *,
    ref_iids: Sequence[int], src_iids: Sequence[int],
    ref_i2m: Dict[int, int], src_i2m: Dict[int, int],
    ref_masks_all: Dict[int, Dict[int, np.ndarray]],
    src_masks_all: Dict[int, Dict[int, np.ndarray]],
    ref_frame_bgr_loader, src_frame_bgr_loader,
    ref_stats: Dict[int, dict], src_stats: Dict[int, dict],
    top_k: int, pad_frac: float, min_side_px: int,
    outline_thickness: int, panel_px: int,
):
    """Returns (PIL.Image, ref_ordered_iids, src_ordered_iids)."""
    ref_top = top_k_by_mean_depth(list(ref_iids), ref_stats, top_k)
    src_top = top_k_by_mean_depth(list(src_iids), src_stats, top_k)
    ref_ordered = sorted(ref_top, key=lambda i: ref_i2m[i])
    src_ordered = sorted(src_top, key=lambda i: src_i2m[i])

    def _make(iid: int, marker_id: int, side_masks_all, side_stats, load_bgr):
        fids = list(side_masks_all.keys())
        fid = select_crop_frame(iid, fids, side_stats)
        if fid is None:
            return None
        m = side_masks_all[fid].get(iid)
        if m is None:
            return None
        frame = load_bgr(int(fid))
        crop = square_crop(frame, m, pad_frac=pad_frac, min_side_px=min_side_px,
                           outline_thickness=outline_thickness)
        if crop is None:
            return None
        stamp_label_top_left(crop, int(marker_id))
        return crop

    ref_crops = [_make(iid, ref_i2m[iid], ref_masks_all, ref_stats, ref_frame_bgr_loader)
                 for iid in ref_ordered]
    src_crops = [_make(iid, src_i2m[iid], src_masks_all, src_stats, src_frame_bgr_loader)
                 for iid in src_ordered]
    img = compose_crops_2xk(ref_crops, src_crops, panel_px=panel_px)
    return img, ref_ordered, src_ordered


def build_bin_crop_grids(
    *,
    ref_iids: Sequence[int], src_iids: Sequence[int],
    ref_i2m: Dict[int, int], src_i2m: Dict[int, int],
    ref_masks_all: Dict[int, Dict[int, np.ndarray]],
    src_masks_all: Dict[int, Dict[int, np.ndarray]],
    ref_frame_bgr_loader, src_frame_bgr_loader,
    ref_stats: Dict[int, dict], src_stats: Dict[int, dict],
    top_k: int, pad_frac: float, min_side_px: int,
    outline_thickness: int, panel_px: int,
):
    """Per-side compact crop grids.

    Same crop building as ``build_bin_crops_mosaic`` but lays each side
    out as its own near-square grid (``compose_crops_grid``) instead of a
    shared 2xK strip — avoids the ultra-wide canvas that loses crop
    resolution under the VLM's image downscale, and drops black padding
    on asymmetric bins.

    Returns ``(ref_grid_img, src_grid_img, ref_ordered, src_ordered)``.
    """
    ref_top = top_k_by_mean_depth(list(ref_iids), ref_stats, top_k)
    src_top = top_k_by_mean_depth(list(src_iids), src_stats, top_k)
    ref_ordered = sorted(ref_top, key=lambda i: ref_i2m[i])
    src_ordered = sorted(src_top, key=lambda i: src_i2m[i])

    def _make(iid: int, marker_id: int, side_masks_all, side_stats, load_bgr):
        fids = list(side_masks_all.keys())
        fid = select_crop_frame(iid, fids, side_stats)
        if fid is None:
            return None
        m = side_masks_all[fid].get(iid)
        if m is None:
            return None
        frame = load_bgr(int(fid))
        crop = square_crop(frame, m, pad_frac=pad_frac, min_side_px=min_side_px,
                           outline_thickness=outline_thickness)
        if crop is None:
            return None
        stamp_label_top_left(crop, int(marker_id))
        return crop

    ref_crops = [_make(iid, ref_i2m[iid], ref_masks_all, ref_stats, ref_frame_bgr_loader)
                 for iid in ref_ordered]
    src_crops = [_make(iid, src_i2m[iid], src_masks_all, src_stats, src_frame_bgr_loader)
                 for iid in src_ordered]
    ref_img = compose_crops_grid(ref_crops, panel_px=panel_px)
    src_img = compose_crops_grid(src_crops, panel_px=panel_px)
    return ref_img, src_img, ref_ordered, src_ordered
