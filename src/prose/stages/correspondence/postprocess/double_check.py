"""Double-check postprocessor: per-pair VLM yes/no verification pass.

For each proposed ``(src_iid, ref_iid)`` correspondence, renders a zoomed
side-by-side crop (src left, ref right) and asks the VLM a binary question:
  "Is the object outlined in green on the LEFT the same physical object
   as the one outlined in green on the RIGHT? Answer [1] yes or [0] no."

All pairs are batched into a single ``call_batch`` for throughput.
Only pairs answered [1] (yes) are retained.

Disabled by default (``enabled=False``); set ``do_double_check=true``
in the YAML to activate.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..context import CorrespondenceResult, PairContext
from ..shared.frame_io import load_rgb
from ..vlm.invoker import VLMInvoker

log = logging.getLogger(__name__)

_DC_SYSTEM = (
    "You compare two object crops from different captures of the same room. "
    "The LEFT crop is from Scene A, the RIGHT crop is from Scene B. "
    "Both objects are outlined in green. "
    "Decide whether they are the same physical object."
)

_DC_USER = (
    "Is the object on the LEFT the same physical object as the object on "
    "the RIGHT? Consider shape, color, material, and size. "
    "A wrong match is worse than a missed one — only say yes if confident. "
    "Answer with a single integer: [1] for yes, [0] for no."
)

_DC_NEG_SYSTEM = (
    "You compare two object crops from different captures of the same room. "
    "The LEFT crop is from Scene A, the RIGHT crop is from Scene B. "
    "Both objects are outlined in green. "
    "Decide whether they are DIFFERENT objects."
)

_DC_NEG_USER = (
    "Are the object on the LEFT and the object on the RIGHT DIFFERENT "
    "physical objects? Consider shape, color, material, and size. "
    "A missed mismatch is worse than a false alarm — say yes if in doubt. "
    "Answer with a single integer: [1] for yes (different), [0] for no (same)."
)

_YES_NO_RE = re.compile(r"\[?\s*([01])\s*\]?")


def _parse_yes_no(raw: str) -> Optional[bool]:
    tokens = _YES_NO_RE.findall(raw.strip().lower())
    if tokens:
        return tokens[-1] == "1"
    return None


def _best_crop_frame(
    per_frame_masks: Dict[int, Dict],
    iid: int,
    color_paths: Dict[int, Path],
) -> Optional[int]:
    """Pick the frame where iid has the largest mask area."""
    best_fid, best_area = None, 0
    for fid, inst in per_frame_masks.items():
        m = inst.get(int(iid))
        if m is None or not bool(np.any(m)):
            continue
        if int(fid) not in color_paths:
            continue
        area = int(np.count_nonzero(m))
        if area > best_area:
            best_area = area
            best_fid = int(fid)
    return best_fid


def _render_crop(
    fid: int,
    iid: int,
    per_frame_masks: Dict,
    color_paths: Dict[int, Path],
    image_rotation_k: int,
    pad_frac: float = 0.4,
    min_side_px: int = 256,
    panel_px: int = 384,
) -> Optional[np.ndarray]:
    """Render a zoomed, outlined crop of one instance. Returns BGR."""
    from ..visuals.crops import square_crop

    bgr = load_rgb(color_paths[fid])
    mask = per_frame_masks.get(fid, {}).get(int(iid))
    if mask is None:
        return None
    if image_rotation_k:
        bgr = np.rot90(bgr, k=image_rotation_k).copy()
        mask = np.rot90(mask.astype(np.uint8), k=image_rotation_k).astype(bool)

    crop = square_crop(
        bgr, mask,
        pad_frac=pad_frac,
        min_side_px=min_side_px,
        outline_thickness=2,
        outline_color_bgr=(0, 255, 0),
    )
    if crop is None:
        return None
    h, w = crop.shape[:2]
    if max(h, w) != panel_px:
        scale = panel_px / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return crop


def _side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray,
                  gap: int = 8) -> Image.Image:
    """Concatenate two BGR crops into one RGB PIL image with a gap."""
    lh, lw = left_bgr.shape[:2]
    rh, rw = right_bgr.shape[:2]
    h = max(lh, rh)
    canvas = np.zeros((h, lw + gap + rw, 3), dtype=np.uint8)
    canvas[:lh, :lw] = left_bgr
    canvas[:rh, lw + gap:] = right_bgr
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@dataclass
class DoubleCheckPostprocessor:
    """Verifies each proposed match with a batched VLM yes/no call.

    Renders side-by-side zoomed crops (src left, ref right) and asks
    "same object?" All queries are batched via ``call_batch``.
    """

    vlm: VLMInvoker
    enabled: bool = False
    negation_pass: bool = False
    keep_on_error: bool = True
    crop_pad_frac: float = 0.4
    crop_min_side_px: int = 256
    crop_panel_px: int = 384
    sample_dir: Optional[Path] = None

    def apply(
        self, result: CorrespondenceResult, ctx: PairContext
    ) -> CorrespondenceResult:
        if not self.enabled or not result.pairs:
            return result

        rot_k = ctx.image_rotation_k or 0
        weights = list(result.weights) if result.weights else [1.0] * len(result.pairs)

        # Build all side-by-side images.
        batch_images: List[List[Image.Image]] = []
        batch_indices: List[int] = []
        skipped: List[Tuple[int, dict]] = []

        for idx, (s_iid, r_iid) in enumerate(result.pairs):
            src_fid = _best_crop_frame(
                ctx.src.per_frame_masks, s_iid, ctx.src_frames.color_paths)
            ref_fid = _best_crop_frame(
                ctx.ref.per_frame_masks, r_iid, ctx.ref_frames.color_paths)

            if src_fid is None or ref_fid is None:
                skipped.append((idx, {"src_iid": int(s_iid), "ref_iid": int(r_iid),
                                      "verdict": "keep_no_frame"}))
                continue

            src_crop = _render_crop(
                src_fid, s_iid, ctx.src.per_frame_masks,
                ctx.src_frames.color_paths, rot_k,
                self.crop_pad_frac, self.crop_min_side_px, self.crop_panel_px)
            ref_crop = _render_crop(
                ref_fid, r_iid, ctx.ref.per_frame_masks,
                ctx.ref_frames.color_paths, rot_k,
                self.crop_pad_frac, self.crop_min_side_px, self.crop_panel_px)

            if src_crop is None or ref_crop is None:
                skipped.append((idx, {"src_iid": int(s_iid), "ref_iid": int(r_iid),
                                      "verdict": "keep_no_crop"}))
                continue

            sbs = _side_by_side(src_crop, ref_crop)
            batch_images.append([sbs])
            batch_indices.append(idx)

        # Batch VLM call.
        audit_dc: List[dict] = []
        kept: List[Tuple[int, int]] = []
        kept_weights: List[float] = []
        verdict_map: Dict[int, Optional[bool]] = {}

        if batch_images:
            try:
                raws = self.vlm.call_batch(
                    [(imgs, _DC_SYSTEM, _DC_USER) for imgs in batch_images],
                    max_new_tokens=64,
                    enable_thinking=False,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("DC batch call failed: %s", exc)
                raws = [None] * len(batch_images)

            for bi, raw in zip(batch_indices, raws):
                verdict_map[bi] = _parse_yes_no(raw) if raw else None

            if self.sample_dir is not None:
                try:
                    dc_dir = self.sample_dir / str(ctx.pair_id) / "double_check"
                    dc_dir.mkdir(parents=True, exist_ok=True)
                    for bi_pos, (bi, imgs) in enumerate(zip(batch_indices, batch_images)):
                        s, r = int(result.pairs[bi][0]), int(result.pairs[bi][1])
                        imgs[0].save(dc_dir / f"dc_{s}_{r}.jpg")
                    raw_lines = []
                    for bi_pos, (bi, raw) in enumerate(zip(batch_indices, raws)):
                        s, r = int(result.pairs[bi][0]), int(result.pairs[bi][1])
                        v = verdict_map.get(bi)
                        raw_lines.append(f"src={s} ref={r} verdict={v} raw={raw}")
                    (dc_dir / "dc_responses.txt").write_text("\n".join(raw_lines))
                except Exception:  # noqa: BLE001
                    pass

        # Assemble results.
        for idx, ((s_iid, r_iid), w) in enumerate(zip(result.pairs, weights)):
            s, r = int(s_iid), int(r_iid)
            entry = {"src_iid": s, "ref_iid": r}

            # Check skipped entries.
            skip_entry = next((e for i, e in skipped if i == idx), None)
            if skip_entry is not None:
                if self.keep_on_error:
                    kept.append((s, r))
                    kept_weights.append(float(w))
                entry.update(skip_entry)
                audit_dc.append(entry)
                continue

            verdict = verdict_map.get(idx)
            if verdict is True:
                kept.append((s, r))
                kept_weights.append(float(w))
                entry["verdict"] = "yes"
            elif verdict is False:
                entry["verdict"] = "no"
            else:
                if self.keep_on_error:
                    kept.append((s, r))
                    kept_weights.append(float(w))
                entry["verdict"] = "ambiguous"
            # Include raw response for debugging.
            raw_idx = next((bi_pos for bi_pos, bi in enumerate(batch_indices)
                            if bi == idx), None)
            if raw_idx is not None and raw_idx < len(raws):
                entry["raw"] = (raws[raw_idx] or "")[:200]
            audit_dc.append(entry)

        log.info(
            "DC [%s]: %d in → %d kept (%d yes, %d no, %d ambiguous/skip)",
            ctx.pair_id, len(result.pairs), len(kept),
            sum(1 for e in audit_dc if e.get("verdict") == "yes"),
            sum(1 for e in audit_dc if e.get("verdict") == "no"),
            sum(1 for e in audit_dc if e.get("verdict") not in ("yes", "no")),
        )

        audit_neg: Optional[List[dict]] = None

        if self.negation_pass and kept:
            neg_images: List[List[Image.Image]] = []
            neg_indices: List[int] = []

            for ki, (s_iid, r_iid) in enumerate(kept):
                src_fid = _best_crop_frame(
                    ctx.src.per_frame_masks, s_iid, ctx.src_frames.color_paths)
                ref_fid = _best_crop_frame(
                    ctx.ref.per_frame_masks, r_iid, ctx.ref_frames.color_paths)
                if src_fid is None or ref_fid is None:
                    continue
                src_crop = _render_crop(
                    src_fid, s_iid, ctx.src.per_frame_masks,
                    ctx.src_frames.color_paths, rot_k,
                    self.crop_pad_frac, self.crop_min_side_px, self.crop_panel_px)
                ref_crop = _render_crop(
                    ref_fid, r_iid, ctx.ref.per_frame_masks,
                    ctx.ref_frames.color_paths, rot_k,
                    self.crop_pad_frac, self.crop_min_side_px, self.crop_panel_px)
                if src_crop is None or ref_crop is None:
                    continue
                neg_images.append([_side_by_side(src_crop, ref_crop)])
                neg_indices.append(ki)

            neg_verdicts: Dict[int, Optional[bool]] = {}
            neg_raws: List[Optional[str]] = []
            if neg_images:
                try:
                    neg_raws = self.vlm.call_batch(
                        [(imgs, _DC_NEG_SYSTEM, _DC_NEG_USER) for imgs in neg_images],
                        max_new_tokens=64,
                        enable_thinking=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("DC negation batch call failed: %s", exc)
                    neg_raws = [None] * len(neg_images)

                for ni, raw in zip(neg_indices, neg_raws):
                    neg_verdicts[ni] = _parse_yes_no(raw) if raw else None

                if self.sample_dir is not None:
                    try:
                        neg_dir = self.sample_dir / str(ctx.pair_id) / "double_check_neg"
                        neg_dir.mkdir(parents=True, exist_ok=True)
                        raw_lines = []
                        for ni_pos, (ni, raw) in enumerate(zip(neg_indices, neg_raws)):
                            s, r = int(kept[ni][0]), int(kept[ni][1])
                            v = neg_verdicts.get(ni)
                            raw_lines.append(f"src={s} ref={r} neg_verdict={v} raw={raw}")
                        (neg_dir / "neg_responses.txt").write_text("\n".join(raw_lines))
                    except Exception:  # noqa: BLE001
                        pass

            final_kept: List[Tuple[int, int]] = []
            final_weights: List[float] = []
            audit_neg = []
            for ki, ((s, r), w) in enumerate(zip(kept, kept_weights)):
                entry = {"src_iid": int(s), "ref_iid": int(r)}
                neg_v = neg_verdicts.get(ki)
                if neg_v is False:
                    final_kept.append((s, r))
                    final_weights.append(w)
                    entry["neg_verdict"] = "same"
                elif neg_v is True:
                    entry["neg_verdict"] = "different"
                else:
                    if self.keep_on_error:
                        final_kept.append((s, r))
                        final_weights.append(w)
                    entry["neg_verdict"] = "ambiguous"
                audit_neg.append(entry)

            log.info(
                "DC-neg [%s]: %d in → %d kept (%d same, %d different, %d ambiguous)",
                ctx.pair_id, len(kept), len(final_kept),
                sum(1 for e in audit_neg if e.get("neg_verdict") == "same"),
                sum(1 for e in audit_neg if e.get("neg_verdict") == "different"),
                sum(1 for e in audit_neg if e.get("neg_verdict") == "ambiguous"),
            )
            kept = final_kept
            kept_weights = final_weights

        return CorrespondenceResult(
            pairs=kept,
            weights=kept_weights if kept_weights else None,
            audit={
                **result.audit,
                "double_check": {
                    "n_in": len(result.pairs),
                    "n_kept": len(kept),
                    "pairs": audit_dc,
                    **({"negation": audit_neg} if audit_neg is not None else {}),
                },
            },
        )
