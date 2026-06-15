"""Stage 3: temporally consistent instance segmentation via SAM3 (paper §3.4).

Per text prompt, SAM3 tracks each matching concept across frames. Instance IDs
from different prompts are offset so they are globally unique. Duplicate masks
within a frame are removed via:
  (Algorithm 1) containment — drop A if A fully inside B
  (Algorithm 2) IoU > 0.5    — drop the smaller one
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from ..utils.io import dump_pickle, ensure_dir
from ..utils.logging import get_logger
from ..utils.mask_ops import dedup_masks_across_frames, instance_id_set

log = get_logger(__name__)


@dataclass
class SegmentationArtifact:
    subscan_id: str
    per_frame_masks: Dict[int, Dict[int, np.ndarray]]
    instance_ids: List[int] = field(default_factory=list)
    text_prompts: List[str] = field(default_factory=list)
    # Per-(frame, iid) SAM3 predicted-IoU signal; sparse, missing keys allowed.
    per_frame_iou: Dict[int, Dict[int, float]] = field(default_factory=dict)
    # Per-instance originating text prompt — used as a cheap class label by
    # the mosaic-matcher Stage 4 backend.
    instance_to_prompt: Dict[int, str] = field(default_factory=dict)

    def save(self, out_dir: Path) -> Path:
        out_dir = ensure_dir(out_dir)
        path = out_dir / f"{self.subscan_id}.pkl"
        # Store masks as packed bits to save disk space.
        packed: Dict[int, Dict[int, np.ndarray]] = {}
        for fidx, inst in self.per_frame_masks.items():
            packed[fidx] = {iid: np.packbits(m, axis=None) for iid, m in inst.items()}

        shapes: Dict[int, tuple] = {}
        for fidx, inst in self.per_frame_masks.items():
            for _, m in inst.items():
                shapes[fidx] = m.shape
                break

        dump_pickle(
            {
                "packed_masks": packed,
                "frame_shapes": shapes,
                "instance_ids": self.instance_ids,
                "text_prompts": self.text_prompts,
                # New optional fields. Older readers ignore them; mosaic
                # backend reads them when present.
                "per_frame_iou": self.per_frame_iou,
                "instance_to_prompt": self.instance_to_prompt,
            },
            path,
        )
        return path


def _load_frames_as_rgb(frame_paths: Sequence[Path]) -> List[np.ndarray]:
    """Read RGB frames (H,W,3 uint8)."""
    from ..utils.visualize import load_frame_bgr
    out: List[np.ndarray] = []
    for p in frame_paths:
        im = load_frame_bgr(p)
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out


def run_segmentation(
    wrapper,
    *,
    subscan_id: str,
    frame_paths: Sequence[Path],
    text_prompts: Sequence[str],
    cfg_stage3,
) -> SegmentationArtifact:
    """Run SAM3 per object name, merge IDs, deduplicate per frame."""
    frames = _load_frames_as_rgb(frame_paths)

    # Cap prompts to avoid SAM3 mask_iou OOM on runaway-tail subscans where
    # the stage-2 VLM emits 100+ items (e.g. InternVL-14B-V1.2 wood-stuffing
    # tail: "wooden trim, wooden corner, wooden detail, ..."). p99 of the val
    # set is 38 prompts; only 6/984 exceed 50 and those triggered 42 GiB
    # allocations in mask_iou. Cap is configurable via cfg_stage3.max_prompts;
    # null/missing keeps all prompts (legacy behavior).
    max_prompts = getattr(cfg_stage3, "max_prompts", None)
    if max_prompts is not None and len(text_prompts) > int(max_prompts):
        log.warning(
            "Stage 3 [%s]: truncating prompts %d → %d to avoid SAM3 OOM",
            subscan_id, len(text_prompts), int(max_prompts),
        )
        text_prompts = list(text_prompts)[: int(max_prompts)]

    raw_masks = wrapper.segment_video_with_text_prompts(
        frames,
        text_prompts,
        max_frame_num_to_track=int(cfg_stage3.max_frames_per_track),
        per_prompt=bool(getattr(cfg_stage3, "per_prompt_sessions", False)),
    )
    extras = getattr(wrapper, "last_extras", None)

    # Discard tiny masks.
    min_px = int(cfg_stage3.min_mask_pixels)
    filtered: Dict[int, Dict[int, np.ndarray]] = {}
    for fidx, inst in raw_masks.items():
        kept = {iid: m for iid, m in inst.items() if m.sum() >= min_px}
        if kept:
            filtered[fidx] = kept

    deduped = dedup_masks_across_frames(
        filtered,
        containment=bool(cfg_stage3.dedup_containment),
        iou_threshold=float(cfg_stage3.dedup_iou_threshold),
    )

    instance_ids = instance_id_set(deduped)
    log.info(
        "Stage 3 [%s]: %d instances across %d frames (prompts=%d)",
        subscan_id, len(instance_ids), len(deduped), len(text_prompts),
    )

    # Carry SAM3 extras through, restricted to (fid, iid) pairs that survived
    # min-mask-pixel filter + dedup. Missing entries are OK — consumers
    # tolerate sparsity.
    per_frame_iou: Dict[int, Dict[int, float]] = {}
    instance_to_prompt: Dict[int, str] = {}
    if extras is not None:
        iid_set = set(instance_ids)
        for fidx, inst in deduped.items():
            iou_src = extras.per_frame_iou.get(int(fidx), {}) if extras.per_frame_iou else {}
            keep_iou = {int(iid): float(iou_src[iid]) for iid in inst if iid in iou_src}
            if keep_iou:
                per_frame_iou[int(fidx)] = keep_iou
        for iid, prompt in extras.instance_to_prompt.items():
            if int(iid) in iid_set:
                instance_to_prompt[int(iid)] = str(prompt)

        n_with_iou = sum(len(v) for v in per_frame_iou.values())
        if n_with_iou or instance_to_prompt:
            log.info(
                "Stage 3 [%s]: extras — iou rows=%d, labelled iids=%d/%d",
                subscan_id, n_with_iou, len(instance_to_prompt), len(iid_set),
            )

    return SegmentationArtifact(
        subscan_id=subscan_id,
        per_frame_masks=deduped,
        instance_ids=instance_ids,
        text_prompts=list(text_prompts),
        per_frame_iou=per_frame_iou,
        instance_to_prompt=instance_to_prompt,
    )
