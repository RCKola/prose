"""SAM3 video predictor wrapper with text-prompt tracking.

Uses `transformers.Sam3VideoModel` + `Sam3VideoProcessor` (transformers >= 5.0).
All prompts are passed in a single session and processed in one propagation pass,
matching the official multi-prompt API.

The wrapper also captures per-mask predicted-IoU on `self.last_extras` for
stage 3 to persist. Transformers SAM3 v5 does NOT surface a stability score
(probed 2026-05-15: returns ['boxes', 'masks', 'object_ids',
'prompt_to_obj_ids', 'scores']) — anywhere downstream that historically
needed stability falls back to a synthetic mask-area-variance signal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from ..utils.gpu import release_gpu, select_torch_dtype
from ..utils.logging import get_logger
from ..utils.profiling import record_model_call

log = get_logger(__name__)


@dataclass
class Sam3VideoExtras:
    """Per-mask quality + per-instance label signals from a SAM3 propagation.

    All maps are keyed by (frame_idx, instance_id) or (instance_id,) as noted.
    Missing entries are allowed — downstream code must tolerate sparsity.
    """

    # frame_idx -> instance_id -> predicted_iou ∈ [0, 1]
    per_frame_iou: Dict[int, Dict[int, float]] = field(default_factory=dict)
    # instance_id -> originating text prompt (e.g. "chair")
    instance_to_prompt: Dict[int, str] = field(default_factory=dict)


def _extract_score(obj: Any, key: str, idx: int) -> Optional[float]:
    """Pull score `key` at row `idx` out of a transformers post-processor output.

    Tolerates dicts, attribute access, torch tensors, numpy arrays, and lists.
    Returns None if the field is absent or unindexable.
    """
    v = None
    if isinstance(obj, dict):
        v = obj.get(key)
    else:
        v = getattr(obj, key, None)
    if v is None:
        return None
    try:
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        arr = np.asarray(v)
        if arr.ndim == 0:
            return float(arr)
        if idx < 0 or idx >= len(arr):
            return None
        item = arr[idx]
        if hasattr(item, "item"):
            return float(item.item())
        return float(item)
    except Exception:
        return None


def make_sam3_wrapper(model_id: str, **kwargs: Any):
    """Construct the SAM3 text-prompted video segmenter (transformers backend)."""
    return Sam3VideoWrapper(model_id=model_id, **kwargs)


class Sam3VideoWrapper:
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        *,
        torch_dtype: str = "float16",
        device: str = "cuda",
        offload_video_to_cpu: bool = False,
    ) -> None:
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        dtype = select_torch_dtype(torch_dtype)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model_id = model_id

        self.offload_video_to_cpu = offload_video_to_cpu

        log.info("Loading SAM3 video model %s (dtype=%s, device=%s)", model_id, torch_dtype, self.device)
        self.model = Sam3VideoModel.from_pretrained(model_id).to(self.device, dtype=dtype)
        self.model.eval()
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)
        # Populated by `segment_video_with_text_prompts`. Stage 3 reads this
        # after each call to persist scores + labels in the artifact.
        self.last_extras: Sam3VideoExtras = Sam3VideoExtras()

    def close(self) -> None:
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        self.model = None
        self.processor = None
        release_gpu(model, processor)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def segment_video_with_text_prompts(
        self,
        frames: Sequence[np.ndarray],
        text_prompts: Sequence[str],
        *,
        max_frame_num_to_track: int = 50,
        per_prompt: bool = False,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Run SAM3 on all prompts in a single propagation pass.

        Args:
            frames: list of (H,W,3) uint8 RGB numpy arrays.
            text_prompts: object names (e.g. ['chair','lamp']).
            per_prompt: when True, run one SAM3 video session per prompt
                (each prompt isolated in its own tracking pool, eliminating
                cross-prompt iid contamination) and merge results with iid
                offsets to keep global uniqueness. Costs ~len(text_prompts)
                × the batched mode but produces stabler per-prompt masks on
                scenes where SAM3 propagation lets a slot drift to a
                different object. Default False keeps the original single
                propagation pass.

        Returns:
            { frame_idx : { global_instance_id : (H,W) bool mask } }
        """
        if not text_prompts:
            self.last_extras = Sam3VideoExtras()
            return {}

        if per_prompt and len(text_prompts) > 1:
            return self._segment_video_per_prompt(
                frames=frames,
                text_prompts=text_prompts,
                max_frame_num_to_track=max_frame_num_to_track,
            )

        log.info("SAM3: %d prompts: %s", len(text_prompts), list(text_prompts))

        t0 = time.perf_counter()
        storage_dev = torch.device("cpu") if self.offload_video_to_cpu else self.device
        session = self.processor.init_video_session(
            video=list(frames),
            inference_device=self.device,
            processing_device=self.device,
            video_storage_device=storage_dev,
            dtype=self.dtype,
        )
        session = self.processor.add_text_prompt(
            inference_session=session,
            text=list(text_prompts),
        )

        combined: Dict[int, Dict[int, np.ndarray]] = {}
        extras = Sam3VideoExtras()
        warned_missing_p2o = False

        with torch.inference_mode():
            for model_outputs in self.model.propagate_in_video_iterator(
                inference_session=session,
                max_frame_num_to_track=max_frame_num_to_track,
            ):
                processed = self.processor.postprocess_outputs(session, model_outputs)
                fidx = int(model_outputs.frame_idx)

                if not isinstance(processed, dict):
                    continue

                masks = processed.get("masks")
                ids = processed.get("object_ids")
                if masks is None or ids is None:
                    continue
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()
                if hasattr(ids, "cpu"):
                    ids = ids.cpu().numpy().tolist()
                else:
                    ids = list(ids)
                if len(ids) == 0:
                    continue

                frame_masks: Dict[int, np.ndarray] = {}
                iou_row: Dict[int, float] = {}
                for k, (iid, m) in enumerate(zip(ids, masks)):
                    iid_int = int(iid)
                    m = np.asarray(m)
                    if m.ndim == 3:
                        m = m[0]
                    frame_masks[iid_int] = m.astype(bool)

                    pi = _extract_score(processed, "predicted_iou", k)
                    if pi is None:
                        pi = _extract_score(processed, "iou_scores", k)
                    if pi is None:
                        pi = _extract_score(processed, "scores", k)
                    if pi is not None:
                        iou_row[iid_int] = pi

                if iou_row:
                    extras.per_frame_iou[fidx] = iou_row

                if frame_masks:
                    combined[fidx] = frame_masks

                # Direct prompt → object-ids map from the v5 post-processor.
                # Set per iid on first emission (setdefault); ignore later
                # frames since each obj_id is owned by exactly one prompt.
                p2o = processed.get("prompt_to_obj_ids")
                if p2o is None:
                    if not warned_missing_p2o:
                        log.warning(
                            "SAM3 post-processor returned no 'prompt_to_obj_ids'; "
                            "instance_to_prompt will be empty."
                        )
                        warned_missing_p2o = True
                    continue
                # transformers Sam3VideoProcessor returns prompt_to_obj_ids
                # keyed by the prompt **text** (e.g. "chair"), not an integer
                # index. Earlier wrapper code assumed int keys and silently
                # dropped every entry — keep the int path for forward-compat
                # and add a string path that passes the text through directly.
                for pid, oids in p2o.items():
                    if isinstance(pid, str):
                        prompt_text = pid
                    else:
                        try:
                            pid_int = int(pid)
                        except (TypeError, ValueError):
                            continue
                        if not (0 <= pid_int < len(text_prompts)):
                            continue
                        prompt_text = str(text_prompts[pid_int])
                    if hasattr(oids, "cpu"):
                        oids = oids.cpu().numpy().tolist()
                    elif hasattr(oids, "tolist"):
                        oids = oids.tolist()
                    else:
                        oids = list(oids)
                    for oid in oids:
                        extras.instance_to_prompt.setdefault(int(oid), prompt_text)

        self.last_extras = extras

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record_model_call(
            self.model_id, "segment_video_with_text_prompts",
            time.perf_counter() - t0, n_inputs=len(frames),
        )
        return combined

    def _segment_video_per_prompt(
        self,
        *,
        frames: Sequence[np.ndarray],
        text_prompts: Sequence[str],
        max_frame_num_to_track: int,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Run one SAM3 video session per prompt, merge with global iid offsets.

        Eliminates cross-prompt iid contamination (a SAM3 slot drifting from
        one prompt's object to another's) at the cost of N sessions.
        """
        log.info(
            "SAM3 per-prompt: %d sessions for prompts %s",
            len(text_prompts), list(text_prompts),
        )
        t0 = time.perf_counter()
        merged: Dict[int, Dict[int, np.ndarray]] = {}
        merged_extras = Sam3VideoExtras()
        next_offset = 0
        for prompt in text_prompts:
            sub = self.segment_video_with_text_prompts(
                frames, [prompt],
                max_frame_num_to_track=max_frame_num_to_track,
                per_prompt=False,
            )
            sub_extras = self.last_extras
            local_iids = sorted({iid for fm in sub.values() for iid in fm})
            remap = {iid: iid + next_offset for iid in local_iids}
            for fid, by_iid in sub.items():
                bucket = merged.setdefault(fid, {})
                for iid, m in by_iid.items():
                    bucket[remap[iid]] = m
            if sub_extras is not None:
                for fid, row in (sub_extras.per_frame_iou or {}).items():
                    out_row = merged_extras.per_frame_iou.setdefault(fid, {})
                    for iid, v in row.items():
                        out_row[remap[iid]] = v
                for iid, lbl in (sub_extras.instance_to_prompt or {}).items():
                    merged_extras.instance_to_prompt[remap[iid]] = lbl
            if local_iids:
                next_offset = max(remap.values()) + 1
        self.last_extras = merged_extras
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record_model_call(
            self.model_id, "segment_video_with_text_prompts_per_prompt",
            time.perf_counter() - t0, n_inputs=len(frames),
        )
        return merged
