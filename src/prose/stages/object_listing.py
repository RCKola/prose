"""Stage 2: object list discovery via Qwen3-VL-8B-Instruct (paper §3.3)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from typing import Protocol

from ..utils.gpu import parse_vlm_list_output
from ..utils.io import dump_json, ensure_dir
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ObjectListArtifact:
    subscan_id: str
    objects: List[str]        # final, deduplicated list
    raw_batches: List[List[str]]  # one list per VLM batch before consolidation

    def save(self, out_dir: Path) -> Path:
        out_dir = ensure_dir(out_dir)
        path = out_dir / f"{self.subscan_id}.json"
        dump_json({"objects": self.objects, "raw_batches": self.raw_batches}, path)
        return path


def _filter_background(objects: Sequence[str], ignored: Sequence[str]) -> List[str]:
    """Drop entries whose lowercase ending matches any banned suffix."""
    out: List[str] = []
    banned = [b.lower() for b in ignored]
    seen = set()
    for raw in objects:
        if not isinstance(raw, str):
            continue
        name = raw.strip().lower().replace("_", " ")
        if not name:
            continue
        if any(name.endswith(b) for b in banned):
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


# Room-type words VLMs list as "objects" despite the prompt asking only to
# identify the room type — SAM3 then segments a whole-room blob (instance 0 =
# "living room"), polluting Stage-4 correspondence.
_ROOM_TYPES = {
    "living room", "livingroom", "bedroom", "bed room", "bathroom", "kitchen",
    "dining room", "dining area", "diningroom", "hallway", "hall", "corridor",
    "office", "study", "closet", "wardrobe room", "entryway", "entrance",
    "foyer", "balcony", "garage", "laundry room", "pantry", "room", "nursery",
    "playroom", "play room", "kids room", "kid's room",
}


def _drop_room_types(objects: Sequence[str]) -> List[str]:
    """Drop room-type pseudo-objects (e.g. 'living room' listed as an object)."""
    return [o for o in objects if str(o).strip().lower() not in _ROOM_TYPES]


class VLMWrapper(Protocol):
    def chat_with_images(
        self, images, prompt, *, max_new_tokens: int, do_sample: bool, **kwargs,
    ) -> str: ...


def run_object_listing(
    wrapper: VLMWrapper,
    *,
    subscan_id: str,
    frame_paths: Sequence,
    cfg_stage2,
) -> ObjectListArtifact:
    """Discover objects over the subscan's frames.

    Frames are processed in batches of `cfg_stage2.max_batch_size`.
    If the subscan has more frames than the batch limit, a consolidation call
    is used to merge partial lists (as in the paper).
    """
    frame_paths = list(frame_paths)
    max_bs = int(cfg_stage2.max_batch_size)
    max_new = int(cfg_stage2.max_new_tokens)
    do_sample = bool(cfg_stage2.do_sample)

    raw_batches: List[List[str]] = []

    for start in range(0, len(frame_paths), max_bs):
        batch = frame_paths[start : start + max_bs]
        log.info("Stage 2 [%s]: frames %d-%d/%d", subscan_id, start, start + len(batch), len(frame_paths))
        text = wrapper.chat_with_images(
            batch,
            cfg_stage2.prompt,
            max_new_tokens=max_new,
            do_sample=do_sample,
        )
        log.info("Stage 2 [%s] raw VLM output (batch %d) [len=%d]: %s", subscan_id, start, len(text), text)
        parsed = parse_vlm_list_output(text)
        log.info("Stage 2 [%s] parsed (batch %d): %d items → %s", subscan_id, start, len(parsed), parsed)
        raw_batches.append([str(x) for x in parsed])

    if len(raw_batches) == 1:
        merged = raw_batches[0]
    else:
        # Consolidation step — feed the list-of-lists to the VLM via text only.
        log.info("Stage 2 [%s]: consolidating %d batches", subscan_id, len(raw_batches))
        merged_prompt = (
            cfg_stage2.consolidation_prompt
            + "\n\nInput lists:\n"
            + "\n".join(repr(b) for b in raw_batches)
        )
        text = wrapper.chat_with_images(
            images=[],  # text-only
            prompt=merged_prompt,
            max_new_tokens=max_new,
            do_sample=do_sample,
        )
        merged = parse_vlm_list_output(text)
        merged = [str(x) for x in merged]

    ignored = list(cfg_stage2.ignored_classes)
    final = _filter_background(merged, ignored)

    if bool(getattr(cfg_stage2, "drop_room_types", False)):
        before = len(final)
        final = _drop_room_types(final)
        if len(final) < before:
            log.info("Stage 2 [%s]: dropped %d room-type pseudo-objects",
                     subscan_id, before - len(final))

    log.info("Stage 2 [%s]: %d objects → %s", subscan_id, len(final), final)

    return ObjectListArtifact(subscan_id=subscan_id, objects=final, raw_batches=raw_batches)


_VISUAL_CONSOLIDATION_PROMPT = """\
You are looking at sample frames from a video of an indoor scene.
Below is a candidate object list extracted by a VLM in batches. It contains
duplicates, near-duplicates, hallucinations, and vague entries.

Candidate list ({n_candidates} items):
{candidate_list}

Your task:
1. Identify the room type from the images.
2. Use the images to understand what the candidates refer to — resolve
   ambiguous names by checking what is actually visible.
3. Merge duplicates and near-duplicates into a single canonical name
   (e.g. "robotic arm" + "robot arm" → "robotic arm",
    "storage bin" + "storage bins" + "plastic bin" → "storage bin").
4. Drop vague/generic entries that don't refer to a specific object
   ("red object", "black object", "office").
5. Keep at most {max_items} items. Prefer landmark-scale objects that anchor
   the scene for relocalization, but keep smaller distinctive items too.
6. Use common lowercase nouns.

Return ONLY a Python list of strings. No explanation, no extra text."""


def consolidate_with_images(
    wrapper: "VLMWrapper",
    candidates: List[str],
    frame_paths: Sequence,
    *,
    max_items: int = 60,
    n_sample_frames: int = 12,
    max_new_tokens: int = 2048,
) -> List[str]:
    """Visual consolidation: VLM sees images + candidate list, returns cleaned list."""
    import random
    paths = list(frame_paths)
    if len(paths) > n_sample_frames:
        step = len(paths) / n_sample_frames
        sample = [paths[int(i * step)] for i in range(n_sample_frames)]
    else:
        sample = paths

    prompt = _VISUAL_CONSOLIDATION_PROMPT.format(
        n_candidates=len(candidates),
        candidate_list=repr(candidates),
        max_items=max_items,
    )

    log.info("Visual consolidation: %d candidates, %d images", len(candidates), len(sample))
    text = wrapper.chat_with_images(
        sample,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    log.info("Visual consolidation raw output [len=%d]: %s", len(text), text)
    result = parse_vlm_list_output(text)
    result = [str(x).strip().lower() for x in result]
    log.info("Visual consolidation: %d → %d items", len(candidates), len(result))
    return result
