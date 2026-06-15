"""Per-bin crops+context prompt for the blocking path.

Pairs with ``visuals.bin_visuals.CropsContextBinVisuals``: the VLM sees a
2×K crop grid plus one context frame per scene, all drawn from a single
height band. The matching set is small, so the prompt asks for a direct,
precision-biased decision (and optionally private reasoning).

Implements ``PromptBuilder`` with the extended blocking signature
``build(ctx, images, *, bin_ctx)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from PIL import Image

from ..context import PairContext, PromptBundle


def _image_guide(use_bev: bool = False, use_context_frame: bool = True,
                  use_crops: bool = True) -> str:
    items: list[str] = []
    if use_crops:
        items.append(
            "REF crop grid — a grid of zoomed crops, one per REF object. "
            "Each crop is centered on ONE object; its silhouette is outlined in "
            "green and its numeric ID is stamped white at the top-left corner.")
        items.append("SRC crop grid — the same, one zoomed crop per SRC object.")
    if use_bev:
        items.append(
            "REF bird's-eye view — an overhead spatial layout of the "
            "point cloud showing REF object positions as numbered red markers. "
            "Grey dots show nearby objects in adjacent bins for context.")
        items.append(
            "SRC bird's-eye view — the same, with blue markers for SRC objects.")
    if use_context_frame:
        items.append(
            "Context frame — top = REF, bottom = SRC — one wide RGB frame "
            "per capture showing those objects in their spatial layout, each "
            "outlined and labelled with the same IDs.")
    n = len(items)
    if n == 1:
        return f"You see one image:\n  (1) {items[0]}\n\n"
    lines = [f"You see up to {['zero','one','two','three','four','five','six'][n]} images:\n"]
    for i, item in enumerate(items, 1):
        lines.append(f"  ({i}) {item}\n")
    return "".join(lines) + "\n"


def _system_core(use_bev: bool = False, use_context_frame: bool = True,
                  use_crops: bool = True) -> str:
    bev_hint = (
        " Use the bird's-eye views to check whether candidate matches "
        "occupy compatible spatial positions among their neighbours."
        if use_bev else ""
    )
    judge_hint = (
        "shape, size, and its neighbours in the context frame), never the "
        if use_context_frame else
        "shape, and size), never the "
    )
    return (
        "You match objects across two captures of the same room. Every "
        "object shown here sits within the SAME narrow height band of the "
        "room, so they are mutually plausible matches.\n\n"
        + _image_guide(use_bev, use_context_frame, use_crops) +
        "REF and SRC IDs are disjoint integers. The outline and label are "
        "localization aids — judge the OBJECT itself (color, material, "
        + judge_hint +
        "marker.\n\n"
        "Pair a REF ID with a SRC ID only when you are confident they tag "
        "the SAME physical object. Most objects appear in only one capture "
        "— leave those unmatched. A wrong match is worse than a missed one."
        + bev_hint
    )


def _system_direct(use_bev: bool = False, use_context_frame: bool = True,
                    use_crops: bool = True) -> str:
    return (
        _system_core(use_bev, use_context_frame, use_crops) + "\n\n"
        'Output ONLY a JSON array of objects: '
        '[{"ref": <id>, "src": <id>}, ...]; emit [] if nothing matches. '
        "No prose, no fences."
    )


def _system_thinking(use_bev: bool = False, use_context_frame: bool = True,
                      use_crops: bool = True) -> str:
    if use_crops:
        reason_hint = (
            "Reason privately inside <think>...</think>: for each REF crop name "
            "its color/material/shape, then scan the SRC crops for the same "
        )
    else:
        reason_hint = (
            "Reason privately inside <think>...</think>: for each REF object name "
            "its color/material/shape, then scan the SRC objects for the same "
        )
    confirm_hint = (
        "object, confirming with the context frame's layout. After "
        if use_context_frame else
        "object. After "
    )
    return (
        _system_core(use_bev, use_context_frame, use_crops) + "\n\n"
        + reason_hint + confirm_hint +
        "</think>, emit one JSON array and nothing else: "
        '[{"ref": <id>, "src": <id>}, ...]. Emit [] if nothing matches. '
        "No fences."
    )

_BODY = """\
Match REF and SRC IDs that tag the SAME physical object.

REF IDs: [{ref_list}]
SRC IDs: [{src_list}]

- Use only the IDs listed above.
- An ID may appear in at most one pair.
- Emit [] if nothing matches.
- Output ONLY a JSON array of objects, one per match:
  [{{"ref": <id>, "src": <id>}}, ...] — no prose, fences, or preamble.
"""

_BODY_ASYMMETRIC = """\
One side has a single object, the other has {n_other} candidates. The
lone {lone_side} object may have no counterpart at all — when in doubt
emit [].

Lone {lone_side} ID: {lone_id}
Candidate {other_side} IDs: [{candidates}]

Compare the lone object's color / material / shape / neighbours against
each candidate. Emit ONLY a JSON array: exactly one object
`[{{"ref": <id>, "src": <id>}}]` if a single candidate is the SAME
object, or `[]` if none. Do not list multiple candidates.
"""


# ---------------------------------------------------------------------------
# Pairwise prompt — 1 REF vs N SRC candidates
# ---------------------------------------------------------------------------

def _pairwise_image_guide(use_bev: bool = False, use_context_frame: bool = True,
                           use_crops: bool = True) -> str:
    items: list[str] = []
    if use_crops:
        items.append(
            "REF crop — a single zoomed crop of ONE reference object, "
            "outlined in green with its numeric ID stamped white at the top-left.")
        items.append(
            "SRC crop grid — a grid of zoomed crops, one per SRC candidate, "
            "each outlined in green with its ID.")
    if use_bev:
        items.append(
            "REF bird's-eye view — overhead spatial layout showing the "
            "REF object as a red marker among its neighbours.")
        items.append(
            "SRC bird's-eye view — the same, with blue markers for SRC candidates.")
    if use_context_frame:
        items.append(
            "Context frame (optional) — top = REF, bottom = SRC — one "
            "wide RGB frame per capture showing the objects in their spatial layout.")
    n = len(items)
    if n == 1:
        return f"You see one image:\n  (1) {items[0]}\n\n"
    lines = [f"You see up to {['zero','one','two','three','four','five','six'][n]} images:\n"]
    for i, item in enumerate(items, 1):
        lines.append(f"  ({i}) {item}\n")
    return "".join(lines) + "\n"


def _system_pairwise(use_bev: bool = False, use_context_frame: bool = True,
                      use_crops: bool = True) -> str:
    bev_hint = (
        " Use the bird's-eye views to check whether the REF object's "
        "spatial position is compatible with each SRC candidate."
        if use_bev else ""
    )
    judge_hint = (
        "shape, size, and its neighbours in the context frame), never the "
        if use_context_frame else
        "shape, and size), never the "
    )
    return (
        "You decide whether a single reference (REF) object matches any of "
        "the candidate source (SRC) objects. All shown objects sit within the "
        "same narrow height band of the room.\n\n"
        + _pairwise_image_guide(use_bev, use_context_frame, use_crops) +
        "REF and SRC IDs are disjoint integers. The outline and label are "
        "localization aids — judge the OBJECT itself (color, material, "
        + judge_hint +
        "marker.\n\n"
        "The REF object may have NO match among the SRC candidates — when in "
        "doubt, emit []. A wrong match is far worse than a missed one."
        + bev_hint + "\n\n"
        'Output ONLY a JSON array: '
        '[{"ref": <id>, "src": <id>}] if exactly one SRC matches, or [] if '
        "none. No prose, no fences."
    )

_BODY_PAIRWISE = """\
Does the REF object match any SRC candidate?

REF ID: {ref_id}
Candidate SRC IDs: [{src_list}]

- Compare the REF object's color, material, shape, and size against each SRC crop.
- Pick at most ONE SRC that is the SAME physical object. Most candidates will NOT match.
- Output ONLY: [{{"ref": {ref_id}, "src": <id>}}] or [].
"""


def build_user_prompt_bin_crops(
    *,
    ref_marker_ids: Sequence[int],
    src_marker_ids: Sequence[int],
    asymmetric_bin: bool = False,
) -> str:
    ref_list = ", ".join(str(m) for m in sorted(ref_marker_ids))
    src_list = ", ".join(str(m) for m in sorted(src_marker_ids))
    if asymmetric_bin:
        if len(ref_marker_ids) == 1:
            lone_id = int(next(iter(ref_marker_ids)))
            lone_side, other_side = "REF", "SRC"
            candidates, n_other = src_list, len(src_marker_ids)
        else:
            lone_id = int(next(iter(src_marker_ids)))
            lone_side, other_side = "SRC", "REF"
            candidates, n_other = ref_list, len(ref_marker_ids)
        return _BODY_ASYMMETRIC.format(
            lone_side=lone_side, lone_id=lone_id, other_side=other_side,
            candidates=candidates, n_other=n_other,
        )
    return _BODY.format(ref_list=ref_list, src_list=src_list)


@dataclass
class BinCropsPrompt:
    """``PromptBuilder`` for the per-bin crops+context blocking path."""

    enable_thinking: bool = False
    use_bev: bool = False
    use_context_frame: bool = True
    use_crops: bool = True

    def build(self, ctx: PairContext, images: List[Image.Image],
              *, bin_ctx=None) -> PromptBundle:
        if bin_ctx is None:
            raise ValueError(
                "BinCropsPrompt requires bin_ctx (call from blocking runner)"
            )
        if self.enable_thinking:
            system = _system_thinking(use_bev=self.use_bev,
                                      use_context_frame=self.use_context_frame,
                                      use_crops=self.use_crops)
        else:
            system = _system_direct(use_bev=self.use_bev,
                                    use_context_frame=self.use_context_frame,
                                    use_crops=self.use_crops)
        user = build_user_prompt_bin_crops(
            ref_marker_ids=list(bin_ctx.ref_marker_ids),
            src_marker_ids=list(bin_ctx.src_marker_ids),
            asymmetric_bin=bool(bin_ctx.asymmetric),
        )
        return PromptBundle(system=system, user=user)


@dataclass
class PairwiseBinCropsPrompt:
    """Prompt for pairwise mode: 1 REF vs N SRC candidates per VLM call."""

    enable_thinking: bool = False
    use_bev: bool = False
    use_context_frame: bool = True
    use_crops: bool = True

    def build(self, ctx: PairContext, images: List[Image.Image],
              *, bin_ctx=None) -> PromptBundle:
        if bin_ctx is None:
            raise ValueError(
                "PairwiseBinCropsPrompt requires bin_ctx (call from blocking runner)"
            )
        if len(bin_ctx.ref_marker_ids) != 1:
            raise ValueError(
                f"Pairwise prompt expects exactly 1 REF marker, "
                f"got {len(bin_ctx.ref_marker_ids)}"
            )
        ref_id = int(bin_ctx.ref_marker_ids[0])
        src_list = ", ".join(str(m) for m in sorted(bin_ctx.src_marker_ids))
        user = _BODY_PAIRWISE.format(ref_id=ref_id, src_list=src_list)
        return PromptBundle(system=_system_pairwise(
            use_bev=self.use_bev, use_context_frame=self.use_context_frame,
            use_crops=self.use_crops),
            user=user)
