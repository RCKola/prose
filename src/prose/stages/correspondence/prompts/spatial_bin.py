"""Spatial-only prompt for the separate-call BEV matching path.

The VLM sees only BEV / orthographic projections (no crops, no RGB frames)
and decides matches based purely on spatial layout — where numbered markers
sit relative to their neighbours.  One call per bin (or per sliding window
of adjacent bins), producing the same ``[{"ref": id, "src": id}]`` output
as the appearance prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image

from ..context import PairContext, PromptBundle


_SYSTEM = (
    "You match objects between two captures of the same room based ONLY on "
    "spatial position.\n\n"
    "You see spatial view images:\n"
    "  - REF views: overhead (and optionally front/side) projections showing "
    "object positions as numbered RED markers. Grey dots show nearby objects "
    "for context.\n"
    "  - SRC views: the same projections, with numbered BLUE markers.\n\n"
    "The two captures observe the same physical space from different "
    "viewpoints. Objects that occupy the same spatial position relative to "
    "their neighbours in both captures are the same physical object.\n\n"
    "Rules:\n"
    "- Match by RELATIVE position among neighbours, not absolute coordinates "
    "(the viewpoints differ).\n"
    "- Each REF marker matches at most one SRC marker and vice versa.\n"
    "- If a REF marker has no plausible spatial counterpart, skip it.\n"
    "- A wrong match is far worse than a missed one.\n\n"
    'Output ONLY a JSON array of matches: '
    '[{"ref": <id>, "src": <id>}, ...]. '
    "Empty array [] if no confident matches. No prose, no fences."
)


def _build_user(ref_marker_ids: List[int], src_marker_ids: List[int]) -> str:
    ref_list = ", ".join(str(m) for m in sorted(ref_marker_ids))
    src_list = ", ".join(str(m) for m in sorted(src_marker_ids))
    return (
        f"REF markers (red): [{ref_list}]\n"
        f"SRC markers (blue): [{src_list}]\n"
        "Which REF markers correspond to which SRC markers based on spatial "
        "position?"
    )


@dataclass
class SpatialBinPrompt:
    """Prompt builder for the spatial-only separate-call path."""

    def build(self, ctx: PairContext, images: List[Image.Image],
              *, bin_ctx=None) -> PromptBundle:
        if bin_ctx is None:
            raise ValueError(
                "SpatialBinPrompt requires bin_ctx (call from blocking runner)"
            )
        user = _build_user(
            ref_marker_ids=list(bin_ctx.ref_marker_ids),
            src_marker_ids=list(bin_ctx.src_marker_ids),
        )
        return PromptBundle(system=_SYSTEM, user=user)
