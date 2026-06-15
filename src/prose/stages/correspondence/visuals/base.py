"""Phase D: visual composition."""
from __future__ import annotations

from typing import List, Protocol

from PIL import Image

from ..context import PairContext


class VisualComposer(Protocol):
    """Phase D contract.

    Returns a list of images (PIL) to attach to the VLM call. The
    blocking meta-composer is a special case: it owns phases C-F per
    bin and returns its aggregated result via the pipeline's escape
    hatch rather than this protocol.
    """

    def compose(self, ctx: PairContext) -> List[Image.Image]: ...
