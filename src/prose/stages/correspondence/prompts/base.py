"""Phase E: prompt construction."""
from __future__ import annotations

from typing import List, Protocol

from PIL import Image

from ..context import PairContext, PromptBundle


class PromptBuilder(Protocol):
    def build(self, ctx: PairContext, images: List[Image.Image]) -> PromptBundle: ...
