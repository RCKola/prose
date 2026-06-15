"""Phase F.parse: VLM response → typed proposals."""
from __future__ import annotations

from typing import Protocol

from ..context import PairContext, VLMResult


class ResponseParser(Protocol):
    def parse(self, raw_text: str, ctx: PairContext) -> VLMResult: ...
