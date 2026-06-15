"""Phase G: correspondence resolution."""
from __future__ import annotations

from typing import Protocol

from ..context import CorrespondenceResult, PairContext, VLMResult


class CorrespondenceResolver(Protocol):
    def resolve(self, vlm: VLMResult, ctx: PairContext) -> CorrespondenceResult: ...
