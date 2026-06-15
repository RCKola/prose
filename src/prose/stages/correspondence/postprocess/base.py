"""Phase G tail: post-processing."""
from __future__ import annotations

from typing import Protocol

from ..context import CorrespondenceResult, PairContext


class Postprocessor(Protocol):
    def apply(
        self,
        result: CorrespondenceResult,
        ctx: PairContext,
    ) -> CorrespondenceResult: ...
