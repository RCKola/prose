"""No-op filter; matches the ADT-baseline default (NS9 disabled)."""
from __future__ import annotations

from ..context import PairContext


class IdentityFilter:
    def __init__(self, **_: object) -> None:
        pass

    def apply(self, ctx: PairContext) -> PairContext:
        return ctx
