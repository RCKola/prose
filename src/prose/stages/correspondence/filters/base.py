"""Phase B: instance filtering / coalescing."""
from __future__ import annotations

from typing import Protocol

from ..context import InstanceSet, PairContext


class InstanceFilter(Protocol):
    """Phase B contract.

    Mutates (or replaces) the ``src`` / ``ref`` ``InstanceSet`` on the
    context. Returning the modified context is fine; returning ``None``
    means in-place is expected.
    """

    def apply(self, ctx: PairContext) -> PairContext: ...
