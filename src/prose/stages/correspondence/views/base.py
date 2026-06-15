"""Phase C: per-side frame selection."""
from __future__ import annotations

from typing import Protocol

from ..context import PairContext


class ViewSelector(Protocol):
    """Phase C contract.

    Populates ``ctx.src_frames`` / ``ctx.ref_frames`` with the frames the
    downstream visuals should render. For the ADT-baseline path, per-bin
    selection runs *inside* the blocking meta-composer, so this phase
    may be a no-op selector that defers to phase D.
    """

    def select(self, ctx: PairContext) -> PairContext: ...
