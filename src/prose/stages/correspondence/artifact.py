"""CorrespondenceArtifact — Stage 5 output consumed by Stage 6 and the metrics.

The ``mosaic: Optional[dict]`` slot carries free-form audit payloads
(blocking audit + resolver audit) rather than a typed schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class CorrespondenceArtifact:
    pair_id: str
    raw_pairs: List[Tuple[int, int]]
    double_checked_pairs: List[Tuple[int, int]]
    vlm_double_checked_pairs: List[Tuple[int, int]] = field(default_factory=list)
    shape_anchor_pairs: List[Tuple[int, int]] = field(default_factory=list)
    fallback_used: bool = False
    correspondence_weights: Optional[List[float]] = None
    mosaic: Optional[dict] = None

    def save(self, out_dir: Path) -> Path:
        from ...utils.io import dump_pickle, ensure_dir
        out_dir = ensure_dir(out_dir)
        dump_pickle(
            {
                "raw_pairs": [list(p) for p in self.raw_pairs],
                "double_checked_pairs": [list(p) for p in self.double_checked_pairs],
                "vlm_double_checked_pairs": [list(p) for p in self.vlm_double_checked_pairs],
                "shape_anchor_pairs": [list(p) for p in self.shape_anchor_pairs],
                "fallback_used": bool(self.fallback_used),
                "correspondence_weights": (
                    list(self.correspondence_weights)
                    if self.correspondence_weights is not None else None
                ),
                "mosaic": self.mosaic,
            },
            out_dir / f"{self.pair_id}.pkl",
        )
        return out_dir / f"{self.pair_id}.pkl"
