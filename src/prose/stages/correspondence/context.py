"""Per-pair data containers passed between Stage 4 phases.

Phases operate on these typed structures, not on raw cache dicts. The
pipeline driver is the only place that reads Stage 1/3/3.5 from disk;
everything downstream sees a ``PairContext``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InstanceSet:
    """One side (ref or src) of a pair: instances with geometry + labels."""
    side: str                                      # "ref" or "src"
    iids: List[int]                                # instance ids on this side
    centroids: Dict[int, np.ndarray]               # iid -> (3,) world-frame centroid
    obbs: Dict[int, np.ndarray]                    # iid -> (3,) OBB extents
    rotations: Dict[int, np.ndarray]               # iid -> (3,3) OBB rotation
    points: Dict[int, np.ndarray]                  # iid -> (Ni, 3) instance points
    label_text: Dict[int, str]                     # iid -> SAM3 class label
    per_frame_masks: Dict[int, Dict[int, np.ndarray]]  # frame_idx -> iid -> mask
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameSet:
    """Frames selected for one side, plus metadata to render them."""
    side: str
    frame_indices: List[int]
    color_paths: Dict[int, Path]
    intrinsics: Optional[Dict[int, np.ndarray]] = None
    extrinsics: Optional[Dict[int, np.ndarray]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBundle:
    """Output of phase E."""
    system: str
    user: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMResult:
    """Output of phase F (raw text + parsed proposals)."""
    raw_text: str
    proposals: List[Tuple[int, int]]               # (src_iid, ref_iid)
    confidences: Optional[List[float]] = None
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrespondenceResult:
    """Output of phase G (resolver)."""
    pairs: List[Tuple[int, int]]                   # final (src_iid, ref_iid)
    weights: Optional[List[float]] = None
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairContext:
    """Everything ``Stage4Pipeline.run_pair`` needs."""
    pair_id: str
    src: InstanceSet
    ref: InstanceSet
    src_frames: FrameSet
    ref_frames: FrameSet
    up_axis: str = "z"                             # "y" for ADT, "z" for 3RScan
    image_rotation_k: int = 0
    depth_dir: Optional[Path] = None
    audit: Dict[str, Any] = field(default_factory=dict)
