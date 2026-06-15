"""Common protocol + dataclasses for datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SubscanPair:
    pair_id: str           # "<src>__<ref>"
    src_id: str
    ref_id: str
    gt_transform: np.ndarray   # 4x4 src→ref
    overlap: float
    anchor_object_ids: Optional[List[int]] = None  # 3RScan GT objectIds shared by src/ref; None if dataset has no anchor schema


@dataclass
class RGBFrames:
    paths: List[Path]
    frame_ids: List[int]


class BaseDataset:
    """Minimal interface every dataset must implement."""

    name: str

    def list_subscan_pairs(self) -> List[SubscanPair]:
        raise NotImplementedError

    def load_rgb_frames(self, subscan_id: str) -> RGBFrames:
        raise NotImplementedError

    def load_gt_pointcloud(
        self, subscan_id: str
    ) -> Tuple[np.ndarray, List[dict], List[int]]:
        """Returns (points, point_to_pixels, frame_ids)."""
        raise NotImplementedError

    def load_gt_object_ids(self, subscan_id: str) -> Optional[np.ndarray]:
        """Per-point GT instance objectId aligned with `load_gt_pointcloud` points.

        Default: None — the dataset has no instance ground truth (e.g. ADT).
        Scan3R overrides to read objectId from data.npy.
        """
        return None
