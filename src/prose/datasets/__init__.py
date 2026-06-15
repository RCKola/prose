"""Dataset loaders.

PROSE ships the Aria Digital Twin (ADT) loader. Every dataset exposes the same
minimal interface:
  - list_subscan_pairs() -> list[SubscanPair]
  - load_rgb_frames(subscan_id) -> RGBFrames
  - load_gt_pointcloud(subscan_id) -> (points, point_to_pixels, frame_ids)
"""
from .adt import ADTDataset
from .base import BaseDataset, RGBFrames, SubscanPair


def build_dataset(cfg_dataset) -> BaseDataset:
    name = cfg_dataset.name.lower()
    if name == "adt":
        return ADTDataset(cfg_dataset)
    raise ValueError(f"Unknown dataset: {name} (PROSE ships the 'adt' loader)")


__all__ = ["ADTDataset", "BaseDataset", "RGBFrames", "SubscanPair", "build_dataset"]
