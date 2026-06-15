"""Per-bin frame selection (self-contained copy).

Two-stage selector used by the blocking meta-composer:

* **Stage A (once per side):** rank frames per iid by visible pixel area.
* **Stage B (per bin):** greedy max-coverage over the bin's iids.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np


@dataclass
class PerBinFrameSelectionConfig:
    enabled: bool = False
    k_per_bin: int = 4
    top_n_per_iid: int = 8
    min_pixel_area: int = 200


@dataclass
class IidFrameCandidates:
    """Stage A artifact: iid -> [(frame_id, pixel_area), ...] desc by area."""
    by_iid: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    def has(self, iid: int) -> bool:
        return int(iid) in self.by_iid and bool(self.by_iid[int(iid)])


def build_iid_frame_ranking(
    per_frame_masks: Mapping[int, Mapping[int, np.ndarray]],
    *,
    top_n: int,
    min_pixel_area: int,
) -> IidFrameCandidates:
    by_iid: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for fid, inst in per_frame_masks.items():
        for iid, mask in inst.items():
            if mask is None:
                continue
            area = int(np.count_nonzero(mask))
            if area < int(min_pixel_area):
                continue
            by_iid[int(iid)].append((int(fid), area))
    out: Dict[int, List[Tuple[int, int]]] = {}
    n = max(1, int(top_n))
    for iid, lst in by_iid.items():
        lst.sort(key=lambda t: -t[1])
        out[int(iid)] = lst[:n]
    return IidFrameCandidates(by_iid=out)


def select_bin_frames(
    bin_iids: Sequence[int],
    candidates: IidFrameCandidates,
    *,
    k: int,
) -> Tuple[List[int], List[int]]:
    """Returns ``(selected_frame_ids, low_visibility_iids)``.

    An iid is flagged low-visibility iff Stage A produced no frame for
    it. Tie-break: when two frames cover the same number of new iids,
    prefer the one with larger total bin-iid pixel area.
    """
    iids = [int(i) for i in bin_iids]
    low_vis = [i for i in iids if not candidates.has(i)]
    coverable: Set[int] = {i for i in iids if candidates.has(i)}
    if not coverable:
        return [], low_vis

    frame_to_iid_area: Dict[int, Dict[int, int]] = defaultdict(dict)
    for iid in coverable:
        for fid, area in candidates.by_iid[iid]:
            prev = frame_to_iid_area[fid].get(iid, 0)
            if area > prev:
                frame_to_iid_area[fid][iid] = area

    covered: Set[int] = set()
    chosen: List[int] = []
    remaining: Dict[int, Dict[int, int]] = dict(frame_to_iid_area)
    K = max(1, int(k))

    while remaining and len(chosen) < K:
        best_fid = None
        best_score: Tuple[int, int] = (-1, -1)
        for fid, iid_area in remaining.items():
            new = sum(1 for i in iid_area if i not in covered)
            tot_area = sum(iid_area.values())
            score = (new, tot_area)
            if score > best_score:
                best_score = score
                best_fid = fid
        if best_fid is None or (best_score[0] == 0 and covered >= coverable):
            break
        chosen.append(int(best_fid))
        covered |= set(remaining[best_fid].keys())
        del remaining[best_fid]

    if len(chosen) < K and remaining:
        ranked = sorted(remaining.items(), key=lambda kv: -sum(kv[1].values()))
        for fid, _ in ranked:
            if len(chosen) >= K:
                break
            chosen.append(int(fid))

    return chosen, sorted(low_vis)
