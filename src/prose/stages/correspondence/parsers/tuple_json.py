"""Tuple-JSON parser for the thinking-mode blocking path (self-contained).

Strips ``<think>...</think>``, locates the first JSON array, walks
entries of shape ``[ref_id, src_id]`` (or legacy ``{"ref": i, "src": j}``),
and emits ``(src_iid, ref_iid)`` proposals after marker→iid translation
via ``bin_ctx``.

Implements ``ResponseParser`` with extended signature
``parse(raw, ctx, *, bin_ctx)``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from ..context import PairContext, VLMResult

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def _strip_fences(raw: str) -> str:
    m = _JSON_FENCE_RE.search(raw)
    return m.group(1).strip() if m else raw.strip()


def _first_json_array_or_object(raw: str):
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = s.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(s)):
            if s[i] == opener:
                depth += 1
            elif s[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return []


def strip_think_block(raw: str) -> Tuple[str, str]:
    """Return ``(think_text, answer_text)``."""
    if not raw:
        return "", ""
    m = _THINK_BLOCK_RE.search(raw)
    if m:
        think = m.group(1).strip()
        answer = (raw[: m.start()] + raw[m.end():]).strip()
        return think, answer
    open_idx = raw.lower().find("<think>")
    if open_idx >= 0:
        return raw[open_idx + len("<think>"):].strip(), ""
    return "", raw.strip()


def _parse_tuple_pairs(
    raw: str,
    *,
    ref_marker_ids: Set[int],
    src_marker_ids: Set[int],
) -> Tuple[List[Tuple[int, int]], int, int]:
    """Returns (accepted_pairs_marker_space, n_input, n_dropped)."""
    _, answer = strip_think_block(raw)
    parsed = _first_json_array_or_object(answer)
    if isinstance(parsed, dict):
        if "pairs" in parsed and isinstance(parsed["pairs"], list):
            parsed = parsed["pairs"]
        elif "ref" in parsed and "src" in parsed:
            parsed = [parsed]
        else:
            return [], 0, 0
    if not isinstance(parsed, list):
        return [], 0, 0

    out: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    n_input = len(parsed)
    n_dropped = 0
    for entry in parsed:
        ref_id: Optional[int] = None
        src_id: Optional[int] = None
        if isinstance(entry, dict):
            try:
                ref_id = int(entry["ref"])
                src_id = int(entry["src"])
            except (KeyError, TypeError, ValueError):
                n_dropped += 1
                continue
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            try:
                ref_id = int(entry[0])
                src_id = int(entry[1])
            except (TypeError, ValueError):
                n_dropped += 1
                continue
        else:
            n_dropped += 1
            continue
        if ref_id not in ref_marker_ids or src_id not in src_marker_ids:
            n_dropped += 1
            continue
        if (ref_id, src_id) in seen:
            n_dropped += 1
            continue
        seen.add((ref_id, src_id))
        out.append((ref_id, src_id))
    return out, n_input, n_dropped


@dataclass
class TupleJsonParser:
    """``ResponseParser`` for thinking-mode tuple schema."""

    def parse(self, raw: str, ctx: PairContext, *, bin_ctx=None) -> VLMResult:
        if bin_ctx is None:
            raise ValueError("TupleJsonParser requires bin_ctx (blocking runner)")
        ref_set = set(int(m) for m in bin_ctx.ref_marker_ids)
        src_set = set(int(m) for m in bin_ctx.src_marker_ids)
        marker_pairs, n_input, n_dropped = _parse_tuple_pairs(
            raw or "", ref_marker_ids=ref_set, src_marker_ids=src_set,
        )
        # Translate marker → iid (context.py spec: proposals = (src_iid, ref_iid)).
        proposals: List[Tuple[int, int]] = []
        for ref_m, src_m in marker_pairs:
            r_iid = bin_ctx.ref_marker_to_iid.get(int(ref_m))
            s_iid = bin_ctx.src_marker_to_iid.get(int(src_m))
            if r_iid is None or s_iid is None:
                continue
            proposals.append((int(s_iid), int(r_iid)))
        return VLMResult(
            raw_text=raw or "",
            proposals=proposals,
            confidences=[1.0] * len(proposals),
            audit={
                "bin_key": [int(bin_ctx.key[0]), int(bin_ctx.key[1])],
                "n_input_entries": int(n_input),
                "n_dropped_invalid": int(n_dropped),
                "n_accepted": int(len(proposals)),
            },
        )
