"""Strict-JSON helper for local VLM wrappers.

Locally hosted Qwen3-VL / InternVL don't have OpenAI-style `response_format=
json_schema` token-level constraint. We approximate it via:
  1. Prompt augmentation — append a schema description with a "respond with
     ONLY JSON, no markdown" instruction.
  2. Post-hoc parsing — extract the first balanced JSON object substring and
     json.loads it.
  3. Single retry on parse failure with a stricter "JSON ONLY" wrapper.

Shared JSON-chat schemas used by the VLM wrappers, so call sites are
interchangeable across backends.
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

log = logging.getLogger(__name__)


PAIRS_SCHEMA_INSTRUCTION = """\

RESPOND WITH ONLY A JSON OBJECT MATCHING THIS EXACT SCHEMA (no markdown, no prose, no code fences):
{
  "descriptions_red": "<one short string>",
  "descriptions_blue": "<one short string>",
  "pairs": [{"red_id": <int>, "blue_id": <int>}, ...]
}
"""

PAIRS_WITH_RELATIONS_SCHEMA_INSTRUCTION = """\

RESPOND WITH ONLY A JSON OBJECT MATCHING THIS EXACT SCHEMA (no markdown, no prose, no code fences):
{
  "descriptions_red": "<one short string describing the scene-A objects you see>",
  "descriptions_blue": "<one short string describing the scene-B objects you see>",
  "relations_red": "<for each scene-A object with visible arrows, list its neighbours and the relation labels — e.g. '0 is same_height 5, above 8'. Skip objects with no visible arrows.>",
  "relations_blue": "<same for scene-B objects with visible arrows.>",
  "pairs": [{"red_id": <int>, "blue_id": <int>}, ...]
}
"""

# Explicit-CoT SoM schema. Per-id description maps + per-blue `matches` array
# (each with a brief rationale). Namespace is fixed: red_id ∈ [0,99], blue_id
# ≥ 100. `pairs` is DERIVED from non-null `matches` by the parser — the model
# does not emit it, which cuts ~20-30% of output tokens on large scenes.
PAIRS_EXPLICIT_COT_SCHEMA_INSTRUCTION = """\

RESPOND WITH ONLY A JSON OBJECT MATCHING THIS EXACT SCHEMA (no markdown, no prose, no code fences):
{
  "descriptions_red": {"<red_id ∈ [0,99]>": "<≤8 words>", ...},
  "descriptions_blue": {"<blue_id ≥ 100>": "<≤8 words>", ...},
  "matches": [
    {"blue_id": <int ≥100>, "matched_red_id": <int ∈ [0,99] | null>, "rationale": "<≤10 words>"},
    ...
  ]
}

Constraints (responses violating these are rejected):
  * red_id ∈ [0,99]; blue_id ≥ 100. Never put a blue id in `descriptions_red`.
  * `matches` has exactly one entry per key in `descriptions_blue`. `matched_red_id` is null when no red matches.
  * `rationale` ≤ 10 words. For null matches: ≤ 5 words (e.g. "no red counterpart").
"""

YES_NO_SCHEMA_INSTRUCTION = """\

RESPOND WITH ONLY A JSON OBJECT MATCHING THIS EXACT SCHEMA (no markdown, no prose, no code fences):
{
  "red_object": "<one short string>",
  "blue_object": "<one short string>",
  "is_same_object": <true | false>
}
"""

STRICT_RETRY_PREFIX = (
    "Your last response was not valid JSON. Output ONLY the JSON object — no "
    "explanation, no markdown fences, no extra text before or after.\n\n"
)


def _extract_json_object(text: str) -> str:
    """Return the first balanced `{...}` substring, stripping code fences."""
    # Strip ```json ... ``` and ``` ... ``` fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Find the first `{`, then walk forward tracking brace depth (ignoring
    # braces inside string literals).
    start = text.find("{")
    if start < 0:
        raise ValueError("no '{' in VLM output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced braces in VLM output")


def parse_pairs_json(
    text: str,
    *,
    red_id_max: int = 99,
    blue_id_min: int = 100,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Parse a pair-list response. Returns (pair_list, parsed_obj).

    Two emission modes:
      * Legacy: top-level `pairs: [{red_id, blue_id}, ...]`. Used by the
        slim SoM/JSON schemas.
      * Explicit-CoT: top-level `matches: [{blue_id, matched_red_id, ...}]`
        with no `pairs` key. Pairs are derived from entries whose
        `matched_red_id` is not null.

    Namespace tightening: pairs where `red_id > red_id_max`,
    `blue_id < blue_id_min`, or `red_id == blue_id` are dropped (and counted
    in `parsed["_dropped_invalid_namespace"]`). The parser does not raise on
    these — it lets `_extract_json_object` raise for structural failures
    (the path that already triggers the strict-retry) and falls through to
    Stage 6 fallback when every pair is dropped.
    """
    blob = _extract_json_object(text)
    parsed = json.loads(blob)

    raw_tuples: List[Tuple[int, int]] = []
    # Some models emit a hybrid: top-level key `pairs` but inner fields from
    # the explicit-CoT schema (`matched_red_id` instead of `red_id`). Accept
    # both inner shapes under either outer key.
    def _harvest(items):
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                rid = it.get("red_id")
                if rid is None:
                    rid = it.get("matched_red_id")
                if rid is None:
                    continue
                bid = it.get("blue_id")
                if bid is None:
                    continue
                raw_tuples.append((int(rid), int(bid)))
            except (TypeError, ValueError):
                continue

    if isinstance(parsed.get("pairs"), list) and parsed["pairs"]:
        _harvest(parsed["pairs"])
    elif isinstance(parsed.get("matches"), list):
        _harvest(parsed["matches"])

    out: List[Tuple[int, int]] = []
    dropped = 0
    for red_id, blue_id in raw_tuples:
        if red_id > red_id_max or blue_id < blue_id_min or red_id == blue_id:
            dropped += 1
            continue
        out.append((red_id, blue_id))
    if dropped:
        parsed["_dropped_invalid_namespace"] = dropped
        log.warning(
            "parse_pairs_json: dropped %d pair(s) violating namespace "
            "(red_id ∈ [0,%d], blue_id ≥ %d, red_id != blue_id). "
            "Likely prompt/schema mismatch — check effective_style routing.",
            dropped, red_id_max, blue_id_min,
        )
    return out, parsed


def parse_yes_no_json(text: str) -> Tuple[bool, dict]:
    """Parse a yes/no response. Returns (is_same_object, parsed_obj)."""
    blob = _extract_json_object(text)
    parsed = json.loads(blob)
    return bool(parsed.get("is_same_object", False)), parsed
