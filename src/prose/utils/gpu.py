"""GPU / dtype helpers."""
from __future__ import annotations

import gc
from typing import Any

import torch


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def select_torch_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype: {name!r}")
    return _DTYPE_MAP[key]


def release_gpu(*objs: Any) -> None:
    """Delete objects, run GC, and empty the CUDA cache.

    Keep arg list of objects to explicitly delete (caller's references); after this
    returns, those references in the caller's scope still need to be cleared via
    `del`, but any lingering CUDA tensors are freed.
    """
    for _ in objs:
        del _
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_vlm_list_output(text: str) -> list:
    """Extract the last python-literal list from raw VLM output.

    Handles wrappers like ```python\n[...]\n``` and <think>...</think> traces.
    Falls back to extracting ``Red N → Blue M`` / ``(N, M)`` / ``red N - blue M``
    patterns from prose, which Qwen3-VL-Thinking sometimes emits instead of a
    proper python list when it self-doubts.
    """
    import ast
    import re

    # Strip thinking traces (Qwen3-VL-Thinking emits <think>...</think>).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Prefer the last fenced code block if present.
    fenced = re.findall(r"```(?:python|json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates = list(reversed(fenced)) if fenced else []
    candidates.append(text)

    list_pat = re.compile(r"\[.*?\]", re.DOTALL)
    for cand in candidates:
        matches = list_pat.findall(cand)
        for m in reversed(matches):
            try:
                val = ast.literal_eval(m)
            except (ValueError, SyntaxError):
                continue
            if isinstance(val, list):
                return val

    # --- Truncated-list fallback ---------------------------------------------
    # Model hit max_new_tokens before closing ]. Find the last '[' and extract
    # all quoted strings from the partial text.
    bracket_pos = text.rfind("[")
    if bracket_pos != -1:
        partial = text[bracket_pos:]
        items = re.findall(r"'([^']*)'|\"([^\"]*)\"", partial)
        recovered = [a or b for a, b in items if (a or b).strip()]
        if recovered:
            return recovered

    # --- Markdown bullet fallback --------------------------------------------
    # InternVL3-14B (and other models) sometimes emit `- foo\n- bar\n...`
    # instead of a Python list, even when the prompt requests one.
    # Match `-`, `*`, or `•` bullets followed by whitespace and a label.
    bullet_pat = re.compile(r"(?:^|\n)\s*[-*•]\s+([^\n]+)")
    bullet_items = [m.strip().rstrip(",.;") for m in bullet_pat.findall(text)]
    bullet_items = [x for x in bullet_items if x]
    if len(bullet_items) >= 2:  # require at least 2 to avoid prose false positives
        return bullet_items

    # --- Bare-tuple sequence fallback ----------------------------------------
    # Some models (gpt-4.1-mini with the structured-reasoning GoM prompt)
    # emit the final answer as bare tuples without enclosing brackets:
    #   `Final matched pairs:\n(0, 2), (1, 100), (5, 101)`
    # Recover the (int, int) tuples directly. We anchor to the last "tuple
    # sequence" in the text (≥ 2 tuples) to avoid false positives from prose
    # like "scene A 0 (chair) matches scene B 5 (chair)".
    tuple_pair = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
    seq_pat = re.compile(
        r"(?:\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*(?:,\s*)?){2,}"
    )
    seq_matches = list(seq_pat.finditer(text))
    if seq_matches:
        last_seq = seq_matches[-1].group(0)
        tuples = [
            (int(a), int(b)) for a, b in tuple_pair.findall(last_seq)
        ]
        if tuples:
            return tuples

    # --- Prose fallback ------------------------------------------------------
    # Look for lines like:
    #   Red 9 → blue 8
    #   red 12: ... blue 3
    #   (red 12, blue 3)
    # Collect every (red_id, blue_id) tuple; deduplicate preserving order.
    prose_pat = re.compile(
        r"red\s*(?P<red>\d+)\s*(?:→|->|:|,|-|=|\s+to\s+|\s+↔\s+|\s+matches\s+)"
        r"[^\n\d]{0,30}?blue\s*(?P<blue>\d+)",
        re.IGNORECASE,
    )
    seen = set()
    pairs: list = []
    for m in prose_pat.finditer(text):
        p = (int(m.group("red")), int(m.group("blue")))
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    if pairs:
        return pairs
    return []
