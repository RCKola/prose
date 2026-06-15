"""Model-call timing accumulator (diagnostic, parallel to utils.io's I/O stats).

Records per-(model, op) wall time, call count, and a free-form input counter
(e.g. n_images, n_frames). For GPU-resident wrappers, callers should call
`torch.cuda.synchronize()` before stopping the timer so the recorded seconds
reflect actual kernel completion rather than launch latency.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict

_LOCK = threading.Lock()
# model -> op -> {count, seconds, n_inputs}
_STATS: Dict[str, Dict[str, list]] = defaultdict(
    lambda: defaultdict(lambda: [0, 0.0, 0])
)


def record_model_call(model: str, op: str, seconds: float, n_inputs: int = 0) -> None:
    with _LOCK:
        slot = _STATS[model][op]
        slot[0] += 1
        slot[1] += seconds
        slot[2] += int(n_inputs)


def reset_model_stats() -> None:
    with _LOCK:
        _STATS.clear()


def model_stats_summary() -> str:
    with _LOCK:
        if not _STATS:
            return "Model-call stats: (no calls recorded)"
        lines = ["=== Model-call timing summary (model: op  count  seconds  inputs  s/call) ==="]
        grand_s = 0.0
        for model in sorted(_STATS):
            ops = _STATS[model]
            model_s = sum(v[1] for v in ops.values())
            grand_s += model_s
            lines.append(f"  {model}  total={model_s:.2f}s")
            for op in sorted(ops):
                count, secs, n_in = ops[op]
                per = secs / count if count else 0.0
                lines.append(
                    f"    {op:<36} n={count:<5} {secs:8.2f}s  inputs={n_in:<6} {per:6.2f}s/call"
                )
        lines.append(f"  GRAND TOTAL: {grand_s:.2f}s")
        return "\n".join(lines)
