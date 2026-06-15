"""Small I/O helpers."""
from __future__ import annotations

import json
import pickle
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple, Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# I/O timing accumulator (diagnostic). Bucketed by stage dir name so we can
# answer "how much wall time went to disk per stage". Thread-safe so a future
# ThreadPoolExecutor refactor still produces correct totals.
# ---------------------------------------------------------------------------

_IO_LOCK = threading.Lock()
# bucket -> op -> (count, seconds, bytes)
_IO_STATS: Dict[str, Dict[str, list]] = defaultdict(
    lambda: defaultdict(lambda: [0, 0.0, 0])
)


def _bucket_for(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("stage"):
            return part
    return path.parent.name or "<root>"


def record_io(op: str, path: PathLike, seconds: float, nbytes: int = 0) -> None:
    bucket = _bucket_for(Path(path))
    with _IO_LOCK:
        slot = _IO_STATS[bucket][op]
        slot[0] += 1
        slot[1] += seconds
        slot[2] += int(nbytes)


def reset_io_stats() -> None:
    with _IO_LOCK:
        _IO_STATS.clear()


def io_stats_summary() -> str:
    with _IO_LOCK:
        if not _IO_STATS:
            return "I/O stats: (no operations recorded)"
        lines = ["=== I/O timing summary (bucket: op  count  seconds  MB) ==="]
        grand_s = 0.0
        for bucket in sorted(_IO_STATS):
            ops = _IO_STATS[bucket]
            bucket_s = sum(v[1] for v in ops.values())
            grand_s += bucket_s
            lines.append(f"  {bucket}  total={bucket_s:.2f}s")
            for op in sorted(ops):
                count, secs, nb = ops[op]
                lines.append(
                    f"    {op:<14} n={count:<4} {secs:7.2f}s  {nb / 1e6:8.1f} MB"
                )
        lines.append(f"  GRAND TOTAL: {grand_s:.2f}s")
        return "\n".join(lines)


def dump_pickle(obj: Any, path: PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    t0 = time.perf_counter()
    with path.open("wb") as f:
        pickle.dump(obj, f)
    record_io("dump_pickle", path, time.perf_counter() - t0, path.stat().st_size)


def load_pickle(path: PathLike) -> Any:
    path = Path(path)
    nbytes = path.stat().st_size
    t0 = time.perf_counter()
    with path.open("rb") as f:
        obj = pickle.load(f)
    record_io("load_pickle", path, time.perf_counter() - t0, nbytes)
    return obj


def dump_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    t0 = time.perf_counter()
    with path.open("w") as f:
        json.dump(obj, f, indent=indent)
    record_io("dump_json", path, time.perf_counter() - t0, path.stat().st_size)


def load_json(path: PathLike) -> Any:
    path = Path(path)
    nbytes = path.stat().st_size
    t0 = time.perf_counter()
    with path.open("r") as f:
        obj = json.load(f)
    record_io("load_json", path, time.perf_counter() - t0, nbytes)
    return obj
