"""Frame loading helpers (self-contained copy)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    """Return BGR uint8 image with dataset rotation already applied."""
    from ....utils.visualize import load_frame_bgr
    return load_frame_bgr(Path(path))
