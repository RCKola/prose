"""Per-instance shape descriptors for the Stage 3.5 prior.

Two cheap CPU-only pieces concatenated into one fixed-length vector:

  * pca_eigen_ratios — 3 dims: linearity, planarity, scattering.
    Scale-invariant by construction (ratios of covariance eigenvalues).
  * esf_histogram   — n_bins (default 64) dims: histogram of pairwise
    point-to-point distances on a normalized (unit-bbox-diag) cloud.
    A cut-down version of the full ESF (Wohlkinger & Vincze 2011),
    keeping only the distance-distribution component — empirically
    sufficient for a similarity prior between cross-scan instances.

Both are pure numpy. No torch, no Open3D.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def pca_eigen_ratios(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return (linearity, planarity, scattering) ∈ [0, 1]^3.

    For a degenerate cloud (≤2 points or zero variance) returns (0, 0, 1).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(pts.shape[0] - 1, 1)
    # eigh returns eigenvalues ascending; reverse to descending.
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.clip(eig, 0.0, None)
    l1 = float(eig[0])
    if l1 < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    l2 = float(eig[1])
    l3 = float(eig[2])
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1
    return np.array([linearity, planarity, scattering], dtype=np.float32)


def esf_histogram(
    points: np.ndarray,
    n_bins: int = 64,
    n_samples: int = 4096,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """L1-normalized histogram of pairwise distances on a (centered, scaled) cloud.

    Caller is expected to pass a bbox-diag normalized cloud — distances are
    binned over [0, sqrt(3)] (the diagonal of a unit AABB).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 2:
        return np.zeros(n_bins, dtype=np.float32)
    rng = rng or np.random.default_rng()
    n = pts.shape[0]
    n_pairs = min(n_samples, n * (n - 1) // 2 if n < 1024 else n_samples)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    same = i == j
    if same.any():
        # Deterministic bump to avoid d=0 from self-pairs.
        j[same] = (j[same] + 1) % n
    diffs = pts[i] - pts[j]
    dists = np.linalg.norm(diffs, axis=1)
    hist, _ = np.histogram(dists, bins=n_bins, range=(0.0, float(np.sqrt(3.0))))
    total = float(hist.sum())
    if total <= 0:
        return np.zeros(n_bins, dtype=np.float32)
    return (hist.astype(np.float32) / total)


def instance_descriptor(
    points_norm: np.ndarray,
    n_bins: int = 64,
    n_samples: int = 4096,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Concat PCA ratios + ESF histogram and L2-normalize the whole vector."""
    pca = pca_eigen_ratios(points_norm)
    esf = esf_histogram(points_norm, n_bins=n_bins, n_samples=n_samples, rng=rng)
    vec = np.concatenate([pca, esf]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec
    return vec / norm


__all__ = ["pca_eigen_ratios", "esf_histogram", "instance_descriptor"]
