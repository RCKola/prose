"""Geometric correction resolver (self-contained copy + bypass flag).

For each REF iid:

  1. Aggregate raw bin commits into soft weights W(i, j) over SRC iids.
  2. q̄_i = Σ_j W(i,j) · centroid_src(j); σ²_i = weighted spatial variance.
  3. k-NN over committed REF iids; per iid:
       Error_i      = mean_{k∈N(i)} | ‖p_i − p_k‖ − ‖q̄_i − q̄_k‖ |
       Volatility_i = σ²_i + mean_{k∈N(i)} σ²_k
       Score_i      = Error_i + λ · √Volatility_i
  4. Keep raw (i, j) commits whose REF iid has Score_i ≤ τ.

The ``bypass`` flag short-circuits steps 1–3 and returns every unique
proposal with weight 1.0 — the canonical ADT-baseline path (legacy
spelling: ``lam=0 tau=1e9``). Configured via Hydra preset.

Operates directly on iid-space proposals; no marker translation needed
(the parser already mapped marker → iid).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ..context import CorrespondenceResult, PairContext, VLMResult


def _aggregate_soft_weights(
    proposals: Sequence[Tuple[int, int]],   # (src_iid, ref_iid)
) -> Dict[int, Dict[int, float]]:
    counts: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for src_iid, ref_iid in proposals:
        counts[int(ref_iid)][int(src_iid)] += 1.0
    weights: Dict[int, Dict[int, float]] = {}
    for ri, dist in counts.items():
        total = float(sum(dist.values()))
        if total <= 0:
            continue
        weights[ri] = {sj: float(c) / total for sj, c in dist.items()}
    return weights


def _expected_pos_and_var(
    W: Mapping[int, Mapping[int, float]],
    src_centroids: Mapping[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    qbar: Dict[int, np.ndarray] = {}
    sigma2: Dict[int, float] = {}
    for ri, dist in W.items():
        pts, ws = [], []
        for sj, w in dist.items():
            c = src_centroids.get(int(sj))
            if c is None:
                continue
            pts.append(np.asarray(c, dtype=np.float64).reshape(-1))
            ws.append(float(w))
        if not pts:
            continue
        pts_arr = np.stack(pts, axis=0)
        ws_arr = np.asarray(ws, dtype=np.float64)
        s = ws_arr.sum()
        if s <= 0:
            continue
        ws_arr = ws_arr / s
        q = (ws_arr[:, None] * pts_arr).sum(axis=0)
        var = float((ws_arr * np.sum((pts_arr - q[None, :]) ** 2, axis=1)).sum())
        qbar[int(ri)] = q
        sigma2[int(ri)] = var
    return qbar, sigma2


def _knn_neighbours(
    iids: Sequence[int],
    ref_centroids: Mapping[int, np.ndarray],
    k: int,
) -> Dict[int, List[int]]:
    valid = [int(i) for i in iids if int(i) in ref_centroids]
    if len(valid) <= 1:
        return {i: [] for i in valid}
    pts = np.stack(
        [np.asarray(ref_centroids[i], dtype=np.float64).reshape(-1) for i in valid],
        axis=0,
    )
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    k_eff = max(1, min(int(k), len(valid) - 1))
    nbrs_idx = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]
    return {valid[row]: [valid[j] for j in nbrs_idx[row]] for row in range(len(valid))}


def _compute_scores(
    W: Mapping[int, Mapping[int, float]],
    ref_centroids: Mapping[int, np.ndarray],
    src_centroids: Mapping[int, np.ndarray],
    *, lam: float, k: int,
) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
    qbar, sigma2 = _expected_pos_and_var(W, src_centroids)
    committed = [ri for ri in W.keys() if ri in qbar and ri in ref_centroids]
    nbrs = _knn_neighbours(committed, ref_centroids, k)
    scores: Dict[int, float] = {}
    for ri in committed:
        N = [j for j in nbrs.get(ri, []) if j in qbar]
        if not N:
            err = 0.0
        else:
            ref_ri = np.asarray(ref_centroids[ri], dtype=np.float64).reshape(-1)
            terms = []
            for kj in N:
                ref_kj = np.asarray(ref_centroids[kj], dtype=np.float64).reshape(-1)
                d_ref = float(np.linalg.norm(ref_ri - ref_kj))
                d_src = float(np.linalg.norm(qbar[ri] - qbar[kj]))
                terms.append(abs(d_ref - d_src))
            err = float(np.mean(terms))
        vol = sigma2[ri] + (
            float(np.mean([sigma2[kj] for kj in N])) if N else 0.0
        )
        scores[ri] = err + float(lam) * float(np.sqrt(max(vol, 0.0)))
    return scores, qbar


@dataclass
class GeoCorrectionResolver:
    """``CorrespondenceResolver``: distance-preservation + volatility filter.

    Scores each committed REF iid by ``Error + λ·√Volatility`` (lower =
    more geometrically consistent), then keeps proposals by one of:

    * ``bypass=True`` — keep every unique proposal (ADT-baseline path).
    * ``top_k > 0``  — keep the ``top_k`` lowest-score REF iids. Per-pair
      coverage is guaranteed (never zeroes a pair) while still dropping
      the least geometrically-consistent matches — the right operating
      point for registration, which needs a few clean anchors, not recall.
    * else (``top_k=0``) — threshold: keep REF iids with score ≤ ``tau``
      (can zero out a pair if everything scores poorly).
    """
    lam: float = 1.0
    tau: float = 0.45
    k: int = 8
    bypass: bool = False
    top_k: int = 0

    def resolve(self, vlm: VLMResult, ctx: PairContext) -> CorrespondenceResult:
        # Dedup proposals (preserve first-seen order).
        seen: set = set()
        unique: List[Tuple[int, int]] = []
        for s, r in vlm.proposals:
            key = (int(s), int(r))
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)

        passthrough = {}
        if vlm.audit.get("ref_features"):
            passthrough["ref_features"] = vlm.audit["ref_features"]
        if vlm.audit.get("src_features"):
            passthrough["src_features"] = vlm.audit["src_features"]

        if self.bypass or not unique:
            return CorrespondenceResult(
                pairs=unique,
                weights=[1.0] * len(unique),
                audit={**passthrough, "resolver": "geo_correction",
                       "bypass": True,
                       "n_in": len(vlm.proposals), "n_out": len(unique)},
            )

        ref_feats = vlm.audit.get("ref_features") or {}
        src_feats = vlm.audit.get("src_features") or {}
        ref_centroids = {int(i): f.centroid for i, f in ref_feats.items()}
        src_centroids = {int(i): f.centroid for i, f in src_feats.items()}
        if not ref_centroids or not src_centroids:
            return CorrespondenceResult(
                pairs=unique,
                weights=[1.0] * len(unique),
                audit={**passthrough, "resolver": "geo_correction",
                       "bypass": True,
                       "reason": "no_centroids", "n_out": len(unique)},
            )

        W = _aggregate_soft_weights(unique)
        scores, _qbar = _compute_scores(
            W, ref_centroids, src_centroids, lam=self.lam, k=self.k,
        )
        tk = int(self.top_k or 0)
        if tk > 0:
            # Top-K mode: keep the tk lowest-score REF iids. Guarantees
            # per-pair coverage — never zeroes a pair that had proposals.
            ranked = sorted(scores.items(), key=lambda kv: kv[1])
            keep_refs = {int(ri) for ri, _ in ranked[:tk]}
            kept = [(int(s), int(r)) for s, r in unique if int(r) in keep_refs]
            if not kept:                       # no scored REF iid → keep all
                kept = list(unique)
            mode = f"top{tk}"
        else:
            kept = [(int(s), int(r)) for s, r in unique
                    if scores.get(int(r)) is not None
                    and scores[int(r)] <= float(self.tau)]
            mode = "tau"
        return CorrespondenceResult(
            pairs=kept,
            weights=[1.0] * len(kept),
            audit={
                **passthrough,
                "resolver": "geo_correction",
                "bypass": False,
                "mode": mode,
                "lam": float(self.lam), "tau": float(self.tau),
                "k": int(self.k), "top_k": tk,
                "n_in": len(vlm.proposals),
                "n_unique": len(unique),
                "n_out": len(kept),
                "scores": {int(ri): float(sc) for ri, sc in scores.items()},
            },
        )
