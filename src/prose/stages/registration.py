"""Stage 6: GeoTransformer per-instance registration + RANSAC (paper §3.6).

For each (src_id, ref_id) correspondence, extract the sub-point-cloud belonging
to that instance in each subscan (via 2D-3D correspondences and SAM3 masks),
run GeoTransformer to get point-level correspondences, and accumulate.

Finally, run pygcransac on the concatenated correspondences to get a robust
4x4 rigid transform from src frame to ref frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.io import dump_pickle, ensure_dir
from ..utils.logging import get_logger
from ..utils.pointcloud import build_point_to_instance, random_downsample, voxel_downsample

log = get_logger(__name__)


@dataclass
class RegistrationArtifact:
    pair_id: str
    est_transform: Optional[np.ndarray]   # (4,4) or None if RANSAC failed
    n_correspondences: int                # total p2p correspondences fed to RANSAC
    per_pair_info: List[dict]             # per instance-pair diagnostics
    src_corr_points: Optional[np.ndarray] = None  # (N,3) RANSAC input src side
    ref_corr_points: Optional[np.ndarray] = None  # (N,3) RANSAC input ref side

    def save(self, out_dir: Path) -> Path:
        out_dir = ensure_dir(out_dir)
        dump_pickle(
            {
                "est_transform": self.est_transform,
                "n_correspondences": self.n_correspondences,
                "per_pair_info": self.per_pair_info,
                "src_corr_points": self.src_corr_points,
                "ref_corr_points": self.ref_corr_points,
            },
            out_dir / f"{self.pair_id}.pkl",
        )
        return out_dir / f"{self.pair_id}.pkl"


class _GeoTransformerRunner:
    """Lazy-loads GeoTransformer and runs per-pair registration."""

    def __init__(self, checkpoint_path: Path, max_points: int,
                 voxel_size: float = 0.025):
        import importlib.util
        import sys

        # Ensure the submodule's parent dir is on sys.path so `geotransformer`
        # package (the built C++ ext) resolves.
        submodule_root = Path(__file__).resolve().parents[3] / "third_party" / "GeoTransformer"
        if str(submodule_root) not in sys.path:
            sys.path.insert(0, str(submodule_root))

        try:
            import geotransformer  # noqa: F401 — triggers the ext load
            # Derive the GeoTransformer root from where the package was actually
            # imported from. Works for both the submodule layout
            # (third_party/GeoTransformer/geotransformer/__init__.py) and the
            # container layout (/opt/GeoTransformer/geotransformer/__init__.py).
            geo_root = Path(geotransformer.__file__).resolve().parent.parent
            # The 3DMatch experiment has the model + config; its directory name
            # is not a legal python identifier, so we load by file path.
            exp_dir = geo_root / "experiments" / (
                "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
            )
            spec_cfg = importlib.util.spec_from_file_location(
                "gt_config", exp_dir / "config.py"
            )
            cfg_mod = importlib.util.module_from_spec(spec_cfg)
            spec_cfg.loader.exec_module(cfg_mod)
            make_cfg = cfg_mod.make_cfg

            # model.py imports backbone and loss from the same dir — we need to
            # make them resolvable too.
            if str(exp_dir) not in sys.path:
                sys.path.insert(0, str(exp_dir))
            spec_model = importlib.util.spec_from_file_location(
                "gt_model", exp_dir / "model.py"
            )
            model_mod = importlib.util.module_from_spec(spec_model)
            spec_model.loader.exec_module(model_mod)
            create_model = model_mod.create_model
        except ImportError as e:
            raise ImportError(
                "GeoTransformer is not installed. Initialize the submodule and build with:\n"
                "  git submodule update --init --recursive\n"
                "  cd third_party/GeoTransformer && python setup.py build_ext --inplace\n"
                f"Original: {e}"
            )

        import torch

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = make_cfg()
        self.model = create_model(self.cfg).to(self.device).eval()
        self.max_points = int(max_points)
        self.voxel_size = float(voxel_size)

        log.info("Loading GeoTransformer checkpoint: %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)

        # Neighbor limits required by registration_collate_fn_stack_mode.
        # Default from the GeoTransformer 3DMatch config.
        self.neighbor_limits = [38, 36, 36, 38]

    def run(self, src_pts: np.ndarray, ref_pts: np.ndarray) -> Optional[dict]:
        """Run one GeoTransformer forward pass; return correspondences or None."""
        from geotransformer.utils.data import (  # type: ignore
            registration_collate_fn_stack_mode,
        )

        torch = self.torch

        # Voxel (uniform-density) downsample — NOT random. A randomly
        # subsampled room-scale cloud is non-uniform, so GeoTransformer's
        # coarse grid never pools and the O(N^2) superpoint matching OOMs.
        # `max_points` is only an OOM safety net; voxel_size sets the density.
        src = voxel_downsample(src_pts, self.voxel_size, self.max_points).astype(np.float32)
        ref = voxel_downsample(ref_pts, self.voxel_size, self.max_points).astype(np.float32)

        data_dict = {
            "ref_points": ref,
            "src_points": src,
            "ref_feats": np.ones_like(ref[:, :1], dtype=np.float32),
            "src_feats": np.ones_like(src[:, :1], dtype=np.float32),
            "transform": np.eye(4, dtype=np.float32),
        }
        try:
            data_dict = registration_collate_fn_stack_mode(
                [data_dict],
                self.cfg.backbone.num_stages,
                self.cfg.backbone.init_voxel_size,
                self.cfg.backbone.init_radius,
                self.neighbor_limits,
            )
        except torch.cuda.OutOfMemoryError as e:
            log.warning(
                "GeoTransformer collate OOM (src=%d, ref=%d): %s",
                src.shape[0], ref.shape[0], e,
            )
            torch.cuda.empty_cache()
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("GeoTransformer collate failed: %s", e)
            return None

        def _to_device(x):
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_to_device(v) for v in x]
            if hasattr(x, "to"):
                return x.to(self.device)
            return x

        try:
            data_dict = _to_device(data_dict)
        except torch.cuda.OutOfMemoryError as e:
            log.warning(
                "GeoTransformer device-transfer OOM (src=%d, ref=%d): %s",
                src.shape[0], ref.shape[0], e,
            )
            del data_dict
            torch.cuda.empty_cache()
            return None

        with torch.no_grad():
            try:
                out = self.model(data_dict)
            except torch.cuda.OutOfMemoryError as e:
                log.warning(
                    "GeoTransformer forward OOM (src=%d, ref=%d): %s",
                    src.shape[0], ref.shape[0], e,
                )
                del data_dict
                torch.cuda.empty_cache()
                return None
            except Exception as e:  # noqa: BLE001
                log.warning("GeoTransformer forward failed: %s", e)
                return None

        src_corr = out["src_corr_points"].detach().cpu().numpy()
        ref_corr = out["ref_corr_points"].detach().cpu().numpy()
        scores = out.get("corr_scores")
        if scores is not None:
            scores = scores.detach().cpu().numpy()
        # Free GPU memory after every forward — the success path previously
        # leaked data_dict + out, so memory accumulated across the per-instance
        # loop and the full-cloud fallback then OOM'd on an already-full GPU.
        del out, data_dict
        torch.cuda.empty_cache()
        return {"src": src_corr, "ref": ref_corr, "scores": scores}


def _ransac_rigid(src_corr: np.ndarray, ref_corr: np.ndarray, cfg_registration) -> Optional[np.ndarray]:
    """Estimate a 4x4 rigid transform from point-to-point correspondences.

    Verbatim from `sgaligner/src/engine/registration_evaluator.py:175-194`.
    pygcransac returns the transform in row-vector convention; the explicit
    matmul with translation-as-bottom-row T1/T2inv plus the trailing
    `.T` converts back to column-vector convention.
    """
    try:
        import pygcransac  # type: ignore
    except ImportError:
        log.warning("pygcransac not available; using Umeyama (no RANSAC).")
        return _umeyama(src_corr, ref_corr)

    corrs_ransac = np.concatenate([src_corr, ref_corr], axis=1).astype(np.float64)
    min_coordinates = np.min(corrs_ransac, axis=0)
    transformed_corrs_ransac = corrs_ransac - min_coordinates

    try:
        est_transform, _ = pygcransac.findRigidTransform(
            np.ascontiguousarray(transformed_corrs_ransac),
            probabilities=[],
            threshold=float(cfg_registration.ransac_threshold),
            neighborhood_size=4,
            sampler=1,
            min_iters=int(cfg_registration.ransac_min_iters),
            max_iters=int(cfg_registration.ransac_max_iters),
            spatial_coherence_weight=0.0,
            use_space_partitioning=not bool(cfg_registration.ransac_use_sprt),
            neighborhood=0,
            conf=0.999,
            use_sprt=bool(cfg_registration.ransac_use_sprt),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("RANSAC failed: %s", e)
        return None

    if est_transform is None:
        return None
    est_transform = np.asarray(est_transform, dtype=np.float64)
    if est_transform.shape != (4, 4) or np.any(np.isnan(est_transform)):
        return None

    T1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [-min_coordinates[0], -min_coordinates[1], -min_coordinates[2], 1],
    ])
    T2inv = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [min_coordinates[3], min_coordinates[4], min_coordinates[5], 1],
    ])

    est_transform = T1 @ est_transform @ T2inv
    est_transform = est_transform.T
    return est_transform


def _umeyama(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Closed-form rigid fit. No outlier rejection."""
    src_c = src.mean(axis=0)
    ref_c = ref.mean(axis=0)
    H = (src - src_c).T @ (ref - ref_c)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = ref_c - R @ src_c
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _count_inliers(
    src_corr: np.ndarray, ref_corr: np.ndarray, transform: np.ndarray, threshold: float
) -> int:
    """Count point correspondences within `threshold` after applying `transform`.

    `transform` is column-vector convention (ref_h ~= transform @ src_h), the
    same as `_ransac_rigid` / `_umeyama` return.
    """
    src_h = np.c_[src_corr, np.ones(len(src_corr))]
    proj = (src_h @ transform.T)[:, :3]
    dist = np.linalg.norm(proj - ref_corr, axis=1)
    return int((dist < threshold).sum())


def _retry_ransac_subsets(
    all_src_corr: List[np.ndarray],
    all_ref_corr: List[np.ndarray],
    all_scores: List[np.ndarray],
    cfg_registration,
    pair_id: str,
) -> Optional[np.ndarray]:
    """Recover a transform after the pooled RANSAC failed.

    Two staged attempts, both purely on the per-instance correspondences
    GeoTransformer already produced (no extra GeoTransformer / VRAM):

      A. Score cap — RANSAC on the top-`ransac_retry_cap` corrs by score.
         Handles the "too many points for pygcransac" failure mode.
      B. Per-instance vote — RANSAC each instance pair alone, then keep the
         candidate transform with the most inliers over the FULL pool.
         Handles instance-level contamination: one clean object pair is
         enough for an over-determined rigid fit, and inlier voting picks it.
    """
    src_corr = np.concatenate(all_src_corr, axis=0)
    ref_corr = np.concatenate(all_ref_corr, axis=0)
    have_scores = len(all_scores) == len(all_src_corr) and len(all_scores) > 0
    scores = np.concatenate(all_scores, axis=0) if have_scores else None

    # --- Attempt A: score/count cap ---
    cap = int(getattr(cfg_registration, "ransac_retry_cap", 8000))
    if src_corr.shape[0] > cap:
        if scores is not None:
            sel = np.argsort(-scores)[:cap]
        else:
            sel = np.random.default_rng(0).choice(src_corr.shape[0], cap, replace=False)
        est_T = _ransac_rigid(src_corr[sel], ref_corr[sel], cfg_registration)
        if est_T is not None:
            log.info(
                "Stage 6 [%s]: retry RANSAC on top-%d/%d corrs → SUCCESS",
                pair_id, cap, src_corr.shape[0],
            )
            return est_T

    # --- Attempt B: per-instance RANSAC + inlier voting ---
    threshold = float(cfg_registration.ransac_threshold)
    best_T: Optional[np.ndarray] = None
    best_inliers = -1
    for sc, rc in zip(all_src_corr, all_ref_corr):
        if sc.shape[0] < 3:
            continue
        cand = _ransac_rigid(sc, rc, cfg_registration)
        if cand is None:
            continue
        n_in = _count_inliers(src_corr, ref_corr, cand, threshold)
        if n_in > best_inliers:
            best_T, best_inliers = cand, n_in
    if best_T is not None:
        log.info(
            "Stage 6 [%s]: retry per-instance RANSAC → SUCCESS "
            "(%d/%d pooled inliers)",
            pair_id, best_inliers, src_corr.shape[0],
        )
    else:
        log.warning("Stage 6 [%s]: RANSAC retry exhausted — no transform", pair_id)
    return best_T


def _full_cloud_fallback(
    *,
    pair_id: str,
    runner: "_GeoTransformerRunner",
    src_points: np.ndarray,
    ref_points: np.ndarray,
    cfg_registration,
    per_pair_info: List[dict],
    reason: str,
) -> RegistrationArtifact:
    """Run GeoTransformer on the FULL src/ref clouds (no instance masking).

    Triggered when Stage 5 returns no correspondences, or when every
    instance pair failed during the per-instance loop. `reason` is recorded
    in per_pair_info under the synthetic `(-1, -1)` instance pair so
    downstream eval can tell a fallback registration apart from a per-
    instance one.
    """
    info = {
        "src_id": -1, "ref_id": -1,
        "src_pts": int(src_points.shape[0]), "ref_pts": int(ref_points.shape[0]),
        "fallback_reason": reason,
    }
    # Cap the fallback clouds. A full room-scale GeoTransformer forward needs
    # >95 GB regardless of voxel density — its coarse superpoint matching
    # scales with the room's spatial extent, which downsampling does not
    # shrink. Voxel-downsample to `fallback_max_points` uniform points so the
    # forward fits in GPU memory WITHOUT cropping the spatial extent.
    fb_cap = int(getattr(cfg_registration, "fallback_max_points", 10000))
    fb_voxel = float(getattr(cfg_registration, "input_voxel_size", 0.025))
    src_fb = voxel_downsample(src_points, fb_voxel, fb_cap)
    ref_fb = voxel_downsample(ref_points, fb_voxel, fb_cap)
    t_geo = time.perf_counter()
    res = runner.run(src_fb, ref_fb)
    t_geo = time.perf_counter() - t_geo
    if res is None or res["src"].shape[0] == 0:
        info["status"] = "fallback_geotransformer_failed"
        info["geo_time_s"] = round(t_geo, 2)
        per_pair_info.append(info)
        log.warning("Stage 6 [%s] FALLBACK (%s): GeoTransformer failed (%.1fs)", pair_id, reason, t_geo)
        return RegistrationArtifact(
            pair_id=pair_id, est_transform=None, n_correspondences=0,
            per_pair_info=per_pair_info,
            src_corr_points=np.zeros((0, 3), dtype=np.float32),
            ref_corr_points=np.zeros((0, 3), dtype=np.float32),
        )

    sc = res["src"]
    rc = res["ref"]
    scores = res.get("scores")
    cap = int(cfg_registration.num_p2p_corrs)
    if sc.shape[0] > cap:
        if scores is not None:
            sel = np.argsort(-scores)[:cap]
            sc, rc, scores = sc[sel], rc[sel], scores[sel]
        else:
            sc, rc = sc[:cap], rc[:cap]

    info["status"] = "fallback_ok"
    info["n_corr"] = int(sc.shape[0])
    info["geo_time_s"] = round(t_geo, 2)
    per_pair_info.append(info)

    t_ransac = time.perf_counter()
    est_T = _ransac_rigid(sc, rc, cfg_registration)
    t_ransac = time.perf_counter() - t_ransac
    log.info(
        "Stage 6 [%s] FALLBACK (%s): %d p2p corrs → %s  [geo: %.1fs, ransac: %.1fs]",
        pair_id, reason, sc.shape[0], "SUCCESS" if est_T is not None else "FAIL",
        t_geo, t_ransac,
    )
    return RegistrationArtifact(
        pair_id=pair_id,
        est_transform=est_T,
        n_correspondences=int(sc.shape[0]),
        per_pair_info=per_pair_info,
        src_corr_points=sc.astype(np.float32),
        ref_corr_points=rc.astype(np.float32),
    )


def _allocate_per_pair_caps(
    *,
    num_p2p_corrs: int,
    n_pairs: int,
    weights: Optional[List[float]],
) -> List[int]:
    """Distribute Stage-5's per-pair correspondence budget.

    Uniform when `weights` is None or malformed; otherwise proportional
    to weight with a floor of 1 per pair (so zero-weight pairs still get
    one correspondence — RANSAC can ignore it). Sum may exceed
    `num_p2p_corrs` by a few units due to rounding; that's fine, the
    real cap is `sc.shape[0]` from GeoTransformer per pair anyway.
    """
    if n_pairs <= 0:
        return []
    uniform = max(1, int(num_p2p_corrs) // n_pairs)
    if weights is None or len(weights) != n_pairs:
        return [uniform] * n_pairs
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0.0:
        return [uniform] * n_pairs
    raw = (w / s) * float(num_p2p_corrs)
    caps = np.maximum(1, np.round(raw).astype(np.int64))
    return [int(c) for c in caps]


_CORR_EXTRACTOR_REQUIREMENTS = {
    "geotransformer": "GeoTransformer C++ extension (bash scripts/setup_geotransformer.sh)",
    "fpfh": "Open3D (pip install -e \".[viz]\") — handcrafted, no weights needed",
    "fcgf": "MinkowskiEngine + ResUNetBN2C 3DMatch weights at weights/fcgf/fcgf_3dmatch.pth",
}


def _make_corr_runner(cfg_registration):
    """Dispatch the per-instance correspondence extractor.

    `cfg_registration.corr_extractor` ∈ {"fcgf" (default), "fpfh", "geotransformer"}
    selects the backend; everything else (voxel input, _filter_corrs,
    verify_transform, _retry_ransac_subsets, full-cloud fallback) is shared.

    Each backend has its own extra dependency — see
    ``_CORR_EXTRACTOR_REQUIREMENTS`` for what each one needs.
    """
    name = str(getattr(cfg_registration, "corr_extractor", "fcgf")).lower()
    voxel = float(getattr(cfg_registration, "input_voxel_size", 0.025))
    max_pts = int(cfg_registration.max_points_per_pc)
    if name not in _CORR_EXTRACTOR_REQUIREMENTS:
        raise ValueError(
            f"Unknown corr_extractor: {name!r}. "
            f"Choose one of: {sorted(_CORR_EXTRACTOR_REQUIREMENTS)}."
        )

    requirement_hint = _CORR_EXTRACTOR_REQUIREMENTS[name]
    try:
        if name == "geotransformer":
            return _GeoTransformerRunner(
                Path(cfg_registration.checkpoint_path), max_pts, voxel_size=voxel,
            )
        if name == "fpfh":
            from ..models.fpfh_extractor import _FPFHRunner
            return _FPFHRunner(max_points=max_pts, voxel_size=voxel)
        if name == "fcgf":
            from ..models.fcgf_extractor import _FCGFRunner
            ckpt = Path(getattr(cfg_registration, "corr_extractor_checkpoint",
                                "weights/fcgf/fcgf_3dmatch.pth"))
            if not ckpt.is_absolute():
                ckpt = Path.cwd() / ckpt
            return _FCGFRunner(checkpoint_path=ckpt, max_points=max_pts, voxel_size=voxel)
    except ImportError as e:
        raise ImportError(
            f"corr_extractor={name!r} requires: {requirement_hint}. "
            f"Original import error: {e}"
        ) from e
    raise ValueError(f"Unhandled corr_extractor: {name!r}")


def _free_runner(runner: Optional["_GeoTransformerRunner"]) -> None:
    """Release a GeoTransformer runner's model + CUDA cache.

    Stage 6 builds one runner per pair; without an explicit release the GPU
    memory from each pair's model accumulates across the run and the
    full-cloud fallback OOMs on later pairs (observed: 69 GB -> 94 GB in use
    across 36 ADT pairs).
    """
    if runner is None:
        return
    try:
        import gc

        import torch

        # FPFH has no torch model; other backends own `.model`.
        if hasattr(runner, "model"):
            runner.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        pass


def _nn_inlier_ratio(src_t: np.ndarray, ref: np.ndarray, radius: float) -> float:
    """Fraction of `src_t` points within `radius` of any `ref` point.

    Pure-numpy chunked nearest-neighbour (no scipy dependency). Both clouds
    should already be subsampled by the caller.
    """
    if src_t.shape[0] == 0 or ref.shape[0] == 0:
        return 0.0
    r2 = float(radius) * float(radius)
    n_in = 0
    chunk = 400
    ref = ref.astype(np.float32)
    for i in range(0, src_t.shape[0], chunk):
        c = src_t[i:i + chunk].astype(np.float32)
        d2 = ((c[:, None, :] - ref[None, :, :]) ** 2).sum(axis=-1)
        n_in += int((d2.min(axis=1) < r2).sum())
    return n_in / float(src_t.shape[0])


def _select_by_cloud_overlap(candidates, src_points, ref_points, cfg_registration):
    """Score each candidate transform by full-cloud overlap; return the best.

    Applies T to a subsample of the src cloud and measures the fraction of
    transformed points landing within `verify_radius` of a ref point. A
    correct transform makes the clouds overlap (high score); a wrong one
    (even from a self-consistent but globally-wrong correspondence set)
    aligns nothing (low score). Returns (best_T, best_score, best_tag).
    """
    if not candidates:
        return None, 0.0, "none"
    radius = float(getattr(cfg_registration, "verify_radius", 0.1))
    rng = np.random.default_rng(0)

    def _sub(p, n):
        if p.shape[0] <= n:
            return p.astype(np.float32)
        return p[rng.choice(p.shape[0], n, replace=False)].astype(np.float32)

    src_s = _sub(src_points, 3000)
    ref_s = _sub(ref_points, 20000)
    best_T, best_score, best_tag = None, -1.0, "none"
    for tag, T in candidates:
        Tf = np.asarray(T, dtype=np.float32)
        src_t = src_s @ Tf[:3, :3].T + Tf[:3, 3]
        score = _nn_inlier_ratio(src_t, ref_s, radius)
        if score > best_score:
            best_T = np.asarray(T, dtype=np.float64)
            best_score = score
            best_tag = tag
    return best_T, best_score, best_tag


def _filter_correspondences(
    correspondences: List[Tuple[int, int]],
    weights: Optional[List[float]],
    src_points: np.ndarray,
    ref_points: np.ndarray,
    src_inst: np.ndarray,
    ref_inst: np.ndarray,
    cfg_registration,
) -> Tuple[List[Tuple[int, int]], Optional[List[float]]]:
    """Geometric-consistency + one-to-one filter on instance correspondences.

    A wrong instance pair (e.g. sofa-A <-> sofa-B) feeds RANSAC a self-
    consistent but incorrect transform; many-to-one matches come from
    indistinguishable repeated objects. Both are removed here using each
    instance's 3D centroid:
      * geometric consistency — a pair is kept only if at least
        `corr_min_consistent_neighbors` other pairs agree on the pairwise
        centroid distance (|d_src - d_ref| / max < tol). Transform-invariant,
        so it works on GT and DA3 clouds alike.
      * one-to-one — each src/ref instance keeps a single partner; conflicts
        are resolved by geometric-consistency vote count.
    """
    tol = float(getattr(cfg_registration, "corr_distance_tolerance", 0.15))
    min_consistent = int(getattr(cfg_registration, "corr_min_consistent_neighbors", 2))

    def _centroid(points, inst, iid):
        m = points[inst == iid]
        return m.mean(axis=0) if m.shape[0] > 0 else None

    src_cen: dict = {}
    ref_cen: dict = {}
    kept0: List[Tuple[int, int]] = []
    kept0_w: List[Optional[float]] = []
    for i, (s, r) in enumerate(correspondences):
        if s not in src_cen:
            src_cen[s] = _centroid(src_points, src_inst, s)
        if r not in ref_cen:
            ref_cen[r] = _centroid(ref_points, ref_inst, r)
        if src_cen[s] is not None and ref_cen[r] is not None:
            kept0.append((s, r))
            kept0_w.append(weights[i] if weights is not None and i < len(weights) else None)

    n = len(kept0)
    if n <= 2:
        return kept0, (kept0_w if weights is not None else None)

    cs = np.stack([src_cen[s] for s, _ in kept0], axis=0)
    cr = np.stack([ref_cen[r] for _, r in kept0], axis=0)
    d_src = np.linalg.norm(cs[:, None, :] - cs[None, :, :], axis=-1)
    d_ref = np.linalg.norm(cr[:, None, :] - cr[None, :, :], axis=-1)
    denom = np.maximum(np.maximum(d_src, d_ref), 1e-6)
    rel = np.abs(d_src - d_ref) / denom
    np.fill_diagonal(rel, np.inf)
    votes = (rel < tol).sum(axis=1)  # geom-consistency vote count per pair

    geom_keep = [i for i in range(n) if votes[i] >= min_consistent]
    # One-to-one: greedily assign by descending vote count.
    geom_keep.sort(key=lambda i: -int(votes[i]))
    used_src: set = set()
    used_ref: set = set()
    final_idx: List[int] = []
    for i in geom_keep:
        s, r = kept0[i]
        if s in used_src or r in used_ref:
            continue
        used_src.add(s)
        used_ref.add(r)
        final_idx.append(i)
    final_idx.sort()

    final_pairs = [kept0[i] for i in final_idx]
    if weights is None:
        final_w = None
    else:
        final_w = [kept0_w[i] for i in final_idx]
        if any(w is None for w in final_w):
            final_w = None
    return final_pairs, final_w


def run_registration(
    *,
    pair_id: str,
    src_points: np.ndarray,
    ref_points: np.ndarray,
    src_point_to_pixels: List[List[dict]],
    ref_point_to_pixels: List[List[dict]],
    src_masks: Dict[int, Dict[int, np.ndarray]],
    ref_masks: Dict[int, Dict[int, np.ndarray]],
    correspondences: List[Tuple[int, int]],
    cfg_registration,
    correspondence_weights: Optional[List[float]] = None,
) -> RegistrationArtifact:
    """Per-instance GeoTransformer + final RANSAC fusion.

    When `cfg_registration.use_geotransformer_fallback=true` (default), Stage 6
    falls back to a full-cloud GeoTransformer + RANSAC run if Stage 5
    produced no correspondences or every instance pair failed. The
    fallback is tagged in `per_pair_info[*].fallback_reason`.

    `correspondence_weights` (optional, parallel to `correspondences`)
    biases the per-pair correspondence budget. When None, all pairs get
    `num_p2p_corrs // len(correspondences)`. When supplied, the budget
    is distributed proportionally to weight (with a floor of 1 per pair),
    so high-confidence Stage-4 pairs dominate the RANSAC inlier pool.
    Weight is echoed into `per_pair_info[i]["weight"]`.
    """
    use_fallback = bool(getattr(cfg_registration, "use_geotransformer_fallback", True))

    if not correspondences:
        if not use_fallback:
            log.warning("Stage 6 [%s]: no correspondences — skipping", pair_id)
            return RegistrationArtifact(pair_id=pair_id, est_transform=None, n_correspondences=0, per_pair_info=[])
        log.warning("Stage 6 [%s]: no correspondences — falling back to full-cloud GeoTransformer", pair_id)
        runner = _make_corr_runner(cfg_registration)
        try:
            return _full_cloud_fallback(
                pair_id=pair_id, runner=runner,
                src_points=src_points, ref_points=ref_points,
                cfg_registration=cfg_registration, per_pair_info=[],
                reason="no_correspondences",
            )
        finally:
            _free_runner(runner)

    src_inst = build_point_to_instance(src_points, _index_point_to_pixels(src_point_to_pixels), src_masks)
    ref_inst = build_point_to_instance(ref_points, _index_point_to_pixels(ref_point_to_pixels), ref_masks)

    # Geometric-consistency + one-to-one filter on the Stage-4 correspondences
    # (drops wrong instance pairs that would otherwise hijack RANSAC, and the
    # many-to-one matches from indistinguishable repeated objects).
    if bool(getattr(cfg_registration, "filter_correspondences", False)):
        n_before = len(correspondences)
        correspondences, correspondence_weights = _filter_correspondences(
            correspondences, correspondence_weights,
            src_points, ref_points, src_inst, ref_inst, cfg_registration,
        )
        log.info("Stage 6 [%s]: correspondence filter %d -> %d (geom-consistency + 1:1)",
                 pair_id, n_before, len(correspondences))
        if not correspondences:
            log.warning("Stage 6 [%s]: all correspondences filtered out", pair_id)
            return RegistrationArtifact(
                pair_id=pair_id, est_transform=None, n_correspondences=0,
                per_pair_info=[],
            )

    runner = _make_corr_runner(cfg_registration)

    all_src_corr: List[np.ndarray] = []
    all_ref_corr: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    per_pair_info: List[dict] = []

    min_pts = int(cfg_registration.min_object_points)
    num_p2p_corrs = int(cfg_registration.num_p2p_corrs)
    per_pair_caps = _allocate_per_pair_caps(
        num_p2p_corrs=num_p2p_corrs,
        n_pairs=len(correspondences),
        weights=correspondence_weights,
    )

    t_geo_total = 0.0
    n_geo_calls = 0
    for idx, (src_id, ref_id) in enumerate(correspondences):
        sp = src_points[src_inst == src_id]
        rp = ref_points[ref_inst == ref_id]
        per_node_cap = per_pair_caps[idx]
        weight = (
            float(correspondence_weights[idx])
            if correspondence_weights is not None and idx < len(correspondence_weights)
            else None
        )
        info = {
            "src_id": src_id, "ref_id": ref_id,
            "src_pts": int(sp.shape[0]), "ref_pts": int(rp.shape[0]),
            "per_node_cap": int(per_node_cap),
        }
        if weight is not None:
            info["weight"] = weight
        if sp.shape[0] < min_pts or rp.shape[0] < min_pts:
            info["status"] = "insufficient_points"
            per_pair_info.append(info)
            continue

        t0 = time.perf_counter()
        res = runner.run(sp, rp)
        dt = time.perf_counter() - t0
        t_geo_total += dt
        n_geo_calls += 1
        info["geo_time_s"] = round(dt, 2)
        if res is None or res["src"].shape[0] == 0:
            info["status"] = "geotransformer_failed"
            per_pair_info.append(info)
            continue

        sc = res["src"]
        rc = res["ref"]
        scores = res.get("scores")

        # Per-node top-k by score (upstream registration_evaluator.py:159-163).
        if scores is not None and sc.shape[0] > per_node_cap:
            sel = np.argsort(-scores)[:per_node_cap]
            sc = sc[sel]
            rc = rc[sel]
            scores = scores[sel]
        elif scores is None and sc.shape[0] > per_node_cap:
            sc = sc[:per_node_cap]
            rc = rc[:per_node_cap]

        all_src_corr.append(sc)
        all_ref_corr.append(rc)
        if scores is not None:
            all_scores.append(scores)
        info["status"] = "ok"
        info["n_corr"] = int(sc.shape[0])
        per_pair_info.append(info)

    if not all_src_corr:
        if not use_fallback:
            _free_runner(runner)
            return RegistrationArtifact(
                pair_id=pair_id, est_transform=None, n_correspondences=0,
                per_pair_info=per_pair_info,
                src_corr_points=np.zeros((0, 3), dtype=np.float32),
                ref_corr_points=np.zeros((0, 3), dtype=np.float32),
            )
        log.warning(
            "Stage 6 [%s]: all %d instance pairs failed — falling back to full-cloud GeoTransformer",
            pair_id, len(correspondences),
        )
        try:
            return _full_cloud_fallback(
                pair_id=pair_id, runner=runner,
                src_points=src_points, ref_points=ref_points,
                cfg_registration=cfg_registration, per_pair_info=per_pair_info,
                reason="instance_pairs_failed",
            )
        finally:
            _free_runner(runner)

    src_corr = np.concatenate(all_src_corr, axis=0)
    ref_corr = np.concatenate(all_ref_corr, axis=0)

    _free_runner(runner)

    # --- Transform hypothesis verification against the clouds ---
    # RANSAC trusts its input p2p correspondences: a globally-coherent but
    # wrong correspondence set yields a confident wrong transform. Build
    # multiple hypotheses (the pooled RANSAC fit + one per instance pair),
    # score each by how well it actually aligns the full src/ref clouds, and
    # keep the best. Recovers a correct transform that a wrong majority would
    # drown out in the single pooled fit, and rejects transforms that align
    # nothing.
    t_ransac = time.perf_counter()
    if bool(getattr(cfg_registration, "verify_transform", False)):
        candidates: List[Tuple[str, np.ndarray]] = []
        pooled_T = _ransac_rigid(src_corr, ref_corr, cfg_registration)
        if pooled_T is not None:
            candidates.append(("pooled", pooled_T))
        for i, (s, r) in enumerate(zip(all_src_corr, all_ref_corr)):
            if s.shape[0] >= 3:
                ti = _ransac_rigid(s, r, cfg_registration)
                if ti is not None:
                    candidates.append((f"inst{i}", ti))
        est_T, best_score, best_tag = _select_by_cloud_overlap(
            candidates, src_points, ref_points, cfg_registration)
        min_score = float(getattr(cfg_registration, "verify_min_inlier_ratio", 0.1))
        if est_T is not None and best_score < min_score:
            log.info("Stage 6 [%s]: best hypothesis (%s) cloud-overlap %.3f < %.2f — rejected",
                     pair_id, best_tag, best_score, min_score)
            est_T = None
        else:
            log.info("Stage 6 [%s]: %d transform hypotheses, selected=%s cloud-overlap=%.3f",
                     pair_id, len(candidates), best_tag, best_score)
    else:
        est_T = _ransac_rigid(src_corr, ref_corr, cfg_registration)
    t_ransac = time.perf_counter() - t_ransac
    log.info(
        "Stage 6 [%s]: %d p2p corrs (n_nodes=%d, per_node_caps=%s) → %s  "
        "[geo: %.1fs/%d calls, ransac: %.1fs]",
        pair_id, src_corr.shape[0], len(correspondences),
        list(per_pair_caps) if len(per_pair_caps) <= 8 else f"<{len(per_pair_caps)} entries>",
        "SUCCESS" if est_T is not None else "FAIL",
        t_geo_total, n_geo_calls, t_ransac,
    )
    if est_T is None:
        t_retry = time.perf_counter()
        est_T = _retry_ransac_subsets(
            all_src_corr, all_ref_corr, all_scores, cfg_registration, pair_id
        )
        t_retry = time.perf_counter() - t_retry
        if est_T is not None:
            per_pair_info.append({
                "src_id": -1, "ref_id": -1,
                "status": "ransac_retry_ok",
                "fallback_reason": "ransac_retry",
                "retry_time_s": round(t_retry, 2),
            })
        else:
            log.info("Stage 6 [%s]: RANSAC retry took %.1fs", pair_id, t_retry)
    return RegistrationArtifact(
        pair_id=pair_id,
        est_transform=est_T,
        n_correspondences=int(src_corr.shape[0]),
        per_pair_info=per_pair_info,
        src_corr_points=src_corr.astype(np.float32),
        ref_corr_points=ref_corr.astype(np.float32),
    )


def _index_point_to_pixels(meta: List[List[dict]]) -> Dict[int, List[dict]]:
    """Turn the per-point list-of-entries into a point_idx → entries map.

    The Stage 1 schema is now `List[List[dict]]` (each point carries all the
    frames it was visible in). Legacy `List[dict]` artifacts on disk are
    auto-wrapped for backwards compatibility.
    """
    out: Dict[int, List[dict]] = {}
    for pt_idx, entries in enumerate(meta):
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            continue
        out[pt_idx] = [e for e in entries if e]
    return out
