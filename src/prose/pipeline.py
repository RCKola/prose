"""PROSE pipeline orchestrator (single-process).

Runs the six PROSE stages for a list of subscan pairs, caching every
intermediate artifact to disk so re-runs resume instantly:

  Scene parsing (per subscan, paper Sec. 3.1)
    1. geometry        — VGGT-Omega / GT point cloud + 2D-to-3D index
    2. object_listing  — VLM object names
    3. segmentation    — SAM3 text-prompted video instance masks
    4. fusion          — per-instance 3D fusion → scene-graph nodes

  Per pair (paper Sec. 3.2-3.3)
    5. correspondence  — height-binned VLM matching + double-check
    6. registration    — per-instance descriptor + RANSAC + pose voting

Designed for a single GPU: heavy models are loaded one at a time and released
between stages.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .datasets import build_dataset
from .datasets.base import BaseDataset, SubscanPair
from .eval import (
    aggregate_discovery_metrics,
    aggregate_metrics,
    aggregate_pair_set_metrics,
    build_sam3_gt_pairs,
    build_sam3_mask_clouds,
    build_sam_to_object_map,
    compute_all_metrics,
    compute_object_discovery,
    compute_pair_set_metrics,
)
from .stages.geometry import run_geometry
from .stages.object_listing import run_object_listing
from .stages.segmentation import run_segmentation
from .stages.fusion import run_fusion, PriorArtifact
from .stages.registration import run_registration, RegistrationArtifact
from .utils.gpu import release_gpu
from .utils.io import dump_json, ensure_dir, load_json, load_pickle
from .utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unpack_masks(path: Path) -> Dict[int, Dict[int, np.ndarray]]:
    """Unpack the packbits masks saved by the segmentation stage."""
    payload = load_pickle(path)
    shapes = payload["frame_shapes"]
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for fidx, packed in payload["packed_masks"].items():
        shape = shapes[fidx]
        per_inst = {}
        for iid, packed_m in packed.items():
            flat = np.unpackbits(packed_m).astype(bool)
            per_inst[iid] = flat[: shape[0] * shape[1]].reshape(shape)
        out[fidx] = per_inst
    return out


def _filter_pairs(all_pairs: List[SubscanPair], cfg) -> List[SubscanPair]:
    selector = cfg.pairs
    if selector == "all":
        return all_pairs
    if selector == "first_n":
        return all_pairs[: int(cfg.max_pairs)]
    # Explicit list of pair IDs (Hydra parses `pairs=[a,b]` as a ListConfig).
    if not isinstance(selector, str) and hasattr(selector, "__iter__"):
        keep = set(selector)
        return [p for p in all_pairs if p.pair_id in keep]
    raise ValueError(f"Unknown pairs selector: {selector!r} (use 'all', 'first_n', or a list)")


def _g(obj, key, default):
    """getattr that also works on OmegaConf / dict nodes."""
    if obj is None:
        return default
    if hasattr(obj, "__contains__") and not isinstance(obj, str):
        try:
            if key in obj:
                return obj[key]
        except TypeError:
            pass
    return getattr(obj, key, default)


@dataclass
class PairResult:
    pair_id: str
    metrics: Dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset: BaseDataset = build_dataset(cfg.dataset)
        root = Path(cfg.output.root)
        self.dirs = {
            "geometry": ensure_dir(root / cfg.output.geometry_dir),
            "object_listing": ensure_dir(root / cfg.output.object_listing_dir),
            "segmentation": ensure_dir(root / cfg.output.segmentation_dir),
            "fusion": ensure_dir(root / cfg.output.fusion_dir),
            "correspondence": ensure_dir(root / cfg.output.correspondence_dir),
            "registration": ensure_dir(root / cfg.output.registration_dir),
            "eval": ensure_dir(root / cfg.output.eval_dir),
            "root": ensure_dir(root),
        }

    # ------------------------------------------------------------------
    # Scene parsing: per-subscan artifacts (stages 1-4)
    # ------------------------------------------------------------------
    def run_scene_parsing(self, subscan_ids: List[str]) -> Dict[str, Dict[str, Path]]:
        cfg = self.cfg
        skip = set(cfg.skip_stages)
        results: Dict[str, Dict[str, Path]] = {sid: {} for sid in subscan_ids}

        # Populate cache hits up front (independent of skip_stages).
        for sid in subscan_ids:
            for stage, ext in (
                ("geometry", "pkl"), ("object_listing", "json"),
                ("segmentation", "pkl"), ("fusion", "pkl"),
            ):
                p = self.dirs[stage] / f"{sid}.{ext}"
                if p.exists():
                    results[sid][stage] = p

        # -- Stage 1: geometry --
        if "geometry" not in skip:
            todo = [s for s in subscan_ids if "geometry" not in results[s]]
            if todo:
                log.info("=== Geometry: %d to run, %d cached ===", len(todo), len(subscan_ids) - len(todo))
                wrapper = None
                if not bool(cfg.use_gt_pointclouds):
                    dev = cfg.pipeline.get("device_map", None) or "cuda"
                    if dev == "auto":
                        dev = "cuda"
                    from .models.vggt_omega import VGGTOmegaWrapper
                    try:
                        wrapper = VGGTOmegaWrapper(
                            checkpoint_path=cfg.geometry.checkpoint_path,
                            device=dev,
                            image_resolution=int(getattr(cfg.geometry, "image_resolution", 512)),
                        )
                    except (ImportError, FileNotFoundError) as e:
                        log.warning("VGGT-Omega unavailable (%s); using GT point clouds.", e)
                        wrapper = None
                try:
                    for sid in todo:
                        art = run_geometry(
                            self.dataset, sid,
                            use_gt=bool(cfg.use_gt_pointclouds),
                            cfg_geometry=cfg.geometry,
                            cfg_pipeline=cfg.pipeline,
                            wrapper=wrapper,
                        )
                        results[sid]["geometry"] = art.save(self.dirs["geometry"])
                finally:
                    if wrapper is not None:
                        wrapper.close()
                        release_gpu()

        # -- Stage 2: object listing --
        if "object_listing" not in skip:
            todo = [s for s in subscan_ids if "object_listing" not in results[s]]
            if todo:
                log.info("=== Object listing: %d to run, %d cached ===", len(todo), len(subscan_ids) - len(todo))
                wrapper = self._build_listing_vlm(cfg.object_listing)
                try:
                    for sid in todo:
                        frames = self.dataset.load_rgb_frames(sid)
                        art = run_object_listing(
                            wrapper, subscan_id=sid,
                            frame_paths=frames.paths, cfg_object_listing=cfg.object_listing,
                        )
                        results[sid]["object_listing"] = art.save(self.dirs["object_listing"])
                finally:
                    wrapper.close()
                    release_gpu()

        # -- Stage 3: segmentation (SAM3) --
        if "segmentation" not in skip:
            todo = [
                s for s in subscan_ids
                if "segmentation" not in results[s] and "object_listing" in results[s]
            ]
            if todo:
                log.info("=== Segmentation (SAM3): %d to run ===", len(todo))
                device = "cuda" if cfg.pipeline.device_map != "cpu" else "cpu"
                from .models.sam3 import make_sam3_wrapper
                wrapper = make_sam3_wrapper(
                    model_id=cfg.segmentation.model_id,
                    torch_dtype=cfg.pipeline.torch_dtype,
                    device=device,
                    offload_video_to_cpu=bool(getattr(cfg.segmentation, "offload_video_to_cpu", False)),
                )
                try:
                    for sid in todo:
                        frames = self.dataset.load_rgb_frames(sid)
                        prompts = load_json(results[sid]["object_listing"])["objects"]
                        art = run_segmentation(
                            wrapper, subscan_id=sid,
                            frame_paths=frames.paths, text_prompts=prompts,
                            cfg_segmentation=cfg.segmentation,
                        )
                        results[sid]["segmentation"] = art.save(self.dirs["segmentation"])
                finally:
                    wrapper.close()
                    release_gpu()

        # -- Stage 4: per-instance fusion (CPU) --
        if "fusion" not in skip:
            todo = [
                s for s in subscan_ids
                if "fusion" not in results[s]
                and "geometry" in results[s] and "segmentation" in results[s]
            ]
            if todo:
                log.info("=== Fusion (per-instance 3D): %d to run ===", len(todo))
                for sid in todo:
                    try:
                        s1 = load_pickle(results[sid]["geometry"])
                        s3 = load_pickle(results[sid]["segmentation"])
                        masks = _unpack_masks(results[sid]["segmentation"])
                        art = run_fusion(
                            subscan_id=sid,
                            points_xyz=s1["points"],
                            point_to_pixels=s1["point_to_pixels"],
                            per_frame_masks=masks,
                            categories=s3.get("instance_to_prompt", {}) or None,
                            cfg=cfg.fusion,
                            stage1_pixel_resolution=s1.get("pixel_resolution"),
                        )
                        results[sid]["fusion"] = art.save(self.dirs["fusion"])
                    except Exception as e:  # noqa: BLE001
                        log.exception("Fusion failed for %s: %s", sid, e)

        return results

    def _build_listing_vlm(self, cfg_listing):
        """Construct the object-listing VLM (vLLM by default; HF Qwen fallback)."""
        model_id = cfg_listing.model_id
        backend = str(getattr(cfg_listing, "vlm_backend", "vllm_local")).lower()
        if backend == "vllm_local":
            from .models.vllm_local import VllmVLMWrapper
            v = getattr(cfg_listing, "vllm", {}) or {}
            mml = int(_g(v, "max_model_len", 0) or 0)
            mns = _g(v, "max_num_seqs", None)
            return VllmVLMWrapper(
                model_id=model_id,
                dtype=str(_g(v, "dtype", "auto")),
                gpu_memory_utilization=float(_g(v, "gpu_memory_utilization", 0.85)),
                max_model_len=mml if mml > 0 else None,
                limit_mm_per_prompt_image=int(_g(v, "limit_mm_per_prompt_image", 16)),
                tensor_parallel_size=int(_g(v, "tensor_parallel_size", 1)),
                enforce_eager=bool(_g(v, "enforce_eager", False)),
                trust_remote_code=bool(_g(v, "trust_remote_code", True)),
                max_num_seqs=(int(mns) if mns else None),
            )
        from .models.qwen_vl import QwenVLWrapper
        return QwenVLWrapper(
            model_id=model_id,
            torch_dtype=self.cfg.pipeline.torch_dtype,
            attn_implementation=self.cfg.pipeline.attn_implementation,
            device_map=self.cfg.pipeline.device_map,
        )

    # ------------------------------------------------------------------
    # Per-pair stages (correspondence + registration + eval)
    # ------------------------------------------------------------------
    def run_pairs(
        self, pairs: List[SubscanPair], per_subscan: Dict[str, Dict[str, Path]],
    ) -> List[PairResult]:
        cfg = self.cfg
        skip = set(cfg.skip_stages)

        # -- Stage 5: correspondence (batch, one VLM load) --
        corr_outputs: Dict[str, Path] = {}
        for pair in pairs:
            p = self.dirs["correspondence"] / f"{pair.pair_id}.pkl"
            if p.exists():
                corr_outputs[pair.pair_id] = p
        todo = [p for p in pairs if p.pair_id not in corr_outputs]
        if "correspondence" not in skip and todo:
            log.info("=== Correspondence: %d to run, %d cached ===", len(todo), len(pairs) - len(todo))
            t0 = time.perf_counter()
            corr_outputs.update(self._run_correspondence(todo, per_subscan))
            log.info("=== Correspondence done: %.1fs ===", time.perf_counter() - t0)

        # -- Stage 6: registration + eval (per pair) --
        results: List[PairResult] = []
        if "registration" not in skip:
            log.info("=== Registration: %d pairs ===", len(pairs))
        for pair in pairs:
            metrics: Dict[str, Any] = {"valid": 0.0}
            reg_path = self.dirs["registration"] / f"{pair.pair_id}.pkl"
            if "registration" not in skip:
                try:
                    if reg_path.exists():
                        payload = load_pickle(reg_path)
                        art = RegistrationArtifact(
                            pair_id=pair.pair_id,
                            est_transform=payload["est_transform"],
                            n_correspondences=payload["n_correspondences"],
                            per_pair_info=payload["per_pair_info"],
                            src_corr_points=payload.get("src_corr_points"),
                            ref_corr_points=payload.get("ref_corr_points"),
                        )
                    else:
                        art = self._run_registration_for_pair(
                            pair, per_subscan, corr_outputs.get(pair.pair_id),
                        )
                        if art is not None:
                            art.save(self.dirs["registration"])
                    if art is not None and cfg.run_evaluation:
                        metrics = self._evaluate(pair, per_subscan, art)
                except Exception as e:  # noqa: BLE001
                    log.exception("Registration failed for %s: %s", pair.pair_id, e)

            if cfg.run_evaluation:
                inst = self._evaluate_instance(pair, per_subscan, corr_outputs.get(pair.pair_id))
                if inst:
                    metrics["instance"] = inst

            results.append(PairResult(pair_id=pair.pair_id, metrics=metrics))
        return results

    # ------------------------------------------------------------------
    # Correspondence (paper Sec. 3.2): height-binned crops + double-check.
    # ------------------------------------------------------------------
    def _run_correspondence(
        self, pairs: List[SubscanPair], per_subscan: Dict[str, Dict[str, Path]],
    ) -> Dict[str, Path]:
        from .stages.correspondence.artifact import CorrespondenceArtifact
        from .stages.correspondence.filters.identity import IdentityFilter
        from .stages.correspondence.parsers.tuple_json import TupleJsonParser
        from .stages.correspondence.pipeline import CorrespondencePipeline
        from .stages.correspondence.postprocess.double_check import DoubleCheckPostprocessor
        from .stages.correspondence.prompts.bin_crops import (
            BinCropsPrompt, PairwiseBinCropsPrompt,
        )
        from .stages.correspondence.resolvers.geo_correction import GeoCorrectionResolver
        from .stages.correspondence.views.per_bin import PerBinFrameSelectionConfig
        from .stages.correspondence.visuals.bin_visuals import CropsContextBinVisuals
        from .stages.correspondence.visuals.blocking_meta import (
            BlockingConfig, run_blocking_pipeline,
        )
        from .stages.correspondence.vlm.invoker import VLMInvoker

        cfg_c = self.cfg.correspondence

        # -- VLM construction (vLLM default; HF Qwen fallback) --
        backend = str(getattr(cfg_c, "vlm_backend", "vllm_local")).lower()
        if backend == "vllm_local":
            from .models.vllm_local import VllmVLMWrapper
            v = getattr(cfg_c, "vllm", {}) or {}
            mml = int(_g(v, "max_model_len", 0) or 0)
            mns = _g(v, "max_num_seqs", None)
            vlm_raw = VllmVLMWrapper(
                model_id=str(getattr(cfg_c, "model_id", "Qwen/Qwen3.6-27B")),
                dtype=str(_g(v, "dtype", "auto")),
                gpu_memory_utilization=float(_g(v, "gpu_memory_utilization", 0.85)),
                max_model_len=(mml if mml > 0 else None),
                limit_mm_per_prompt_image=int(_g(v, "limit_mm_per_prompt_image", 256)),
                tensor_parallel_size=int(_g(v, "tensor_parallel_size", 1)),
                enforce_eager=bool(_g(v, "enforce_eager", True)),
                trust_remote_code=bool(_g(v, "trust_remote_code", True)),
                max_num_seqs=(int(mns) if mns else None),
            )
        else:
            from .models.qwen_vl import QwenVLWrapper
            vlm_raw = QwenVLWrapper(
                model_id=str(getattr(cfg_c, "model_id", "Qwen/Qwen3.6-27B")),
                torch_dtype=self.cfg.pipeline.torch_dtype,
                attn_implementation=self.cfg.pipeline.attn_implementation,
                device_map=self.cfg.pipeline.device_map,
            )

        # -- BlockingConfig from cfg.correspondence.blocking --
        bg = getattr(cfg_c, "blocking", None) or {}
        pbfs = _g(bg, "per_bin_frame_selection", {}) or {}
        block_cfg = BlockingConfig(
            enabled=True,
            n_bins=int(_g(bg, "n_bins", 5)),
            overlap_frac=float(_g(bg, "overlap_frac", 0.2)),
            bin_mode=str(_g(bg, "bin_mode", "quantile") or "quantile"),
            coalesce_pop_threshold=int(_g(bg, "coalesce_pop_threshold", 12)),
            coalesce_radius_frac=float(_g(bg, "coalesce_radius_frac", 0.5)),
            coalesce_obb_extent_frac=float(_g(bg, "coalesce_obb_extent_frac", 0.40)),
            use_bev=bool(getattr(cfg_c, "use_bev", False)),
            subpanel_longside_px=int(getattr(cfg_c, "subpanel_longside_px", 512)),
            bev_resolution_px=int(getattr(cfg_c, "bev_resolution_px", 1024)),
            bev_point_size_px=int(getattr(cfg_c, "bev_point_size_px", 6)),
            marker_namespace=str(getattr(cfg_c, "marker_namespace", "shared_distinct")),
            draw_circles=bool(_g(bg, "draw_circles", False)),
            mosaic_font_scale=float(_g(bg, "mosaic_font_scale", 0.9)),
            mosaic_mask_outline_thickness=int(_g(bg, "mosaic_mask_outline_thickness", 2)),
            crops_enabled=bool(_g(bg, "crops_enabled", True)),
            crops_pad_frac=float(_g(bg, "crops_pad_frac", 0.4)),
            crops_min_side_px=int(_g(bg, "crops_min_side_px", 256)),
            crops_outline_thickness=int(_g(bg, "crops_outline_thickness", 2)),
            crops_panel_px=int(_g(bg, "crops_panel_px", 384)),
            bin_visuals="crops_context",
            context_frame_enabled=bool(_g(bg, "context_frame_enabled", True)),
            enable_thinking_mode=bool(_g(bg, "enable_thinking_mode", False)),
            thinking_max_new_tokens=int(_g(bg, "thinking_max_new_tokens", 3000)),
            per_bin_frame_selection=PerBinFrameSelectionConfig(
                enabled=bool(_g(pbfs, "enabled", True)),
                k_per_bin=int(_g(pbfs, "k_per_bin", 6)),
                top_n_per_iid=int(_g(pbfs, "top_n_per_iid", 8)),
                min_pixel_area=int(_g(pbfs, "min_pixel_area", 200)),
            ),
            pairwise=bool(_g(bg, "pairwise", False)),
            cross_bin_stitch=bool(_g(bg, "cross_bin_stitch", True)),
            cross_bin_max_per_side=int(_g(bg, "cross_bin_max_per_side", 15)),
        )

        gc = _g(bg, "geo_correction", {}) or {}
        resolver = GeoCorrectionResolver(
            lam=float(_g(gc, "lam", 1.0)),
            tau=float(_g(gc, "tau", 0.45)),
            k=int(_g(gc, "k", 8)),
            bypass=bool(_g(gc, "bypass", True)),
            top_k=int(_g(gc, "top_k", 0)),
        )

        temp = getattr(cfg_c, "temperature", None)
        invoker = VLMInvoker(
            vlm_raw,
            max_new_tokens=int(getattr(cfg_c, "vlm_max_new_tokens", 2000)),
            enable_thinking=bool(block_cfg.enable_thinking_mode),
            temperature=(float(temp) if temp is not None else None),
        )

        prompt_cls = PairwiseBinCropsPrompt if block_cfg.pairwise else BinCropsPrompt
        blk_prompt = prompt_cls(
            enable_thinking=bool(block_cfg.enable_thinking_mode),
            use_bev=bool(block_cfg.use_bev),
            use_context_frame=bool(block_cfg.context_frame_enabled),
            use_crops=bool(block_cfg.crops_enabled),
        )

        postprocessors = []
        if bool(getattr(cfg_c, "do_double_check", True)):
            dc_invoker = VLMInvoker(
                vlm_raw,
                max_new_tokens=int(getattr(cfg_c, "dc_max_new_tokens", 64)),
                enable_thinking=False,
            )
            postprocessors.append(DoubleCheckPostprocessor(
                vlm=dc_invoker,
                enabled=True,
                negation_pass=bool(getattr(cfg_c, "dc_negation_pass", True)),
            ))

        pipeline = CorrespondencePipeline(
            filter=IdentityFilter(),
            views=None,
            visuals=[],
            prompt=blk_prompt,
            parser=TupleJsonParser(),
            resolver=resolver,
            postprocess=postprocessors,
            vlm=invoker,
            blocking_enabled=True,
            blocking_runner=run_blocking_pipeline,
        )
        pipeline.blocking_cfg = block_cfg
        pipeline.bin_visuals = CropsContextBinVisuals()

        outputs: Dict[str, Path] = {}
        try:
            for pair in pairs:
                try:
                    ctx = self._assemble_pair_context(pair, per_subscan, cfg_c)
                    if ctx is None:
                        art = CorrespondenceArtifact(
                            pair_id=pair.pair_id, raw_pairs=[], double_checked_pairs=[],
                        )
                    else:
                        art = pipeline.run_pair(ctx)
                    outputs[pair.pair_id] = art.save(self.dirs["correspondence"])
                except Exception as e:  # noqa: BLE001
                    log.exception("Correspondence failed for %s: %s", pair.pair_id, e)
        finally:
            vlm_raw.close()
            release_gpu()
        return outputs

    def _assemble_pair_context(self, pair, per_subscan, cfg_c):
        from .stages.correspondence.context import FrameSet, InstanceSet, PairContext
        from .utils.instance_fusion import (
            apply_alias_to_instance_map, apply_alias_to_per_frame_masks,
        )

        src_stage1 = load_pickle(per_subscan[pair.src_id]["geometry"])
        ref_stage1 = load_pickle(per_subscan[pair.ref_id]["geometry"])
        prior_src = self._load_prior(per_subscan, pair.src_id)
        prior_ref = self._load_prior(per_subscan, pair.ref_id)
        if prior_src is None or prior_ref is None:
            log.warning("Correspondence [%s]: missing fusion prior; emitting empty artifact", pair.pair_id)
            return None

        src_masks = _unpack_masks(per_subscan[pair.src_id]["segmentation"])
        ref_masks = _unpack_masks(per_subscan[pair.ref_id]["segmentation"])
        src_alias = dict(getattr(prior_src, "iid_alias", {}) or {})
        ref_alias = dict(getattr(prior_ref, "iid_alias", {}) or {})
        if src_alias:
            src_masks = apply_alias_to_per_frame_masks(src_masks, src_alias)
        if ref_alias:
            ref_masks = apply_alias_to_per_frame_masks(ref_masks, ref_alias)

        src_stage3 = load_pickle(per_subscan[pair.src_id]["segmentation"])
        ref_stage3 = load_pickle(per_subscan[pair.ref_id]["segmentation"])
        src_labels = src_stage3.get("instance_to_prompt", {}) or {}
        ref_labels = ref_stage3.get("instance_to_prompt", {}) or {}
        if src_alias:
            src_labels = apply_alias_to_instance_map(src_labels, src_alias, on_conflict="canonical")
        if ref_alias:
            ref_labels = apply_alias_to_instance_map(ref_labels, ref_alias, on_conflict="canonical")

        def _build_side(side, prior, stage1, masks, labels) -> InstanceSet:
            return InstanceSet(
                side=side,
                iids=[int(i) for i in prior.instance_ids],
                centroids={int(k): np.asarray(v, dtype=np.float64) for k, v in prior.instance_centroids.items()},
                obbs={int(k): np.asarray(v, dtype=np.float64) for k, v in prior.bbox_extents.items()},
                rotations={int(k): np.asarray(v, dtype=np.float64) for k, v in prior.bbox_orientation.items()},
                points={int(k): np.asarray(v, dtype=np.float64) for k, v in prior.instance_points.items()},
                label_text={int(k): str(v) for k, v in labels.items()},
                per_frame_masks=masks,
                extra={"points": stage1["points"], "colors": stage1.get("colors")},
            )

        src_iset = _build_side("src", prior_src, src_stage1, src_masks, src_labels)
        ref_iset = _build_side("ref", prior_ref, ref_stage1, ref_masks, ref_labels)

        src_paths = self._frame_paths_dict(pair.src_id)
        ref_paths = self._frame_paths_dict(pair.ref_id)
        src_fset = FrameSet(
            side="src",
            frame_indices=[int(f) for f in src_stage1["frame_ids"]],
            color_paths={int(k): Path(v) for k, v in src_paths.items()},
        )
        ref_fset = FrameSet(
            side="ref",
            frame_indices=[int(f) for f in ref_stage1["frame_ids"]],
            color_paths={int(k): Path(v) for k, v in ref_paths.items()},
        )
        return PairContext(
            pair_id=pair.pair_id,
            src=src_iset, ref=ref_iset,
            src_frames=src_fset, ref_frames=ref_fset,
            up_axis=str(getattr(cfg_c, "up_axis", "y")),
            image_rotation_k=int(getattr(cfg_c, "image_rotation_k", 0) or 0),
            depth_dir=None,
        )

    def _load_prior(self, per_subscan, sid):
        path = per_subscan.get(sid, {}).get("fusion")
        if path is None:
            return None
        try:
            return PriorArtifact.load(path)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load fusion prior %s: %s", path, e)
            return None

    def _frame_paths_dict(self, subscan_id: str) -> Dict[int, Path]:
        frames = self.dataset.load_rgb_frames(subscan_id)
        return {i: p for i, p in enumerate(frames.paths)}

    # ------------------------------------------------------------------
    # Registration (paper Sec. 3.3)
    # ------------------------------------------------------------------
    def _run_registration_for_pair(self, pair, per_subscan, corr_path) -> Optional[RegistrationArtifact]:
        src_art = load_pickle(per_subscan[pair.src_id]["geometry"])
        ref_art = load_pickle(per_subscan[pair.ref_id]["geometry"])

        corrs: List[Tuple[int, int]] = []
        weights = None
        if corr_path is not None:
            payload = load_pickle(corr_path)
            corrs = [tuple(p) for p in payload.get("double_checked_pairs", [])]
            wr = payload.get("correspondence_weights")
            if wr is not None and len(wr) == len(corrs):
                weights = [float(w) for w in wr]

        return run_registration(
            pair_id=pair.pair_id,
            src_points=src_art["points"],
            ref_points=ref_art["points"],
            src_point_to_pixels=src_art["point_to_pixels"],
            ref_point_to_pixels=ref_art["point_to_pixels"],
            src_masks=_unpack_masks(per_subscan[pair.src_id]["segmentation"]) if corr_path else {},
            ref_masks=_unpack_masks(per_subscan[pair.ref_id]["segmentation"]) if corr_path else {},
            correspondences=corrs,
            cfg_registration=self.cfg.registration,
            correspondence_weights=weights,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, pair, per_subscan, reg: RegistrationArtifact) -> Dict[str, float]:
        src_art = load_pickle(per_subscan[pair.src_id]["geometry"])
        ref_art = load_pickle(per_subscan[pair.ref_id]["geometry"])
        src_points = src_art["points"]
        ref_points = ref_art["points"]
        try:
            raw_points = self.dataset.load_raw_points(pair.src_id)
        except (AttributeError, FileNotFoundError):
            raw_points = src_points

        cfg5 = self.cfg.registration
        metrics = compute_all_metrics(
            est_transform=reg.est_transform,
            gt_transform=pair.gt_transform,
            src_points=src_points,
            ref_points=ref_points,
            raw_points=raw_points,
            src_corr_points=reg.src_corr_points,
            ref_corr_points=reg.ref_corr_points,
            gt_src_corr_points=None,
            gt_ref_corr_points=None,
            positive_radius=float(getattr(cfg5, "positive_radius", 0.1)),
            inlier_ratio_thresh=float(getattr(cfg5, "inlier_ratio_thresh", 0.05)),
            rmse_thresh=float(getattr(cfg5, "rmse_thresh", 0.2)),
        )
        return metrics

    def _evaluate_instance(self, pair, per_subscan, corr_path) -> Dict[str, Any]:
        """Correspondence (instance-matching) metrics against a SAM3↔SAM3 IoU GT.

        Computed whenever GT clouds + a correspondence artifact are available
        (no anchor schema required). The discovery block is added only when the
        dataset supplies per-point GT object ids.
        """
        if corr_path is None:
            return {}
        try:
            src_stage1 = load_pickle(per_subscan[pair.src_id]["geometry"])
            ref_stage1 = load_pickle(per_subscan[pair.ref_id]["geometry"])
            src_masks = _unpack_masks(per_subscan[pair.src_id]["segmentation"])
            ref_masks = _unpack_masks(per_subscan[pair.ref_id]["segmentation"])
        except (KeyError, FileNotFoundError) as e:
            log.warning("Instance eval [%s]: prerequisite missing: %s", pair.pair_id, e)
            return {}

        payload = load_pickle(corr_path)
        raw = [tuple(p) for p in payload.get("raw_pairs", [])]
        dc = [tuple(p) for p in payload.get("double_checked_pairs", [])]

        src_clouds = build_sam3_mask_clouds(src_stage1, src_masks)
        ref_clouds = build_sam3_mask_clouds(ref_stage1, ref_masks)
        eval_cfg = getattr(self.cfg, "eval", None)
        gt_iou_method = str(getattr(eval_cfg, "gt_iou_method", "voxel")).lower()
        gt_voxel_size = float(getattr(eval_cfg, "gt_voxel_size", 0.05))
        gt_iou_min = getattr(eval_cfg, "gt_iou_min", None)
        gt_iou_min = float(gt_iou_min) if gt_iou_min is not None else None
        gt_pairs = build_sam3_gt_pairs(
            src_clouds, ref_clouds, pair.gt_transform,
            iou_method=gt_iou_method, voxel_size=gt_voxel_size, min_iou=gt_iou_min,
        )

        out: Dict[str, Any] = {
            "instance": {
                "raw": compute_pair_set_metrics(raw, gt_pairs),
                "dc": compute_pair_set_metrics(dc, gt_pairs),
            }
        }

        # Optional discovery block (only when GT object ids exist).
        if src_stage1.get("object_ids") is not None and ref_stage1.get("object_ids") is not None \
                and pair.anchor_object_ids is not None:
            src_sam2obj = build_sam_to_object_map(src_stage1, src_masks)
            ref_sam2obj = build_sam_to_object_map(ref_stage1, ref_masks)
            out["discovery"] = compute_object_discovery(src_sam2obj, ref_sam2obj, pair.anchor_object_ids)

        log.info(
            "Correspondence eval [%s]: dc P=%.2f R=%.2f F1=%.2f (gt=%d)",
            pair.pair_id, out["instance"]["dc"]["precision"],
            out["instance"]["dc"]["recall"], out["instance"]["dc"]["f1"],
            out["instance"]["dc"]["n_gt"],
        )
        return out

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        pairs = self.dataset.list_subscan_pairs()
        log.info("Dataset %s: %d candidate pairs", self.cfg.dataset.name, len(pairs))
        pairs = _filter_pairs(pairs, self.cfg)
        log.info("After filter: %d pairs", len(pairs))
        if not pairs:
            log.error(
                "No subscan pairs found. Check the anchors file:\n  %s\n"
                "See README.md for dataset download + preprocessing.",
                self.cfg.dataset.anchors_file,
            )
            return

        subscan_ids = sorted({p.src_id for p in pairs} | {p.ref_id for p in pairs})
        t0 = time.perf_counter()
        per_subscan = self.run_scene_parsing(subscan_ids)
        results = self.run_pairs(pairs, per_subscan)
        log.info("=== Total pipeline time: %.1fs ===", time.perf_counter() - t0)

        if not self.cfg.run_evaluation:
            return

        # -- Aggregate --
        per_pair = [r.metrics for r in results]
        reg_agg = aggregate_metrics(per_pair)

        def _collect(block, phase=None):
            out = []
            for r in results:
                inst = r.metrics.get("instance")
                if not inst or block not in inst:
                    continue
                payload = inst[block]
                if phase is not None:
                    payload = payload.get(phase) if isinstance(payload, dict) else None
                if payload is not None:
                    out.append(payload)
            return out

        corr_agg = {}
        for phase in ("raw", "dc"):
            pp = _collect("instance", phase)
            if pp:
                corr_agg[phase] = aggregate_pair_set_metrics(pp)
        disc_pp = _collect("discovery")
        disc_agg = aggregate_discovery_metrics(disc_pp) if disc_pp else {}

        per_pair_reg = {}
        per_pair_corr = {}
        reg_keys = {"valid", "CD", "IR", "FMR", "RRE", "RTE", "recall"}
        for r in results:
            per_pair_reg[r.pair_id] = {k: v for k, v in r.metrics.items() if k in reg_keys}
            inst = r.metrics.get("instance", {})
            if inst:
                per_pair_corr[r.pair_id] = {k: v for k, v in inst.items() if k != "discovery"}

        payload = {
            "registration": {"aggregate": reg_agg, "per_pair": per_pair_reg},
            "correspondence": {"aggregate": corr_agg, "per_pair": per_pair_corr},
            "discovery": {"aggregate": disc_agg},
        }
        dump_json(payload, self.dirs["eval"] / "metrics.json")
        log.info("Aggregated registration: %s", reg_agg)
        if corr_agg:
            log.info("Aggregated correspondence: %s", corr_agg)
