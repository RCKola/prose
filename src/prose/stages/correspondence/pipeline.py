"""Correspondence pipeline orchestrator.

The canonical per-pair pipeline is:

    A. assemble PairContext                (pipeline driver, outside this class)
    B. filter.apply(ctx)
    C. views.select(ctx)
    D. images = [v.compose(ctx) for v in visuals]
    E. prompt = prompt.build(ctx, images)
    F. raw   = vlm.call(images, prompt.system, prompt.user)
       vlm_result = parser.parse(raw, ctx)
    G. result = resolver.resolve(vlm_result, ctx)
       for p in postprocess: result = p.apply(result, ctx)

When ``blocking_enabled`` is set, ``run_pair`` delegates to
``run_blocking_pipeline`` which owns phases C–F per height bin and
returns an aggregated result. This is the path the default ADT config
takes (``blocking.enabled: true``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)

from .artifact import CorrespondenceArtifact
from .context import (
    CorrespondenceResult,
    PairContext,
    PromptBundle,
    VLMResult,
)
from .vlm.invoker import VLMInvoker


@dataclass
class CorrespondencePipeline:
    """Composition of phase objects + a VLM invoker.

    The pipeline driver (``prose.pipeline.Pipeline``) is responsible for
    assembling the ``PairContext`` from the Stage 1/3/4 caches (geometry,
    segmentation, fusion) and calling ``run_pair`` once per pair. It also
    owns the Hydra-side construction of the phase objects.
    """

    filter: Any                          # InstanceFilter
    views: Any                           # ViewSelector
    visuals: List[Any]                   # List[VisualComposer]
    prompt: Any                          # PromptBuilder
    parser: Any                          # ResponseParser
    resolver: Any                        # CorrespondenceResolver
    postprocess: List[Any]               # List[Postprocessor]
    vlm: VLMInvoker

    # When True, run_pair delegates to blocking_runner (phases C-F per bin).
    blocking_enabled: bool = False
    blocking_runner: Optional[Callable[["CorrespondencePipeline", PairContext], CorrespondenceResult]] = None

    # Optional post-DC geo correction resolver.
    geo_after_dc_resolver: Optional[Any] = None

    # CLI-only: when set, saves mosaic image + prompt + raw VLM text per pair.
    # Not in any YAML default; pass via correspondence.save_vlm_samples_to=<dir>.
    vlm_sample_dir: Optional[Path] = None

    def run_pair(self, ctx: PairContext) -> CorrespondenceArtifact:
        ctx = self.filter.apply(ctx)

        if self.blocking_enabled:
            if self.blocking_runner is None:
                raise RuntimeError(
                    "blocking_enabled=True but no blocking_runner provided"
                )
            result = self.blocking_runner(self, ctx)
        else:
            ctx = self.views.select(ctx)
            images = []
            for v in self.visuals:
                images.extend(v.compose(ctx))
            prompt: PromptBundle = self.prompt.build(ctx, images)
            raw = self.vlm.call(images, prompt.system, prompt.user)
            if self.vlm_sample_dir is not None:
                _save_vlm_sample(self.vlm_sample_dir, ctx.pair_id, images, prompt, raw)
            vlm_result: VLMResult = self.parser.parse(raw, ctx)
            result = self.resolver.resolve(vlm_result, ctx)

        pre_dc_pairs = list(result.pairs)

        for p in self.postprocess:
            result = p.apply(result, ctx)

        if self.geo_after_dc_resolver is not None:
            vlm_for_geo = VLMResult(
                raw_text="",
                proposals=list(result.pairs),
                audit=result.audit,
            )
            result = self.geo_after_dc_resolver.resolve(vlm_for_geo, ctx)

        return self._to_artifact(ctx, result, pre_dc_pairs=pre_dc_pairs)

    @staticmethod
    def _to_artifact(
        ctx: PairContext,
        result: CorrespondenceResult,
        *,
        pre_dc_pairs: Optional[List] = None,
    ) -> CorrespondenceArtifact:
        return CorrespondenceArtifact(
            pair_id=ctx.pair_id,
            raw_pairs=list(pre_dc_pairs) if pre_dc_pairs is not None else list(result.pairs),
            double_checked_pairs=list(result.pairs),
            vlm_double_checked_pairs=list(result.pairs),
            shape_anchor_pairs=[],
            fallback_used=False,
            correspondence_weights=(
                list(result.weights) if result.weights is not None else None
            ),
            mosaic={"audit": ctx.audit, "resolver_audit": result.audit},
        )


def _save_vlm_sample(
    sample_dir: Path,
    pair_id: str,
    images: List[Any],
    prompt: PromptBundle,
    raw: str,
) -> None:
    """Save mosaic image(s), prompt text, and raw VLM response for one pair."""
    try:
        out = Path(sample_dir) / str(pair_id).replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        (out / "prompt.txt").write_text(
            f"=== SYSTEM ===\n{prompt.system or ''}\n\n=== USER ===\n{prompt.user or ''}\n",
            encoding="utf-8",
        )
        for i, img in enumerate(images):
            try:
                img.save(out / f"mosaic_{i:02d}.jpg")
            except Exception as exc:  # noqa: BLE001
                log.debug("sample dump: image %d: %s", i, exc)
        (out / "vlm_raw.txt").write_text(raw or "", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        log.warning("VLM sample dump failed for %s: %s", pair_id, exc)
