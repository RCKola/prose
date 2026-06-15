"""vLLM-backed Qwen3-VL wrapper.

Drop-in replacement for `QwenVLWrapper` that uses vLLM's offline `LLM` engine
instead of HuggingFace `generate`. Exposes the same surface
(`chat_with_images`, `chat_for_pairs`, `chat_yes_no`, `close`) so it can be
swapped in by selecting `vlm_backend: vllm_local` at the Stage-4 config layer.

Requires the `prose_vllm.sqsh` container (EDF: `prose_vllm.toml`). The
default `prose.sqsh` image does NOT include vLLM. Note that the vLLM image
also does NOT include GeoTransformer — runs using this wrapper must split
Stage 5 fallback into a second job using the canonical prose image. See
CLAUDE.md "Optional vLLM image (Stage 4 only)" for the two-job pattern.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

from PIL import Image

from ..utils.logging import get_logger
from ..utils.profiling import record_model_call

log = get_logger(__name__)

ImageSource = Union[str, Path, Image.Image]


def _to_pil(src: ImageSource) -> Image.Image:
    if isinstance(src, (str, Path)):
        return Image.open(src).convert("RGB")
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    raise TypeError(f"Unsupported image source: {type(src)}")


class VllmVLMWrapper:
    """Multimodal vLLM wrapper for Qwen3-VL (and any vLLM-supported VLM).

    Uses vLLM's offline `LLM.chat([messages], sampling_params)` interface with
    OpenAI-style multimodal message content. One instance per process; a
    single GPU is dedicated to the engine.
    """

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        limit_mm_per_prompt_image: int = 16,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        trust_remote_code: bool = True,
        max_num_seqs: Optional[int] = None,
    ) -> None:
        from vllm import LLM  # imported lazily — only the vllm image has it

        log.info(
            "Loading vLLM engine model=%s dtype=%s tp=%d gpu_mem_util=%.2f max_model_len=%s max_num_seqs=%s",
            model_id, dtype, tensor_parallel_size, gpu_memory_utilization,
            max_model_len, max_num_seqs,
        )
        engine_kwargs = dict(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt={"image": int(limit_mm_per_prompt_image)},
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = int(max_model_len)
        # max_num_seqs caps the per-step decode batch. Required for hybrid
        # (Mamba/SSM) models like Qwen3.6-27B: vLLM allocates one Mamba cache
        # block per concurrent decode sequence, and the default of 1024 will
        # exceed available blocks unless gpu_memory_utilization is raised
        # accordingly. We only run batch-1 calls, so 8 is plenty.
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = int(max_num_seqs)
        self.llm = LLM(**engine_kwargs)
        self.model_id = model_id

    def close(self) -> None:
        # vLLM owns the GPU memory; releasing requires del + gc. Best effort.
        llm = getattr(self, "llm", None)
        self.llm = None
        if llm is not None:
            try:
                del llm
            except Exception:  # noqa: BLE001
                pass
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _pil_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _build_messages(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        extra_system: Optional[str],
    ) -> list:
        # `image_url` with a base64 data URI is the universal chat-content
        # type accepted by vLLM's offline LLM.chat across every supported
        # multimodal model (Qwen3-VL, InternVL3, etc.). The older
        # `image_pil` content type silently drops images for some models
        # (InternVL3 emits token-soup gibberish), so use this format
        # everywhere.
        content: List[dict] = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._pil_to_data_url(img)},
            })
        content.append({"type": "text", "text": prompt})
        messages = []
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        messages.append({"role": "user", "content": content})
        return messages

    def chat_with_images(
        self,
        images: Sequence[ImageSource],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        extra_system: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> str:
        """Single-turn multi-image chat. Returns the generated text only.

        Mirrors `QwenVLWrapper.chat_with_images`. `do_sample=False` maps to
        `temperature=0`. Empty image list is supported (text-only chat).
        `enable_thinking=True` opts into Qwen3-VL's `<think>...</think>`
        CoT — callers must pass a `max_new_tokens` large enough to cover
        the CoT + answer (≥ 4096) and strip the `<think>` block from the
        returned string themselves.
        """
        from vllm import SamplingParams

        pil_images = [_to_pil(x) for x in images]
        messages = self._build_messages(pil_images, prompt, extra_system)
        sampling = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=0.0 if not do_sample else 1.0,
        )

        t0 = time.perf_counter()
        # Default is thinking-off: prior runs showed thinking-on with a
        # 512-token budget burned the entire allowance on <think> and
        # emitted no parseable answer. Callers that pass
        # `enable_thinking=True` are responsible for bumping max_new_tokens
        # and parsing the CoT block out of the response. Harmless for
        # non-thinking models — the kwarg is silently dropped when the
        # chat template doesn't reference it.
        outputs = self.llm.chat(
            [messages],
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": bool(enable_thinking)},
        )
        elapsed = time.perf_counter() - t0
        record_model_call(
            self.model_id, "chat_with_images", elapsed, n_inputs=len(pil_images),
        )

        out = outputs[0].outputs[0]
        return str(out.text)

    def chat_with_images_batch(
        self,
        items: Sequence[tuple],
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        extra_system: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> List[str]:
        """Batched multi-conversation chat.

        `items` is a sequence of `(images, prompt)` tuples. Submits all
        conversations to vLLM in a single `LLM.chat` call so the engine
        can apply continuous batching across them. Returns the generated
        text per item in the same order.
        """
        from vllm import SamplingParams

        if not items:
            return []
        conversations = []
        for images, prompt in items:
            pil_images = [_to_pil(x) for x in images]
            conversations.append(self._build_messages(pil_images, prompt, extra_system))
        sampling = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=0.0 if not do_sample else 1.0,
        )
        t0 = time.perf_counter()
        outputs = self.llm.chat(
            conversations,
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": bool(enable_thinking)},
        )
        elapsed = time.perf_counter() - t0
        record_model_call(
            self.model_id, "chat_with_images_batch", elapsed, n_inputs=len(items),
        )
        return [str(o.outputs[0].text) for o in outputs]

    def chat_yes_no_batch(
        self,
        items: Sequence[tuple],
        *,
        max_output_tokens: int = 64,
    ) -> List[tuple]:
        """Batched yes/no chat with strict-JSON parsing + per-item retry.

        Returns a list of `(is_same: bool, raw_text: str)` parallel to
        `items`. Items whose first response fails to parse are collected
        into a single retry batch under the `STRICT_RETRY_PREFIX` wrapper;
        items still failing after retry default to `False`.
        """
        from ._json_chat import (
            STRICT_RETRY_PREFIX,
            YES_NO_SCHEMA_INSTRUCTION,
            parse_yes_no_json,
        )

        if not items:
            return []
        augmented = [(imgs, prompt + YES_NO_SCHEMA_INSTRUCTION) for imgs, prompt in items]
        texts = self.chat_with_images_batch(augmented, max_new_tokens=max_output_tokens)

        results: List[Optional[tuple]] = [None] * len(items)
        retry_idx: List[int] = []
        for i, text in enumerate(texts):
            try:
                is_same, _ = parse_yes_no_json(text)
                results[i] = (bool(is_same), text)
            except (ValueError, json.JSONDecodeError):
                retry_idx.append(i)
        if retry_idx:
            log.warning("vLLM chat_yes_no_batch: %d/%d items failed first-pass JSON; retrying",
                        len(retry_idx), len(items))
            retry_items = [
                (augmented[i][0], STRICT_RETRY_PREFIX + augmented[i][1])
                for i in retry_idx
            ]
            retry_texts = self.chat_with_images_batch(
                retry_items, max_new_tokens=max_output_tokens,
            )
            for j, i in enumerate(retry_idx):
                text = retry_texts[j]
                try:
                    is_same, _ = parse_yes_no_json(text)
                    results[i] = (bool(is_same), text)
                except (ValueError, json.JSONDecodeError):
                    results[i] = (False, text)
        return [r if r is not None else (False, "") for r in results]

    # Strict-JSON surfaces — mirror QwenVLWrapper.chat_for_pairs/chat_yes_no.

    def chat_for_pairs(
        self,
        images: Sequence[ImageSource],
        prompt: str,
        *,
        max_output_tokens: int = 4000,
        with_relations: bool = False,
        explicit_cot: bool = False,
    ) -> tuple:
        from ._json_chat import (
            PAIRS_EXPLICIT_COT_SCHEMA_INSTRUCTION,
            PAIRS_SCHEMA_INSTRUCTION,
            PAIRS_WITH_RELATIONS_SCHEMA_INSTRUCTION,
            STRICT_RETRY_PREFIX,
            parse_pairs_json,
        )

        if explicit_cot:
            schema_instruction = PAIRS_EXPLICIT_COT_SCHEMA_INSTRUCTION
        elif with_relations:
            schema_instruction = PAIRS_WITH_RELATIONS_SCHEMA_INSTRUCTION
        else:
            schema_instruction = PAIRS_SCHEMA_INSTRUCTION
        augmented = prompt + schema_instruction
        text = self.chat_with_images(images, augmented, max_new_tokens=max_output_tokens)
        try:
            pair_list, _ = parse_pairs_json(text)
            return pair_list, text
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("vLLM chat_for_pairs JSON parse failed (%s); retrying", e)
        retry = self.chat_with_images(
            images, STRICT_RETRY_PREFIX + augmented, max_new_tokens=max_output_tokens,
        )
        try:
            pair_list, _ = parse_pairs_json(retry)
            return pair_list, retry
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("vLLM chat_for_pairs JSON parse failed on retry (%s); returning empty", e)
            return [], retry

    def chat_yes_no(
        self,
        images: Sequence[ImageSource],
        prompt: str,
        *,
        max_output_tokens: int = 1000,
    ) -> tuple:
        from ._json_chat import (
            STRICT_RETRY_PREFIX,
            YES_NO_SCHEMA_INSTRUCTION,
            parse_yes_no_json,
        )

        augmented = prompt + YES_NO_SCHEMA_INSTRUCTION
        text = self.chat_with_images(images, augmented, max_new_tokens=max_output_tokens)
        try:
            is_same, _ = parse_yes_no_json(text)
            return is_same, text
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("vLLM chat_yes_no JSON parse failed (%s); retrying", e)
        retry = self.chat_with_images(
            images, STRICT_RETRY_PREFIX + augmented, max_new_tokens=max_output_tokens,
        )
        try:
            is_same, _ = parse_yes_no_json(retry)
            return is_same, retry
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("vLLM chat_yes_no JSON parse failed on retry (%s); defaulting to False", e)
            return False, retry
