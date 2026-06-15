"""Qwen3-VL wrapper.

Uses the official HuggingFace `Qwen3VLForConditionalGeneration` + `AutoProcessor`
interface (transformers >= 4.52).

Supports both Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Thinking from a single class —
they share the same API; only the chat template and stopping differ, which
`AutoProcessor.apply_chat_template` handles for us.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image

from ..utils.gpu import release_gpu, select_torch_dtype
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


class QwenVLWrapper:
    def __init__(
        self,
        model_id: str,
        *,
        torch_dtype: str = "float16",
        attn_implementation: str = "sdpa",
        device_map: str = "auto",
    ) -> None:
        from transformers import AutoProcessor

        # Qwen3VLForConditionalGeneration is the canonical class.
        # Fall back to AutoModelForImageTextToText for compatibility.
        try:
            from transformers import Qwen3VLForConditionalGeneration as _ModelCls
        except ImportError:  # older transformers
            from transformers import AutoModelForImageTextToText as _ModelCls  # type: ignore

        dtype = select_torch_dtype(torch_dtype)

        log.info(
            "Loading Qwen VL model %s (dtype=%s, attn=%s, device_map=%s)",
            model_id, torch_dtype, attn_implementation, device_map,
        )
        self.model = _ModelCls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model_id = model_id

    def close(self) -> None:
        model = getattr(self, "model", None)
        processor = getattr(self, "processor", None)
        self.model = None
        self.processor = None
        release_gpu(model, processor)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def chat_with_images(
        self,
        images: Sequence[ImageSource],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        extra_system: Optional[str] = None,
    ) -> str:
        """Run a single-turn multi-image chat. Returns the generated text only.

        Supports text-only calls by passing `images=[]` (used by Stage 2's
        consolidation step). New transformers' image processor crashes on an
        empty image list, so we route around it entirely when there are no
        images.
        """
        pil_images = [_to_pil(x) for x in images]

        content: List[dict] = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = []
        if extra_system:
            messages.append({"role": "system", "content": extra_system})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if pil_images:
            proc_kwargs["images"] = pil_images
        inputs = self.processor(**proc_kwargs)

        # Move to the model's primary device.
        primary_device = self._primary_device()
        inputs = {k: v.to(primary_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record_model_call(
            self.model_id, "chat_with_images",
            time.perf_counter() - t0, n_inputs=len(pil_images),
        )

        # Strip the prompt from the generated tokens.
        input_len = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        out_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text

    def _primary_device(self) -> torch.device:
        # With device_map="auto", pick the device of the first param.
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Strict-JSON surfaces — mirror OpenAIVLMWrapper.chat_for_pairs/chat_yes_no.
    # Local Qwen has no token-level schema constraint; we prompt-augment +
    # post-parse with a single retry on parse failure.

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
            log.warning("Qwen chat_for_pairs JSON parse failed (%s); retrying", e)
        retry = self.chat_with_images(
            images, STRICT_RETRY_PREFIX + augmented, max_new_tokens=max_output_tokens,
        )
        try:
            pair_list, _ = parse_pairs_json(retry)
            return pair_list, retry
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("Qwen chat_for_pairs JSON parse failed on retry (%s); returning empty", e)
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
            log.warning("Qwen chat_yes_no JSON parse failed (%s); retrying", e)
        retry = self.chat_with_images(
            images, STRICT_RETRY_PREFIX + augmented, max_new_tokens=max_output_tokens,
        )
        try:
            is_same, _ = parse_yes_no_json(retry)
            return is_same, retry
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("Qwen chat_yes_no JSON parse failed on retry (%s); defaulting to False", e)
            return False, retry
