"""Thin wrapper around the project VLM wrapper's chat-with-images call.

All Stage 4 phases route VLM I/O through this module. The wrapper
itself (HF / vLLM / OpenAI) is constructed by the pipeline driver and
passed in — this module does not know about backends.
"""
from __future__ import annotations

from typing import List, Optional

from PIL import Image


class VLMInvoker:
    def __init__(
        self,
        vlm,
        *,
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
        enable_thinking: bool = False,
    ) -> None:
        if not hasattr(vlm, "chat_with_images"):
            raise ValueError(
                "VLMInvoker requires a wrapper exposing chat_with_images(...)"
            )
        self.vlm = vlm
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = temperature
        self.enable_thinking = bool(enable_thinking)

    def call(
        self,
        images: List[Image.Image],
        system_text: str,
        user_prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        prompt = (
            f"{system_text}\n\n{user_prompt}"
            if system_text else user_prompt
        )
        kwargs = {"max_new_tokens": int(max_new_tokens or self.max_new_tokens)}
        think = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        if think:
            kwargs["enable_thinking"] = True
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        return str(self.vlm.chat_with_images(list(images), prompt, **kwargs))

    def call_batch(
        self,
        items: List[tuple],
        *,
        max_new_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        """Batched multi-conversation call.

        ``items`` is a list of ``(images, system_text, user_prompt)``
        tuples. When the wrapper exposes ``chat_with_images_batch`` (vLLM),
        all conversations go out in one ``LLM.chat`` so the engine applies
        continuous batching. Otherwise falls back to serial ``call`` —
        keeps non-batching backends (HF) working unchanged. Returns the
        raw text per item, in order.
        """
        if not items:
            return []
        think = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        mnt = int(max_new_tokens or self.max_new_tokens)
        if not hasattr(self.vlm, "chat_with_images_batch"):
            return [
                self.call(im, sy, us, max_new_tokens=mnt, enable_thinking=think)
                for im, sy, us in items
            ]
        batch = []
        for images, system_text, user_prompt in items:
            prompt = (
                f"{system_text}\n\n{user_prompt}" if system_text else user_prompt
            )
            batch.append((list(images), prompt))
        kwargs = {"max_new_tokens": mnt, "enable_thinking": think}
        if self.temperature is not None and float(self.temperature) > 0.0:
            kwargs["do_sample"] = True
        return [str(t) for t in self.vlm.chat_with_images_batch(batch, **kwargs)]
