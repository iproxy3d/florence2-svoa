"""Programmatic Florence-2 SVOA inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .parsing import ParseResult, parse_generation


DEFAULT_TASK_PROMPT = (
    "Extract the main event or communicative gesture as "
    "subject|action|object|adjuncts:"
)


class Florence2SVOA:
    """Load a fine-tuned checkpoint once and reuse it for many images."""

    def __init__(
        self,
        model: str | Path,
        *,
        device: str = "auto",
        task_prompt: str | None = None,
        num_beams: int = 3,
        max_new_tokens: int = 32,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install inference dependencies: pip install -e '.[inference]'"
            ) from exc

        selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if selected == "auto":
            selected = "cpu"
        if selected == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        self.torch = torch
        self.device = torch.device(selected)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        config_prompt = getattr(self.model.config, "svoa_task_prompt", None)
        self.task_prompt = task_prompt or config_prompt or DEFAULT_TASK_PROMPT
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

    def predict(self, image: str | Path | Any) -> ParseResult:
        from PIL import Image

        close_image = False
        if isinstance(image, (str, Path)):
            image = Image.open(image)
            close_image = True
        try:
            rgb = image.convert("RGB")
            inputs = self.processor(
                text=self.task_prompt,
                images=rgb,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
            text = self.processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            return parse_generation(text)
        finally:
            if close_image:
                image.close()
