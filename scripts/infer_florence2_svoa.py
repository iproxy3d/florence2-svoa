#!/usr/bin/env python3
"""Run a fine-tuned Florence-2 SVOA model on one or more images.

Example:

    python infer_florence2_svoa.py ^
      --model ./florence2_svoa_smoke/best_model ^
      --image ./example.jpg

Several images and an optional JSONL output:

    python infer_florence2_svoa.py ^
      --model ./florence2_svoa_smoke/best_model ^
      --image ./one.jpg ^
      --image ./two.jpg ^
      --output-jsonl ./predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TASK_PROMPT = (
    "Extract the main event or communicative gesture as "
    "subject|action|object|adjuncts:"
)
DEFAULT_FIELDS = ("subject", "action", "object", "adjuncts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="input image; repeat this option for several images",
    )
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--task-prompt",
        help="override the prompt stored in the trained model config",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
    )
    return parser.parse_args()


def choose_device(requested: str, torch_module: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)


def parse_prediction(text: str, fields: tuple[str, ...]) -> dict[str, Any]:
    parts = [
        " ".join(part.strip().lower().split())
        for part in text.strip().split("|")
    ]
    normalized = "|".join(parts)
    result: dict[str, Any] = {
        "prediction": text.strip(),
        "normalized": normalized,
        "valid_format": len(parts) == len(fields) and all(parts),
    }
    if result["valid_format"]:
        result.update(dict(zip(fields, parts)))
    return result


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def load_florence_config(
    model_path: Path,
    auto_config_class: Any,
    pretrained_config_class: Any,
) -> Any:
    """Load a checkpoint while repairing old Florence-2 vision metadata.

    Some Florence-2 remote-code revisions serialize the nested vision config
    as a generic PretrainedConfig and lose ``model_type='davit'``. The weights
    remain valid, but modeling_florence2.py rejects the config before loading
    them. Repairing the in-memory config is sufficient for existing models.
    """

    config = auto_config_class.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise RuntimeError(
            f"Florence-2 checkpoint has no vision_config: {model_path / 'config.json'}"
        )
    if isinstance(vision_config, dict):
        vision_config = pretrained_config_class(**vision_config)
        config.vision_config = vision_config
    if getattr(vision_config, "model_type", None) != "davit":
        vision_config.model_type = "davit"
        print(
            "repaired Florence-2 vision_config.model_type='davit' in memory; "
            "model weights were not changed"
        )
    return config


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    image_paths = [path.expanduser().resolve() for path in args.image]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"image does not exist: {missing[0]}")
    if args.max_new_tokens < 4 or args.num_beams < 1:
        raise ValueError("generation parameters must be positive")

    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            PretrainedConfig,
        )
    except ImportError as exc:
        raise SystemExit(f"missing inference dependency: {exc}") from exc

    device = choose_device(args.device, torch)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    config = load_florence_config(model_path, AutoConfig, PretrainedConfig)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    task_prompt = (
        args.task_prompt
        or getattr(model.config, "svoa_task_prompt", None)
        or DEFAULT_TASK_PROMPT
    )
    configured_fields = getattr(model.config, "svoa_output_fields", DEFAULT_FIELDS)
    fields = tuple(str(field) for field in configured_fields)
    records: list[dict[str, Any]] = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            inputs = processor(
                text=task_prompt,
                images=rgb,
                return_tensors="pt",
            )
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
            elif key == "pixel_values":
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)

        generation_inputs = {
            key: value
            for key, value in moved.items()
            if key in {"input_ids", "pixel_values"}
        }
        # Legacy Florence-2 remote code merges image and text embeddings inside
        # generate(), but does not replace a caller-supplied text-only mask in
        # **kwargs. Omitting it lets the language model build a mask for the
        # complete image+text sequence (for example 595 tokens instead of 18).
        with torch.inference_mode():
            generated_ids = model.generate(
                **generation_inputs,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
            )
        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        record = {
            "image_path": image_path.as_posix(),
            **parse_prediction(prediction, fields),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, records)
        print(f"saved: {args.output_jsonl.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
