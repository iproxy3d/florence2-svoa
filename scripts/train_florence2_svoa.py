#!/usr/bin/env python3
"""Fine-tune microsoft/Florence-2-base-ft for image -> SVOA generation.

The input is dataset.jsonl produced by prepare_svoa_dataset.py. Every record
must contain image_path, subject, action, object, and adjuncts. The target text
is a compact deterministic sequence:

    subject|action|object|adjuncts

The script creates reproducible train/validation/test JSONL files, trains the
model, stores the best validation checkpoint, and writes test predictions and
field-level accuracy metrics. The same command automatically resumes from the
last completed epoch in OUTPUT_DIR. It appends raw and rolling-window loss to
loss_history.csv. Pass --restart only when a genuinely new run is required.

Install (choose the PyTorch build appropriate for your CUDA installation):

    python -m pip install torch torchvision transformers accelerate pillow tqdm timm einops safetensors

Smoke test with the small dataset:

    python train_florence2_svoa.py ^
      --dataset-jsonl ./svoa_work/dataset.jsonl ^
      --output-dir ./florence2_svoa_smoke ^
      --epochs 1 ^
      --batch-size 1 ^
      --gradient-accumulation 4

Only prepare and inspect the split files without loading Florence-2:

    python train_florence2_svoa.py ^
      --dataset-jsonl ./svoa_work/dataset.jsonl ^
      --output-dir ./florence2_svoa_smoke ^
      --prepare-only
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import sys
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_MODEL = "microsoft/Florence-2-base-ft"
DEFAULT_TASK_PROMPT = (
    "Extract the main event or communicative gesture as "
    "subject|action|object|adjuncts:"
)
FIELDS = ("subject", "action", "object", "adjuncts")
LOSS_LOG_FIELDS = (
    "timestamp_utc",
    "session_id",
    "event",
    "epoch",
    "batch_in_epoch",
    "global_batch_step",
    "optimizer_step",
    "raw_loss",
    "rolling_loss",
    "epoch_train_loss",
    "validation_loss",
    "learning_rate",
    "next_learning_rate",
)


@dataclass(frozen=True)
class Example:
    image_id: str
    image_path: Path
    subject: str
    action: str
    object: str
    adjuncts: str

    @property
    def target(self) -> str:
        return "|".join(getattr(self, field) for field in FIELDS)

    @property
    def dataset(self) -> str:
        return self.image_id.split(":", 1)[0] if ":" in self.image_id else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="maximum epochs; early stopping may finish sooner",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-8)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--loss-window",
        type=int,
        default=20,
        help="window size for the rolling training-loss average written to CSV",
    )
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="auto selects CUDA when available",
    )
    parser.add_argument(
        "--unfreeze-vision",
        action="store_true",
        help="train the vision encoder too; needs more memory and more data",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="reduce memory usage at the cost of additional compute",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate data and write split files without loading the model",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard this output directory's training checkpoint/log and train anew",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise SystemExit("batch size and gradient accumulation must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if not 0 <= args.validation_ratio < 1 or not 0 <= args.test_ratio < 1:
        raise SystemExit("validation and test ratios must be in [0, 1)")
    if args.validation_ratio + args.test_ratio >= 1:
        raise SystemExit("validation-ratio + test-ratio must be less than 1")
    if not 0 < args.lr_factor < 1:
        raise SystemExit("--lr-factor must be in (0, 1)")
    if args.lr_patience < 0:
        raise SystemExit("--lr-patience must be non-negative")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        raise SystemExit("--min-learning-rate must be in (0, learning-rate]")
    if args.early_stopping_patience <= args.lr_patience:
        raise SystemExit(
            "--early-stopping-patience must be greater than --lr-patience "
            "so LR can decrease before training stops"
        )
    if args.early_stopping_min_delta < 0:
        raise SystemExit("--early-stopping-min-delta must be non-negative")
    if args.loss_window < 1:
        raise SystemExit("--loss-window must be at least 1")
    if args.max_target_length < 4 or args.max_new_tokens < 4:
        raise SystemExit("target and generation lengths are too small")


def normalize_word(value: Any, field: str, line_number: int) -> str:
    word = " ".join(str(value).strip().lower().split())
    has_forbidden_space = field != "action" and " " in word
    if not word or has_forbidden_space or "|" in word:
        requirement = (
            "a non-empty label, optionally containing spaces"
            if field == "action"
            else "one non-empty word"
        )
        raise ValueError(
            f"line {line_number}: {field} must be {requirement} without '|': {word!r}"
        )
    return word


def load_examples(dataset_jsonl: Path) -> list[Example]:
    dataset_jsonl = dataset_jsonl.expanduser().resolve()
    if not dataset_jsonl.is_file():
        raise FileNotFoundError(dataset_jsonl)

    examples: list[Example] = []
    seen_ids: set[str] = set()
    with dataset_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc

            image_id = str(row.get("image_id", "")).strip()
            raw_image_path = str(row.get("image_path", "")).strip()
            if not image_id or not raw_image_path:
                raise ValueError(f"line {line_number}: image_id and image_path are required")
            if image_id in seen_ids:
                raise ValueError(f"line {line_number}: duplicate image_id {image_id!r}")
            seen_ids.add(image_id)

            image_path = Path(raw_image_path).expanduser()
            if not image_path.is_absolute():
                image_path = (dataset_jsonl.parent / image_path).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"line {line_number}: image does not exist: {image_path}"
                )

            values = {
                field: normalize_word(row.get(field, ""), field, line_number)
                for field in FIELDS
            }
            examples.append(
                Example(
                    image_id=image_id,
                    image_path=image_path,
                    **values,
                )
            )

    if len(examples) < 3:
        raise ValueError("at least three examples are required for train/validation/test")
    return examples


def holdout_size(total: int, ratio: float) -> int:
    if ratio == 0:
        return 0
    return max(1, int(round(total * ratio)))


def split_examples(
    examples: Sequence[Example],
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Example], list[Example], list[Example]]:
    def split_group(
        group: Sequence[Example],
        group_seed: int,
    ) -> tuple[list[Example], list[Example], list[Example]]:
        ordered = sorted(group, key=lambda item: item.image_id)
        random.Random(group_seed).shuffle(ordered)
        validation_count = holdout_size(len(ordered), validation_ratio)
        test_count = holdout_size(len(ordered), test_ratio)
        while validation_count + test_count >= len(ordered):
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("dataset is too small for the requested split ratios")
        test_part = ordered[:test_count]
        validation_part = ordered[test_count : test_count + validation_count]
        train_part = ordered[test_count + validation_count :]
        return train_part, validation_part, test_part

    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.dataset, []).append(example)

    # With at least three samples per source, stratify every source separately.
    # If this is only a seven-image smoke test (one sample per source), use one
    # global split instead of putting entire sources exclusively in holdouts.
    if all(len(group) < 3 for group in groups.values()):
        train, validation, test = split_group(examples, seed)
    else:
        train = []
        validation = []
        test = []
        small_groups: list[Example] = []
        for dataset, group in sorted(groups.items()):
            if len(group) < 3:
                small_groups.extend(group)
                continue
            digest = hashlib.sha256(f"{seed}:{dataset}".encode("utf-8")).digest()
            group_seed = int.from_bytes(digest[:8], "big")
            train_part, validation_part, test_part = split_group(group, group_seed)
            train.extend(train_part)
            validation.extend(validation_part)
            test.extend(test_part)
        train.extend(small_groups)
        random.Random(seed + 1).shuffle(train)
        random.Random(seed + 2).shuffle(validation)
        random.Random(seed + 3).shuffle(test)

    if not train or not validation or not test:
        raise ValueError(
            "train, validation, and test must all be non-empty; use positive "
            "validation/test ratios and at least three examples"
        )
    return train, validation, test


def portable_record(example: Example, dataset_root: Path) -> dict[str, str]:
    try:
        image_path = example.image_path.relative_to(dataset_root).as_posix()
    except ValueError:
        image_path = example.image_path.as_posix()
    return {
        "image_id": example.image_id,
        "image_path": image_path,
        "subject": example.subject,
        "action": example.action,
        "object": example.object,
        "adjuncts": example.adjuncts,
        "target": example.target,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def patch_saved_florence_config(
    model_dir: Path,
    task_prompt: str,
    output_fields: Sequence[str],
) -> None:
    """Make a saved Florence-2 checkpoint reloadable across remote-code versions."""

    config_path = model_dir / "config.json"
    config = read_json(config_path)
    vision_config = config.get("vision_config")
    if not isinstance(vision_config, dict):
        raise RuntimeError(f"saved Florence-2 config lacks vision_config: {config_path}")
    vision_config["model_type"] = "davit"
    config["svoa_task_prompt"] = task_prompt
    config["svoa_output_fields"] = list(output_fields)
    write_json(config_path, config)


def load_florence_config(
    model_source: str | Path,
    auto_config_class: Any,
    pretrained_config_class: Any,
) -> Any:
    """Repair old Florence-2 vision metadata before loading model weights."""

    config = auto_config_class.from_pretrained(
        model_source,
        trust_remote_code=True,
    )
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise RuntimeError(f"Florence-2 model has no vision_config: {model_source}")
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def resume_configuration(args: argparse.Namespace, dataset_sha256: str) -> dict[str, Any]:
    """Parameters that must stay identical for a mathematically valid resume."""
    return {
        "dataset_sha256": dataset_sha256,
        "model": args.model,
        "task_prompt": args.task_prompt,
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "min_learning_rate": args.min_learning_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "loss_window": args.loss_window,
        "max_target_length": args.max_target_length,
        "max_grad_norm": args.max_grad_norm,
        "vision_frozen": not args.unfreeze_vision,
        "gradient_checkpointing": args.gradient_checkpointing,
    }


def validate_resume_configuration(
    saved: dict[str, Any], current: dict[str, Any]
) -> None:
    differences = [
        key
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]
    if differences:
        details = ", ".join(
            f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
            for key in differences
        )
        raise RuntimeError(
            "training settings or dataset changed; automatic resume would be invalid: "
            f"{details}. Use a different --output-dir or pass --restart."
        )


def reset_training_artifacts(output_dir: Path) -> None:
    """Remove only files owned by this trainer, never the output root itself."""
    for directory_name in (
        "last_checkpoint",
        "last_checkpoint.tmp",
        "last_checkpoint.previous",
        "best_model",
    ):
        directory = output_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)
    for file_name in (
        "loss_history.csv",
        "training_summary.json",
        "test_predictions.jsonl",
        "test_metrics.json",
    ):
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def dataset_counts(examples: Sequence[Example]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.dataset] = counts.get(example.dataset, 0) + 1
    return dict(sorted(counts.items()))


def save_splits(
    output_dir: Path,
    dataset_root: Path,
    train: Sequence[Example],
    validation: Sequence[Example],
    test: Sequence[Example],
    seed: int,
) -> None:
    split_dir = output_dir / "splits"
    for name, examples in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        write_jsonl(
            split_dir / f"{name}.jsonl",
            (portable_record(example, dataset_root) for example in examples),
        )
    metadata = {
        "seed": seed,
        "train": {"size": len(train), "datasets": dataset_counts(train)},
        "validation": {
            "size": len(validation),
            "datasets": dataset_counts(validation),
        },
        "test": {"size": len(test), "datasets": dataset_counts(test)},
    }
    (split_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def set_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def choose_device(requested: str, torch_module: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch_module.device(requested)


def find_vision_tower(model: Any) -> Any | None:
    if hasattr(model, "vision_tower"):
        return model.vision_tower
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "vision_tower"):
        return inner.vision_tower
    return None


def freeze_vision_encoder(model: Any) -> int:
    tower = find_vision_tower(model)
    if tower is None:
        raise RuntimeError(
            "Florence-2 vision_tower was not found; use --unfreeze-vision only "
            "after checking the installed Transformers/model implementation"
        )
    frozen = 0
    for parameter in tower.parameters():
        parameter.requires_grad_(False)
        frozen += parameter.numel()
    return frozen


def trainable_parameters(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def normalize_prediction(text: str) -> str:
    # Keep spaces inside a field because HaGRID gesture labels such as
    # "little finger" and "take picture" are intentional.  Only normalize
    # whitespace around and inside the four pipe-delimited fields.
    return "|".join(
        " ".join(part.strip().lower().split()) for part in text.strip().split("|")
    )


def field_metrics(targets: Sequence[str], predictions: Sequence[str]) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions have different lengths")
    total = len(targets)
    exact = 0
    valid_format = 0
    correct = {field: 0 for field in FIELDS}
    for target, prediction in zip(targets, predictions):
        target_parts = normalize_prediction(target).split("|")
        prediction_parts = normalize_prediction(prediction).split("|")
        if prediction_parts == target_parts:
            exact += 1
        if len(prediction_parts) == len(FIELDS):
            valid_format += 1
            for index, field in enumerate(FIELDS):
                if prediction_parts[index] == target_parts[index]:
                    correct[field] += 1
    denominator = max(total, 1)
    return {
        "examples": total,
        "exact_match": exact / denominator,
        "valid_format": valid_format / denominator,
        "field_accuracy": {
            field: correct[field] / denominator for field in FIELDS
        },
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_jsonl = args.dataset_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.restart:
        reset_training_artifacts(output_dir)
        print("restart requested: previous checkpoint and loss history removed")

    examples = load_examples(dataset_jsonl)
    train_examples, validation_examples, test_examples = split_examples(
        examples,
        args.validation_ratio,
        args.test_ratio,
        args.seed,
    )
    save_splits(
        output_dir,
        dataset_jsonl.parent,
        train_examples,
        validation_examples,
        test_examples,
        args.seed,
    )
    print(
        f"split: train={len(train_examples)}, "
        f"validation={len(validation_examples)}, test={len(test_examples)}"
    )
    print(f"dataset counts: {dataset_counts(examples)}")
    if args.prepare_only:
        print(f"split files written to {output_dir / 'splits'}")
        return 0

    dataset_sha256 = file_sha256(dataset_jsonl)
    current_resume_config = resume_configuration(args, dataset_sha256)
    last_checkpoint_dir = output_dir / "last_checkpoint"
    checkpoint_temporary_dir = output_dir / "last_checkpoint.tmp"
    checkpoint_backup_dir = output_dir / "last_checkpoint.previous"

    def checkpoint_is_complete(directory: Path) -> bool:
        return (
            directory.is_dir()
            and (directory / "trainer_state.json").is_file()
            and (directory / "trainer_state.pt").is_file()
        )

    # Recover an atomic directory swap interrupted between rename operations.
    if not checkpoint_is_complete(last_checkpoint_dir):
        recovery_source = None
        if checkpoint_is_complete(checkpoint_temporary_dir):
            recovery_source = checkpoint_temporary_dir
        elif checkpoint_is_complete(checkpoint_backup_dir):
            recovery_source = checkpoint_backup_dir
        if recovery_source is not None:
            if last_checkpoint_dir.exists():
                shutil.rmtree(last_checkpoint_dir)
            os.replace(recovery_source, last_checkpoint_dir)
            print(f"recovered complete checkpoint: {last_checkpoint_dir}")
    if checkpoint_is_complete(last_checkpoint_dir):
        for stale_directory in (checkpoint_temporary_dir, checkpoint_backup_dir):
            if stale_directory.exists():
                shutil.rmtree(stale_directory)

    trainer_state_json_path = last_checkpoint_dir / "trainer_state.json"
    trainer_state_pt_path = last_checkpoint_dir / "trainer_state.pt"
    loss_log_path = output_dir / "loss_history.csv"
    best_dir = output_dir / "best_model"

    checkpoint_parts = (
        last_checkpoint_dir.is_dir(),
        trainer_state_json_path.is_file(),
        trainer_state_pt_path.is_file(),
    )
    if any(checkpoint_parts) and not all(checkpoint_parts):
        raise RuntimeError(
            f"incomplete checkpoint in {last_checkpoint_dir}; pass --restart to train anew"
        )
    resume_training = all(checkpoint_parts)
    saved_trainer_state: dict[str, Any] | None = None
    if resume_training:
        saved_trainer_state = read_json(trainer_state_json_path)
        saved_config = saved_trainer_state.get("resume_configuration")
        if not isinstance(saved_config, dict):
            raise RuntimeError(
                f"checkpoint lacks resume_configuration: {trainer_state_json_path}"
            )
        validate_resume_configuration(saved_config, current_resume_config)
        if not loss_log_path.is_file():
            raise RuntimeError(
                f"checkpoint exists but loss log is missing: {loss_log_path}; "
                "restore it or pass --restart"
            )
        committed_log_size = int(saved_trainer_state.get("loss_log_size", -1))
        actual_log_size = loss_log_path.stat().st_size
        if committed_log_size < 0 or actual_log_size < committed_log_size:
            raise RuntimeError(
                "loss log is older or shorter than the checkpoint; refusing to create "
                "a discontinuous history"
            )
        if actual_log_size > committed_log_size:
            # Rows from an interrupted, uncommitted epoch must not be duplicated.
            with loss_log_path.open("r+b") as stream:
                stream.truncate(committed_log_size)
            print("removed uncommitted loss rows from an interrupted epoch")
        print(
            f"resuming from epoch {int(saved_trainer_state['epoch']) + 1}: "
            f"{last_checkpoint_dir}"
        )
    else:
        stale_artifacts = [
            path
            for path in (
                best_dir,
                loss_log_path,
                output_dir / "training_summary.json",
            )
            if path.exists()
        ]
        if stale_artifacts:
            joined = ", ".join(str(path) for path in stale_artifacts)
            raise RuntimeError(
                "training artifacts exist without a resumable last checkpoint: "
                f"{joined}. Pass --restart to train anew or use another --output-dir."
            )
        print(f"no checkpoint found; initializing base model {args.model}")

    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoProcessor,
            PretrainedConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "missing training dependency; install packages listed in the script docstring: "
            f"{exc}"
        ) from exc

    set_seed(args.seed, torch)
    device = choose_device(args.device, torch)
    print(f"device: {device}")

    model_source = last_checkpoint_dir if resume_training else args.model
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
    )
    model_config = load_florence_config(
        model_source,
        AutoConfig,
        PretrainedConfig,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.config.svoa_task_prompt = args.task_prompt
    model.config.svoa_output_fields = list(FIELDS)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError("this Florence-2 implementation lacks gradient checkpointing")
        model.gradient_checkpointing_enable()
    if not args.unfreeze_vision:
        frozen = freeze_vision_encoder(model)
        print(f"frozen vision parameters: {frozen:,}")

    trainable, total = trainable_parameters(model)
    print(f"trainable parameters: {trainable:,}/{total:,} ({trainable / total:.2%})")

    class SVOADataset(Dataset):
        def __init__(self, rows: Sequence[Example]) -> None:
            self.rows = list(rows)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Example, Any]:
            example = self.rows[index]
            with Image.open(example.image_path) as image:
                rgb = image.convert("RGB")
            return example, rgb

    def collate(batch: Sequence[tuple[Example, Any]]) -> tuple[dict[str, Any], Any, list[Example]]:
        batch_examples, images = zip(*batch)
        prompts = [args.task_prompt] * len(batch_examples)
        targets = [example.target for example in batch_examples]
        inputs = processor(
            text=prompts,
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        labels = processor.tokenizer(
            text=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_target_length,
            return_token_type_ids=False,
        )["input_ids"]
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        return inputs, labels, list(batch_examples)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        SVOADataset(train_examples),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        SVOADataset(validation_examples),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        SVOADataset(test_examples),
        shuffle=False,
        **loader_kwargs,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=args.min_learning_rate,
    )

    use_amp = device.type == "cuda"
    use_bfloat16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp and not use_bfloat16,
        )
    except (AttributeError, TypeError):
        # Compatibility with older supported PyTorch releases.
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp and not use_bfloat16,
        )

    def amp_context() -> Any:
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    def move_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(device, non_blocking=True)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

    def forward_loss(inputs: dict[str, Any], labels: Any) -> Any:
        moved = move_inputs(inputs)
        labels = labels.to(device, non_blocking=True)
        accepted = {
            key: value
            for key, value in moved.items()
            if key in {"input_ids", "pixel_values", "attention_mask"}
        }
        return model(**accepted, labels=labels).loss

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    start_epoch = 1
    global_batch_step = 0
    optimizer_step = 0
    rolling_losses: deque[float] = deque(maxlen=args.loss_window)

    if resume_training:
        assert saved_trainer_state is not None
        try:
            binary_state = torch.load(
                trainer_state_pt_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            binary_state = torch.load(trainer_state_pt_path, map_location="cpu")
        optimizer.load_state_dict(binary_state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)
        scheduler.load_state_dict(binary_state["scheduler"])
        if binary_state.get("scaler"):
            scaler.load_state_dict(binary_state["scaler"])
        random.setstate(binary_state["python_random_state"])
        torch.set_rng_state(binary_state["torch_rng_state"])
        if device.type == "cuda" and binary_state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(binary_state["cuda_rng_state_all"])
        generator.set_state(binary_state["data_loader_generator_state"])
        rolling_losses.extend(float(value) for value in binary_state["rolling_losses"])

        history = list(saved_trainer_state.get("history", []))
        best_validation_loss = float(saved_trainer_state["best_validation_loss"])
        best_epoch = int(saved_trainer_state["best_epoch"])
        epochs_without_improvement = int(
            saved_trainer_state["epochs_without_improvement"]
        )
        stopped_early = bool(saved_trainer_state.get("stopped_early", False))
        start_epoch = int(saved_trainer_state["epoch"]) + 1
        global_batch_step = int(saved_trainer_state["global_batch_step"])
        optimizer_step = int(saved_trainer_state["optimizer_step"])
        if best_epoch and not best_dir.is_dir():
            raise RuntimeError(
                f"best checkpoint referenced by trainer state is missing: {best_dir}"
            )

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if resume_training:
        with loss_log_path.open("r", encoding="utf-8", newline="") as check_stream:
            existing_fields = next(csv.reader(check_stream), [])
        if existing_fields != list(LOSS_LOG_FIELDS):
            raise RuntimeError(
                f"unexpected columns in {loss_log_path}; refusing to append incompatible data"
            )
        loss_stream = loss_log_path.open("a", encoding="utf-8", newline="")
    else:
        loss_stream = loss_log_path.open("w", encoding="utf-8", newline="")
    loss_writer = csv.DictWriter(loss_stream, fieldnames=LOSS_LOG_FIELDS)
    if not resume_training:
        loss_writer.writeheader()
        loss_stream.flush()

    def write_loss_row(**values: Any) -> None:
        row = {field: "" for field in LOSS_LOG_FIELDS}
        row.update(values)
        row["timestamp_utc"] = utc_now()
        row["session_id"] = session_id
        loss_writer.writerow(row)
        # A crash should lose at most the current batch, not an entire epoch's curve.
        loss_stream.flush()

    def save_last_checkpoint(epoch: int, early_stopped: bool) -> None:
        if checkpoint_temporary_dir.exists():
            shutil.rmtree(checkpoint_temporary_dir)
        checkpoint_temporary_dir.mkdir(parents=True)
        model.save_pretrained(checkpoint_temporary_dir, safe_serialization=True)
        patch_saved_florence_config(
            checkpoint_temporary_dir,
            args.task_prompt,
            FIELDS,
        )
        processor.save_pretrained(checkpoint_temporary_dir)
        binary_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "data_loader_generator_state": generator.get_state(),
            "rolling_losses": list(rolling_losses),
        }
        torch.save(binary_state, checkpoint_temporary_dir / "trainer_state.pt")

        loss_stream.flush()
        os.fsync(loss_stream.fileno())
        state = {
            "format_version": 1,
            "epoch": epoch,
            "global_batch_step": global_batch_step,
            "optimizer_step": optimizer_step,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "stopped_early": early_stopped,
            "history": history,
            "loss_log_size": loss_log_path.stat().st_size,
            "resume_configuration": current_resume_config,
        }
        write_json(checkpoint_temporary_dir / "trainer_state.json", state)

        if checkpoint_backup_dir.exists():
            shutil.rmtree(checkpoint_backup_dir)
        if last_checkpoint_dir.exists():
            os.replace(last_checkpoint_dir, checkpoint_backup_dir)
        os.replace(checkpoint_temporary_dir, last_checkpoint_dir)
        if checkpoint_backup_dir.exists():
            shutil.rmtree(checkpoint_backup_dir)

    try:
        if stopped_early:
            print(
                f"checkpoint already stopped early at epoch {start_epoch - 1}; "
                "use --restart for a new run"
            )
        elif start_epoch > args.epochs:
            print(
                f"checkpoint already completed {start_epoch - 1} epochs; "
                f"--epochs={args.epochs}, so no additional training is needed"
            )
        else:
            for epoch in range(start_epoch, args.epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running_train_loss = 0.0
                progress = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}")
                for step, (inputs, labels, _) in enumerate(progress, 1):
                    with amp_context():
                        loss = forward_loss(inputs, labels)
                        scaled_loss = loss / args.gradient_accumulation
                    scaler.scale(scaled_loss).backward()
                    raw_loss = float(loss.detach().cpu())
                    running_train_loss += raw_loss
                    rolling_losses.append(raw_loss)
                    rolling_loss = sum(rolling_losses) / len(rolling_losses)
                    global_batch_step += 1

                    should_update = (
                        step % args.gradient_accumulation == 0
                        or step == len(train_loader)
                    )
                    if should_update:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            (
                                parameter
                                for parameter in model.parameters()
                                if parameter.requires_grad
                            ),
                            args.max_grad_norm,
                        )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1

                    write_loss_row(
                        event="train_batch",
                        epoch=epoch,
                        batch_in_epoch=step,
                        global_batch_step=global_batch_step,
                        optimizer_step=optimizer_step,
                        raw_loss=f"{raw_loss:.10g}",
                        rolling_loss=f"{rolling_loss:.10g}",
                        learning_rate=f"{float(optimizer.param_groups[0]['lr']):.10g}",
                    )
                    progress.set_postfix(
                        loss=f"{raw_loss:.4f}", rolling=f"{rolling_loss:.4f}"
                    )

                average_train_loss = running_train_loss / len(train_loader)
                model.eval()
                running_validation_loss = 0.0
                with torch.no_grad():
                    for inputs, labels, _ in tqdm(
                        validation_loader,
                        desc=f"validation {epoch}/{args.epochs}",
                    ):
                        with amp_context():
                            loss = forward_loss(inputs, labels)
                        running_validation_loss += float(loss.detach().cpu())
                average_validation_loss = (
                    running_validation_loss / len(validation_loader)
                )
                learning_rate_before = float(optimizer.param_groups[0]["lr"])
                improved = (
                    average_validation_loss
                    < best_validation_loss - args.early_stopping_min_delta
                )
                if improved:
                    best_validation_loss = average_validation_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(best_dir, safe_serialization=True)
                    patch_saved_florence_config(
                        best_dir,
                        args.task_prompt,
                        FIELDS,
                    )
                    processor.save_pretrained(best_dir)
                    print(f"saved best checkpoint: {best_dir}")
                else:
                    epochs_without_improvement += 1

                scheduler.step(average_validation_loss)
                learning_rate_after = float(optimizer.param_groups[0]["lr"])
                lr_reduced = learning_rate_after < learning_rate_before
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": average_train_loss,
                        "validation_loss": average_validation_loss,
                        "learning_rate": learning_rate_before,
                        "next_learning_rate": learning_rate_after,
                        "improved": improved,
                    }
                )
                write_loss_row(
                    event="epoch_end",
                    epoch=epoch,
                    batch_in_epoch=len(train_loader),
                    global_batch_step=global_batch_step,
                    optimizer_step=optimizer_step,
                    rolling_loss=(
                        f"{sum(rolling_losses) / len(rolling_losses):.10g}"
                        if rolling_losses
                        else ""
                    ),
                    epoch_train_loss=f"{average_train_loss:.10g}",
                    validation_loss=f"{average_validation_loss:.10g}",
                    learning_rate=f"{learning_rate_before:.10g}",
                    next_learning_rate=f"{learning_rate_after:.10g}",
                )
                stopped_early = (
                    epochs_without_improvement >= args.early_stopping_patience
                )
                save_last_checkpoint(epoch, stopped_early)

                print(
                    f"epoch {epoch}: train_loss={average_train_loss:.6f}, "
                    f"validation_loss={average_validation_loss:.6f}, "
                    f"lr={learning_rate_before:.3e}, "
                    f"without_improvement={epochs_without_improvement}"
                )
                if lr_reduced:
                    print(
                        "validation plateau: learning rate reduced "
                        f"{learning_rate_before:.3e} -> {learning_rate_after:.3e}"
                    )
                if stopped_early:
                    print(
                        f"early stopping at epoch {epoch}; best epoch was {best_epoch}"
                    )
                    break
    finally:
        loss_stream.close()

    training_summary = {
        "model": args.model,
        "task_prompt": args.task_prompt,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "resumed": resume_training,
        "last_checkpoint": last_checkpoint_dir.as_posix(),
        "loss_history": loss_log_path.as_posix(),
        "history": history,
        "split_sizes": {
            "train": len(train_examples),
            "validation": len(validation_examples),
            "test": len(test_examples),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "lr_factor": args.lr_factor,
            "lr_patience": args.lr_patience,
            "min_learning_rate": args.min_learning_rate,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "loss_window": args.loss_window,
            "vision_frozen": not args.unfreeze_vision,
            "amp_dtype": str(amp_dtype) if use_amp else "float32",
        },
    }
    write_json(output_dir / "training_summary.json", training_summary)

    model_class = model.__class__
    del optimizer, scheduler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Reuse the already loaded class and processor. This also works when the
    # checkpoint was loaded through Florence-2 remote code.
    model = model_class.from_pretrained(
        best_dir,
        torch_dtype=torch.float32,
    ).to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval()

    predictions: list[dict[str, Any]] = []
    targets: list[str] = []
    generated_texts: list[str] = []
    with torch.no_grad():
        for inputs, _, batch_examples in tqdm(test_loader, desc="test generation"):
            moved = move_inputs(inputs)
            generation_inputs = {
                key: value
                for key, value in moved.items()
                if key in {"input_ids", "pixel_values"}
            }
            # Legacy Florence-2 generate() keeps a caller-supplied text-only
            # mask after prepending visual tokens. Let it construct the mask
            # from the merged image+text embeddings instead.
            with amp_context():
                generated_ids = model.generate(
                    **generation_inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                )
            decoded = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            for example, prediction in zip(batch_examples, decoded):
                target = example.target
                prediction = prediction.strip()
                targets.append(target)
                generated_texts.append(prediction)
                predictions.append(
                    {
                        "image_id": example.image_id,
                        "image_path": example.image_path.as_posix(),
                        "target": target,
                        "prediction": prediction,
                        "exact": normalize_prediction(target)
                        == normalize_prediction(prediction),
                    }
                )

    metrics = field_metrics(targets, generated_texts)
    write_jsonl(output_dir / "test_predictions.jsonl", predictions)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"best model: {best_dir}")
    print(f"loss history: {loss_log_path}")
    print(f"last checkpoint: {last_checkpoint_dir}")
    print(f"test predictions: {output_dir / 'test_predictions.jsonl'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
