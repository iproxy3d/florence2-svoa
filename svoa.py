#!/usr/bin/env python3
"""Single command-line entry point for the Florence-2 SVOA project.

The public workflow intentionally has four commands only::

    python svoa.py download --limit-per-dataset 5000
    python svoa.py prepare --env-file .env --limit-per-dataset 5000
    python svoa.py train --output-dir ./florence2_svoa
    python svoa.py infer --model ./florence2_svoa/best_model --image ./example.jpg

SWiG and HaGRID correction scripts are implementation details. ``prepare``
runs them automatically and writes the final training file to
``WORK_DIR/dataset.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_DOWNLOAD_DIR = Path("downloaded_images_v2")
DEFAULT_WORK_DIR = Path("svoa_work_v2")
DEFAULT_MODEL_DIR = Path("florence2_svoa")

HAGRID_GESTURES = (
    "grabbing",
    "grip",
    "holy",
    "point",
    "call",
    "three3",
    "timeout",
    "xsign",
    "hand_heart",
    "hand_heart2",
    "little_finger",
    "middle_finger",
    "take_picture",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "stop_inverted",
    "three",
    "three2",
    "two_up",
    "two_up_inverted",
    "three_gun",
    "thumb_index",
    "thumb_index2",
    "no_gesture",
)

SWIG_ANNOTATION_BASE = (
    "https://raw.githubusercontent.com/allenai/swig/master/SWiG_jsons"
)
SWIG_ANNOTATION_FILES = ("train.json", "dev.json", "test.json", "imsitu_space.json")


def run_script(script_name: str, arguments: Sequence[str]) -> None:
    command = [sys.executable, str(SCRIPTS_DIR / script_name), *map(str, arguments)]
    print("\n> " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def download_file(url: str, destination: Path, timeout: float) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"SWiG annotations: already exists {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "svoa-dataset-pipeline/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"empty response from {url}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"SWiG annotations: downloaded {destination.name}")


def download_swig_annotations(download_dir: Path, timeout: float) -> Path:
    target = download_dir / "swig_annotations" / "SWiG_jsons"
    for filename in SWIG_ANNOTATION_FILES:
        download_file(
            f"{SWIG_ANNOTATION_BASE}/{filename}",
            target / filename,
            timeout,
        )
    return target


def validate_jsonl(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows += 1
    if rows == 0:
        raise ValueError(f"empty dataset: {path}")


def command_download(args: argparse.Namespace, extra: Sequence[str]) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    # Keep the requested HaGRID contribution close to one ordinary dataset:
    # 5000 total means 147 images from each of the 34 official classes.
    per_gesture = max(1, args.limit_per_dataset // len(HAGRID_GESTURES))
    run_script(
        "download_svoa_images.py",
        [
            "--output-dir",
            output_dir,
            "--limit-per-dataset",
            str(args.limit_per_dataset),
            "--hagrid-gestures",
            *HAGRID_GESTURES,
            "--hagrid-limit-per-gesture",
            str(per_gesture),
            "--hagrid-download-annotations",
            "--request-timeout",
            str(args.request_timeout),
            *extra,
        ],
    )
    download_swig_annotations(output_dir, args.request_timeout)
    print("\nDownload completed")
    print(f"manifest: {output_dir / 'manifest.jsonl'}")
    print(f"SWiG annotations: {output_dir / 'swig_annotations' / 'SWiG_jsons'}")
    print(f"HaGRID annotations: {output_dir / 'hagrid_annotations'}")


def command_prepare(args: argparse.Namespace, extra: Sequence[str]) -> None:
    download_dir = args.download_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = download_dir / "manifest.jsonl"
    swig_annotations = download_dir / "swig_annotations" / "SWiG_jsons"
    hagrid_annotations = download_dir / "hagrid_annotations"
    required = [
        manifest,
        swig_annotations / "train.json",
        swig_annotations / "dev.json",
        swig_annotations / "test.json",
        swig_annotations / "imsitu_space.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "download is incomplete; run `python svoa.py download` first. "
            f"Missing:\n{details}"
        )
    if not hagrid_annotations.is_dir():
        raise FileNotFoundError(
            "HaGRID annotations are missing; run `python svoa.py download` first: "
            f"{hagrid_annotations}"
        )

    vlm_dataset = work_dir / "dataset_vlm.jsonl"
    swig_dataset = work_dir / "dataset_swig_corrected.jsonl"
    final_dataset = work_dir / "dataset.jsonl"
    run_script(
        "prepare_svoa_dataset.py",
        [
            "--env-file",
            args.env_file,
            "--manifest",
            manifest,
            "--work-dir",
            work_dir,
            "--output",
            vlm_dataset,
            "--limit-per-dataset",
            str(args.limit_per_dataset),
            *extra,
        ],
    )
    run_script(
        "correct_swig_svoa.py",
        [
            "--dataset-jsonl",
            vlm_dataset,
            "--swig-json-dir",
            swig_annotations,
            "--output",
            swig_dataset,
            "--audit",
            work_dir / "swig_corrections.jsonl",
            "--summary",
            work_dir / "swig_correction_summary.json",
        ],
    )
    run_script(
        "correct_hagrid_svoa.py",
        [
            "--dataset-jsonl",
            swig_dataset,
            "--hagrid-annotations-dir",
            hagrid_annotations,
            "--output",
            final_dataset,
            "--audit",
            work_dir / "hagrid_corrections.jsonl",
            "--summary",
            work_dir / "hagrid_correction_summary.json",
            "--require-all-hagrid-matched",
        ],
    )
    validate_jsonl(final_dataset)
    print("\nDataset preparation completed")
    print(f"training dataset: {final_dataset}")


def command_train(args: argparse.Namespace, extra: Sequence[str]) -> None:
    dataset = args.work_dir.expanduser().resolve() / "dataset.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(
            f"training dataset not found: {dataset}; run `python svoa.py prepare` first"
        )
    run_script(
        "train_florence2_svoa.py",
        [
            "--dataset-jsonl",
            dataset,
            "--output-dir",
            args.output_dir,
            *extra,
        ],
    )


def command_infer(args: argparse.Namespace, extra: Sequence[str]) -> None:
    run_script(
        "infer_florence2_svoa.py",
        ["--model", args.model, "--image", args.image, *extra],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download images and annotations")
    download.add_argument("--output-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    download.add_argument("--limit-per-dataset", type=int, default=5000)
    download.add_argument("--request-timeout", type=float, default=120.0)

    prepare = subparsers.add_parser(
        "prepare", help="run VLM labelling and official corrections"
    )
    prepare.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    prepare.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    prepare.add_argument("--env-file", type=Path, default=Path(".env"))
    prepare.add_argument("--limit-per-dataset", type=int, default=5000)

    train = subparsers.add_parser("train", help="train Florence-2 on dataset.jsonl")
    train.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)

    infer = subparsers.add_parser("infer", help="run inference for one image")
    infer.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR / "best_model")
    infer.add_argument("--image", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args()
    if getattr(args, "limit_per_dataset", 1) < 1:
        parser.error("--limit-per-dataset must be at least 1")
    if extra and extra[0] == "--":
        extra = extra[1:]
    commands = {
        "download": command_download,
        "prepare": command_prepare,
        "train": command_train,
        "infer": command_infer,
    }
    try:
        commands[args.command](args, extra)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nERROR: stage {args.command!r} failed with exit code {exc.returncode}",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
