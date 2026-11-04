#!/usr/bin/env python3
"""Recalculate SVOA metrics from test_predictions.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_florence2_svoa import field_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    targets: list[str] = []
    predictions: list[str] = []
    with args.predictions.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict) or "target" not in row or "prediction" not in row:
                raise ValueError(
                    f"{args.predictions}:{line_number}: target and prediction required"
                )
            targets.append(str(row["target"]))
            predictions.append(str(row["prediction"]))
    metrics = field_metrics(targets, predictions)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
