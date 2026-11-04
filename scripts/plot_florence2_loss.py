#!/usr/bin/env python3
"""Plot Florence-2 SVOA training history from loss_history.csv.

The CSV is produced by ``train_florence2_svoa.py`` and may contain several
continued training sessions. The script preserves the global step/epoch axes,
marks session boundaries and ignores blank cells that belong to other event
types.

Example:
    python plot_florence2_loss.py \
        ./florence2_svoa_smoke/loss_history.csv \
        --output ./florence2_svoa_smoke/loss_history.png
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "session_id",
    "event",
    "epoch",
    "global_batch_step",
    "raw_loss",
    "rolling_loss",
    "epoch_train_loss",
    "validation_loss",
    "learning_rate",
    "next_learning_rate",
}


@dataclass(frozen=True)
class BatchHistory:
    global_step: np.ndarray
    epoch: np.ndarray
    raw_loss: np.ndarray
    rolling_loss: np.ndarray
    learning_rate: np.ndarray
    session_id: tuple[str, ...]


@dataclass(frozen=True)
class EpochHistory:
    epoch: np.ndarray
    global_step: np.ndarray
    train_loss: np.ndarray
    validation_loss: np.ndarray
    learning_rate: np.ndarray
    next_learning_rate: np.ndarray
    session_id: tuple[str, ...]


@dataclass(frozen=True)
class TrainingHistory:
    path: Path
    batches: BatchHistory
    epochs: EpochHistory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loss_history", type=Path, help="path to loss_history.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path; default: <CSV directory>/loss_history.png",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--max-batch-points",
        type=int,
        default=50000,
        help="maximum raw batch points drawn; rolling loss remains complete",
    )
    parser.add_argument(
        "--title",
        default="Florence-2 SVOA training history",
        help="figure title",
    )
    parser.add_argument("--show", action="store_true", help="also open the figure")
    return parser.parse_args()


def _float(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _int(value: str | None) -> int:
    parsed = _float(value)
    return int(parsed) if math.isfinite(parsed) else -1


def load_history(path: Path) -> TrainingHistory:
    if not path.is_file():
        raise FileNotFoundError(path)

    batch_rows: list[dict[str, str]] = []
    epoch_rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for row in reader:
            event = (row.get("event") or "").strip()
            if event == "train_batch":
                batch_rows.append(row)
            elif event == "epoch_end":
                epoch_rows.append(row)

    if not batch_rows:
        raise ValueError(f"{path}: no train_batch rows")
    if not epoch_rows:
        raise ValueError(f"{path}: no completed epoch_end rows")

    batches = BatchHistory(
        global_step=np.asarray([_int(row.get("global_batch_step")) for row in batch_rows]),
        epoch=np.asarray([_int(row.get("epoch")) for row in batch_rows]),
        raw_loss=np.asarray([_float(row.get("raw_loss")) for row in batch_rows]),
        rolling_loss=np.asarray([_float(row.get("rolling_loss")) for row in batch_rows]),
        learning_rate=np.asarray([_float(row.get("learning_rate")) for row in batch_rows]),
        session_id=tuple((row.get("session_id") or "").strip() for row in batch_rows),
    )
    epochs = EpochHistory(
        epoch=np.asarray([_int(row.get("epoch")) for row in epoch_rows]),
        global_step=np.asarray([_int(row.get("global_batch_step")) for row in epoch_rows]),
        train_loss=np.asarray([_float(row.get("epoch_train_loss")) for row in epoch_rows]),
        validation_loss=np.asarray([_float(row.get("validation_loss")) for row in epoch_rows]),
        learning_rate=np.asarray([_float(row.get("learning_rate")) for row in epoch_rows]),
        next_learning_rate=np.asarray([_float(row.get("next_learning_rate")) for row in epoch_rows]),
        session_id=tuple((row.get("session_id") or "").strip() for row in epoch_rows),
    )

    if np.any(batches.global_step < 0) or np.any(np.diff(batches.global_step) < 0):
        raise ValueError(f"{path}: global_batch_step must be non-decreasing")
    if np.any(epochs.epoch < 1) or np.any(np.diff(epochs.epoch) <= 0):
        raise ValueError(f"{path}: completed epochs must be strictly increasing")
    return TrainingHistory(path=path, batches=batches, epochs=epochs)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _downsample_indices(length: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _session_boundaries(
    global_steps: np.ndarray,
    session_ids: tuple[str, ...],
) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    previous: str | None = None
    session_number = 0
    for index, session_id in enumerate(session_ids):
        if session_id != previous:
            session_number += 1
            if index > 0:
                boundaries.append((int(global_steps[index]), session_number))
            previous = session_id
    return boundaries


def _mark_session_boundaries(axes: list[object], boundaries: list[tuple[int, int]]) -> None:
    for step, session_number in boundaries:
        for axis in axes:
            axis.axvline(step, color="#777777", linestyle=":", linewidth=1.0, alpha=0.75)
        axes[0].annotate(
            f"resume {session_number}",
            xy=(step, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color="#555555",
        )


def plot_history(
    history: TrainingHistory,
    output: Path,
    title: str,
    dpi: int,
    max_batch_points: int,
) -> dict[str, float | int]:
    import matplotlib.pyplot as plt

    batches = history.batches
    epochs = history.epochs
    figure, axes_array = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        constrained_layout=True,
    )
    axes = list(axes_array)
    figure.suptitle(title, fontsize=16)

    # Panel 1: batch loss. Raw loss is deliberately faint because it is noisy;
    # the training script's persistent rolling loss is the main trend line.
    raw_x, raw_y = _finite_xy(batches.global_step.astype(float), batches.raw_loss)
    selected = _downsample_indices(len(raw_x), max_batch_points)
    axes[0].scatter(
        raw_x[selected],
        raw_y[selected],
        s=7,
        alpha=0.18,
        color="#4C78A8",
        edgecolors="none",
        label="raw batch loss",
    )
    rolling_x, rolling_y = _finite_xy(
        batches.global_step.astype(float), batches.rolling_loss
    )
    axes[0].plot(
        rolling_x,
        rolling_y,
        linewidth=1.8,
        color="#1F4E8C",
        label="rolling loss",
    )
    axes[0].set_ylabel("loss")
    axes[0].set_title("Batch training loss")
    axes[0].legend(loc="best")

    # Panel 2: one comparable point per completed epoch.
    train_x, train_y = _finite_xy(epochs.epoch.astype(float), epochs.train_loss)
    validation_x, validation_y = _finite_xy(
        epochs.epoch.astype(float), epochs.validation_loss
    )
    axes[1].plot(
        train_x,
        train_y,
        marker="o",
        markersize=4,
        linewidth=1.7,
        color="#2A6FBB",
        label="epoch train loss",
    )
    axes[1].plot(
        validation_x,
        validation_y,
        marker="o",
        markersize=4,
        linewidth=1.7,
        color="#D64B3C",
        label="validation loss",
    )
    if len(validation_y):
        best_index = int(np.argmin(validation_y))
        best_epoch = int(validation_x[best_index])
        best_loss = float(validation_y[best_index])
        axes[1].scatter(
            [best_epoch],
            [best_loss],
            s=90,
            marker="*",
            color="#18864B",
            zorder=5,
            label=f"best validation: epoch {best_epoch}",
        )
        axes[1].annotate(
            f"{best_loss:.5g}",
            (best_epoch, best_loss),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            color="#146B3A",
        )
    else:
        best_epoch = -1
        best_loss = math.nan
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mean loss")
    axes[1].set_title("Train and validation loss by epoch")
    axes[1].legend(loc="best")

    # Panel 3: scheduler output. The step is located at the end of each epoch.
    lr_mask = (
        np.isfinite(epochs.learning_rate)
        & np.isfinite(epochs.next_learning_rate)
        & (epochs.learning_rate > 0)
        & (epochs.next_learning_rate > 0)
    )
    lr_epochs = epochs.epoch[lr_mask]
    lr_before = epochs.learning_rate[lr_mask]
    lr_after = epochs.next_learning_rate[lr_mask]
    if len(lr_epochs):
        axes[2].step(
            lr_epochs,
            lr_before,
            where="pre",
            linewidth=1.6,
            color="#7A5195",
            label="learning rate used",
        )
        axes[2].scatter(
            lr_epochs,
            lr_after,
            s=22,
            color="#EF8A17",
            label="learning rate for next epoch",
            zorder=3,
        )
        axes[2].set_yscale("log")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate, log scale")
    axes[2].set_title("Learning-rate scheduler")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.grid(True, alpha=0.25)

    # Session boundaries are meaningful only on the global-step panel.
    boundaries = _session_boundaries(batches.global_step, batches.session_id)
    _mark_session_boundaries([axes[0]], boundaries)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)

    latest_train = float(train_y[-1]) if len(train_y) else math.nan
    latest_validation = float(validation_y[-1]) if len(validation_y) else math.nan
    return {
        "batch_rows": int(len(batches.global_step)),
        "completed_epochs": int(len(epochs.epoch)),
        "sessions": int(len(set(batches.session_id))),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "latest_train_loss": latest_train,
        "latest_validation_loss": latest_validation,
    }


def main() -> int:
    args = _parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.max_batch_points < 0:
        raise SystemExit("--max-batch-points must be non-negative")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    output = args.output or args.loss_history.with_name("loss_history.png")
    history = load_history(args.loss_history)
    summary = plot_history(
        history,
        output,
        title=args.title,
        dpi=args.dpi,
        max_batch_points=args.max_batch_points,
    )

    print(f"history: {args.loss_history.resolve()}")
    print(f"batch rows: {summary['batch_rows']}")
    print(f"completed epochs: {summary['completed_epochs']}")
    print(f"training sessions: {summary['sessions']}")
    print(
        "best validation: "
        f"epoch {summary['best_epoch']}, "
        f"loss={summary['best_validation_loss']:.6g}"
    )
    print(
        "latest losses: "
        f"train={summary['latest_train_loss']:.6g}, "
        f"validation={summary['latest_validation_loss']:.6g}"
    )
    print(f"plot: {output.resolve()}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
