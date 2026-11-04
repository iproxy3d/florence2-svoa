#!/usr/bin/env python3
"""Correct HaGRID actions with official gesture annotations.

The input is the ``dataset.jsonl`` produced by ``prepare_svoa_dataset.py``.
Rows from other datasets are copied byte-for-byte at the JSON value level.

For a matched HaGRID image, an official gesture label replaces only the
``action`` field.  The label is not culturally interpreted: underscores are
replaced with spaces, so ``little_finger`` becomes ``little finger`` and
``three_gun`` becomes ``three gun``.  Distinct official variants remain
distinct.  Subject, object and adjuncts stay as produced by the VLM.

``no_gesture`` has a special meaning: the complete VLM SVOA label is retained.
The converter never invents an ``unknown`` action.

For images containing several hands, the primary action is selected as
follows:

1. a non-``no_gesture`` ``united_label`` has priority;
2. ``no_gesture`` labels are ignored when another gesture is present;
3. identical labels collapse to one action;
4. for distinct gestures, the label with the largest bounding box is primary.

All raw labels and boxes remain in the output metadata and audit file.

Example::

    python correct_hagrid_svoa.py \
        --dataset-jsonl ./svoa_work/dataset.jsonl \
        --hagrid-annotations-dir ./downloaded_images/hagrid_annotations \
        --output ./svoa_work/dataset_hagrid_corrected.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


FIELDS = ("subject", "action", "object", "adjuncts")
NO_GESTURE = "no_gesture"
SPLIT_NAMES = {"train": "train", "val": "val", "validation": "val", "test": "test"}
SPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class AtomicJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.stream: Any = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.temporary.open("w", encoding="utf-8", newline="\n")
        return self

    def write(self, value: Mapping[str, Any]) -> None:
        assert self.stream is not None
        self.stream.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        assert self.stream is not None
        if exc_type is None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            os.replace(self.temporary, self.path)
        else:
            self.stream.close()
            self.temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class HagridAnnotation:
    image_key: str
    image_stem: str
    gesture_file: str
    split: str
    value: dict[str, Any]


@dataclass(frozen=True)
class GestureDecision:
    primary: str | None
    raw_labels: tuple[str, ...]
    non_no_labels: tuple[str, ...]
    united_label: str | None
    reason: str


def normalize_raw_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    return label or None


def values_as_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        label = normalize_raw_label(value)
        return [label] if label else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            label = normalize_raw_label(item)
            if label:
                result.append(label)
        return result
    return []


def canonical_action(raw_label: str) -> str:
    """Convert syntax only; do not assign a meaning to the gesture."""

    return SPACE_RE.sub(" ", raw_label.replace("_", " ").strip().lower())


def box_area(box: Any) -> float:
    """Return area for HaGRID's ``[x, y, width, height]`` box."""

    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return -1.0
    try:
        width = float(box[2])
        height = float(box[3])
    except (TypeError, ValueError):
        return -1.0
    if width < 0 or height < 0:
        return -1.0
    return width * height


def select_gesture(annotation: Mapping[str, Any]) -> GestureDecision:
    labels = values_as_labels(annotation.get("labels"))
    united = next(
        (
            label
            for label in values_as_labels(annotation.get("united_label"))
            if label != NO_GESTURE
        ),
        None,
    )
    non_no = [label for label in labels if label != NO_GESTURE]

    if united is not None:
        return GestureDecision(
            united,
            tuple(labels),
            tuple(dict.fromkeys(non_no)),
            united,
            "united_label",
        )
    if not non_no:
        return GestureDecision(
            None,
            tuple(labels),
            (),
            None,
            "no_gesture",
        )

    distinct = list(dict.fromkeys(non_no))
    if len(distinct) == 1:
        return GestureDecision(
            distinct[0],
            tuple(labels),
            tuple(distinct),
            None,
            "single_or_identical_label",
        )

    boxes = annotation.get("bboxes")
    boxes = boxes if isinstance(boxes, list) else []
    candidates: list[tuple[float, int, str]] = []
    for index, label in enumerate(labels):
        if label == NO_GESTURE:
            continue
        area = box_area(boxes[index]) if index < len(boxes) else -1.0
        candidates.append((area, -index, label))
    primary = max(candidates)[2] if candidates else distinct[0]
    return GestureDecision(
        primary,
        tuple(labels),
        tuple(distinct),
        None,
        "largest_bbox",
    )


def split_from_path(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        normalized = SPLIT_NAMES.get(part.lower())
        if normalized:
            return normalized
    return "unknown"


def load_official_annotations(
    root: Path,
) -> tuple[
    dict[tuple[str, str], HagridAnnotation],
    dict[str, list[HagridAnnotation]],
]:
    if not root.is_dir():
        raise FileNotFoundError(f"HaGRID annotation directory not found: {root}")

    by_class_and_stem: dict[tuple[str, str], HagridAnnotation] = {}
    by_stem: dict[str, list[HagridAnnotation]] = defaultdict(list)
    loaded = 0
    for path in sorted(root.rglob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        gesture_file = path.stem.lower()
        split = split_from_path(path, root)
        for image_key, value in payload.items():
            if not isinstance(value, dict):
                continue
            stem = Path(str(image_key)).stem.lower()
            annotation = HagridAnnotation(
                image_key=str(image_key),
                image_stem=stem,
                gesture_file=gesture_file,
                split=split,
                value=dict(value),
            )
            compound = (gesture_file, stem)
            if compound in by_class_and_stem:
                raise ValueError(
                    f"duplicate HaGRID annotation for {gesture_file}/{stem}"
                )
            by_class_and_stem[compound] = annotation
            by_stem[stem].append(annotation)
            loaded += 1
    if loaded == 0:
        raise ValueError(f"no HaGRID annotations found below {root}")
    return by_class_and_stem, dict(by_stem)


def is_hagrid_row(row: Mapping[str, Any]) -> bool:
    dataset = str(row.get("dataset", "")).strip().lower()
    image_id = str(row.get("image_id", "")).strip().lower()
    return dataset == "hagrid" or image_id.startswith("hagrid:")


def row_class_hint(row: Mapping[str, Any]) -> str | None:
    explicit = normalize_raw_label(row.get("hagrid_label"))
    if explicit:
        return explicit
    image_id = str(row.get("image_id", ""))
    if image_id.lower().startswith("hagrid:"):
        relative = image_id.split(":", 1)[1].replace("\\", "/")
        first = relative.split("/", 1)[0].lower()
        return first or None
    return None


def row_stem_candidates(row: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for field in ("image_id", "image_path", "source_ref"):
        text = str(row.get(field, "")).strip()
        if not text:
            continue
        if field == "source_ref" and "#" in text:
            text = text.rsplit("#", 1)[1]
        text = text.split("?", 1)[0].replace("\\", "/")
        stem = Path(text.split(":", 1)[-1]).stem.lower()
        if stem and stem not in result:
            result.append(stem)
    return result


def match_annotation(
    row: Mapping[str, Any],
    by_class_and_stem: Mapping[tuple[str, str], HagridAnnotation],
    by_stem: Mapping[str, list[HagridAnnotation]],
) -> tuple[HagridAnnotation | None, str]:
    hint = row_class_hint(row)
    stems = row_stem_candidates(row)
    if hint:
        matches = [
            by_class_and_stem[(hint, stem)]
            for stem in stems
            if (hint, stem) in by_class_and_stem
        ]
        unique = {item.image_key: item for item in matches}
        if len(unique) == 1:
            return next(iter(unique.values())), "class_and_stem"
        if len(unique) > 1:
            return None, "ambiguous_class_and_stem"

    matches = []
    for stem in stems:
        matches.extend(by_stem.get(stem, []))
    unique = {
        (item.gesture_file, item.image_key, item.split): item for item in matches
    }
    if len(unique) == 1:
        return next(iter(unique.values())), "unique_stem"
    if len(unique) > 1:
        return None, "ambiguous_stem"
    return None, "image_not_found"


def correct_row(
    row: Mapping[str, Any], annotation: HagridAnnotation
) -> tuple[dict[str, Any], dict[str, Any]]:
    decision = select_gesture(annotation.value)
    original = {field: str(row.get(field, "")) for field in FIELDS}
    corrected = dict(row)
    changed_fields: list[str] = []

    if decision.primary is not None:
        action = canonical_action(decision.primary)
        if action != str(row.get("action", "")):
            corrected["action"] = action
            changed_fields.append("action")

    corrected.update(
        {
            "split": annotation.split,
            "annotation_source": (
                "hagrid_official+vlm"
                if decision.primary is not None
                else "vlm"
            ),
            "hagrid_image_key": annotation.image_key,
            "hagrid_gesture_file": annotation.gesture_file,
            "hagrid_primary_label": decision.primary,
            "hagrid_labels": list(decision.raw_labels),
            "hagrid_non_no_labels": list(decision.non_no_labels),
            "hagrid_united_label": annotation.value.get("united_label"),
            "hagrid_bboxes": annotation.value.get("bboxes"),
            "hagrid_united_bbox": annotation.value.get("united_bbox"),
            "hagrid_user_id": annotation.value.get("user_id"),
            "hagrid_meta": annotation.value.get("meta"),
            "vlm_original": original,
            "corrected_fields": changed_fields,
        }
    )
    audit = {
        "image_id": row.get("image_id"),
        "hagrid_image_key": annotation.image_key,
        "split": annotation.split,
        "status": "changed" if changed_fields else "unchanged",
        "reason": decision.reason,
        "primary_label": decision.primary,
        "raw_labels": list(decision.raw_labels),
        "non_no_labels": list(decision.non_no_labels),
        "united_label": annotation.value.get("united_label"),
        "bboxes": annotation.value.get("bboxes"),
        "user_id": annotation.value.get("user_id"),
        "vlm_original": original,
        "final_action": corrected.get("action"),
        "changed_fields": changed_fields,
    }
    return corrected, audit


def read_dataset(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            missing = [field for field in FIELDS if field not in value]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing SVOA fields: {', '.join(missing)}"
                )
            yield line_number, value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--hagrid-annotations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--require-all-hagrid-matched",
        action="store_true",
        help="fail instead of retaining an unmatched HaGRID row",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_jsonl.expanduser().resolve()
    annotation_root = args.hagrid_annotations_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    audit_path = (
        args.audit.expanduser().resolve()
        if args.audit
        else output_path.with_name(output_path.stem + "_audit.jsonl")
    )
    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary
        else output_path.with_name(output_path.stem + "_summary.json")
    )
    if output_path == dataset_path:
        raise SystemExit("--output must differ from --dataset-jsonl")
    if output_path == audit_path:
        raise SystemExit("--output and --audit must differ")

    by_class_and_stem, by_stem = load_official_annotations(annotation_root)
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    with AtomicJsonlWriter(output_path) as output, AtomicJsonlWriter(audit_path) as audit:
        for line_number, row in read_dataset(dataset_path):
            counts["rows_total"] += 1
            if not is_hagrid_row(row):
                counts["rows_non_hagrid"] += 1
                output.write(row)
                continue

            counts["rows_hagrid"] += 1
            annotation, match_reason = match_annotation(
                row, by_class_and_stem, by_stem
            )
            if annotation is None:
                counts["hagrid_unmatched"] += 1
                reasons[match_reason] += 1
                audit.write(
                    {
                        "image_id": row.get("image_id"),
                        "line": line_number,
                        "status": "unmatched",
                        "reason": match_reason,
                        "changed_fields": [],
                    }
                )
                if args.require_all_hagrid_matched:
                    raise ValueError(
                        f"{dataset_path}:{line_number}: unmatched HaGRID image: "
                        f"{match_reason}"
                    )
                output.write(row)
                continue

            corrected, record = correct_row(row, annotation)
            counts["hagrid_matched"] += 1
            if record["reason"] == "no_gesture":
                counts["hagrid_vlm_preserved_no_gesture"] += 1
            if record["changed_fields"]:
                counts["hagrid_rows_changed"] += 1
            else:
                counts["hagrid_rows_unchanged"] += 1
            reasons[record["reason"]] += 1
            audit.write(record)
            output.write(corrected)

    summary = {
        "dataset_jsonl": str(dataset_path),
        "hagrid_annotations_dir": str(annotation_root),
        "output": str(output_path),
        "audit": str(audit_path),
        "counts": dict(sorted(counts.items())),
        "reasons": dict(sorted(reasons.items())),
    }
    atomic_write_json(summary_path, summary)
    print(f"corrected dataset: {output_path}")
    print(f"audit:             {audit_path}")
    print(f"summary:           {summary_path}")
    print(
        "HaGRID rows: "
        f"{counts['rows_hagrid']}; matched: {counts['hagrid_matched']}; "
        f"changed: {counts['hagrid_rows_changed']}; "
        f"no_gesture kept: {counts['hagrid_vlm_preserved_no_gesture']}; "
        f"unmatched: {counts['hagrid_unmatched']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
