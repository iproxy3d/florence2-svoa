#!/usr/bin/env python3
"""Correct existing VLM SVOA labels with official SWiG annotations.

The input dataset is the ``dataset.jsonl`` produced by
``prepare_svoa_dataset.py``.  Non-SWiG rows are copied unchanged.  A SWiG
field is replaced only when the official annotation can be converted with
high confidence:

* the image is found unambiguously in train/dev/test.json;
* a semantic role can be mapped unambiguously to an SVOA field;
* at least ``--min-votes`` SWiG annotators agree on the noun;
* the noun has a usable gloss in imsitu_space.json.

For ``subject`` and ``object`` the corresponding SWiG role must also have a
valid grounding box.  This prevents an occluded but annotated participant from
becoming a target for a model that is trained to describe directly visible
facts.  ``--allow-ungrounded`` disables this guard when a different task needs
the complete semantic frame.

The original file is never modified.  The program writes a corrected JSONL,
an audit JSONL with every changed/skipped SWiG row, and a summary JSON.

Recommended installation for reliable conversion of SWiG gerunds to base
English verbs::

    python -m pip install lemminflect

Without ``lemminflect`` an action is changed only when ``--action-map``
contains it.  A VLM action that is already a valid base form of the official
SWiG gerund is retained and marked as confirmed.

Example::

    python correct_swig_svoa.py \
        --dataset-jsonl ./svoa_work/dataset.jsonl \
        --swig-json-dir ./swig/SWiG_jsons \
        --output ./svoa_work/dataset_swig_corrected.jsonl \
        --audit ./svoa_work/swig_corrections.jsonl \
        --summary ./svoa_work/swig_correction_summary.json

Optional mapping files make every project-specific choice explicit.

``action_map.json``::

    {"drinking": "drink", "hitchhiking": "hitchhike"}

``role_map.json``::

    {
      "drinking": {
        "subject": "agent",
        "object": "liquid",
        "adjuncts": "place"
      }
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


FIELDS = ("subject", "action", "object", "adjuncts")
WORD_RE = re.compile(r"^(?:[a-z0-9]+(?:-[a-z0-9]+)*|none|unknown)$")
SWIG_IMAGE_RE = re.compile(r"([a-z][a-z0-9_-]*_\d+\.(?:jpe?g|png))", re.I)

# These rules are intentionally conservative.  They are not intended to map
# all FrameNet roles.  Ambiguous fields stay as their original VLM values.
SUBJECT_ROLE_NAMES = {
    "agent",
    "actor",
    "performer",
    "speaker",
    "experiencer",
    "perceiver",
    "communicator",
}
SUBJECT_FRAMENET_NAMES = {
    "agent",
    "actor",
    "speaker",
    "experiencer",
    "perceiver_agentive",
    "communicator",
    "cognizer",
    "ingestor",
}
OBJECT_ROLE_NAMES = {
    "item",
    "object",
    "target",
    "patient",
    "theme",
    "goods",
    "food",
    "liquid",
    "substance",
    "message",
    "content",
    "vehicle",
    "garment",
    "bodypart",
    "product",
}
OBJECT_FRAMENET_NAMES = {
    "patient",
    "theme",
    "goods",
    "ingestibles",
    "content",
    "message",
    "stimulus",
    "phenomenon",
    "entity",
}
ADJUNCT_ROLE_NAMES = {"place", "location"}
ADJUNCT_FRAMENET_NAMES = {"place", "location"}
NON_OBJECT_ROLE_NAMES = {
    "place",
    "location",
    "tool",
    "instrument",
    "manner",
    "time",
    "source",
    "destination",
    "path",
    "goal",
    "container",
}
NON_OBJECT_FRAMENET_NAMES = {
    "place",
    "location",
    "instrument",
    "manner",
    "time",
    "source",
    "destination",
    "path",
    "goal",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_object(path: Path | None, description: str) -> dict[str, Any]:
    if path is None:
        return {}
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


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
class OfficialAnnotation:
    image_name: str
    split: str
    value: dict[str, Any]


@dataclass
class FieldDecision:
    field: str
    original: str
    replacement: str | None = None
    source: str = "vlm"
    reason: str = "no_official_replacement"
    role: str | None = None
    votes: int | None = None
    alternatives: dict[str, int] = field(default_factory=dict)

    @property
    def final(self) -> str:
        return self.replacement if self.replacement is not None else self.original

    @property
    def changed(self) -> bool:
        return self.replacement is not None and self.replacement != self.original

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "final": self.final,
            "changed": self.changed,
            "source": self.source,
            "reason": self.reason,
            "role": self.role,
            "votes": self.votes,
            "alternatives": self.alternatives,
        }


def normalize_word(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = text.replace("&", " and ")
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    text = re.sub(r"-+", "-", text)
    if not text or not WORD_RE.fullmatch(text):
        return None
    return text


def read_dataset(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            missing = [name for name in ("image_id", *FIELDS) if name not in row]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing required fields: {', '.join(missing)}"
                )
            yield line_number, row


def load_official_annotations(directory: Path) -> tuple[
    dict[str, OfficialAnnotation], dict[str, list[str]]
]:
    annotations: dict[str, OfficialAnnotation] = {}
    stems: dict[str, list[str]] = {}
    for filename, split in (("train.json", "train"), ("dev.json", "validation"), ("test.json", "test")):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing official SWiG annotation file: {path}")
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"official SWiG file must contain an object: {path}")
        for raw_name, raw_annotation in payload.items():
            name = PurePosixPath(str(raw_name).replace("\\", "/")).name
            key = name.lower()
            if key in annotations:
                previous = annotations[key]
                raise ValueError(
                    f"duplicate SWiG image {name!r} in {previous.split} and {split}"
                )
            if not isinstance(raw_annotation, dict):
                raise ValueError(f"invalid annotation for {name!r} in {path}")
            annotations[key] = OfficialAnnotation(name, split, raw_annotation)
            stems.setdefault(Path(key).stem, []).append(key)
    return annotations, stems


def is_swig_row(row: Mapping[str, Any]) -> bool:
    dataset = str(row.get("dataset", "")).strip().lower()
    image_id = str(row.get("image_id", "")).strip().lower()
    return dataset == "swig" or image_id == "swig" or image_id.startswith("swig:")


def match_image(
    row: Mapping[str, Any],
    annotations: Mapping[str, OfficialAnnotation],
    stems: Mapping[str, list[str]],
) -> tuple[OfficialAnnotation | None, str]:
    texts = [str(row.get("image_id", "")), str(row.get("image_path", ""))]
    candidates: set[str] = set()
    for raw in texts:
        normalized = raw.replace("\\", "/").lower()
        basename = PurePosixPath(normalized).name
        for candidate in (normalized, basename, normalized.split(":", 1)[-1]):
            key = PurePosixPath(candidate).name
            if key in annotations:
                candidates.add(key)
            stem = Path(key).stem
            if stem in stems and len(stems[stem]) == 1:
                candidates.add(stems[stem][0])
        for found in SWIG_IMAGE_RE.findall(normalized):
            key = PurePosixPath(found).name.lower()
            if key in annotations:
                candidates.add(key)
    if len(candidates) == 1:
        key = next(iter(candidates))
        return annotations[key], "matched"
    if not candidates:
        return None, "official_image_not_found"
    return None, "ambiguous_image_match:" + ",".join(sorted(candidates))


def role_order(verb_info: Mapping[str, Any], frames: list[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    raw_order = verb_info.get("order", [])
    if isinstance(raw_order, list):
        result.extend(str(role).lower() for role in raw_order)
    for frame_value in frames:
        result.extend(str(role).lower() for role in frame_value)
    return list(dict.fromkeys(result))


def role_metadata(verb_info: Mapping[str, Any], role: str) -> tuple[str, str]:
    roles = verb_info.get("roles", {})
    value = roles.get(role, {}) if isinstance(roles, dict) else {}
    if not isinstance(value, dict):
        return "", ""
    return str(value.get("framenet", "")).strip().lower(), str(
        value.get("def", "")
    ).strip().lower()


def infer_role(
    field_name: str,
    verb_info: Mapping[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[str | None, str]:
    order = role_order(verb_info, frames)
    candidates: list[tuple[int, str]] = []
    for index, role in enumerate(order):
        framenet, definition = role_metadata(verb_info, role)
        score = 0
        if field_name == "subject":
            if role in SUBJECT_ROLE_NAMES:
                score = 100
            elif framenet in SUBJECT_FRAMENET_NAMES:
                score = 90
            elif re.search(
                r"\b(entity|person|animal|group)\s+(doing|performing|carrying out)",
                definition,
            ):
                score = 80
        elif field_name == "object":
            if role in NON_OBJECT_ROLE_NAMES or framenet in NON_OBJECT_FRAMENET_NAMES:
                score = 0
            elif role in OBJECT_ROLE_NAMES:
                score = 100
            elif role.endswith("item"):
                score = 95
            elif framenet in OBJECT_FRAMENET_NAMES:
                score = 85
            elif re.search(r"\b(entity|item|object|person|animal)\s+(being|that|which)", definition):
                score = 70
        elif field_name == "adjuncts":
            if role in ADJUNCT_ROLE_NAMES:
                score = 100
            elif framenet in ADJUNCT_FRAMENET_NAMES:
                score = 90
        if score:
            # Earlier roles win only after semantic confidence.
            candidates.append((score * 1000 - index, role))

    if not candidates:
        return None, "no_confident_role"
    candidates.sort(reverse=True)
    best_score, best_role = candidates[0]
    if len(candidates) > 1 and candidates[1][0] // 1000 == best_score // 1000:
        return None, "ambiguous_roles:" + ",".join(
            role for _, role in candidates if _ // 1000 == best_score // 1000
        )
    return best_role, "inferred_role"


def resolve_role(
    field_name: str,
    verb: str,
    verb_info: Mapping[str, Any],
    frames: list[dict[str, Any]],
    role_map: Mapping[str, Any],
) -> tuple[str | None, str]:
    configured = role_map.get(verb, {})
    if configured is not None and not isinstance(configured, dict):
        return None, "invalid_role_map_entry"
    if isinstance(configured, dict) and field_name in configured:
        raw_role = configured[field_name]
        if raw_role is None or str(raw_role).strip().lower() in {"", "none"}:
            return None, "role_disabled_by_map"
        return str(raw_role).strip().lower(), "explicit_role_map"
    return infer_role(field_name, verb_info, frames)


def consensus_synset(
    frames: list[dict[str, Any]], role: str, min_votes: int
) -> tuple[str | None, int, dict[str, int], str]:
    values = [str(frame_value.get(role, "")).strip() for frame_value in frames]
    counts = Counter(value for value in values if value)
    if not counts:
        return None, 0, {}, "role_empty_in_official_frames"
    most_common = counts.most_common()
    top_value, top_votes = most_common[0]
    alternatives = dict(sorted(counts.items()))
    if top_votes < min_votes:
        return None, top_votes, alternatives, "insufficient_annotator_agreement"
    if len(most_common) > 1 and most_common[1][1] == top_votes:
        return None, top_votes, alternatives, "tied_annotator_votes"
    return top_value, top_votes, alternatives, "annotator_consensus"


def noun_label(synset: str, nouns: Mapping[str, Any]) -> tuple[str | None, str]:
    info = nouns.get(synset)
    if not isinstance(info, dict):
        return None, "noun_missing_from_imsitu_space"
    glosses = info.get("gloss", [])
    if not isinstance(glosses, list):
        return None, "invalid_noun_glosses"
    for gloss in glosses:
        label = normalize_word(gloss)
        if label:
            return label, "first_usable_official_gloss"
    return None, "no_usable_noun_gloss"


def role_is_grounded(boxes: Any, role: str) -> bool:
    if not isinstance(boxes, dict):
        return False
    value = boxes.get(role)
    # Some tools may preserve several boxes for a role.  Accept either one
    # [x1, y1, x2, y2] box or a list of such boxes.
    candidates = value if isinstance(value, list) else []
    if len(candidates) == 4 and all(isinstance(item, (int, float)) for item in candidates):
        candidates = [candidates]
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        if (
            isinstance(candidate, list)
            and len(candidate) == 4
            and all(isinstance(item, (int, float)) for item in candidate)
            and candidate[0] >= 0
            and candidate[1] >= 0
            and candidate[2] > candidate[0]
            and candidate[3] > candidate[1]
        ):
            return True
    return False


def gerund_forms(base: str) -> set[str]:
    forms = {base + "ing"}
    if base.endswith("ie") and len(base) > 2:
        forms.add(base[:-2] + "ying")
    if base.endswith("e") and not base.endswith(("ee", "ye", "oe")):
        forms.add(base[:-1] + "ing")
    if (
        len(base) >= 3
        and base[-1] not in "aeiouwxy"
        and base[-2] in "aeiou"
        and base[-3] not in "aeiou"
    ):
        forms.add(base + base[-1] + "ing")
    return forms


class ActionResolver:
    def __init__(self, explicit_map: Mapping[str, Any]) -> None:
        self.explicit_map = {
            str(key).strip().lower(): normalize_word(value)
            for key, value in explicit_map.items()
        }
        invalid = [key for key, value in self.explicit_map.items() if value is None]
        if invalid:
            raise ValueError("invalid values in action map: " + ", ".join(invalid[:10]))
        try:
            from lemminflect import getLemma  # type: ignore
        except ImportError:
            self.get_lemma = None
        else:
            self.get_lemma = getLemma

    def resolve(self, official_verb: str, original: str) -> tuple[str | None, str]:
        verb = official_verb.strip().lower()
        if verb in self.explicit_map:
            return self.explicit_map[verb], "explicit_action_map"
        if verb in gerund_forms(original):
            return original, "vlm_action_confirmed_by_official_verb"
        if self.get_lemma is not None:
            lemmas = self.get_lemma(verb, upos="VERB")
            labels = [normalize_word(value) for value in lemmas]
            labels = [value for value in labels if value]
            if len(set(labels)) == 1:
                return labels[0], "lemminflect_from_official_verb"
            if labels:
                return None, "ambiguous_action_lemmas:" + ",".join(sorted(set(labels)))
        return None, "action_map_or_lemminflect_required"


def decide_noun_field(
    field_name: str,
    original: str,
    verb: str,
    verb_info: Mapping[str, Any],
    frames: list[dict[str, Any]],
    nouns: Mapping[str, Any],
    role_map: Mapping[str, Any],
    min_votes: int,
    boxes: Any,
    allow_ungrounded: bool,
) -> FieldDecision:
    decision = FieldDecision(field_name, original)
    role, role_reason = resolve_role(field_name, verb, verb_info, frames, role_map)
    decision.role = role
    if role is None:
        decision.reason = role_reason
        return decision
    if (
        field_name in {"subject", "object"}
        and not allow_ungrounded
        and not role_is_grounded(boxes, role)
    ):
        decision.reason = "official_role_is_not_visually_grounded"
        return decision
    synset, votes, alternatives, vote_reason = consensus_synset(frames, role, min_votes)
    decision.votes = votes
    decision.alternatives = alternatives
    if synset is None:
        decision.reason = vote_reason
        return decision
    replacement, gloss_reason = noun_label(synset, nouns)
    if replacement is None:
        decision.reason = gloss_reason
        return decision
    decision.replacement = replacement
    decision.source = "swig_official"
    decision.reason = f"{role_reason};{vote_reason};{gloss_reason}"
    return decision


def correct_row(
    row: dict[str, Any],
    annotation: OfficialAnnotation,
    space: Mapping[str, Any],
    role_map: Mapping[str, Any],
    action_resolver: ActionResolver,
    min_votes: int,
    allow_ungrounded: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    official = annotation.value
    verb = str(official.get("verb", "")).strip().lower()
    frames_value = official.get("frames", [])
    if not verb or not isinstance(frames_value, list):
        raise ValueError(f"invalid official annotation for {annotation.image_name}")
    frames = [value for value in frames_value if isinstance(value, dict)]
    if not frames:
        raise ValueError(f"official annotation has no frames: {annotation.image_name}")

    verbs = space.get("verbs", {})
    nouns = space.get("nouns", {})
    if not isinstance(verbs, dict) or not isinstance(nouns, dict):
        raise ValueError("imsitu_space.json must contain objects 'verbs' and 'nouns'")
    verb_info = verbs.get(verb, {})
    if not isinstance(verb_info, dict):
        verb_info = {}

    decisions: dict[str, FieldDecision] = {}
    original_action = str(row["action"]).strip().lower()
    action, action_reason = action_resolver.resolve(verb, original_action)
    decisions["action"] = FieldDecision(
        field="action",
        original=original_action,
        replacement=action,
        source="swig_official" if action is not None else "vlm",
        reason=action_reason,
    )
    for field_name in ("subject", "object", "adjuncts"):
        decisions[field_name] = decide_noun_field(
            field_name=field_name,
            original=str(row[field_name]).strip().lower(),
            verb=verb,
            verb_info=verb_info,
            frames=frames,
            nouns=nouns,
            role_map=role_map,
            min_votes=min_votes,
            boxes=official.get("bb"),
            allow_ungrounded=allow_ungrounded,
        )

    corrected = dict(row)
    original_labels = {field_name: str(row[field_name]) for field_name in FIELDS}
    for field_name, decision in decisions.items():
        corrected[field_name] = decision.final
    changed_fields = [name for name in FIELDS if decisions[name].changed]
    official_fields = [
        name for name in FIELDS if decisions[name].source == "swig_official"
    ]
    corrected["split"] = annotation.split
    corrected["annotation_source"] = (
        "swig_official" if len(official_fields) == len(FIELDS) else "swig_official+vlm"
    )
    corrected["swig_image_name"] = annotation.image_name
    corrected["swig_verb"] = verb
    corrected["vlm_original"] = original_labels
    corrected["corrected_fields"] = changed_fields

    audit = {
        "image_id": row.get("image_id"),
        "swig_image_name": annotation.image_name,
        "split": annotation.split,
        "verb": verb,
        "changed_fields": changed_fields,
        "fields": {name: decisions[name].as_dict() for name in FIELDS},
    }
    return corrected, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument(
        "--swig-json-dir",
        type=Path,
        required=True,
        help="directory containing train.json, dev.json, test.json and imsitu_space.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--action-map", type=Path)
    parser.add_argument("--role-map", type=Path)
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        choices=(2, 3),
        help="minimum number of SWiG annotators agreeing on a noun (default: 2)",
    )
    parser.add_argument(
        "--require-all-swig-matched",
        action="store_true",
        help="fail instead of preserving a SWiG row when no official image matches",
    )
    parser.add_argument(
        "--allow-ungrounded",
        action="store_true",
        help=(
            "allow ungrounded official subject/object nouns to replace VLM labels; "
            "disabled by default because the target describes visible facts"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_jsonl.expanduser().resolve()
    swig_dir = args.swig_json_dir.expanduser().resolve()
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
        raise SystemExit("--output must differ from --dataset-jsonl; source is immutable")
    if output_path == audit_path:
        raise SystemExit("--output and --audit must be different files")

    annotations, stems = load_official_annotations(swig_dir)
    space_path = swig_dir / "imsitu_space.json"
    if not space_path.is_file():
        raise FileNotFoundError(f"missing official SWiG metadata file: {space_path}")
    space = load_json(space_path)
    if not isinstance(space, dict):
        raise ValueError(f"imsitu_space.json must contain an object: {space_path}")
    action_map = load_object(args.action_map, "action map")
    role_map = load_object(args.role_map, "role map")
    action_resolver = ActionResolver(action_map)

    counts: Counter[str] = Counter()
    changes_by_field: Counter[str] = Counter()
    unresolved_reasons: Counter[str] = Counter()
    with AtomicJsonlWriter(output_path) as output, AtomicJsonlWriter(audit_path) as audit:
        for line_number, row in read_dataset(dataset_path):
            counts["rows_total"] += 1
            if not is_swig_row(row):
                counts["rows_non_swig"] += 1
                output.write(row)
                continue
            counts["rows_swig"] += 1
            annotation, match_reason = match_image(row, annotations, stems)
            if annotation is None:
                counts["swig_unmatched"] += 1
                unresolved_reasons[match_reason] += 1
                audit.write(
                    {
                        "image_id": row.get("image_id"),
                        "line": line_number,
                        "changed_fields": [],
                        "status": "unmatched",
                        "reason": match_reason,
                    }
                )
                if args.require_all_swig_matched:
                    raise ValueError(
                        f"{dataset_path}:{line_number}: cannot match SWiG image: {match_reason}"
                    )
                output.write(row)
                continue

            corrected, audit_record = correct_row(
                row=row,
                annotation=annotation,
                space=space,
                role_map=role_map,
                action_resolver=action_resolver,
                min_votes=args.min_votes,
                allow_ungrounded=args.allow_ungrounded,
            )
            changed_fields = audit_record["changed_fields"]
            counts["swig_matched"] += 1
            if changed_fields:
                counts["swig_rows_changed"] += 1
                changes_by_field.update(changed_fields)
            else:
                counts["swig_rows_unchanged"] += 1
            for field_name, decision in audit_record["fields"].items():
                if decision["source"] != "swig_official":
                    unresolved_reasons[f"{field_name}:{decision['reason']}"] += 1
            audit_record["status"] = "changed" if changed_fields else "unchanged"
            audit.write(audit_record)
            output.write(corrected)

    summary = {
        "dataset_jsonl": str(dataset_path),
        "swig_json_dir": str(swig_dir),
        "output": str(output_path),
        "audit": str(audit_path),
        "min_votes": args.min_votes,
        "allow_ungrounded": args.allow_ungrounded,
        "lemminflect_available": action_resolver.get_lemma is not None,
        "counts": dict(sorted(counts.items())),
        "changes_by_field": dict(sorted(changes_by_field.items())),
        "unresolved_reasons": dict(unresolved_reasons.most_common()),
    }
    atomic_write_json(summary_path, summary)

    print(f"corrected dataset: {output_path}")
    print(f"audit:             {audit_path}")
    print(f"summary:           {summary_path}")
    print(
        "SWiG rows: "
        f"{counts['rows_swig']}; matched: {counts['swig_matched']}; "
        f"changed: {counts['swig_rows_changed']}; unmatched: {counts['swig_unmatched']}"
    )
    if action_resolver.get_lemma is None and not action_map:
        print(
            "warning: lemminflect is not installed and no --action-map was supplied; "
            "mismatching VLM actions were preserved",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
