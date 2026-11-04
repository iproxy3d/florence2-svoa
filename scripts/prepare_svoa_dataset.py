#!/usr/bin/env python3
"""Build a resumable image -> SVOA dataset through an OpenAI-compatible API.

Install:
    python -m pip install "openai>=1.40" "Pillow>=10.0"

Create the local environment file once:
    cp .env.example .env
    # Edit .env and insert a newly issued API key.

Input can be dataset image directories:
    python prepare_svoa_dataset.py \
        --source-dir swig=/data/swig/images_512 \
        --source-dir gqa=/data/gqa/images \
        --source-dir openimages=/data/openimages \
        --work-dir ./svoa_work \
        --limit-per-dataset 1

Or a JSONL manifest. Each line must contain:
    {"dataset":"swig","image_id":"abc","image":"/path/or/https://..."}

Run one image from every dataset in the manifest:
    python prepare_svoa_dataset.py \
        --manifest sources.jsonl \
        --work-dir ./svoa_work \
        --limit-per-dataset 1

Run all remaining images later:
    python prepare_svoa_dataset.py \
        --manifest sources.jsonl \
        --work-dir ./svoa_work

The SQLite database in work-dir is the source of truth. Completed rows are
not sent to the API again while the teacher model and prompt version remain
unchanged. A model or prompt update automatically queues them for relabelling.
dataset.jsonl is rebuilt atomically from SQLite.
Configuration is read from process environment variables and optionally from
.env. Existing environment variables override values from .env. The script
never writes secrets to its cache, SQLite database, or output dataset.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_BASE_URL = "https://api.selectel.ru/aig/v1"
DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
DEFAULT_SIZE = 768
PROMPT_VERSION = "event_roles_v4"
FIELDS = ("subject", "action", "object", "adjuncts")
WORD_RE = re.compile(r"^(?:[a-z]+(?:-[a-z]+)*|none|unknown)$")
ACTION_RE = re.compile(
    r"^(?:[a-z][a-z0-9]*(?:[- ][a-z0-9]+)*|none|unknown)$"
)

PROMPT = """Extract one main directly observable event from the visually dominant coherent region of this image.

Return exactly one JSON object with exactly these four keys:
{"subject":"...","action":"...","object":"...","adjuncts":"..."}

Rules:
- subject, object and adjuncts must each be exactly one lowercase English word
  with no spaces. A hyphenated word is allowed.
- subject: always name the main visible entity or coherent group, even when
  action is "none". Use "none" only when no entity can be identified.
- action: a directly observable base-form English verb or a canonical
  communicative gesture label. A gesture label may contain multiple lowercase
  words, for example "little finger", "thumb index2", or "take picture".
  Prefer a meaningful interaction such as hold, feed, cut, touch, look,
  follow, smile, run, like, dislike, or ok over a generic static posture.
- If no action is directly observable, use "none". Never invent sit, stand,
  pose, or another action only to fill the field.
- object: only the visible target or patient of that exact action. Do not put a
  nearby object, background, location, or companion here. If the action has no
  direct object, use "none".
- adjuncts: one useful visible circumstance. Prefer a meaningful companion,
  concrete setting, attribute, place, time, or manner over generic indoors or
  outdoors.
- Use "none" when a role is absent.
- Use "unknown" when a role exists but cannot be identified reliably.
- For a face or person, visible actions such as smile, frown, look, cry, shout,
  hold, or touch are valid. Do not replace them with emotion labels.
- If the image is a collage or split screen, describe only the main event in
  the largest coherent visual region and ignore the smaller regions.
- Describe only visible facts. Do not infer emotion, intention, personality,
  danger, symbolism, relationships, or the viewer's reaction.
- Choose one main event only.
- Do not add markdown, explanations, alternatives, or extra keys.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_env_file(path: Path) -> None:
    """Load a small KEY=VALUE .env file without overriding real environment."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"{path}:{line_number}: invalid variable name {key!r}")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_name(dataset: str, image_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._-")[:80]
    readable = readable or "image"
    suffix = hashlib.sha256(f"{dataset}\0{image_id}".encode()).hexdigest()[:12]
    return f"{readable}-{suffix}"


@dataclass(frozen=True)
class SourceItem:
    dataset: str
    image_id: str
    image: str
    metadata: dict[str, Any]
    manifest_dir: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.image_id}"


def read_manifest(path: Path) -> list[SourceItem]:
    items: list[SourceItem] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            dataset = str(row.get("dataset", "")).strip()
            image_id = str(row.get("image_id", "")).strip()
            image = str(
                row.get("image") or row.get("image_url") or row.get("image_path") or ""
            ).strip()
            if not dataset or not image_id or not image:
                raise ValueError(
                    f"{path}:{line_number}: dataset, image_id and image are required"
                )
            key = f"{dataset}:{image_id}"
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
            seen.add(key)
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {"dataset", "image_id", "image", "image_url", "image_path"}
            }
            items.append(SourceItem(dataset, image_id, image, metadata, path.parent))
    if not items:
        raise ValueError(f"manifest is empty: {path}")
    return items


def scan_source_dirs(specifications: list[str]) -> list[SourceItem]:
    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    items: list[SourceItem] = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(
                f"invalid --source-dir {specification!r}; expected DATASET=/image/directory"
            )
        dataset, raw_directory = specification.split("=", 1)
        dataset = dataset.strip()
        directory = Path(raw_directory).expanduser().resolve()
        if not dataset or not directory.is_dir():
            raise ValueError(f"invalid source directory: {specification!r}")
        for image_path in sorted(directory.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in supported:
                continue
            relative = image_path.relative_to(directory).as_posix()
            items.append(
                SourceItem(
                    dataset=dataset,
                    image_id=relative,
                    image=str(image_path),
                    metadata={},
                    manifest_dir=directory,
                )
            )
    return items


def merge_unique_items(groups: Iterable[Iterable[SourceItem]]) -> list[SourceItem]:
    result: list[SourceItem] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.key in seen:
                raise ValueError(f"duplicate dataset/image id: {item.key}")
            seen.add(item.key)
            result.append(item)
    return result


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS items (
                item_key TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                image_id TEXT NOT NULL,
                source_image TEXT NOT NULL,
                source_metadata_json TEXT NOT NULL,
                original_path TEXT,
                normalized_path TEXT,
                source_sha256 TEXT,
                normalized_sha256 TEXT,
                original_width INTEGER,
                original_height INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                api_attempts INTEGER NOT NULL DEFAULT 0,
                subject TEXT,
                action TEXT,
                object TEXT,
                adjuncts TEXT,
                raw_response TEXT,
                last_error TEXT,
                teacher_model TEXT,
                prompt_version TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(dataset, image_id)
            );
            CREATE INDEX IF NOT EXISTS idx_items_dataset_status
                ON items(dataset, status);
            """
        )

    def close(self) -> None:
        self.connection.close()

    def register(self, items: Iterable[SourceItem]) -> None:
        now = utc_now()
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO items (
                    item_key, dataset, image_id, source_image,
                    source_metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_key) DO UPDATE SET
                    source_image = excluded.source_image,
                    source_metadata_json = excluded.source_metadata_json,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        item.key,
                        item.dataset,
                        item.image_id,
                        item.image,
                        json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                    )
                    for item in items
                ],
            )

    def completed_count(
        self,
        dataset: str,
        teacher_model: str,
        prompt_version: str,
    ) -> int:
        row = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM items
            WHERE dataset=? AND status='done'
              AND teacher_model=? AND prompt_version=?
            """,
            (dataset, teacher_model, prompt_version),
        ).fetchone()
        return int(row["count"])

    def get(self, item_key: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM items WHERE item_key=?", (item_key,)
        ).fetchone()
        if row is None:
            raise KeyError(item_key)
        return row

    def update(self, item_key: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{name}=?" for name in values)
        with self.connection:
            self.connection.execute(
                f"UPDATE items SET {columns} WHERE item_key=?",
                (*values.values(), item_key),
            )

    def increment_attempt(self, item_key: str) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE items
                SET api_attempts=api_attempts+1, status='processing', updated_at=?
                WHERE item_key=?
                """,
                (utc_now(), item_key),
            )

    def done_rows(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM items WHERE status='done' ORDER BY dataset, image_id"
            )
        )


def is_http_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme.lower() in {"http", "https"}


def cache_source(item: SourceItem, destination: Path, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        if is_http_url(item.image):
            request = urllib.request.Request(
                item.image,
                headers={"User-Agent": "svoa-dataset-builder/1.0"},
            )
            with closing(urllib.request.urlopen(request, timeout=timeout)) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
        else:
            source = Path(item.image)
            if not source.is_absolute():
                source = (item.manifest_dir / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copyfile(source, temporary)
        if temporary.stat().st_size == 0:
            raise ValueError("empty image file")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def normalize_image(source: Path, destination: Path, size: int) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        with Image.open(source) as original:
            return original.size

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            original_size = image.size
            if image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = image.convert("RGB")
            # Florence-2-base-ft uses an exact 768x768 bicubic resize and no crop.
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            image.save(temporary, format="JPEG", quality=95, optimize=True)
        os.replace(temporary, destination)
        return original_size
    except (UnidentifiedImageError, OSError, ValueError):
        temporary.unlink(missing_ok=True)
        raise


def image_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                chunks.append(str(part.text))
        return "".join(chunks).strip()
    return str(content).strip()


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def validate_svoa(value: dict[str, Any]) -> dict[str, str]:
    if set(value) != set(FIELDS):
        raise ValueError(f"expected exactly keys {FIELDS}, got {tuple(value)}")
    normalized: dict[str, str] = {}
    for field in FIELDS:
        label = " ".join(str(value[field]).strip().lower().split())
        pattern = ACTION_RE if field == "action" else WORD_RE
        if not pattern.fullmatch(label):
            requirement = (
                "a lowercase English verb or canonical gesture label"
                if field == "action"
                else "one lowercase English word"
            )
            raise ValueError(f"{field} must be {requirement}: {label!r}")
        normalized[field] = label
    return normalized


def request_svoa(client: Any, model: str, image_path: Path, max_tokens: int) -> tuple[dict[str, str], str]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                ],
            }
        ],
    )
    raw = response_text(response)
    return validate_svoa(extract_json(raw)), raw


def export_jsonl(db: StateDB, output_path: Path, work_dir: Path, model: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in db.done_rows():
            normalized_path = Path(row["normalized_path"])
            try:
                image_path = normalized_path.relative_to(work_dir).as_posix()
            except ValueError:
                image_path = normalized_path.as_posix()
            record = {
                "image_id": row["item_key"],
                "image_path": image_path,
                "subject": row["subject"],
                "action": row["action"],
                "object": row["object"],
                "adjuncts": row["adjuncts"],
                "teacher_model": model,
                "prompt_version": row["prompt_version"] or PROMPT_VERSION,
            }
            record["teacher_model"] = row["teacher_model"] or model
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, output_path)
    return count


def selected_items(
    items: list[SourceItem],
    db: StateDB,
    limit_per_dataset: int,
    max_total_attempts: int,
    teacher_model: str,
    prompt_version: str,
) -> list[SourceItem]:
    remaining: dict[str, int | None] = {}
    selected: list[SourceItem] = []
    for item in items:
        if item.dataset not in remaining:
            if limit_per_dataset > 0:
                remaining[item.dataset] = max(
                    0,
                    limit_per_dataset
                    - db.completed_count(
                        item.dataset,
                        teacher_model,
                        prompt_version,
                    ),
                )
            else:
                remaining[item.dataset] = None
        row = db.get(item.key)
        result_is_current = (
            row["status"] == "done"
            and row["teacher_model"] == teacher_model
            and row["prompt_version"] == prompt_version
        )
        if result_is_current:
            continue
        # The attempt limit applies to retries of the current labelling setup.
        # A changed model or prompt is a new setup and must be allowed to run.
        if (
            row["teacher_model"] == teacher_model
            and row["prompt_version"] == prompt_version
            and int(row["api_attempts"]) >= max_total_attempts
        ):
            continue
        quota = remaining[item.dataset]
        if quota is not None and quota <= 0:
            continue
        selected.append(item)
        if quota is not None:
            remaining[item.dataset] = quota - 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="recursively scan a local image directory; may be repeated",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("svoa_work"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env", default="SELECTEL_API_KEY")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--api-attempts", type=int, default=3)
    parser.add_argument("--max-total-attempts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="cache/resize only; do not call API")
    parser.add_argument("--export-only", action="store_true", help="only rebuild dataset.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.api_attempts <= 0 or args.max_total_attempts <= 0:
        raise SystemExit("attempt counts must be positive")

    if args.env_file.is_file():
        load_env_file(args.env_file)

    base_url = args.base_url or os.environ.get("SELECTEL_BASE_URL") or DEFAULT_BASE_URL
    model = args.model or os.environ.get("SELECTEL_MODEL") or DEFAULT_MODEL

    work_dir = args.work_dir.resolve()
    output_path = (args.output or (work_dir / "dataset.jsonl")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    groups: list[list[SourceItem]] = []
    if args.manifest:
        groups.append(read_manifest(args.manifest.resolve()))
    if args.source_dir:
        groups.append(scan_source_dirs(args.source_dir))
    if not groups:
        raise SystemExit("provide --manifest and/or at least one --source-dir")
    items = merge_unique_items(groups)
    if not items:
        raise SystemExit("no supported images found")
    db = StateDB(work_dir / "state.sqlite3")
    db.register(items)

    if args.export_only:
        count = export_jsonl(db, output_path, work_dir, model)
        print(f"exported {count} rows to {output_path}")
        db.close()
        return 0

    client = None
    if not args.dry_run:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"environment variable {args.api_key_env} is required; "
                "the API key must not be stored in source code"
            )
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

    queue = selected_items(
        items,
        db,
        args.limit_per_dataset,
        args.max_total_attempts,
        model,
        PROMPT_VERSION,
    )
    if not queue:
        count = export_jsonl(db, output_path, work_dir, model)
        print(f"nothing to process; dataset contains {count} rows")
        db.close()
        return 0

    failures = 0
    for index, item in enumerate(queue, 1):
        stem = stable_name(item.dataset, item.image_id)
        original_path = work_dir / "cache" / "original" / item.dataset / f"{stem}.bin"
        normalized_path = work_dir / "cache" / "normalized" / item.dataset / f"{stem}.jpg"
        prefix = f"[{index}/{len(queue)}] {item.key}"
        try:
            cache_source(item, original_path, args.download_timeout)
            original_width, original_height = normalize_image(
                original_path, normalized_path, args.size
            )
            db.update(
                item.key,
                original_path=str(original_path),
                normalized_path=str(normalized_path),
                source_sha256=sha256_file(original_path),
                normalized_sha256=sha256_file(normalized_path),
                original_width=original_width,
                original_height=original_height,
                status="normalized" if args.dry_run else "pending",
                last_error=None,
            )
            if args.dry_run:
                print(f"{prefix}: cached and normalized")
                continue

            current = db.get(item.key)
            if (
                current["teacher_model"] != model
                or current["prompt_version"] != PROMPT_VERSION
            ):
                # Retry accounting belongs to one exact teacher/prompt setup.
                # A new setup must not inherit exhausted attempts from an old one.
                db.update(
                    item.key,
                    status="pending",
                    api_attempts=0,
                    last_error=None,
                )

            last_error: Exception | None = None
            for attempt in range(1, args.api_attempts + 1):
                if int(db.get(item.key)["api_attempts"]) >= args.max_total_attempts:
                    last_error = RuntimeError("maximum total API attempts reached")
                    break
                db.increment_attempt(item.key)
                try:
                    result, raw = request_svoa(
                        client, model, normalized_path, args.max_tokens
                    )
                    db.update(
                        item.key,
                        status="done",
                        raw_response=raw,
                        last_error=None,
                        teacher_model=model,
                        prompt_version=PROMPT_VERSION,
                        **result,
                    )
                    print(
                        f"{prefix}: "
                        + "|".join(result[field] for field in FIELDS)
                    )
                    last_error = None
                    break
                except Exception as exc:  # API and validation failures are retryable.
                    last_error = exc
                    db.update(item.key, status="failed", last_error=str(exc))
                    if attempt < args.api_attempts:
                        time.sleep(min(2 ** (attempt - 1), 8))
            if last_error is not None:
                failures += 1
                print(f"{prefix}: failed: {last_error}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            db.update(item.key, status="failed", last_error=str(exc))
            print(f"{prefix}: failed: {exc}", file=sys.stderr)

    count = export_jsonl(db, output_path, work_dir, model)
    db.close()
    print(f"exported {count} completed rows to {output_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
