#!/usr/bin/env python3
"""Probe SVOA image sources and download a small resumable sample.

The default command tries to keep exactly one image from every configured
source::

    python download_svoa_images.py \
        --output-dir ./downloaded_images \
        --limit-per-dataset 1

The program never downloads a complete large archive as a fallback. Sources
are handled in one of two ways:

* individual images are requested by URL/ID (GQA, Open Images, Mutual Gaze);
* selected members are read from a remote ZIP with HTTP Range requests
  (SWiG, EasyPortrait, WIDER FACE, HaGRIDv2).

If selective access is unavailable, the dataset directory is still created
and receives ``_PLACE_IMAGES_HERE.txt``. Images can be copied there manually
at a later stage. A failed probe is expected and does not make the command
fail unless ``--strict`` is supplied.

Progress is stored in SQLite. ``manifest.jsonl`` contains every successfully
downloaded or previously cached image, and ``source_status.json`` records
which sources supported selective access during the latest run.

Install dependencies::

    python -m pip install "Pillow>=10" "remotezip>=0.12"

``--limit-per-dataset`` is a target total, not an amount added on every run.
For HaGRID it is the target for each selected gesture; override it with
``--hagrid-limit-per-gesture``.  The default still probes only ``no_gesture``.
Select useful interaction labels explicitly, for example::

    python download_svoa_images.py \
        --datasets hagrid \
        --hagrid-gestures like dislike ok little_finger thumb_index \
        --hagrid-limit-per-gesture 100 \
        --hagrid-download-annotations

Images are preserved at their source dimensions. Resizing and SVOA labelling
remain the responsibility of ``prepare_svoa_dataset.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


DEFAULT_SWIG_URL = (
    "https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip"
)
DEFAULT_GQA_DATASET = "wliafe/GQA200"
DEFAULT_OPENIMAGES_DATASET = "bitmind/open-images-v7"
DEFAULT_DATASET_VIEWER_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_MUTUAL_GAZE_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "google-research-datasets/Google-Open-Images-Mutual-Gaze-dataset/"
    "main/train_data.csv"
)
DEFAULT_EASYPORTRAIT_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/easyportrait/v2/train.zip"
)
DEFAULT_WIDERFACE_URL = (
    "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/"
    "833d07e7bf3860f294242312fe95eed0561eeb17/data/WIDER_train.zip"
)
DEFAULT_HAGRID_URL_TEMPLATE = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/hagrid_v2_zip/{gesture}.zip"
)
DEFAULT_HAGRID_ANNOTATIONS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)
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

DATASETS = (
    "swig",
    "gqa",
    "openimages",
    "mutual_gaze",
    "easyportrait",
    "widerface",
    "hagrid",
)

SOURCE_MODES = {
    "swig": "remote_zip",
    "gqa": "individual",
    "openimages": "individual",
    "mutual_gaze": "individual",
    "easyportrait": "remote_zip",
    "widerface": "remote_zip",
    "hagrid": "remote_zip",
}

MANUAL_SOURCE_NOTES = {
    "swig": "Copy SWiG image files here if the official remote ZIP is unavailable.",
    "gqa": "Copy ordinary GQA scene images here if Dataset Viewer is unavailable.",
    "openimages": "Copy selected Open Images JPG/PNG files here.",
    "mutual_gaze": (
        "Copy full Open Images frames containing two visible people here; "
        "do not crop the faces separately."
    ),
    "easyportrait": (
        "Copy selected EasyPortrait RGB images here. Do not copy segmentation "
        "masks. Prefer the official train split."
    ),
    "widerface": (
        "Copy selected WIDER FACE images here. Prefer scenes with 1-4 "
        "sufficiently large faces and avoid dense crowds."
    ),
    "hagrid": (
        "Copy selected HaGRIDv2 RGB images here. Preserve the official class "
        "directory or image identifier so annotations can be matched later."
    ),
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_name(dataset: str, image_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", image_id).strip("._-")[:80]
    readable = readable or "image"
    suffix = hashlib.sha256(f"{dataset}\0{image_id}".encode()).hexdigest()[:12]
    return f"{readable}-{suffix}.jpg"


def image_id_from_url(index: Any, url: str) -> str:
    if index is not None and str(index).strip():
        return str(index).strip()
    return hashlib.sha256(url.encode()).hexdigest()[:24]


def dataset_seed(seed: int, dataset: str) -> int:
    digest = hashlib.sha256(dataset.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class DownloadedImage:
    dataset: str
    image_id: str
    source_ref: str
    local_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class SourceStatus:
    dataset: str
    mode: str
    status: str
    downloaded: int
    target: int
    directory: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "mode": self.mode,
            "status": self.status,
            "downloaded": self.downloaded,
            "target": self.target,
            "directory": self.directory,
            "detail": self.detail,
        }


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS downloads (
                dataset TEXT NOT NULL,
                image_id TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                local_path TEXT,
                width INTEGER,
                height INTEGER,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(dataset, image_id)
            );

            CREATE INDEX IF NOT EXISTS idx_downloads_dataset_status
                ON downloads(dataset, status);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        self.connection.close()

    def success_count(self, dataset: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM downloads "
            "WHERE dataset = ? AND status = 'done'",
            (dataset,),
        ).fetchone()
        return int(row["count"])

    def successful_ids(self, dataset: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT image_id FROM downloads "
            "WHERE dataset = ? AND status = 'done'",
            (dataset,),
        )
        return {str(row["image_id"]) for row in rows}

    def successful_paths(self, dataset: str) -> set[Path]:
        rows = self.connection.execute(
            "SELECT local_path FROM downloads "
            "WHERE dataset = ? AND status = 'done' AND local_path IS NOT NULL",
            (dataset,),
        )
        return {Path(str(row["local_path"])).resolve() for row in rows}

    def get_offset(self, dataset: str) -> int:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (f"{dataset}.next_offset",),
        ).fetchone()
        return int(row["value"]) if row else 0

    def set_offset(self, dataset: str, value: int) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (f"{dataset}.next_offset", str(value)),
        )
        self.connection.commit()

    def record_success(self, item: DownloadedImage) -> None:
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO downloads(
                dataset, image_id, source_ref, local_path, width, height,
                status, attempts, last_error, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, 'done', 1, NULL, ?, ?)
            ON CONFLICT(dataset, image_id) DO UPDATE SET
                source_ref = excluded.source_ref,
                local_path = excluded.local_path,
                width = excluded.width,
                height = excluded.height,
                status = 'done',
                attempts = downloads.attempts + 1,
                last_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                item.dataset,
                item.image_id,
                item.source_ref,
                str(item.local_path.resolve()),
                item.width,
                item.height,
                now,
                now,
            ),
        )
        self.connection.commit()

    def record_failure(
        self, dataset: str, image_id: str, source_ref: str, error: Exception
    ) -> None:
        now = utc_now()
        message = f"{type(error).__name__}: {error}"[:2000]
        self.connection.execute(
            """
            INSERT INTO downloads(
                dataset, image_id, source_ref, status, attempts, last_error,
                created_at, updated_at
            ) VALUES(?, ?, ?, 'failed', 1, ?, ?, ?)
            ON CONFLICT(dataset, image_id) DO UPDATE SET
                source_ref = excluded.source_ref,
                status = 'failed',
                attempts = downloads.attempts + 1,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (dataset, image_id, source_ref, message, now, now),
        )
        self.connection.commit()

    def export_manifest(self, path: Path) -> None:
        rows = self.connection.execute(
            """
            SELECT dataset, image_id, source_ref, local_path, width, height
            FROM downloads
            WHERE status = 'done'
            ORDER BY dataset, image_id
            """
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                local_path = Path(str(row["local_path"]))
                if not local_path.is_file():
                    continue
                record = {
                    "dataset": row["dataset"],
                    "image_id": row["image_id"],
                    "image": str(local_path.resolve()),
                    "source_ref": row["source_ref"],
                    "width": row["width"],
                    "height": row["height"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def save_image(
    image: Image.Image,
    dataset: str,
    image_id: str,
    source_ref: str,
    output_dir: Path,
) -> DownloadedImage:
    """Validate and store an image without changing its dimensions."""
    image = ImageOps.exif_transpose(image)
    image.load()
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    if image.mode != "RGB":
        image = image.convert("RGB")

    destination_dir = output_dir / dataset
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / stable_name(dataset, image_id)
    temporary = destination.with_suffix(destination.suffix + ".part")
    image.save(temporary, format="JPEG", quality=95, subsampling=0)
    temporary.replace(destination)
    return DownloadedImage(
        dataset=dataset,
        image_id=image_id,
        source_ref=source_ref,
        local_path=destination,
        width=width,
        height=height,
    )


def request_headers(accept: str) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "svoa-dataset-builder/3.0",
    }
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def retry_delay(attempt: int) -> float:
    return min(8.0, 2.0**attempt)


def fetch_bytes(
    url: str,
    timeout: float,
    maximum: int,
    accept: str,
    attempts: int = 5,
) -> bytes:
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=request_headers(accept))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read(maximum + 1)
            if len(payload) > maximum:
                raise ValueError(f"response exceeds the {maximum}-byte safety limit")
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if retryable and attempt + 1 < attempts:
                time.sleep(retry_delay(attempt))
                continue
            try:
                detail = exc.read(2000).decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"HTTP {exc.code} for {url}{suffix}") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 < attempts:
                time.sleep(retry_delay(attempt))
                continue
            raise RuntimeError(f"cannot access {url}: {exc.reason}") from exc
    raise RuntimeError(f"cannot access {url} after {attempts} attempts")


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    payload = fetch_bytes(
        url,
        timeout=timeout,
        maximum=10 * 1024 * 1024,
        accept="application/json",
    )
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("server returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("server returned an unexpected JSON response")
    return result


def dataset_viewer_rows(
    api_url: str,
    dataset_name: str,
    split: str,
    offset: int,
    length: int,
    timeout: float,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset_name,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": max(1, min(100, length)),
        }
    )
    result = fetch_json(f"{api_url}?{query}", timeout)
    rows = result.get("rows")
    if not isinstance(rows, list):
        error = result.get("error") or result.get("message") or "missing rows"
        raise RuntimeError(f"Dataset Viewer error for {dataset_name}: {error}")
    return [item for item in rows if isinstance(item, dict)]


def fetch_remote_image(url: str, timeout: float, attempts: int = 3) -> Image.Image:
    maximum = 50 * 1024 * 1024
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers=request_headers("image/*"))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type().lower()
                if content_type.startswith("text/") or content_type in {
                    "application/json",
                    "application/xml",
                    "text/xml",
                }:
                    raise ValueError(
                        f"URL returned {content_type!r}, not binary image data"
                    )
                payload = response.read(maximum + 1)
            if len(payload) > maximum:
                raise ValueError("image exceeds the 50 MiB safety limit")
            return Image.open(BytesIO(payload))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if retryable and attempt + 1 < attempts:
                time.sleep(retry_delay(attempt))
                continue
            raise RuntimeError(f"HTTP {exc.code} while downloading image") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 < attempts:
                time.sleep(retry_delay(attempt))
                continue
            raise RuntimeError(f"cannot download image: {exc.reason}") from exc
    raise RuntimeError(f"cannot download image after {attempts} attempts")


def shuffled_image_members(names: Iterable[str], seed: int) -> list[str]:
    images = [
        name
        for name in names
        if not name.endswith("/") and Path(name).suffix.lower() in IMAGE_SUFFIXES
    ]
    randomizer = random.Random(seed)
    randomizer.shuffle(images)
    return images


def download_remote_zip(
    dataset: str,
    db: StateDB,
    output_dir: Path,
    target: int,
    seed: int,
    archive_url: str,
    max_errors: int,
) -> None:
    if db.success_count(dataset) >= target:
        print(f"{dataset}: target already reached ({target})")
        return
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError(
            'missing dependency: install it with `python -m pip install "remotezip>=0.12"`'
        ) from exc

    known = db.successful_ids(dataset)
    errors = 0
    archive = RemoteZip(archive_url)
    try:
        members = shuffled_image_members(
            (info.filename for info in archive.infolist()),
            dataset_seed(seed, dataset),
        )
        if not members:
            raise RuntimeError("remote ZIP contains no supported image files")
        for member in members:
            image_id = member
            if image_id in known:
                continue
            source_ref = f"{archive_url}#{member}"
            try:
                with archive.open(member) as source:
                    image = Image.open(BytesIO(source.read()))
                    item = save_image(
                        image, dataset, image_id, source_ref, output_dir
                    )
                db.record_success(item)
                known.add(image_id)
                print(f"{dataset}: {db.success_count(dataset)}/{target} {member}")
            except Exception as exc:
                db.record_failure(dataset, image_id, source_ref, exc)
                errors += 1
                print(f"{dataset}: skipped {member}: {exc}", file=sys.stderr)
                if errors >= max_errors:
                    raise RuntimeError(
                        f"{dataset}: maximum error count reached"
                    ) from exc
            if db.success_count(dataset) >= target:
                return
    finally:
        archive.close()

    raise RuntimeError(
        f"{dataset}: archive ended before target {target}; "
        f"only {db.success_count(dataset)} images are available locally"
    )


def hagrid_success_count(db: StateDB, gesture: str) -> int:
    """Return the number of downloaded HaGRID images for one gesture."""

    prefix = f"{gesture}/"
    return sum(
        image_id.startswith(prefix)
        for image_id in db.successful_ids("hagrid")
    )


def hagrid_selected_success_count(
    db: StateDB, gestures: Iterable[str]
) -> int:
    return sum(hagrid_success_count(db, gesture) for gesture in gestures)


def hagrid_archive_url(args: argparse.Namespace, gesture: str) -> str:
    """Resolve the official or explicitly supplied archive URL."""

    custom_url = getattr(args, "hagrid_url", None)
    gestures = list(getattr(args, "hagrid_gestures", []))
    if custom_url:
        if len(gestures) != 1:
            raise ValueError(
                "--hagrid-url can only be used with exactly one "
                "--hagrid-gestures value"
            )
        return str(custom_url)
    try:
        return str(args.hagrid_url_template).format(gesture=gesture)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "--hagrid-url-template must contain a valid {gesture} placeholder"
        ) from exc


def download_hagrid(
    args: argparse.Namespace,
    db: StateDB,
    output_dir: Path,
) -> None:
    """Download an equal, resumable sample from selected HaGRID classes."""

    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise RuntimeError(
            'missing dependency: install it with `python -m pip install '
            '"remotezip>=0.12"`'
        ) from exc

    target = args.hagrid_limit_per_gesture or args.limit_per_dataset
    known = db.successful_ids("hagrid")
    for gesture in args.hagrid_gestures:
        current = hagrid_success_count(db, gesture)
        if current >= target:
            print(f"hagrid/{gesture}: target already reached ({target})")
            continue

        archive_url = hagrid_archive_url(args, gesture)
        errors = 0
        archive = RemoteZip(archive_url)
        try:
            members = shuffled_image_members(
                (info.filename for info in archive.infolist()),
                dataset_seed(args.seed, f"hagrid/{gesture}"),
            )
            if not members:
                raise RuntimeError(
                    f"HaGRID archive {gesture!r} contains no supported images"
                )
            for member in members:
                image_id = f"{gesture}/{member}"
                if image_id in known:
                    continue
                source_ref = f"{archive_url}#{member}"
                try:
                    with archive.open(member) as source:
                        image = Image.open(BytesIO(source.read()))
                        item = save_image(
                            image,
                            "hagrid",
                            image_id,
                            source_ref,
                            output_dir,
                        )
                    db.record_success(item)
                    known.add(image_id)
                    current = hagrid_success_count(db, gesture)
                    print(f"hagrid/{gesture}: {current}/{target} {member}")
                except Exception as exc:
                    db.record_failure("hagrid", image_id, source_ref, exc)
                    errors += 1
                    print(
                        f"hagrid/{gesture}: skipped {member}: {exc}",
                        file=sys.stderr,
                    )
                    if errors >= args.max_errors:
                        raise RuntimeError(
                            f"hagrid/{gesture}: maximum error count reached"
                        ) from exc
                if hagrid_success_count(db, gesture) >= target:
                    break
        finally:
            archive.close()

        current = hagrid_success_count(db, gesture)
        if current < target:
            raise RuntimeError(
                f"hagrid/{gesture}: archive ended before target {target}; "
                f"only {current} images are available locally"
            )


def download_and_extract_hagrid_annotations(
    url: str,
    output_dir: Path,
    timeout: float,
) -> Path:
    """Download and safely extract the official HaGRID annotation archive."""

    target = output_dir / "hagrid_annotations"
    marker = target / "_COMPLETE"
    if marker.exists():
        print(f"hagrid annotations: already available in {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "hagrid_annotations.zip.part"
    request = urllib.request.Request(url, headers=request_headers("application/zip"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with archive_path.open("wb") as destination:
                shutil.copyfileobj(response, destination)

        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        f"unsafe path in annotation archive: {member.filename}"
                    )
            archive.extractall(target)
        marker.write_text(f"source={url}\n", encoding="utf-8")
    finally:
        archive_path.unlink(missing_ok=True)

    print(f"hagrid annotations: extracted to {target}")
    return target


def download_gqa(
    db: StateDB,
    output_dir: Path,
    target: int,
    dataset_name: str,
    viewer_url: str,
    timeout: float,
    max_errors: int,
) -> None:
    if db.success_count("gqa") >= target:
        print(f"gqa: target already reached ({target})")
        return
    offset = db.get_offset("gqa")
    known = db.successful_ids("gqa")
    errors = 0
    while db.success_count("gqa") < target:
        page_length = min(100, max(1, target - db.success_count("gqa")))
        wrappers = dataset_viewer_rows(
            viewer_url, dataset_name, "train", offset, page_length, timeout
        )
        if not wrappers:
            raise RuntimeError(
                f"gqa: Dataset Viewer ended at offset {offset} before target {target}"
            )
        for wrapper in wrappers:
            index = int(wrapper.get("row_idx", offset))
            row = wrapper.get("row")
            if not isinstance(row, dict):
                db.set_offset("gqa", index + 1)
                offset = index + 1
                continue
            image_id = str(row.get("id") or f"row-{index}")
            source_ref = f"hf://datasets/{dataset_name}/train/{image_id}"
            try:
                if image_id in known:
                    continue
                image_cell = row.get("image")
                if not isinstance(image_cell, dict):
                    raise ValueError("row has no image object")
                image_url = str(image_cell.get("src") or "").strip()
                if not image_url.startswith(("http://", "https://")):
                    raise ValueError("row has no valid image URL")
                image = fetch_remote_image(image_url, timeout)
                item = save_image(image, "gqa", image_id, source_ref, output_dir)
                db.record_success(item)
                known.add(image_id)
                print(f"gqa: {db.success_count('gqa')}/{target} {image_id}")
            except Exception as exc:
                db.record_failure("gqa", image_id, source_ref, exc)
                errors += 1
                print(f"gqa: skipped {image_id}: {exc}", file=sys.stderr)
                if errors >= max_errors:
                    raise RuntimeError("gqa: maximum error count reached") from exc
            finally:
                db.set_offset("gqa", index + 1)
                offset = index + 1
            if db.success_count("gqa") >= target:
                return


def download_openimages(
    db: StateDB,
    output_dir: Path,
    target: int,
    dataset_name: str,
    viewer_url: str,
    timeout: float,
    max_errors: int,
) -> None:
    if db.success_count("openimages") >= target:
        print(f"openimages: target already reached ({target})")
        return
    offset = db.get_offset("openimages")
    known = db.successful_ids("openimages")
    errors = 0
    while db.success_count("openimages") < target:
        page_length = min(100, max(1, target - db.success_count("openimages")))
        wrappers = dataset_viewer_rows(
            viewer_url, dataset_name, "train", offset, page_length, timeout
        )
        if not wrappers:
            raise RuntimeError(
                "openimages: Dataset Viewer ended at offset "
                f"{offset} before target {target}"
            )
        for wrapper in wrappers:
            index = int(wrapper.get("row_idx", offset))
            row = wrapper.get("row")
            if not isinstance(row, dict):
                db.set_offset("openimages", index + 1)
                offset = index + 1
                continue
            url = str(row.get("url") or "").strip()
            image_id = image_id_from_url(row.get("index"), url)
            try:
                if image_id in known:
                    continue
                if not url.startswith(("http://", "https://")):
                    raise ValueError("row has no valid image URL")
                image = fetch_remote_image(url, timeout)
                item = save_image(image, "openimages", image_id, url, output_dir)
                db.record_success(item)
                known.add(image_id)
                print(
                    f"openimages: {db.success_count('openimages')}/{target} "
                    f"{image_id}"
                )
            except Exception as exc:
                db.record_failure("openimages", image_id, url or "missing-url", exc)
                errors += 1
                print(f"openimages: skipped {image_id}: {exc}", file=sys.stderr)
                if errors >= max_errors:
                    raise RuntimeError(
                        "openimages: maximum error count reached"
                    ) from exc
            finally:
                db.set_offset("openimages", index + 1)
                offset = index + 1
            if db.success_count("openimages") >= target:
                return


def get_case_insensitive(row: dict[str, str], *names: str) -> str:
    normalized = {key.strip().lower(): value for key, value in row.items() if key}
    for name in names:
        value = normalized.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def mutual_gaze_image_urls(image_id: str, preview_url: str) -> list[str]:
    """Return stable Open Images URLs before the rate-limited preview URL."""
    encoded_id = urllib.parse.quote(image_id, safe="")
    candidates = [
        f"https://open-images-dataset.s3.amazonaws.com/train/{encoded_id}.jpg",
        f"https://open-images-dataset.s3.amazonaws.com/validation/{encoded_id}.jpg",
        f"https://open-images-dataset.s3.amazonaws.com/test/{encoded_id}.jpg",
    ]
    if preview_url.startswith(("http://", "https://")):
        candidates.append(preview_url)
    # Preserve priority while avoiding a duplicate preview URL.
    return list(dict.fromkeys(candidates))


def fetch_first_available_image(
    urls: list[str], timeout: float, preview_url: str = ""
) -> tuple[Image.Image, str]:
    errors: list[str] = []
    for url in urls:
        try:
            # A preview host that has already rate-limited the client should not
            # receive repeated retries. Stable Open Images storage keeps the
            # ordinary retry policy for transient errors.
            attempts = 1 if url == preview_url else 3
            return fetch_remote_image(url, timeout, attempts=attempts), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    summary = "; ".join(errors)
    if len(summary) > 1800:
        summary = summary[:1800] + "..."
    raise RuntimeError(f"all image locations failed: {summary}")


def load_mutual_gaze_rows(
    csv_url: str, cache_path: Path, timeout: float
) -> list[dict[str, str]]:
    if cache_path.is_file():
        payload = cache_path.read_bytes()
    else:
        payload = fetch_bytes(
            csv_url,
            timeout=timeout,
            maximum=25 * 1024 * 1024,
            accept="text/csv,*/*;q=0.8",
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
    text = payload.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(StringIO(text))]


def download_mutual_gaze(
    db: StateDB,
    output_dir: Path,
    target: int,
    csv_url: str,
    timeout: float,
    max_errors: int,
) -> None:
    dataset = "mutual_gaze"
    if db.success_count(dataset) >= target:
        print(f"{dataset}: target already reached ({target})")
        return
    rows = load_mutual_gaze_rows(
        csv_url, output_dir / "_cache" / "mutual_gaze_train.csv", timeout
    )
    if not rows:
        raise RuntimeError("mutual_gaze: CSV contains no rows")

    offset = db.get_offset(dataset)
    known = db.successful_ids(dataset)
    errors = 0
    while offset < len(rows) and db.success_count(dataset) < target:
        row = rows[offset]
        image_id = get_case_insensitive(row, "ImageID", "image_id") or f"row-{offset}"
        preview_url = get_case_insensitive(row, "ImageUrl", "image_url", "url")
        source_ref = f"openimages://{image_id}"
        try:
            if image_id in known:
                continue
            image, source_ref = fetch_first_available_image(
                mutual_gaze_image_urls(image_id, preview_url),
                timeout,
                preview_url=preview_url,
            )
            item = save_image(image, dataset, image_id, source_ref, output_dir)
            db.record_success(item)
            known.add(image_id)
            print(f"{dataset}: {db.success_count(dataset)}/{target} {image_id}")
        except Exception as exc:
            db.record_failure(dataset, image_id, source_ref, exc)
            errors += 1
            print(f"{dataset}: skipped {image_id}: {exc}", file=sys.stderr)
            if errors >= max_errors:
                raise RuntimeError(
                    "mutual_gaze: maximum error count reached"
                ) from exc
        finally:
            offset += 1
            db.set_offset(dataset, offset)

    if db.success_count(dataset) < target:
        raise RuntimeError(
            f"mutual_gaze: CSV ended before target {target}; "
            f"only {db.success_count(dataset)} images were downloaded"
        )


def write_manual_note(dataset: str, output_dir: Path, error: str) -> Path:
    directory = output_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "_PLACE_IMAGES_HERE.txt"
    source_note = MANUAL_SOURCE_NOTES[dataset]
    content = (
        f"Dataset: {dataset}\n\n"
        "Selective download did not succeed during the latest probe.\n"
        f"Reason: {error}\n\n"
        f"{source_note}\n\n"
        "Supported extensions: .jpg, .jpeg, .png, .webp, .bmp\n"
        "The preparation script may read images directly from this directory.\n"
    )
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def clear_manual_note(dataset: str, output_dir: Path) -> None:
    path = output_dir / dataset / "_PLACE_IMAGES_HERE.txt"
    if path.is_file():
        path.unlink()


def register_manual_images(dataset: str, output_dir: Path, db: StateDB) -> int:
    """Add user-supplied images from a dataset folder to the shared manifest."""
    directory = output_dir / dataset
    known_paths = db.successful_paths(dataset)
    registered = 0
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        if resolved in known_paths:
            continue
        relative = path.relative_to(directory).as_posix()
        image_id = f"manual:{relative}"
        if dataset == "hagrid":
            class_name = relative.split("/", 1)[0]
            if class_name in HAGRID_GESTURES:
                image_id = f"{class_name}/manual:{relative}"
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                image.load()
                width, height = image.size
            if width < 1 or height < 1:
                raise ValueError(f"invalid image dimensions: {width}x{height}")
            db.record_success(
                DownloadedImage(
                    dataset=dataset,
                    image_id=image_id,
                    source_ref=f"manual://{dataset}/{relative}",
                    local_path=resolved,
                    width=width,
                    height=height,
                )
            )
            known_paths.add(resolved)
            registered += 1
            print(f"{dataset}: registered manual image {relative}")
        except Exception as exc:
            db.record_failure(
                dataset,
                image_id,
                f"manual://{dataset}/{relative}",
                exc,
            )
            print(f"{dataset}: invalid manual image {relative}: {exc}", file=sys.stderr)
    return registered


def export_status(path: Path, statuses: list[SourceStatus]) -> None:
    payload = {
        "generated_at": utc_now(),
        "sources": [status.as_dict() for status in statuses],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe selective access and download a resumable image sample from "
            "seven SVOA sources."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloaded_images"),
        help="dataset folders, manifest, report and SQLite output directory",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=1,
        help="target successful image count for every selected dataset",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="sources to probe; defaults to all seven",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code if any source needs manual images",
    )
    parser.add_argument(
        "--skip-archive-probes",
        action="store_true",
        help="create manual folders for remote-ZIP sources without contacting them",
    )
    parser.add_argument("--swig-url", default=DEFAULT_SWIG_URL)
    parser.add_argument("--gqa-dataset", default=DEFAULT_GQA_DATASET)
    parser.add_argument("--openimages-dataset", default=DEFAULT_OPENIMAGES_DATASET)
    parser.add_argument("--mutual-gaze-csv-url", default=DEFAULT_MUTUAL_GAZE_CSV_URL)
    parser.add_argument("--easyportrait-url", default=DEFAULT_EASYPORTRAIT_URL)
    parser.add_argument("--widerface-url", default=DEFAULT_WIDERFACE_URL)
    parser.add_argument(
        "--hagrid-gestures",
        nargs="+",
        choices=HAGRID_GESTURES,
        default=["no_gesture"],
        help=(
            "official HaGRID classes to sample; defaults to no_gesture. "
            "Use all 34 names explicitly only when they are really needed"
        ),
    )
    parser.add_argument(
        "--hagrid-limit-per-gesture",
        type=int,
        help="target image count for each selected HaGRID class",
    )
    parser.add_argument(
        "--hagrid-url-template",
        default=DEFAULT_HAGRID_URL_TEMPLATE,
        help="HaGRID class archive URL template containing {gesture}",
    )
    parser.add_argument(
        "--hagrid-url",
        help="custom HaGRID ZIP URL; valid only when one class is selected",
    )
    parser.add_argument(
        "--hagrid-download-annotations",
        action="store_true",
        help="also download and extract the official annotation archive",
    )
    parser.add_argument(
        "--hagrid-annotations-url",
        default=DEFAULT_HAGRID_ANNOTATIONS_URL,
    )
    parser.add_argument(
        "--dataset-viewer-url",
        default=DEFAULT_DATASET_VIEWER_URL,
        help="Hugging Face Dataset Viewer rows endpoint",
    )
    return parser


def run_dataset(
    args: argparse.Namespace,
    dataset: str,
    db: StateDB,
    output_dir: Path,
) -> None:
    target = args.limit_per_dataset
    if dataset == "hagrid":
        if args.skip_archive_probes:
            raise RuntimeError("remote ZIP probe skipped by --skip-archive-probes")
        download_hagrid(args, db, output_dir)
        return
    if db.success_count(dataset) >= target:
        print(f"{dataset}: target already reached ({target})")
        return
    if dataset in {"swig", "easyportrait", "widerface"}:
        if args.skip_archive_probes:
            raise RuntimeError("remote ZIP probe skipped by --skip-archive-probes")
        archive_url = {
            "swig": args.swig_url,
            "easyportrait": args.easyportrait_url,
            "widerface": args.widerface_url,
        }[dataset]
        download_remote_zip(
            dataset,
            db,
            output_dir,
            target,
            args.seed,
            archive_url,
            args.max_errors,
        )
    elif dataset == "gqa":
        download_gqa(
            db,
            output_dir,
            target,
            args.gqa_dataset,
            args.dataset_viewer_url,
            args.request_timeout,
            args.max_errors,
        )
    elif dataset == "openimages":
        download_openimages(
            db,
            output_dir,
            target,
            args.openimages_dataset,
            args.dataset_viewer_url,
            args.request_timeout,
            args.max_errors,
        )
    elif dataset == "mutual_gaze":
        download_mutual_gaze(
            db,
            output_dir,
            target,
            args.mutual_gaze_csv_url,
            args.request_timeout,
            args.max_errors,
        )
    else:
        raise AssertionError(f"unsupported dataset: {dataset}")


def main() -> int:
    args = build_parser().parse_args()
    if args.limit_per_dataset < 1:
        raise SystemExit("--limit-per-dataset must be at least 1")
    if (
        args.hagrid_limit_per_gesture is not None
        and args.hagrid_limit_per_gesture < 1
    ):
        raise SystemExit("--hagrid-limit-per-gesture must be at least 1")
    if args.max_errors < 1:
        raise SystemExit("--max-errors must be at least 1")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        (output_dir / dataset).mkdir(parents=True, exist_ok=True)

    db = StateDB(output_dir / "download_state.sqlite3")
    manifest = output_dir / "manifest.jsonl"
    report = output_dir / "source_status.json"
    statuses: list[SourceStatus] = []
    manual_sources: list[str] = []
    try:
        if "hagrid" in args.datasets and args.hagrid_download_annotations:
            download_and_extract_hagrid_annotations(
                args.hagrid_annotations_url,
                output_dir,
                args.request_timeout,
            )
        for dataset in args.datasets:
            register_manual_images(dataset, output_dir, db)
            if dataset == "hagrid":
                before = hagrid_selected_success_count(
                    db, args.hagrid_gestures
                )
            else:
                before = db.success_count(dataset)
            target = args.limit_per_dataset
            if dataset == "hagrid":
                per_gesture = (
                    args.hagrid_limit_per_gesture or args.limit_per_dataset
                )
                target = per_gesture * len(args.hagrid_gestures)
            try:
                run_dataset(args, dataset, db, output_dir)
                if dataset == "hagrid":
                    after = hagrid_selected_success_count(
                        db, args.hagrid_gestures
                    )
                else:
                    after = db.success_count(dataset)
                clear_manual_note(dataset, output_dir)
                state = "cached" if before >= target else "available"
                detail = (
                    "target already existed locally"
                    if state == "cached"
                    else "selective download succeeded"
                )
            except Exception as exc:
                if dataset == "hagrid":
                    after = hagrid_selected_success_count(
                        db, args.hagrid_gestures
                    )
                else:
                    after = db.success_count(dataset)
                detail = f"{type(exc).__name__}: {exc}"
                state = "manual_required"
                manual_sources.append(dataset)
                note = write_manual_note(dataset, output_dir, detail)
                print(f"MANUAL: {dataset}: {detail}", file=sys.stderr)
                print(f"        folder: {note.parent}", file=sys.stderr)
            finally:
                db.export_manifest(manifest)

            statuses.append(
                SourceStatus(
                    dataset=dataset,
                    mode=SOURCE_MODES[dataset],
                    status=state,
                    downloaded=after,
                    target=target,
                    directory=str((output_dir / dataset).resolve()),
                    detail=detail,
                )
            )
            export_status(report, statuses)

        print("\nSelective access report:")
        for status in statuses:
            print(
                f"- {status.dataset}: {status.status} "
                f"({status.downloaded}/{status.target}, {status.mode})"
            )
        print(f"manifest: {manifest}")
        print(f"report:   {report}")
        if manual_sources:
            print("manual folders: " + ", ".join(manual_sources))
    finally:
        db.close()

    return 1 if args.strict and manual_sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
