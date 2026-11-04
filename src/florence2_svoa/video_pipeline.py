"""Optional frame sampling pipeline; this is not a motion model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .inference import Florence2SVOA
from .parsing import ParseResult


@dataclass(frozen=True)
class FrameEvent:
    time_s: float
    frame_index: int
    result: ParseResult


def iter_video_events(
    model: Florence2SVOA,
    video: str | Path,
    *,
    every_seconds: float = 1.0,
) -> Iterator[FrameEvent]:
    """Sample frames and yield independent SVOA events in time order."""

    if every_seconds <= 0:
        raise ValueError("every_seconds must be positive")
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install video dependencies: pip install -e '.[video]'") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise ValueError("video reports an invalid frame rate")
    step = max(1, round(fps * every_seconds))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield FrameEvent(index / fps, index, model.predict(Image.fromarray(rgb)))
            index += 1
    finally:
        capture.release()
