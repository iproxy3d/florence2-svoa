"""Parsing and strict validation of generated SVOA text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schema import SVOAEvent


CODE_FENCE_RE = re.compile(r"^```(?:json|text)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    raw: str
    normalized: str
    valid: bool
    event: SVOAEvent | None
    error: str | None


def parse_generation(text: str) -> ParseResult:
    """Parse pipe or JSON output without inventing absent fields."""

    raw = text.strip()
    cleaned = CODE_FENCE_RE.sub("", raw).strip()
    try:
        if cleaned.startswith("{"):
            value = json.loads(cleaned)
            if not isinstance(value, dict):
                raise ValueError("JSON output is not an object")
            event = SVOAEvent.from_mapping(value)
        else:
            event = SVOAEvent.from_text(cleaned)
        return ParseResult(raw, event.to_text(), True, event, None)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        normalized = "|".join(
            " ".join(part.strip().lower().split()) for part in cleaned.split("|")
        )
        return ParseResult(raw, normalized, False, None, str(exc))
