"""Formal SVOA output schema."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


FIELDS = ("subject", "action", "object", "adjuncts")
WORD_RE = re.compile(r"^(?:[a-z]+(?:-[a-z]+)*|none|unknown)$")
ACTION_RE = re.compile(
    r"^(?:[a-z][a-z0-9]*(?:[- ][a-z0-9]+)*|none|unknown)$"
)


def normalize_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


@dataclass(frozen=True)
class SVOAEvent:
    """One directly observable visual event."""

    subject: str
    action: str
    object: str
    adjuncts: str

    def __post_init__(self) -> None:
        for field in FIELDS:
            value = normalize_label(getattr(self, field))
            pattern = ACTION_RE if field == "action" else WORD_RE
            if not pattern.fullmatch(value):
                raise ValueError(f"invalid {field}: {value!r}")
            object.__setattr__(self, field, value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SVOAEvent":
        missing = [field for field in FIELDS if field not in value]
        extra = [field for field in value if field not in FIELDS]
        if missing or extra:
            raise ValueError(f"schema mismatch; missing={missing}, extra={extra}")
        return cls(**{field: value[field] for field in FIELDS})

    @classmethod
    def from_text(cls, text: str) -> "SVOAEvent":
        parts = [normalize_label(part) for part in text.strip().split("|")]
        if len(parts) != len(FIELDS):
            raise ValueError(f"expected four pipe-separated fields, got {len(parts)}")
        return cls(**dict(zip(FIELDS, parts)))

    def to_text(self) -> str:
        return "|".join(getattr(self, field) for field in FIELDS)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)
