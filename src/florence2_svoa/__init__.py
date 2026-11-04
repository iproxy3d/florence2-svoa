"""Structured visual event extraction with a fine-tuned Florence-2 model."""

from .parsing import ParseResult, parse_generation
from .schema import FIELDS, SVOAEvent

__all__ = ["FIELDS", "ParseResult", "SVOAEvent", "parse_generation"]
__version__ = "0.1.0"
