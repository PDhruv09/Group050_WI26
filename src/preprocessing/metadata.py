"""Metadata extraction for prompt-level records."""

from __future__ import annotations

import hashlib
import re


QUESTION_STARTERS = re.compile(
    r"\b(what|why|how|when|where|who|which|can|could|would|should|is|are|do|does)\b",
    flags=re.IGNORECASE,
)
CODE_PATTERNS = re.compile(
    r"(```|def\s+\w+\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import|function\s+\w+\(|print\s*\(|console\.log|SELECT\s+.+\s+FROM)",
    flags=re.IGNORECASE | re.DOTALL,
)


def normalize_whitespace(text: object) -> str:
    """Convert arbitrary text-like values into normalized single-line strings."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def text_hash(text: str) -> str:
    """Create a deterministic hash for normalized prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_question(text: str) -> bool:
    """Detect whether the prompt appears question-like."""
    return "?" in text or bool(QUESTION_STARTERS.search(text))


def contains_code(text: str) -> bool:
    """Detect whether the prompt appears to include programming syntax."""
    return bool(CODE_PATTERNS.search(text))
