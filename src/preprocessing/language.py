"""Lightweight language detection helpers.

This is intentionally conservative. Phase 2 avoids adding a heavyweight
language model dependency and uses this only as a first-pass English heuristic.
"""

from __future__ import annotations

import re


COMMON_ENGLISH_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "the",
    "to",
    "what",
    "with",
    "you",
}


def detect_language(text: str) -> str:
    """Return a coarse language code for prompt filtering."""
    if not text:
        return "unknown"

    ascii_ratio = sum(1 for char in text if ord(char) < 128) / max(len(text), 1)
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    english_hits = len(words & COMMON_ENGLISH_WORDS)

    if ascii_ratio >= 0.85 and (english_hits > 0 or len(words) <= 3):
        return "en"
    return "unknown"


def normalize_language_label(value: object) -> str:
    """Normalize common dataset language labels for filtering."""
    if value is None:
        return "unknown"

    label = str(value).strip()
    if not label:
        return "unknown"

    lowered = label.lower()
    if lowered in {"en", "eng", "english"}:
        return "en"

    return lowered
