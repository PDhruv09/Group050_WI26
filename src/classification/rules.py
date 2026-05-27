"""Rule-based behavioral classification baseline."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True)
class LabelScore:
    """A selected taxonomy label and its confidence-like score."""

    label: str
    score: float


def normalize_text(text: object) -> str:
    """Normalize text for deterministic lexical matching."""
    return " ".join(str(text or "").lower().split())


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count matching keywords and phrases in normalized text."""
    if not text:
        return 0
    hits = 0
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue
        if " " in normalized_keyword:
            hits += int(normalized_keyword in text)
        else:
            hits += int(re.search(rf"\b{re.escape(normalized_keyword)}\b", text) is not None)
    return hits


def keyword_score(text: str, keywords: list[str]) -> float:
    """Score keyword coverage on a bounded 0-1 scale."""
    hits = count_keyword_hits(text, keywords)
    if hits == 0:
        return 0.0
    denominator = max(2.0, math.sqrt(len(keywords)))
    return min(1.0, hits / denominator)


def score_section(text: str, section: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Score every label in one taxonomy section."""
    return {label: keyword_score(text, payload["keywords"]) for label, payload in section.items()}


def select_label(scores: dict[str, float], fallback: str) -> LabelScore:
    """Select the highest scoring label, or fallback when every score is zero."""
    if not scores:
        return LabelScore(fallback, 0.0)
    label, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if score == 0:
        return LabelScore(fallback, 0.0)
    return LabelScore(label, float(score))


def bounded_mean(values: list[float]) -> float:
    """Return the bounded mean of non-empty values."""
    if not values:
        return 0.0
    return max(0.0, min(1.0, sum(values) / len(values)))


def classify_prompt(text: object, taxonomy: dict[str, Any], threshold: float = 0.35) -> dict[str, object]:
    """Classify one prompt into Phase 4 behavioral labels and scores."""
    normalized = normalize_text(text)
    interaction_scores = score_section(normalized, taxonomy["interaction_modes"])
    outsourcing_scores = score_section(normalized, taxonomy["cognitive_outsourcing"])
    emotion_scores = score_section(normalized, taxonomy["emotional_signals"])
    composite_scores = score_section(normalized, taxonomy["composite_signals"])

    interaction = select_label(interaction_scores, "assistant_mode")
    outsourcing = select_label(outsourcing_scores, "none")
    emotion = select_label(emotion_scores, "none")

    companionship_score = max(
        composite_scores.get("companionship", 0.0),
        interaction_scores.get("companion_mode", 0.0),
        emotion_scores.get("loneliness", 0.0),
        emotion_scores.get("affection", 0.0),
    )
    vulnerability_score = max(
        composite_scores.get("vulnerability", 0.0),
        interaction_scores.get("therapist_surrogate_mode", 0.0),
        emotion_scores.get("vulnerability", 0.0),
        bounded_mean([emotion_scores.get("fear", 0.0), emotion_scores.get("sadness", 0.0)]),
    )
    dependency_score = max(
        composite_scores.get("dependency", 0.0),
        emotion_scores.get("dependency", 0.0),
        bounded_mean([companionship_score, vulnerability_score]),
    )

    return {
        "interaction_mode": interaction.label,
        "interaction_mode_score": interaction.score,
        "cognitive_outsourcing_type": outsourcing.label,
        "cognitive_outsourcing_score": outsourcing.score,
        "emotion_primary": emotion.label,
        "emotion_score": emotion.score,
        "companionship_score": float(companionship_score),
        "vulnerability_score": float(vulnerability_score),
        "dependency_score": float(dependency_score),
        "is_companionship": companionship_score >= threshold,
        "is_vulnerable": vulnerability_score >= threshold,
        "is_dependency_signal": dependency_score >= threshold,
        "is_cognitive_outsourcing": outsourcing.score >= threshold,
    }


def classify_texts(texts: list[object], taxonomy: dict[str, Any], threshold: float = 0.35) -> list[dict[str, object]]:
    """Classify a batch of prompt texts."""
    return [classify_prompt(text, taxonomy, threshold) for text in texts]
