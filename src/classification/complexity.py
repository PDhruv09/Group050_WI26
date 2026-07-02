"""Conversational complexity and prompt sophistication metrics."""

from __future__ import annotations

import re

import pandas as pd


ROLE_PROMPT_PATTERN = r"\b(?:act as|you are|pretend to be|roleplay as)\b"
INSTRUCTION_PATTERN = r"\b(?:step by step|format as|do not|make sure|include|avoid|use a|write in)\b"
RECURSION_PATTERN = r"\b(?:previous|above|again|continue|revise|rewrite|follow up|as before|same as)\b"


def bounded(value: float) -> float:
    """Bound a score to the 0-1 interval."""
    return max(0.0, min(1.0, value))


def add_complexity_metrics(frame: pd.DataFrame, text_column: str = "prompt_text") -> pd.DataFrame:
    """Add prompt sophistication, recursive interaction, and self-disclosure metrics."""
    output = frame.copy()
    text = output[text_column].fillna("").astype(str)
    word_count = text.str.split().str.len().fillna(0)
    char_count = text.str.len()
    sentence_count = text.str.count(r"[.!?]+").clip(lower=1)
    unique_ratio = text.map(lambda value: len(set(value.lower().split())) / max(1, len(value.split())))
    instruction_hits = text.str.count(INSTRUCTION_PATTERN, flags=re.IGNORECASE)
    role_prompt = text.str.contains(ROLE_PROMPT_PATTERN, flags=re.IGNORECASE, na=False)
    recursive_hits = text.str.count(RECURSION_PATTERN, flags=re.IGNORECASE)

    output["sentence_count"] = sentence_count.astype(int)
    output["avg_words_per_sentence"] = (word_count / sentence_count).round(3)
    output["lexical_diversity"] = unique_ratio.round(3)
    output["instruction_specificity_score"] = (instruction_hits / 4).clip(upper=1).astype(float)
    output["role_prompting_score"] = role_prompt.astype(float)
    output["recursive_interaction_score"] = (recursive_hits / 3).clip(upper=1).astype(float)
    output["prompt_sophistication_score"] = (
        (char_count / 1500).clip(upper=1) * 0.25
        + (word_count / 250).clip(upper=1) * 0.25
        + output["lexical_diversity"].astype(float) * 0.15
        + output["instruction_specificity_score"] * 0.15
        + output["role_prompting_score"] * 0.1
        + output["recursive_interaction_score"] * 0.1
    ).map(bounded)
    output["conversational_depth_score"] = (
        output["recursive_interaction_score"] * 0.35
        + output.get("self_disclosure_score", 0.0) * 0.35
        + output["prompt_sophistication_score"] * 0.3
    ).map(bounded)
    return output
