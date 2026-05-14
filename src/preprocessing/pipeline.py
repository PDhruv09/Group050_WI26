"""Phase 2 preprocessing pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.preprocessing.io import read_dataset, write_dataset
from src.preprocessing.language import detect_language
from src.preprocessing.metadata import contains_code, has_question, normalize_whitespace, text_hash


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate present in the dataset."""
    column_set = set(columns)
    return next((candidate for candidate in candidates if candidate in column_set), None)


def require_column(columns: Iterable[str], candidates: Iterable[str], purpose: str) -> str:
    """Find a required semantic column or raise an actionable error."""
    column = first_existing_column(columns, candidates)
    if column is None:
        candidate_list = ", ".join(candidates)
        raise ValueError(f"No {purpose} column found. Tried: {candidate_list}")
    return column


def normalize_records(frame: pd.DataFrame, config: dict, source_dataset: str) -> pd.DataFrame:
    """Normalize raw records into the canonical prompt-level schema."""
    preprocessing = config["preprocessing"]
    text_column = require_column(
        frame.columns,
        preprocessing["text_column_candidates"],
        "prompt text",
    )
    timestamp_column = first_existing_column(frame.columns, preprocessing["timestamp_column_candidates"])
    conversation_column = first_existing_column(
        frame.columns,
        preprocessing.get("conversation_id_column_candidates", []),
    )
    turn_column = first_existing_column(frame.columns, preprocessing.get("turn_index_column_candidates", []))

    cleaned_at = datetime.now(timezone.utc).isoformat()
    normalized = pd.DataFrame()
    normalized["prompt_text"] = frame[text_column].map(normalize_whitespace)
    normalized["raw_text_hash"] = normalized["prompt_text"].map(text_hash)
    normalized["record_id"] = "prompt_" + normalized["raw_text_hash"].str[:16]
    normalized["conversation_id"] = (
        frame[conversation_column].astype(str)
        if conversation_column
        else "conversation_" + normalized["raw_text_hash"].str[:16]
    )
    normalized["turn_index"] = (
        pd.to_numeric(frame[turn_column], errors="coerce").astype("Int64")
        if turn_column
        else pd.Series(range(len(frame)), dtype="Int64")
    )

    if timestamp_column:
        normalized["timestamp"] = pd.to_datetime(frame[timestamp_column], errors="coerce", utc=True)
    else:
        normalized["timestamp"] = pd.NaT

    normalized["source_dataset"] = source_dataset
    normalized["prompt_length"] = normalized["prompt_text"].str.len()
    normalized["prompt_word_count"] = normalized["prompt_text"].str.split().str.len().fillna(0).astype(int)
    normalized["language"] = normalized["prompt_text"].map(detect_language)
    normalized["has_question"] = normalized["prompt_text"].map(has_question)
    normalized["contains_code"] = normalized["prompt_text"].map(contains_code)
    normalized["complexity_score"] = calculate_complexity_score(normalized)
    normalized["disclosure_score"] = 0.0
    normalized["dependency_score"] = 0.0
    normalized["interaction_type"] = pd.NA
    normalized["semantic_cluster"] = pd.NA
    normalized["emotion_scores"] = pd.NA
    normalized["cleaned_at"] = cleaned_at

    return normalized


def calculate_complexity_score(frame: pd.DataFrame) -> pd.Series:
    """Create a simple bounded prompt sophistication proxy."""
    length_component = (frame["prompt_length"] / 1000).clip(upper=1)
    word_component = (frame["prompt_word_count"] / 150).clip(upper=1)
    question_component = frame["has_question"].astype(float) * 0.1
    code_component = frame["contains_code"].astype(float) * 0.1
    return (0.45 * length_component + 0.45 * word_component + question_component + code_component).clip(upper=1)


def filter_records(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply Phase 2 quality filters."""
    preprocessing = config["preprocessing"]
    min_prompt_chars = int(preprocessing.get("min_prompt_chars", 1))
    filtered = frame[frame["prompt_length"] >= min_prompt_chars].copy()

    if preprocessing.get("require_language_match", False):
        filtered = filtered[filtered["language"] == preprocessing.get("language", "en")].copy()

    if preprocessing.get("drop_duplicates", True):
        filtered = filtered.drop_duplicates(subset=["raw_text_hash"]).copy()

    return filtered.reset_index(drop=True)


def run_pipeline(input_file: Path, output_file: Path, config: dict, source_dataset: str | None = None) -> dict:
    """Read, normalize, filter, and persist a dataset."""
    raw = read_dataset(input_file)
    source_label = source_dataset or input_file.stem
    normalized = normalize_records(raw, config, source_label)
    processed = filter_records(normalized, config)
    write_dataset(processed, output_file)

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "source_dataset": source_label,
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(processed)),
        "dropped_rows": int(len(raw) - len(processed)),
        "columns": list(processed.columns),
    }

