"""Advanced Phase 4 behavioral analytics artifacts."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


def write_taxonomy_coverage(frame: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    """Measure how much of the dataset receives confident taxonomy labels."""
    rows = []
    score_columns = {
        "interaction_mode": "interaction_mode_score",
        "cognitive_outsourcing_type": "cognitive_outsourcing_score",
        "emotion_primary": "emotion_score",
        "companionship": "companionship_score",
        "vulnerability": "vulnerability_score",
        "dependency": "dependency_score",
        "anthropomorphism": "anthropomorphism_score",
        "reassurance_seeking": "reassurance_seeking_score",
        "self_disclosure": "self_disclosure_score",
    }
    for label, score_column in score_columns.items():
        scores = frame[score_column].fillna(0)
        rows.append(
            {
                "dimension": label,
                "num_records": int(len(frame)),
                "covered_records": int((scores > 0).sum()),
                "coverage_rate": float((scores > 0).mean()) if len(frame) else 0.0,
                "mean_score": float(scores.mean()) if len(frame) else 0.0,
            }
        )
    coverage = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_file, index=False)
    return coverage


def write_behavioral_overlap(frame: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    """Measure overlap among binary behavioral indicators."""
    flags = [
        "is_companionship",
        "is_vulnerable",
        "is_dependency_signal",
        "is_cognitive_outsourcing",
        "is_reassurance_seeking",
        "is_anthropomorphic",
        "is_self_disclosure",
    ]
    rows = []
    for left in flags:
        for right in flags:
            left_values = frame[left].astype(bool)
            right_values = frame[right].astype(bool)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "overlap_count": int((left_values & right_values).sum()),
                    "overlap_rate": float((left_values & right_values).mean()) if len(frame) else 0.0,
                }
            )
    overlap = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(output_file, index=False)
    return overlap


def write_temporal_trends(frame: pd.DataFrame, output_files: dict[str, Path]) -> dict[str, int]:
    """Write temporal behavioral trend summaries when a time column exists."""
    if "year_month" not in frame.columns:
        return {}
    trend_frame = frame[frame["year_month"].notna()].copy()
    if trend_frame.empty:
        return {}

    outputs = {}
    grouping = trend_frame.groupby("year_month", dropna=False)

    emotional = grouping.agg(
        num_records=("record_id", "count"),
        vulnerability_rate=("is_vulnerable", "mean"),
        dependency_rate=("is_dependency_signal", "mean"),
        companionship_rate=("is_companionship", "mean"),
        mean_emotion_score=("emotion_score", "mean"),
    ).reset_index()
    emotional.to_csv(output_files["emotional"], index=False)
    outputs["emotional"] = len(emotional)

    outsourcing = (
        trend_frame.groupby(["year_month", "cognitive_outsourcing_type"], dropna=False)
        .size()
        .reset_index(name="num_records")
    )
    outsourcing.to_csv(output_files["outsourcing"], index=False)
    outputs["outsourcing"] = len(outsourcing)

    dependency = grouping.agg(
        num_records=("record_id", "count"),
        dependency_rate=("is_dependency_signal", "mean"),
        anthropomorphism_rate=("is_anthropomorphic", "mean"),
        reassurance_rate=("is_reassurance_seeking", "mean"),
        mean_dependency_score=("dependency_score", "mean"),
    ).reset_index()
    dependency.to_csv(output_files["dependency"], index=False)
    outputs["dependency"] = len(dependency)

    complexity = grouping.agg(
        num_records=("record_id", "count"),
        mean_prompt_sophistication=("prompt_sophistication_score", "mean"),
        mean_conversational_depth=("conversational_depth_score", "mean"),
        mean_recursive_interaction=("recursive_interaction_score", "mean"),
        mean_self_disclosure=("self_disclosure_score", "mean"),
    ).reset_index()
    complexity.to_csv(output_files["complexity"], index=False)
    outputs["complexity"] = len(complexity)
    return outputs


def write_unstable_regions(frame: pd.DataFrame, output_file: Path, margin: float = 0.08) -> pd.DataFrame:
    """Identify records whose top behavioral scores are close and likely unstable."""
    rows = []
    score_sets = {
        "interaction": [
            "interaction_mode_score",
            "cognitive_outsourcing_score",
            "emotion_score",
            "companionship_score",
            "vulnerability_score",
            "dependency_score",
        ],
        "social_dependency": [
            "companionship_score",
            "vulnerability_score",
            "dependency_score",
            "anthropomorphism_score",
            "reassurance_seeking_score",
            "self_disclosure_score",
        ],
    }
    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        record_id = row_dict.get("record_id")
        for dimension, columns in score_sets.items():
            values = sorted([(column, float(row_dict.get(column) or 0.0)) for column in columns], key=lambda item: item[1], reverse=True)
            if len(values) < 2:
                continue
            margin_value = values[0][1] - values[1][1]
            if values[0][1] > 0 and margin_value <= margin:
                rows.append(
                    {
                        "record_id": record_id,
                        "dimension": dimension,
                        "top_signal": values[0][0],
                        "second_signal": values[1][0],
                        "top_score": values[0][1],
                        "second_score": values[1][1],
                        "margin": margin_value,
                    }
                )
    unstable = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    unstable.to_csv(output_file, index=False)
    return unstable


def write_benchmark(start_time: float, frame: pd.DataFrame, output_file: Path, method: str) -> pd.DataFrame:
    """Write runtime and throughput benchmark metrics."""
    elapsed = max(time.perf_counter() - start_time, 0.000001)
    benchmark = pd.DataFrame(
        [
            {
                "method": method,
                "num_records": int(len(frame)),
                "elapsed_seconds": elapsed,
                "records_per_second": float(len(frame) / elapsed) if len(frame) else 0.0,
            }
        ]
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_file, index=False)
    return benchmark
