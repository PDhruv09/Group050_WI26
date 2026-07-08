"""Data loading and transformation helpers for the interactive dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ArtifactStatus:
    """A dashboard artifact path and availability status."""

    name: str
    path: Path
    exists: bool
    rows: int | None = None


def load_config(path: Path = Path("configs/project.yaml")) -> dict[str, Any]:
    """Load project configuration."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dashboard_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return dashboard configuration."""
    return config.get("dashboard", {})


CLASSIFIED_DASHBOARD_COLUMNS = [
    "record_id",
    "conversation_id",
    "prompt_text",
    "year_month",
    "language",
    "interaction_mode",
    "emotion_primary",
    "cognitive_outsourcing_type",
    "is_companionship",
    "is_dependency_signal",
    "is_cognitive_outsourcing",
    "dependency_score",
    "prompt_sophistication_score",
]


def read_table(
    path: Path,
    max_rows: int | None = None,
    random_state: int | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, usecols=columns)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path, columns=columns)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dashboard table type: {path}")
    if max_rows is not None and len(frame) > int(max_rows):
        if random_state is not None:
            return frame.sample(n=int(max_rows), random_state=int(random_state)).sort_index().reset_index(drop=True)
        return frame.head(int(max_rows)).copy()
    return frame


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_status(config: dict[str, Any]) -> list[ArtifactStatus]:
    """Return availability status for dashboard input artifacts."""
    dash = dashboard_config(config)
    keys = [
        "sample_file",
        "classified_data_file",
        "behavior_trends_file",
        "interaction_transition_file",
        "emotion_transition_file",
        "archetype_summary_file",
        "archetype_assignments_file",
        "network_nodes_file",
        "network_edges_file",
        "event_window_file",
        "taxonomy_coverage_file",
        "classification_summary_file",
        "classification_benchmark_file",
        "phase5_manifest_file",
    ]
    statuses = []
    for key in keys:
        path = Path(dash.get(key, ""))
        rows = None
        if path.exists() and path.suffix.lower() in {".csv", ".parquet"}:
            try:
                rows = len(read_table(path))
            except Exception:
                rows = None
        statuses.append(ArtifactStatus(key, path, path.exists(), rows))
    return statuses


def load_dashboard_tables(config: dict[str, Any], prompt_rows: int | None = None) -> dict[str, pd.DataFrame]:
    """Load all dashboard tables that are available."""
    dash = dashboard_config(config)
    max_prompt_rows = prompt_rows if prompt_rows is not None else dash.get("max_prompt_rows")
    random_state = dash.get("sample_random_state", 42)
    sample_file = Path(dash.get("sample_file", ""))
    classified_source = sample_file if sample_file.exists() else Path(dash["classified_data_file"])
    return {
        "classified": read_table(
            classified_source,
            max_prompt_rows,
            random_state=random_state,
            columns=CLASSIFIED_DASHBOARD_COLUMNS,
        ),
        "behavior_trends": read_table(Path(dash["behavior_trends_file"])),
        "interaction_transition": read_table(Path(dash["interaction_transition_file"])),
        "emotion_transition": read_table(Path(dash["emotion_transition_file"])),
        "archetype_summary": read_table(Path(dash["archetype_summary_file"])),
        "archetype_assignments": read_table(Path(dash["archetype_assignments_file"])),
        "network_nodes": read_table(Path(dash["network_nodes_file"])),
        "network_edges": read_table(Path(dash["network_edges_file"])),
        "event_windows": read_table(Path(dash["event_window_file"])),
        "taxonomy_coverage": read_table(Path(dash["taxonomy_coverage_file"])),
        "classification_summary": read_table(Path(dash["classification_summary_file"])),
        "classification_benchmark": read_table(Path(dash["classification_benchmark_file"])),
    }


def unique_sorted(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted string values for a dashboard filter."""
    if frame.empty or column not in frame.columns:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique())


def filter_classified_data(
    frame: pd.DataFrame,
    languages: list[str] | None = None,
    interaction_modes: list[str] | None = None,
    emotions: list[str] | None = None,
    outsourcing_types: list[str] | None = None,
    year_months: list[str] | None = None,
    keyword: str | None = None,
    min_dependency: float | None = None,
    min_complexity: float | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters to classified prompts."""
    if frame.empty:
        return frame
    filtered = frame.copy()
    filters = [
        ("language", languages),
        ("interaction_mode", interaction_modes),
        ("emotion_primary", emotions),
        ("cognitive_outsourcing_type", outsourcing_types),
        ("year_month", year_months),
    ]
    for column, values in filters:
        if values and column in filtered.columns:
            filtered = filtered[filtered[column].astype(str).isin(values)]
    if keyword and "prompt_text" in filtered.columns:
        filtered = filtered[filtered["prompt_text"].fillna("").str.contains(keyword, case=False, regex=False)]
    if min_dependency is not None and "dependency_score" in filtered.columns:
        filtered = filtered[filtered["dependency_score"].fillna(0) >= float(min_dependency)]
    if min_complexity is not None and "prompt_sophistication_score" in filtered.columns:
        filtered = filtered[filtered["prompt_sophistication_score"].fillna(0) >= float(min_complexity)]
    return filtered


def compute_kpis(frame: pd.DataFrame) -> dict[str, float]:
    """Compute dashboard KPI values from filtered classified data."""
    if frame.empty:
        return {
            "records": 0,
            "conversations": 0,
            "companionship_rate": 0.0,
            "dependency_rate": 0.0,
            "outsourcing_rate": 0.0,
            "mean_complexity": 0.0,
        }
    return {
        "records": int(len(frame)),
        "conversations": int(frame["conversation_id"].nunique()) if "conversation_id" in frame.columns else 0,
        "companionship_rate": float(frame.get("is_companionship", pd.Series(dtype=float)).mean() or 0.0),
        "dependency_rate": float(frame.get("is_dependency_signal", pd.Series(dtype=float)).mean() or 0.0),
        "outsourcing_rate": float(frame.get("is_cognitive_outsourcing", pd.Series(dtype=float)).mean() or 0.0),
        "mean_complexity": float(frame.get("prompt_sophistication_score", pd.Series(dtype=float)).mean() or 0.0),
    }


def top_counts(frame: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """Return top category counts."""
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[column, "count"])
    return frame[column].fillna("missing").astype(str).value_counts().head(n).reset_index(name="count")


def prompt_explorer_columns(frame: pd.DataFrame) -> list[str]:
    """Return dashboard-safe prompt explorer columns."""
    preferred = [
        "record_id",
        "year_month",
        "language",
        "interaction_mode",
        "emotion_primary",
        "cognitive_outsourcing_type",
        "dependency_score",
        "prompt_sophistication_score",
        "prompt_text",
    ]
    return [column for column in preferred if column in frame.columns]
