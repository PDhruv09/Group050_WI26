"""Temporal evolution and transition analysis for Phase 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import linregress


RATE_COLUMNS = [
    "is_companionship",
    "is_vulnerable",
    "is_dependency_signal",
    "is_cognitive_outsourcing",
    "is_reassurance_seeking",
    "is_anthropomorphic",
    "is_self_disclosure",
]

SCORE_COLUMNS = [
    "dependency_score",
    "companionship_score",
    "vulnerability_score",
    "anthropomorphism_score",
    "reassurance_seeking_score",
    "prompt_sophistication_score",
    "conversational_depth_score",
]


def compute_behavior_trends(frame: pd.DataFrame, time_column: str, rolling_window: int) -> pd.DataFrame:
    """Compute monthly behavioral rates, scores, and rolling averages."""
    if time_column not in frame.columns:
        raise ValueError(f"Missing time column: {time_column}")
    data = frame[frame[time_column].notna()].copy()
    grouped = data.groupby(time_column, dropna=False)
    aggregations = {"num_records": ("record_id", "count")}
    for column in RATE_COLUMNS:
        if column in data.columns:
            aggregations[f"{column}_rate"] = (column, "mean")
    for column in SCORE_COLUMNS:
        if column in data.columns:
            aggregations[f"mean_{column}"] = (column, "mean")
    trends = grouped.agg(**aggregations).reset_index().sort_values(time_column)

    numeric_columns = [column for column in trends.columns if column != time_column]
    for column in numeric_columns:
        trends[f"{column}_rolling_{rolling_window}"] = trends[column].rolling(rolling_window, min_periods=1).mean()
        trends[f"{column}_delta"] = trends[column].diff()
    return trends


def compute_transition_matrix(
    frame: pd.DataFrame,
    state_column: str,
    conversation_column: str,
    turn_column: str,
) -> pd.DataFrame:
    """Compute within-conversation transition counts between adjacent states."""
    required = {state_column, conversation_column, turn_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing transition columns: {', '.join(sorted(missing))}")
    ordered = frame.dropna(subset=[state_column, conversation_column]).sort_values([conversation_column, turn_column])
    ordered["next_state"] = ordered.groupby(conversation_column)[state_column].shift(-1)
    transitions = ordered.dropna(subset=["next_state"])
    if transitions.empty:
        return pd.DataFrame(columns=["from_state", "to_state", "count", "probability"])
    counts = transitions.groupby([state_column, "next_state"]).size().reset_index(name="count")
    counts = counts.rename(columns={state_column: "from_state", "next_state": "to_state"})
    counts["probability"] = counts["count"] / counts.groupby("from_state")["count"].transform("sum")
    return counts.sort_values(["from_state", "count"], ascending=[True, False]).reset_index(drop=True)


def compute_trend_statistics(trends: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """Compute linear trend statistics for behavioral rates and scores."""
    rows = []
    ordered = trends.sort_values(time_column).reset_index(drop=True)
    x = pd.Series(range(len(ordered)), dtype=float)
    for column in ordered.columns:
        if column == time_column or column.endswith("_delta") or "_rolling_" in column:
            continue
        if column == "num_records":
            continue
        y = pd.to_numeric(ordered[column], errors="coerce")
        valid = y.notna()
        if valid.sum() < 3:
            continue
        result = linregress(x[valid], y[valid])
        rows.append(
            {
                "metric": column,
                "slope_per_period": float(result.slope),
                "intercept": float(result.intercept),
                "r_value": float(result.rvalue),
                "p_value": float(result.pvalue),
                "std_error": float(result.stderr),
                "num_periods": int(valid.sum()),
            }
        )
    columns = ["metric", "slope_per_period", "intercept", "r_value", "p_value", "std_error", "num_periods"]
    return pd.DataFrame(rows, columns=columns)


def write_temporal_outputs(frame: pd.DataFrame, config: dict) -> dict[str, int]:
    """Write Phase 5 temporal trend and transition outputs."""
    analytics_config = config["advanced_analytics"]
    temporal_config = analytics_config["temporal"]
    time_column = analytics_config.get("time_column", "year_month")
    rolling_window = int(analytics_config.get("rolling_window", 3))
    conversation_column = analytics_config.get("conversation_id_column", "conversation_id")
    turn_column = analytics_config.get("turn_column", "turn_index")

    trends = compute_behavior_trends(frame, time_column, rolling_window)
    trends_file = Path(temporal_config["behavior_trends_file"])
    trends_file.parent.mkdir(parents=True, exist_ok=True)
    trends.to_csv(trends_file, index=False)

    interaction_transitions = compute_transition_matrix(frame, "interaction_mode", conversation_column, turn_column)
    interaction_file = Path(temporal_config["transition_matrix_file"])
    interaction_file.parent.mkdir(parents=True, exist_ok=True)
    interaction_transitions.to_csv(interaction_file, index=False)

    emotion_transitions = compute_transition_matrix(frame, "emotion_primary", conversation_column, turn_column)
    emotion_file = Path(temporal_config["emotion_transition_matrix_file"])
    emotion_file.parent.mkdir(parents=True, exist_ok=True)
    emotion_transitions.to_csv(emotion_file, index=False)

    trend_statistics = compute_trend_statistics(trends, time_column)
    statistics_file = Path(temporal_config["statistical_tests_file"])
    statistics_file.parent.mkdir(parents=True, exist_ok=True)
    trend_statistics.to_csv(statistics_file, index=False)

    return {
        "behavior_trend_rows": int(len(trends)),
        "interaction_transition_rows": int(len(interaction_transitions)),
        "emotion_transition_rows": int(len(emotion_transitions)),
        "trend_statistic_rows": int(len(trend_statistics)),
    }
