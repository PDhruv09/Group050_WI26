"""Event-window analysis for Phase 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EVENT_METRICS = [
    "is_companionship",
    "is_vulnerable",
    "is_dependency_signal",
    "is_cognitive_outsourcing",
    "dependency_score",
    "prompt_sophistication_score",
    "conversational_depth_score",
]


def summarize_window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    """Summarize behavioral metrics inside a date window."""
    subset = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    summary: dict[str, float] = {"num_records": int(len(subset))}
    for column in EVENT_METRICS:
        if column in subset.columns:
            summary[column] = float(subset[column].mean()) if len(subset) else 0.0
    return summary


def write_event_window_analysis(frame: pd.DataFrame, config: dict) -> dict[str, int]:
    """Compare before/after behavioral metrics around configured events."""
    events_config = config["advanced_analytics"]["events"]
    if "timestamp" not in frame.columns:
        return {"event_window_rows": 0}

    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    rows = []
    for event in events_config.get("windows", []):
        event_date = pd.Timestamp(event["date"], tz="UTC")
        before_start = event_date - pd.Timedelta(days=int(event.get("days_before", 30)))
        before_end = event_date - pd.Timedelta(days=1)
        after_start = event_date
        after_end = event_date + pd.Timedelta(days=int(event.get("days_after", 30)))
        before = summarize_window(data, before_start, before_end)
        after = summarize_window(data, after_start, after_end)
        for metric in sorted(set(before) | set(after)):
            rows.append(
                {
                    "event": event["name"],
                    "event_date": event["date"],
                    "metric": metric,
                    "before_value": before.get(metric, 0.0),
                    "after_value": after.get(metric, 0.0),
                    "difference": after.get(metric, 0.0) - before.get(metric, 0.0),
                }
            )
    output = pd.DataFrame(rows)
    output_file = Path(events_config["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    return {"event_window_rows": int(len(output))}

