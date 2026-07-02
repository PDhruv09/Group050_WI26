"""Markdown report generation for Phase 5 analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def read_head(path: Path, rows: int = 5) -> str:
    """Render the head of a CSV file as markdown."""
    if not path.exists():
        return "_Not generated._"
    try:
        frame = pd.read_csv(path).head(rows)
    except EmptyDataError:
        return "_Generated with no rows._"
    if frame.empty:
        return "_Generated with no rows._"
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, separator, *body])


def write_phase5_report(config: dict, manifest: dict) -> Path:
    """Write a compact Phase 5 analytics report."""
    analytics_config = config["advanced_analytics"]
    report_file = Path(analytics_config["report_file"])
    temporal = analytics_config["temporal"]
    network = analytics_config["network"]
    events = analytics_config["events"]
    archetypes = analytics_config["archetypes"]

    lines = [
        "# Phase 5 Advanced Analytics Report",
        "",
        "## Run Summary",
        "",
        f"- Input records: {manifest.get('num_records', 0):,}",
        f"- Behavior trend rows: {manifest.get('behavior_trend_rows', 0):,}",
        f"- Interaction transition rows: {manifest.get('interaction_transition_rows', 0):,}",
        f"- Emotion transition rows: {manifest.get('emotion_transition_rows', 0):,}",
        f"- Archetype rows: {manifest.get('archetype_rows', 0):,}",
        f"- Network nodes: {manifest.get('network_nodes', 0):,}",
        f"- Network edges: {manifest.get('network_edges', 0):,}",
        "",
        "## Behavioral Trends",
        "",
        read_head(Path(temporal["behavior_trends_file"])),
        "",
        "## Trend Statistics",
        "",
        read_head(Path(temporal["statistical_tests_file"])),
        "",
        "## Archetype Summary",
        "",
        read_head(Path(archetypes["summary_file"])),
        "",
        "## Behavior Network Nodes",
        "",
        read_head(Path(network["node_file"])),
        "",
        "## Event Windows",
        "",
        read_head(Path(events["output_file"])),
        "",
        "## Limitations",
        "",
        "Phase 5 analytics are computational indicators for exploratory research. They should be interpreted alongside sampling context, taxonomy limitations, model uncertainty, and later human-labeled validation.",
        "",
    ]
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")
    return report_file
