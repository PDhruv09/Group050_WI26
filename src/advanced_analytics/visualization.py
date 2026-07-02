"""Phase 5 visualization outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_phase5_visualizations(config: dict) -> dict[str, str]:
    """Write interactive HTML figures for Phase 5 analytics."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError("plotly is required for Phase 5 visualizations.") from error

    analytics_config = config["advanced_analytics"]
    figures_config = analytics_config["figures"]
    outputs = {}

    trends_file = Path(analytics_config["temporal"]["behavior_trends_file"])
    if trends_file.exists():
        trends = pd.read_csv(trends_file)
        y_columns = [column for column in trends.columns if column.endswith("_rate")]
        if y_columns:
            fig = px.line(trends, x=analytics_config.get("time_column", "year_month"), y=y_columns, title="Phase 5 Behavioral Trends")
            output_file = Path(figures_config["behavior_trends_file"])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_file)
            outputs["behavior_trends"] = str(output_file)

    archetype_file = Path(analytics_config["archetypes"]["summary_file"])
    if archetype_file.exists():
        archetypes = pd.read_csv(archetype_file)
        if not archetypes.empty:
            fig = px.bar(archetypes, x="archetype_name", y="num_conversations", color="archetype", title="Conversation Archetype Distribution")
            output_file = Path(figures_config["archetype_file"])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_file)
            outputs["archetypes"] = str(output_file)

    edge_file = Path(analytics_config["network"]["edge_file"])
    node_file = Path(analytics_config["network"]["node_file"])
    if edge_file.exists() and node_file.exists():
        edges = pd.read_csv(edge_file).sort_values("weight", ascending=False).head(30)
        if not edges.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=edges["weight"],
                    y=edges["source"] + " -> " + edges["target"],
                    orientation="h",
                )
            )
            fig.update_layout(title="Top Behavioral Network Edges", yaxis={"categoryorder": "total ascending"})
            output_file = Path(figures_config["network_file"])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_file)
            outputs["network"] = str(output_file)

    return outputs

