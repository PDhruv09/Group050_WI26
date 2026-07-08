"""Plotly chart builders for the interactive research dashboard."""

from __future__ import annotations

import pandas as pd


COLOR_SEQUENCE = ["#2fb8ac", "#e5b451", "#ef6f6c", "#7a8cff", "#b86adf", "#6bbf59", "#d27d2d"]
TEMPLATE = "plotly_dark"
PLOT_BG = "#101820"
PAPER_BG = "#101820"
GRID = "rgba(226, 235, 241, 0.12)"
TEXT = "#e6edf3"


def style_figure(fig, title: str | None = None, legend: str = "bottom"):
    """Apply the dashboard visual identity to a Plotly figure."""
    legend_config = (
        {"orientation": "v", "yanchor": "top", "y": 0.98, "xanchor": "left", "x": 1.02, "title": None}
        if legend == "right"
        else {"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0, "title": None}
    )
    right_margin = 170 if legend == "right" else 24
    fig.update_layout(
        template=TEMPLATE,
        title={"text": title, "y": 0.96, "x": 0.01, "xanchor": "left", "yanchor": "top"},
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": TEXT, "family": "Inter, Segoe UI, sans-serif", "size": 12},
        colorway=COLOR_SEQUENCE,
        margin={"l": 34, "r": right_margin, "t": 70, "b": 72},
        legend=legend_config,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def empty_figure(title: str):
    """Create an empty Plotly figure."""
    import plotly.graph_objects as go

    fig = go.Figure()
    return style_figure(fig, title)


def behavior_trends_figure(trends: pd.DataFrame):
    """Create behavioral trend lines."""
    if trends.empty or "year_month" not in trends.columns:
        return empty_figure("Behavioral Trends")
    import plotly.express as px

    plot_frame = trends.copy()
    plot_frame["year_month"] = plot_frame["year_month"].astype(str)
    columns = [
        column
        for column in [
            "is_companionship_rate",
            "is_vulnerable_rate",
            "is_dependency_signal_rate",
            "is_cognitive_outsourcing_rate",
            "is_self_disclosure_rate",
        ]
        if column in trends.columns
    ]
    if not columns:
        return empty_figure("Behavioral Trends")
    fig = px.line(plot_frame, x="year_month", y=columns, markers=True, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=plot_frame["year_month"].tolist())
    return style_figure(fig, "Behavioral Signal Trends")


def category_bar(frame: pd.DataFrame, category: str, title: str):
    """Create a category count bar chart."""
    if frame.empty or category not in frame.columns:
        return empty_figure(title)
    import plotly.express as px

    counts = frame[category].fillna("missing").astype(str).value_counts().head(12).reset_index(name="count")
    fig = px.bar(counts, x=category, y="count", color=category, color_discrete_sequence=COLOR_SEQUENCE)
    fig.update_layout(showlegend=False)
    return style_figure(fig, title)


def transition_heatmap(transitions: pd.DataFrame, title: str):
    """Create transition matrix heatmap."""
    if transitions.empty or not {"from_state", "to_state", "probability"}.issubset(transitions.columns):
        return empty_figure(title)
    import plotly.express as px

    matrix = transitions.pivot_table(index="from_state", columns="to_state", values="probability", fill_value=0)
    fig = px.imshow(matrix, text_auto=".2f", aspect="auto", color_continuous_scale="Tealrose")
    return style_figure(fig, title)


def archetype_distribution(archetypes: pd.DataFrame):
    """Create archetype distribution chart."""
    if archetypes.empty or "num_conversations" not in archetypes.columns:
        return empty_figure("Archetypes")
    import plotly.express as px

    x = "archetype_name" if "archetype_name" in archetypes.columns else "archetype"
    fig = px.bar(archetypes, x=x, y="num_conversations", color="archetype", color_continuous_scale="Tealrose")
    return style_figure(fig, "Conversation Archetype Distribution")


def network_edges_bar(edges: pd.DataFrame):
    """Create top behavioral network edge chart."""
    if edges.empty or not {"source", "target", "weight"}.issubset(edges.columns):
        return empty_figure("Behavior Network")
    import plotly.express as px

    top = edges.sort_values("weight", ascending=False).head(20).copy()
    top["edge"] = top["source"] + " -> " + top["target"]
    fig = px.bar(top, x="weight", y="edge", orientation="h", color="weight", color_continuous_scale="Tealrose")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return style_figure(fig, "Top Behavioral Network Edges")


def event_window_chart(events: pd.DataFrame):
    """Create event-window difference chart."""
    if events.empty or not {"event", "metric", "difference"}.issubset(events.columns):
        return empty_figure("Event Windows")
    import plotly.express as px

    plot_frame = events.copy()
    plot_frame["metric_label"] = plot_frame["metric"].astype(str).str.replace("_", " ").str.replace(" score", "").str.title()
    fig = px.bar(
        plot_frame,
        x="difference",
        y="metric_label",
        color="event",
        barmode="group",
        orientation="h",
        color_discrete_sequence=COLOR_SEQUENCE,
        hover_data=["event_date", "before_value", "after_value", "metric"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="After minus before", yaxis_title="Metric")
    return style_figure(fig, "Event Window Differences", legend="right")


def complexity_scatter(frame: pd.DataFrame):
    """Create complexity versus dependency scatter."""
    required = {"prompt_sophistication_score", "dependency_score"}
    if frame.empty or not required.issubset(frame.columns):
        return empty_figure("Complexity and Dependency")
    import plotly.express as px

    sample = frame.head(5000).copy()
    fig = px.scatter(
        sample,
        x="prompt_sophistication_score",
        y="dependency_score",
        color="interaction_mode" if "interaction_mode" in sample.columns else None,
        opacity=0.55,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    return style_figure(fig, "Prompt Sophistication vs Dependency")
