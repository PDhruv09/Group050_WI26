"""Phase 3 semantic visualization workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def write_cluster_map(config: dict) -> Path:
    """Create an interactive UMAP cluster map."""
    try:
        import plotly.express as px
    except ImportError as error:
        raise ImportError("plotly is required. Install dependencies with: pip install -r requirements.txt") from error

    cluster_file = Path(config["clustering"]["output_dir"]) / "semantic_cluster_assignments.parquet"
    output_file = Path(config["visualization"]["cluster_map_file"])
    frame = pd.read_parquet(cluster_file)
    color = "hdbscan_cluster" if "hdbscan_cluster" in frame.columns else None
    hover_columns = [column for column in ["prompt_text", "year_month", "language"] if column in frame.columns]
    fig = px.scatter(
        frame,
        x="umap_x",
        y="umap_y",
        color=color,
        hover_data=hover_columns,
        title="Semantic Prompt Cluster Map",
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_file)
    return output_file


def write_topic_evolution(config: dict) -> Path:
    """Create topic frequency over time visualization."""
    try:
        import plotly.express as px
    except ImportError as error:
        raise ImportError("plotly is required. Install dependencies with: pip install -r requirements.txt") from error

    topic_config = config["topic_modeling"]
    data = pd.read_parquet(topic_config["input_file"])
    assignments = pd.read_parquet(Path(topic_config["output_dir"]) / "bertopic_assignments.parquet")
    frame = data[["year_month"]].iloc[: len(assignments)].copy()
    frame["topic"] = assignments["topic"].to_numpy()
    counts = frame.groupby(["year_month", "topic"]).size().reset_index(name="count")
    output_file = Path(config["visualization"]["topic_evolution_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig = px.line(counts, x="year_month", y="count", color="topic", title="Topic Evolution Over Time")
    fig.write_html(output_file)
    return output_file


def write_drift_plot(config: dict) -> Path:
    """Create semantic drift metric visualization."""
    try:
        import plotly.express as px
    except ImportError as error:
        raise ImportError("plotly is required. Install dependencies with: pip install -r requirements.txt") from error

    metrics_file = Path(config["semantic_drift"]["metrics_file"])
    frame = pd.read_csv(metrics_file)
    output_file = Path(config["visualization"]["drift_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig = px.line(frame, x="to_period", y="cosine_distance", title="Consecutive-Period Semantic Drift")
    fig.write_html(output_file)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 semantic visualizations.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument(
        "--kind",
        choices=["cluster-map", "topic-evolution", "drift", "all"],
        default="all",
    )
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    outputs = []
    if args.kind in {"cluster-map", "all"}:
        outputs.append(write_cluster_map(config))
    if args.kind in {"topic-evolution", "all"}:
        outputs.append(write_topic_evolution(config))
    if args.kind in {"drift", "all"}:
        outputs.append(write_drift_plot(config))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
