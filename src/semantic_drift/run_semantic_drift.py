"""Generate semantic drift centroid artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import pandas as pd

from src.embeddings.io import load_embeddings
from src.semantic_drift.centroids import (
    compute_rolling_centroids,
    compute_temporal_drift_metrics,
    write_centroids,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embedding centroids by time group.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    embedding_config = config["embeddings"]
    drift_config = config["semantic_drift"]
    centroids = write_centroids(
        embeddings_file=Path(embedding_config["output_file"]),
        metadata_file=Path(embedding_config["metadata_file"]),
        output_file=Path(drift_config["output_file"]),
        group_column=drift_config["embedding_group_column"],
    )
    metrics = compute_temporal_drift_metrics(centroids, drift_config["embedding_group_column"])
    metrics_file = Path(drift_config["metrics_file"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_file, index=False)

    embeddings = load_embeddings(Path(embedding_config["output_file"]))
    metadata = pd.read_parquet(embedding_config["metadata_file"])
    rolling = compute_rolling_centroids(
        embeddings,
        metadata,
        drift_config["embedding_group_column"],
        int(drift_config.get("rolling_window", 3)),
    )
    rolling_file = Path(drift_config["output_file"]).with_name("semantic_drift_rolling_centroids.parquet")
    rolling.to_parquet(rolling_file, index=False)

    print("Semantic drift centroids complete.")
    print(f"Groups: {len(centroids)}")
    print(f"Output: {drift_config['output_file']}")
    print(f"Metrics: {metrics_file}")
    print(f"Rolling centroids: {rolling_file}")


if __name__ == "__main__":
    main()
