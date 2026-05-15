"""Generate semantic drift centroid artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.semantic_drift.centroids import write_centroids


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

    print("Semantic drift centroids complete.")
    print(f"Groups: {len(centroids)}")
    print(f"Output: {drift_config['output_file']}")


if __name__ == "__main__":
    main()

