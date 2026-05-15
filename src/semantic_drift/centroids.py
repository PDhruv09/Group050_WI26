"""Semantic drift utilities based on embedding centroids over time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings.io import load_embeddings


def compute_group_centroids(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Compute one embedding centroid per metadata group."""
    if len(embeddings) != len(metadata):
        raise ValueError("Embeddings and metadata must have the same number of rows.")
    if group_column not in metadata.columns:
        raise ValueError(f"Missing group column: {group_column}")

    rows = []
    for group_value, indices in metadata.groupby(group_column, dropna=True).groups.items():
        index_array = np.array(list(indices), dtype=int)
        centroid = embeddings[index_array].mean(axis=0)
        rows.append(
            {
                group_column: group_value,
                "num_records": int(len(index_array)),
                "centroid": centroid.tolist(),
            }
        )
    return pd.DataFrame(rows)


def write_centroids(
    embeddings_file: Path,
    metadata_file: Path,
    output_file: Path,
    group_column: str,
) -> pd.DataFrame:
    """Load embeddings/metadata, compute centroids, and write them to Parquet."""
    embeddings = load_embeddings(embeddings_file)
    metadata = pd.read_parquet(metadata_file)
    centroids = compute_group_centroids(embeddings, metadata, group_column)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    centroids.to_parquet(output_file, index=False)
    return centroids

