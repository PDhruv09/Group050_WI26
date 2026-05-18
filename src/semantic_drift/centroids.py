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


def cosine_distance(left: list[float], right: list[float]) -> float:
    """Compute cosine distance between two centroid vectors."""
    left_array = np.array(left, dtype=float)
    right_array = np.array(right, dtype=float)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return 0.0
    return float(1 - ((left_array @ right_array) / denominator))


def compute_temporal_drift_metrics(centroids: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Measure consecutive-period centroid movement."""
    ordered = centroids.sort_values(group_column).reset_index(drop=True)
    rows = []
    for index in range(1, len(ordered)):
        previous = ordered.iloc[index - 1]
        current = ordered.iloc[index]
        rows.append(
            {
                "from_period": previous[group_column],
                "to_period": current[group_column],
                "from_records": int(previous["num_records"]),
                "to_records": int(current["num_records"]),
                "cosine_distance": cosine_distance(previous["centroid"], current["centroid"]),
            }
        )
    return pd.DataFrame(rows)


def compute_rolling_centroids(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    group_column: str,
    window: int,
) -> pd.DataFrame:
    """Compute rolling-window centroids across ordered temporal groups."""
    groups = sorted(metadata[group_column].dropna().unique())
    rows = []
    for end_index in range(window - 1, len(groups)):
        window_groups = groups[end_index - window + 1 : end_index + 1]
        mask = metadata[group_column].isin(window_groups).to_numpy()
        centroid = embeddings[mask].mean(axis=0)
        rows.append(
            {
                "window_start": window_groups[0],
                "window_end": window_groups[-1],
                "num_records": int(mask.sum()),
                "centroid": centroid.tolist(),
            }
        )
    return pd.DataFrame(rows)
