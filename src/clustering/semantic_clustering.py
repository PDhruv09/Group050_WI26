"""Standalone UMAP/HDBSCAN and clustering comparison workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.embeddings.io import load_embeddings


def deterministic_sample_indices(total: int, sample_size: int | None) -> np.ndarray:
    """Return deterministic sample indices."""
    if sample_size is None or sample_size >= total:
        return np.arange(total)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(total, size=sample_size, replace=False))


def run_umap(embeddings: np.ndarray, umap_config: dict) -> np.ndarray:
    """Reduce embeddings with UMAP."""
    try:
        from umap import UMAP
    except ImportError as error:
        raise ImportError("umap-learn is required. Install dependencies with: pip install -r requirements.txt") from error

    reducer = UMAP(
        n_neighbors=int(umap_config.get("n_neighbors", 15)),
        n_components=int(umap_config.get("n_components", 2)),
        min_dist=float(umap_config.get("min_dist", 0.0)),
        metric=umap_config.get("metric", "cosine"),
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(reduced_embeddings: np.ndarray, hdbscan_config: dict) -> np.ndarray:
    """Cluster reduced embeddings with HDBSCAN."""
    try:
        import hdbscan
    except ImportError as error:
        raise ImportError("hdbscan is required. Install dependencies with: pip install -r requirements.txt") from error

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(hdbscan_config.get("min_cluster_size", 50)),
        min_samples=int(hdbscan_config.get("min_samples", 10)),
        metric=hdbscan_config.get("metric", "euclidean"),
    )
    return clusterer.fit_predict(reduced_embeddings)


def evaluate_labels(features: np.ndarray, labels: np.ndarray) -> dict:
    """Compute cluster quality metrics when labels contain at least two clusters."""
    clustered_mask = labels != -1
    unique_labels = set(labels[clustered_mask])
    result = {
        "num_records": int(len(labels)),
        "num_clusters": int(len(unique_labels)),
        "noise_fraction": float((labels == -1).mean()) if len(labels) else 0.0,
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(unique_labels) >= 2 and clustered_mask.sum() > len(unique_labels):
        result["silhouette"] = float(silhouette_score(features[clustered_mask], labels[clustered_mask]))
        result["davies_bouldin"] = float(davies_bouldin_score(features[clustered_mask], labels[clustered_mask]))
        result["calinski_harabasz"] = float(calinski_harabasz_score(features[clustered_mask], labels[clustered_mask]))
    return result


def label_clusters(metadata: pd.DataFrame, labels: np.ndarray, text_column: str = "prompt_text") -> pd.DataFrame:
    """Generate lightweight semantic cluster labels from frequent terms."""
    rows = []
    assigned = metadata.copy()
    assigned["cluster"] = labels
    for cluster_id, group in assigned[assigned["cluster"] != -1].groupby("cluster"):
        text = " ".join(group[text_column].dropna().astype(str).head(500).tolist()).lower()
        tokens = [token for token in text.replace("\n", " ").split() if len(token) > 3]
        counts = pd.Series(tokens).value_counts().head(5)
        rows.append(
            {
                "cluster": int(cluster_id),
                "num_records": int(len(group)),
                "label": ", ".join(counts.index.tolist()),
            }
        )
    return pd.DataFrame(rows)


def run_semantic_clustering(config: dict) -> dict:
    """Run UMAP + HDBSCAN and KMeans comparison artifacts."""
    cluster_config = config["clustering"]
    output_dir = Path(cluster_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings(Path(cluster_config["embeddings_file"]))
    metadata = pd.read_parquet(cluster_config["metadata_file"])
    indices = deterministic_sample_indices(len(embeddings), cluster_config.get("sample_size"))
    sampled_embeddings = embeddings[indices]
    sampled_metadata = metadata.iloc[indices].reset_index(drop=True)

    reduced = run_umap(sampled_embeddings, cluster_config["umap"])
    hdbscan_labels = run_hdbscan(reduced, cluster_config["hdbscan"])
    assignments = sampled_metadata.copy()
    assignments["umap_x"] = reduced[:, 0]
    assignments["umap_y"] = reduced[:, 1]
    assignments["hdbscan_cluster"] = hdbscan_labels
    assignments.to_parquet(output_dir / "semantic_cluster_assignments.parquet", index=False)

    labels = label_clusters(assignments, hdbscan_labels)
    labels.to_csv(output_dir / "semantic_cluster_labels.csv", index=False)

    evaluations = [{"method": "hdbscan", **evaluate_labels(reduced, hdbscan_labels)}]
    for k in cluster_config.get("methods", {}).get("kmeans_clusters", []):
        kmeans_labels = KMeans(n_clusters=int(k), random_state=42, n_init="auto").fit_predict(reduced)
        evaluations.append({"method": f"kmeans_{k}", **evaluate_labels(reduced, kmeans_labels)})
    pd.DataFrame(evaluations).to_csv(output_dir / "cluster_quality.csv", index=False)
    return {
        "num_records": int(len(assignments)),
        "output_dir": str(output_dir),
        "hdbscan_clusters": int(len(set(hdbscan_labels[hdbscan_labels != -1]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic clustering workflows.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    summary = run_semantic_clustering(config)
    print("Semantic clustering complete.")
    print(summary)


if __name__ == "__main__":
    main()
