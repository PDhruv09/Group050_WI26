import numpy as np
import pandas as pd

from src.embeddings.semantic_search import cosine_top_k
from src.semantic_drift.centroids import compute_group_centroids


def test_cosine_top_k_returns_best_matches() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ]
    )
    query = np.array([1.0, 0.0])

    matches = cosine_top_k(query, embeddings, top_k=2)

    assert matches[0][0] == 0
    assert matches[0][1] == 1.0
    assert matches[1][0] == 2


def test_compute_group_centroids() -> None:
    embeddings = np.array(
        [
            [1.0, 1.0],
            [3.0, 3.0],
            [10.0, 0.0],
        ]
    )
    metadata = pd.DataFrame({"year_month": ["2024-01", "2024-01", "2024-02"]})

    centroids = compute_group_centroids(embeddings, metadata, "year_month")

    january = centroids[centroids["year_month"] == "2024-01"].iloc[0]
    assert january["num_records"] == 2
    assert january["centroid"] == [2.0, 2.0]

