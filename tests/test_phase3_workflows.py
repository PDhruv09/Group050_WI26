import numpy as np
import pandas as pd

from src.clustering.semantic_clustering import deterministic_sample_indices, evaluate_labels, label_clusters
from src.clustering.topic_evaluation import count_non_outlier_topics, simple_topic_diversity, top_terms_by_topic
from src.embeddings.benchmark import sample_texts
from src.semantic_drift.centroids import compute_rolling_centroids, compute_temporal_drift_metrics, cosine_distance


def test_sample_texts_is_deterministic_size() -> None:
    frame = pd.DataFrame({"prompt_text": [f"text {index}" for index in range(10)]})

    sampled = sample_texts(frame, "prompt_text", 4)

    assert len(sampled) == 4


def test_deterministic_sample_indices() -> None:
    first = deterministic_sample_indices(100, 10)
    second = deterministic_sample_indices(100, 10)

    assert np.array_equal(first, second)
    assert len(first) == 10


def test_evaluate_labels_handles_noise() -> None:
    features = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
    labels = np.array([0, 0, -1])

    result = evaluate_labels(features, labels)

    assert result["num_records"] == 3
    assert result["num_clusters"] == 1
    assert result["noise_fraction"] == 1 / 3


def test_label_clusters_creates_keywords() -> None:
    metadata = pd.DataFrame({"prompt_text": ["write code function", "code debug function", "lonely support"]})
    labels = np.array([0, 0, -1])

    result = label_clusters(metadata, labels)

    assert result.loc[0, "cluster"] == 0
    assert "function" in result.loc[0, "label"]


def test_topic_terms_and_diversity() -> None:
    texts = pd.Series(["write python code", "debug python function", "lonely emotional support"])
    topics = pd.Series([0, 0, 1])

    terms = top_terms_by_topic(texts, topics, top_n=3)

    assert set(terms["topic"]) == {0, 1}
    assert 0 < simple_topic_diversity(terms) <= 1
    assert count_non_outlier_topics(pd.Series([-1, 0, 0, 1, 2])) == 3


def test_temporal_drift_metrics_and_rolling_centroids() -> None:
    centroids = pd.DataFrame(
        {
            "year_month": ["2024-01", "2024-02"],
            "num_records": [2, 2],
            "centroid": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    metrics = compute_temporal_drift_metrics(centroids, "year_month")

    assert metrics.loc[0, "cosine_distance"] == 1.0
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    metadata = pd.DataFrame({"year_month": ["2024-01", "2024-02", "2024-03"]})
    rolling = compute_rolling_centroids(embeddings, metadata, "year_month", window=2)

    assert len(rolling) == 2
