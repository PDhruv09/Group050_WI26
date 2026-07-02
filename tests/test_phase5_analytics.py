import os
from pathlib import Path

import pandas as pd
import yaml

from src.advanced_analytics.archetypes import build_conversation_features, discover_archetypes
from src.advanced_analytics.network import build_behavior_network
from src.advanced_analytics.temporal import (
    compute_behavior_trends,
    compute_transition_matrix,
    compute_trend_statistics,
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def sample_classified_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "conversation_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "turn_index": [0, 1, 0, 1, 0, 1],
            "timestamp": pd.to_datetime(
                ["2023-04-01", "2023-04-02", "2023-05-01", "2023-05-02", "2023-06-01", "2023-06-02"],
                utc=True,
            ),
            "year_month": ["2023-04", "2023-04", "2023-05", "2023-05", "2023-06", "2023-06"],
            "prompt_word_count": [10, 20, 30, 40, 50, 60],
            "interaction_mode": [
                "tool_mode",
                "assistant_mode",
                "companion_mode",
                "therapist_surrogate_mode",
                "collaborator_mode",
                "tool_mode",
            ],
            "emotion_primary": ["none", "confusion", "loneliness", "fear", "joy", "none"],
            "cognitive_outsourcing_type": ["coding", "studying", "none", "emotional_regulation", "creative_ideation", "coding"],
            "is_companionship": [False, False, True, True, False, False],
            "is_vulnerable": [False, False, True, True, False, False],
            "is_dependency_signal": [False, False, True, False, False, False],
            "is_cognitive_outsourcing": [True, True, False, True, True, True],
            "is_reassurance_seeking": [False, False, True, False, False, False],
            "is_anthropomorphic": [False, False, False, True, False, False],
            "is_self_disclosure": [False, False, True, True, False, False],
            "dependency_score": [0.0, 0.0, 0.6, 0.2, 0.0, 0.0],
            "companionship_score": [0.0, 0.0, 0.7, 0.4, 0.0, 0.0],
            "vulnerability_score": [0.0, 0.0, 0.5, 0.8, 0.0, 0.0],
            "anthropomorphism_score": [0.0, 0.0, 0.0, 0.7, 0.0, 0.0],
            "reassurance_seeking_score": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
            "prompt_sophistication_score": [0.2, 0.3, 0.4, 0.5, 0.7, 0.8],
            "conversational_depth_score": [0.1, 0.2, 0.6, 0.7, 0.4, 0.5],
        }
    )


def test_phase5_config_exists() -> None:
    with Path("configs/project.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["project"]["phase"] == 5
    assert "advanced_analytics" in config


def test_behavior_trends_and_statistics() -> None:
    frame = sample_classified_frame()

    trends = compute_behavior_trends(frame, "year_month", 2)
    stats = compute_trend_statistics(trends, "year_month")

    assert len(trends) == 3
    assert "is_companionship_rate" in trends.columns
    assert "mean_prompt_sophistication_score" in trends.columns
    assert not stats.empty


def test_transition_matrix_counts_adjacent_states() -> None:
    frame = sample_classified_frame()

    transitions = compute_transition_matrix(frame, "interaction_mode", "conversation_id", "turn_index")

    assert len(transitions) == 3
    assert {"from_state", "to_state", "count", "probability"}.issubset(transitions.columns)


def test_archetype_feature_builder() -> None:
    frame = sample_classified_frame()

    features = build_conversation_features(frame, "conversation_id")

    assert len(features) == 3
    assert "num_turns" in features.columns
    assert "mean_prompt_word_count" in features.columns


def test_discover_archetypes_writes_outputs() -> None:
    frame = sample_classified_frame()
    scratch = Path("data/processed/_test_phase5")
    scratch.mkdir(parents=True, exist_ok=True)
    config = {
        "advanced_analytics": {
            "conversation_id_column": "conversation_id",
            "archetypes": {
                "n_clusters": 2,
                "min_conversation_records": 2,
                "random_state": 42,
                "output_file": str(scratch / "archetypes.parquet"),
                "summary_file": str(scratch / "archetype_summary.csv"),
            },
        }
    }

    summary = discover_archetypes(frame, config)

    assert summary["archetype_rows"] == 3
    assert Path(config["advanced_analytics"]["archetypes"]["output_file"]).exists()
    assert Path(config["advanced_analytics"]["archetypes"]["summary_file"]).exists()


def test_behavior_network_contains_signal_edges() -> None:
    frame = sample_classified_frame()

    nodes, edges, graph = build_behavior_network(frame)

    assert not nodes.empty
    assert not edges.empty
    assert graph.number_of_nodes() == len(nodes)
