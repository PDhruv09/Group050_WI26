"""Conversation archetype discovery for Phase 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "is_companionship",
    "is_vulnerable",
    "is_dependency_signal",
    "is_cognitive_outsourcing",
    "is_reassurance_seeking",
    "is_anthropomorphic",
    "is_self_disclosure",
    "dependency_score",
    "companionship_score",
    "vulnerability_score",
    "prompt_sophistication_score",
    "conversational_depth_score",
]

def build_conversation_features(frame: pd.DataFrame, conversation_column: str) -> pd.DataFrame:
    """Aggregate prompt-level behavioral features to conversation-level vectors."""
    available = [column for column in FEATURE_COLUMNS if column in frame.columns]
    grouped = frame.groupby(conversation_column)
    features = grouped[available].mean().reset_index()
    features["num_turns"] = grouped.size().to_numpy()
    if "prompt_word_count" in frame.columns:
        features["mean_prompt_word_count"] = grouped["prompt_word_count"].mean().to_numpy()
    return features


def assign_archetype_names(summary: pd.DataFrame) -> pd.DataFrame:
    """Assign interpretable names to archetype clusters from dominant features."""
    names = []
    for row in summary.itertuples(index=False):
        row_dict = row._asdict()
        candidates = {
            "Emotional Support Seekers": row_dict.get("is_vulnerable", 0) + row_dict.get("companionship_score", 0),
            "Cognitive Outsourcers": row_dict.get("is_cognitive_outsourcing", 0),
            "Dependency-Oriented Users": row_dict.get("dependency_score", 0) + row_dict.get("is_dependency_signal", 0),
            "Conversational Collaborators": row_dict.get("prompt_sophistication_score", 0)
            + row_dict.get("conversational_depth_score", 0),
            "Casual/Utility Users": 1 - row_dict.get("is_vulnerable", 0),
        }
        names.append(max(candidates.items(), key=lambda item: item[1])[0])
    output = summary.copy()
    output["archetype_name"] = names
    return output


def discover_archetypes(frame: pd.DataFrame, config: dict) -> dict[str, int]:
    """Cluster conversations into behavioral archetypes."""
    analytics_config = config["advanced_analytics"]
    archetype_config = analytics_config["archetypes"]
    conversation_column = analytics_config.get("conversation_id_column", "conversation_id")
    min_records = int(archetype_config.get("min_conversation_records", 2))
    n_clusters = int(archetype_config.get("n_clusters", 8))
    random_state = int(archetype_config.get("random_state", 42))

    features = build_conversation_features(frame, conversation_column)
    features = features[features["num_turns"] >= min_records].reset_index(drop=True)
    feature_columns = [column for column in features.columns if column != conversation_column]
    if features.empty or len(features) < 2:
        assignments = features.copy()
        assignments["archetype"] = pd.Series(dtype="Int64")
        summary = pd.DataFrame(columns=["archetype", "num_conversations", "archetype_name"])
    else:
        effective_clusters = min(n_clusters, len(features))
        scaled = StandardScaler().fit_transform(features[feature_columns])
        labels = KMeans(n_clusters=effective_clusters, random_state=random_state, n_init="auto").fit_predict(scaled)
        assignments = features.copy()
        assignments["archetype"] = labels
        summary = assignments.groupby("archetype")[feature_columns].mean().reset_index()
        summary.insert(1, "num_conversations", assignments.groupby("archetype").size().to_numpy())
        summary = assign_archetype_names(summary)

    output_file = Path(archetype_config["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output_file, index=False)

    summary_file = Path(archetype_config["summary_file"])
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    return {
        "conversation_feature_rows": int(len(features)),
        "archetype_rows": int(len(assignments)),
        "archetype_summary_rows": int(len(summary)),
    }
