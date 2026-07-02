"""Optional transformer-assisted zero-shot behavioral classification."""

from __future__ import annotations

from typing import Any

import pandas as pd


def label_descriptions(section: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Build human-readable candidate labels from taxonomy definitions."""
    labels = {}
    for label, payload in section.items():
        description = payload.get("description", label.replace("_", " "))
        labels[label] = f"{label.replace('_', ' ')}: {description}"
    return labels


def best_zero_shot_label(result: dict[str, Any], description_map: dict[str, str]) -> tuple[str, float]:
    """Map the highest zero-shot description back to a taxonomy label."""
    if not result.get("labels"):
        return "none", 0.0
    best_description = result["labels"][0]
    score = float(result["scores"][0])
    reverse_map = {description: label for label, description in description_map.items()}
    return reverse_map.get(best_description, best_description), score


def apply_transformer_behavior(frame: pd.DataFrame, config: dict[str, Any], taxonomy: dict[str, Any]) -> pd.DataFrame:
    """Add optional zero-shot transformer interaction and outsourcing predictions."""
    behavior_config = config.get("transformer_behavior", {})
    output = frame.copy()
    default_columns = {
        "transformer_interaction_mode": pd.NA,
        "transformer_interaction_score": pd.NA,
        "transformer_cognitive_outsourcing_type": pd.NA,
        "transformer_cognitive_outsourcing_score": pd.NA,
        "transformer_behavior_model": pd.NA,
    }
    if not behavior_config.get("enabled", False):
        for column, value in default_columns.items():
            output[column] = value
        return output

    try:
        from transformers import pipeline
    except ImportError as error:
        raise ImportError("transformers is required for transformer-assisted behavior classification.") from error

    model_name = behavior_config.get("model_name", "facebook/bart-large-mnli")
    batch_size = int(behavior_config.get("batch_size", 8))
    max_records = behavior_config.get("max_records")
    text_column = config.get("text_column", "prompt_text")
    classifier = pipeline("zero-shot-classification", model=model_name)
    selected_index = output.index if max_records is None else output.index[: int(max_records)]
    texts = output.loc[selected_index, text_column].fillna("").astype(str).tolist()

    interaction_labels = label_descriptions(taxonomy["interaction_modes"])
    outsourcing_labels = label_descriptions(taxonomy["cognitive_outsourcing"])
    interaction_predictions = []
    outsourcing_predictions = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        interaction_predictions.extend(classifier(batch, list(interaction_labels.values()), multi_label=False))
        outsourcing_predictions.extend(classifier(batch, list(outsourcing_labels.values()), multi_label=False))

    interaction_best = [best_zero_shot_label(result, interaction_labels) for result in interaction_predictions]
    outsourcing_best = [best_zero_shot_label(result, outsourcing_labels) for result in outsourcing_predictions]

    for column, value in default_columns.items():
        output[column] = value
    output.loc[selected_index, "transformer_interaction_mode"] = [label for label, _ in interaction_best]
    output.loc[selected_index, "transformer_interaction_score"] = [score for _, score in interaction_best]
    output.loc[selected_index, "transformer_cognitive_outsourcing_type"] = [label for label, _ in outsourcing_best]
    output.loc[selected_index, "transformer_cognitive_outsourcing_score"] = [score for _, score in outsourcing_best]
    output.loc[selected_index, "transformer_behavior_model"] = model_name
    return output
