"""Optional transformer-assisted emotion classification."""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_transformer_emotion(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add transformer emotion predictions when explicitly enabled."""
    emotion_config = config.get("transformer_emotion", {})
    if not emotion_config.get("enabled", False):
        output = frame.copy()
        output["transformer_emotion_label"] = pd.NA
        output["transformer_emotion_score"] = pd.NA
        output["transformer_emotion_model"] = pd.NA
        return output

    try:
        from transformers import pipeline
    except ImportError as error:
        raise ImportError("transformers is required for transformer emotion classification.") from error

    model_name = emotion_config.get("model_name", "j-hartmann/emotion-english-distilroberta-base")
    batch_size = int(emotion_config.get("batch_size", 16))
    max_records = emotion_config.get("max_records")
    text_column = config.get("text_column", "prompt_text")
    classifier = pipeline("text-classification", model=model_name, top_k=emotion_config.get("top_k"))

    output = frame.copy()
    selected_index = output.index if max_records is None else output.index[: int(max_records)]
    texts = output.loc[selected_index, text_column].fillna("").astype(str).tolist()
    labels = []
    scores = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        predictions = classifier(batch, truncation=True)
        for prediction in predictions:
            if isinstance(prediction, list):
                best = max(prediction, key=lambda item: item["score"])
            else:
                best = prediction
            labels.append(best["label"])
            scores.append(float(best["score"]))

    output["transformer_emotion_label"] = pd.NA
    output["transformer_emotion_score"] = pd.NA
    output["transformer_emotion_model"] = model_name
    output.loc[selected_index, "transformer_emotion_label"] = labels
    output.loc[selected_index, "transformer_emotion_score"] = scores
    return output
