"""Evaluation helpers for behavioral classification outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


DEFAULT_TARGETS = {
    "interaction_mode": "interaction_mode",
    "cognitive_outsourcing_type": "cognitive_outsourcing_type",
    "emotion_primary": "emotion_primary",
    "is_companionship": "is_companionship",
    "is_vulnerable": "is_vulnerable",
    "is_dependency_signal": "is_dependency_signal",
    "is_cognitive_outsourcing": "is_cognitive_outsourcing",
}


def write_json(payload: dict, path: Path) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate_against_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    id_column: str,
    output_file: Path,
    confusion_matrix_dir: Path,
    targets: dict[str, str] | None = None,
) -> dict:
    """Evaluate predictions against a labeled dataset with matching IDs."""
    target_map = targets or DEFAULT_TARGETS
    merged = predictions.merge(labels, on=id_column, suffixes=("_pred", "_true"))
    results = {
        "num_prediction_rows": int(len(predictions)),
        "num_label_rows": int(len(labels)),
        "num_matched_rows": int(len(merged)),
        "targets": {},
    }
    confusion_matrix_dir.mkdir(parents=True, exist_ok=True)

    for prediction_column, label_column in target_map.items():
        true_column = f"{label_column}_true" if label_column in predictions.columns else label_column
        pred_column = f"{prediction_column}_pred" if prediction_column in labels.columns else prediction_column
        if true_column not in merged.columns or pred_column not in merged.columns:
            continue
        target_frame = merged[[true_column, pred_column]].dropna()
        if target_frame.empty:
            continue

        y_true = target_frame[true_column].astype(str)
        y_pred = target_frame[pred_column].astype(str)
        labels_sorted = sorted(set(y_true) | set(y_pred))
        matrix = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        matrix_frame = pd.DataFrame(matrix, index=labels_sorted, columns=labels_sorted)
        matrix_frame.to_csv(confusion_matrix_dir / f"{prediction_column}.csv")
        results["targets"][prediction_column] = {
            "num_rows": int(len(target_frame)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        }

    write_json(results, output_file)
    return results


def write_unlabeled_evaluation(predictions: pd.DataFrame, output_file: Path) -> dict:
    """Write coverage diagnostics when no labeled evaluation set is available."""
    payload = {
        "evaluation_type": "unlabeled_coverage",
        "num_rows": int(len(predictions)),
        "interaction_modes": predictions["interaction_mode"].value_counts(dropna=False).to_dict(),
        "cognitive_outsourcing_types": predictions["cognitive_outsourcing_type"].value_counts(dropna=False).to_dict(),
        "primary_emotions": predictions["emotion_primary"].value_counts(dropna=False).to_dict(),
        "rates": {
            "companionship": float(predictions["is_companionship"].mean()) if len(predictions) else 0.0,
            "vulnerable": float(predictions["is_vulnerable"].mean()) if len(predictions) else 0.0,
            "dependency_signal": float(predictions["is_dependency_signal"].mean()) if len(predictions) else 0.0,
            "cognitive_outsourcing": float(predictions["is_cognitive_outsourcing"].mean()) if len(predictions) else 0.0,
        },
    }
    write_json(payload, output_file)
    return payload
