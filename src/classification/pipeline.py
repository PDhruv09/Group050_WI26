"""Phase 4 behavioral classification pipeline."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.classification.analytics import (
    write_behavioral_overlap,
    write_benchmark,
    write_taxonomy_coverage,
    write_temporal_trends,
    write_unstable_regions,
)
from src.classification.complexity import add_complexity_metrics
from src.classification.evaluation import evaluate_against_labels, write_json, write_unlabeled_evaluation
from src.classification.rules import classify_texts
from src.classification.taxonomy import load_taxonomy
from src.classification.transformer_behavior import apply_transformer_behavior
from src.classification.transformer_emotion import apply_transformer_emotion
from src.preprocessing.io import read_dataset, write_dataset


def load_input_frame(input_file: Path, max_records: int | None) -> pd.DataFrame:
    """Load classification input and optionally limit records for development."""
    frame = pd.read_parquet(input_file)
    if max_records is not None:
        return frame.head(int(max_records)).copy()
    return frame


def add_behavioral_columns(frame: pd.DataFrame, config: dict, taxonomy: dict) -> pd.DataFrame:
    """Add Phase 4 classification columns to a processed prompt dataset."""
    text_column = config.get("text_column", "prompt_text")
    if text_column not in frame.columns:
        raise ValueError(f"Classification text column not found: {text_column}")

    threshold = float(config.get("score_threshold", 0.35))
    predictions = classify_texts(frame[text_column].fillna("").tolist(), taxonomy, threshold)
    prediction_frame = pd.DataFrame(predictions)
    output = frame.reset_index(drop=True).copy()
    for column in prediction_frame.columns:
        output[column] = prediction_frame[column]

    output["interaction_type"] = output["interaction_mode"]
    output["disclosure_score"] = output["vulnerability_score"]
    output = add_complexity_metrics(output, text_column)
    output = apply_transformer_emotion(output, config)
    output = apply_transformer_behavior(output, config, taxonomy)
    output["classified_at"] = datetime.now(timezone.utc).isoformat()
    return output


def summarize_classification(frame: pd.DataFrame, summary_file: Path) -> pd.DataFrame:
    """Write compact count and rate summaries for behavioral outputs."""
    rows = []
    count_columns = ["interaction_mode", "cognitive_outsourcing_type", "emotion_primary"]
    for column in count_columns:
        for label, count in frame[column].value_counts(dropna=False).items():
            rows.append({"metric": column, "label": str(label), "value": int(count)})

    rate_columns = [
        "is_companionship",
        "is_vulnerable",
        "is_dependency_signal",
        "is_cognitive_outsourcing",
        "is_reassurance_seeking",
        "is_anthropomorphic",
        "is_self_disclosure",
    ]
    for column in rate_columns:
        rows.append({"metric": column, "label": "rate", "value": float(frame[column].mean()) if len(frame) else 0.0})

    summary = pd.DataFrame(rows)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    return summary


def run_classification_pipeline(config: dict) -> dict:
    """Run the Phase 4 behavioral classification system."""
    start_time = time.perf_counter()
    classification_config = config["classification"]
    input_file = Path(classification_config["input_file"])
    output_file = Path(classification_config["output_file"])
    taxonomy_file = Path(classification_config["taxonomy_file"])
    summary_file = Path(classification_config["summary_file"])
    manifest_file = Path(classification_config["manifest_file"])
    evaluation_file = Path(classification_config["evaluation_file"])
    confusion_matrix_dir = Path(classification_config["confusion_matrix_dir"])
    benchmark_file = Path(classification_config.get("benchmark_file", "reports/classification_benchmark.csv"))
    coverage_file = Path(classification_config.get("coverage_file", "reports/taxonomy_coverage.csv"))
    overlap_file = Path(classification_config.get("overlap_file", "reports/behavioral_overlap.csv"))
    unstable_regions_file = Path(
        classification_config.get("unstable_regions_file", "reports/unstable_classification_regions.csv")
    )

    print(f"Loading processed dataset: {input_file}")
    frame = load_input_frame(input_file, classification_config.get("max_records"))
    print(f"Loaded {len(frame):,} records for behavioral classification.")

    print(f"Loading behavioral taxonomy: {taxonomy_file}")
    taxonomy = load_taxonomy(taxonomy_file)

    print("Applying interaction, emotion, dependency, vulnerability, and outsourcing labels...")
    classified = add_behavioral_columns(frame, classification_config, taxonomy)

    print(f"Writing classified behavioral dataset: {output_file}")
    write_dataset(classified, output_file)

    print(f"Writing classification summary: {summary_file}")
    summarize_classification(classified, summary_file)

    print(f"Writing taxonomy coverage diagnostics: {coverage_file}")
    coverage = write_taxonomy_coverage(classified, coverage_file)

    print(f"Writing behavioral overlap diagnostics: {overlap_file}")
    overlap = write_behavioral_overlap(classified, overlap_file)

    print(f"Writing unstable classification region report: {unstable_regions_file}")
    unstable_regions = write_unstable_regions(classified, unstable_regions_file)

    print("Writing temporal behavior trend reports...")
    temporal_outputs = write_temporal_trends(
        classified,
        {
            "emotional": Path(classification_config.get("emotional_trends_file", "reports/emotional_trends.csv")),
            "outsourcing": Path(
                classification_config.get("outsourcing_trends_file", "reports/cognitive_outsourcing_trends.csv")
            ),
            "dependency": Path(classification_config.get("dependency_trends_file", "reports/dependency_trends.csv")),
            "complexity": Path(
                classification_config.get("complexity_trends_file", "reports/conversational_complexity_trends.csv")
            ),
        },
    )

    print(f"Writing classification benchmark: {benchmark_file}")
    benchmark = write_benchmark(
        start_time,
        classified,
        benchmark_file,
        classification_config.get("method", "rule_based_behavioral_taxonomy"),
    )

    labeled_file = classification_config.get("labeled_data_file")
    if labeled_file:
        print(f"Evaluating against labeled data: {labeled_file}")
        labels = read_dataset(Path(labeled_file))
        evaluation = evaluate_against_labels(
            classified,
            labels,
            classification_config.get("id_column", "record_id"),
            evaluation_file,
            confusion_matrix_dir,
        )
    else:
        print(f"No labeled dataset configured. Writing unlabeled coverage diagnostics: {evaluation_file}")
        evaluation = write_unlabeled_evaluation(classified, evaluation_file)

    manifest = {
        "phase": 4,
        "method": classification_config.get("method", "rule_based_behavioral_taxonomy"),
        "input_file": str(input_file),
        "output_file": str(output_file),
        "taxonomy_file": str(taxonomy_file),
        "summary_file": str(summary_file),
        "evaluation_file": str(evaluation_file),
        "benchmark_file": str(benchmark_file),
        "coverage_file": str(coverage_file),
        "overlap_file": str(overlap_file),
        "unstable_regions_file": str(unstable_regions_file),
        "num_records": int(len(classified)),
        "score_threshold": float(classification_config.get("score_threshold", 0.35)),
        "labeled_evaluation": bool(labeled_file),
        "transformer_emotion_enabled": bool(
            classification_config.get("transformer_emotion", {}).get("enabled", False)
        ),
        "transformer_behavior_enabled": bool(
            classification_config.get("transformer_behavior", {}).get("enabled", False)
        ),
        "taxonomy_coverage_rows": int(len(coverage)),
        "behavioral_overlap_rows": int(len(overlap)),
        "unstable_region_rows": int(len(unstable_regions)),
        "temporal_outputs": temporal_outputs,
        "benchmark": benchmark.to_dict(orient="records")[0] if not benchmark.empty else {},
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": evaluation,
    }
    print(f"Writing classification manifest: {manifest_file}")
    write_json(manifest, manifest_file)
    print("Phase 4 behavioral classification complete.")
    return manifest
