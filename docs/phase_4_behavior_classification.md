# Phase 4 Behavior Classification

Phase 4 turns the processed WildChat prompt dataset into a behavioral classification dataset. The first implementation is a transparent rule-based baseline built around the project taxonomy, so later transformer models can be compared against a reproducible starting point.

## What Phase 4 Produces

- interaction mode labels:
  - `tool_mode`
  - `assistant_mode`
  - `collaborator_mode`
  - `companion_mode`
  - `therapist_surrogate_mode`
- cognitive outsourcing labels:
  - `writing`
  - `coding`
  - `studying`
  - `decision_making`
  - `relationship_advice`
  - `creative_ideation`
  - `life_planning`
  - `memory`
  - `emotional_regulation`
- primary emotional signal labels:
  - `sadness`
  - `loneliness`
  - `fear`
  - `vulnerability`
  - `joy`
  - `confusion`
  - `frustration`
  - `affection`
  - `dependency`
- score columns:
  - `interaction_mode_score`
  - `cognitive_outsourcing_score`
  - `emotion_score`
  - `companionship_score`
  - `vulnerability_score`
  - `dependency_score`
- boolean flags:
  - `is_companionship`
  - `is_vulnerable`
  - `is_dependency_signal`
  - `is_cognitive_outsourcing`

## Run Phase 4

Run the complete classifier on the Phase 2 master dataset:

```bash
python -m src.classification.run_classification --config configs/project.yaml
```

Run a smaller development sample:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 10000
```

Override input and output paths:

```bash
python -m src.classification.run_classification --config configs/project.yaml --input data/processed/train.parquet --output data/processed/classification/train_classified.parquet
```

## Outputs

Default outputs are configured in `configs/project.yaml`:

- `data/processed/classification/classified_prompts.parquet`
- `reports/classification_summary.csv`
- `reports/classification_evaluation.json`
- `reports/classification_manifest.json`

If `classification.labeled_data_file` is configured, Phase 4 will also write confusion matrices to `reports/confusion_matrices/`.

## Evaluation Notes

The current classifier is not intended to be the final research model. It is a baseline that makes the behavioral taxonomy operational and creates data artifacts for dashboarding, exploratory analysis, and later supervised evaluation.

When human-labeled examples become available, configure `classification.labeled_data_file` with a CSV, JSONL, JSON, or Parquet file containing `record_id` plus one or more target columns. The evaluation helper will compute accuracy, macro F1, weighted F1, and confusion matrices for matching targets.
