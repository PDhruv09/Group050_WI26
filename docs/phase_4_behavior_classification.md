# Phase 4 Behavior Classification

Phase 4 turns the processed WildChat prompt dataset into a behavioral classification dataset. It includes a transparent rule-based taxonomy classifier, optional transformer-assisted emotion classification, dependency and anthropomorphism indicators, cognitive outsourcing labels, conversational complexity metrics, temporal trend outputs, and validation artifacts.

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
  - `reassurance_seeking_score`
  - `anthropomorphism_score`
  - `self_disclosure_score`
  - `prompt_sophistication_score`
  - `conversational_depth_score`
  - `recursive_interaction_score`
- boolean flags:
  - `is_companionship`
  - `is_vulnerable`
  - `is_dependency_signal`
  - `is_cognitive_outsourcing`
  - `is_reassurance_seeking`
  - `is_anthropomorphic`
  - `is_self_disclosure`

Optional transformer columns are added when transformer emotion classification is enabled:

- `transformer_emotion_label`
- `transformer_emotion_score`
- `transformer_emotion_model`

Optional zero-shot behavior columns are added when transformer-assisted behavior classification is enabled:

- `transformer_interaction_mode`
- `transformer_interaction_score`
- `transformer_cognitive_outsourcing_type`
- `transformer_cognitive_outsourcing_score`
- `transformer_behavior_model`

## Run Phase 4

Run the complete classifier on the Phase 2 master dataset:

```bash
python -m src.classification.run_classification --config configs/project.yaml
```

Run a smaller development sample:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 10000
```

Run transformer-assisted emotion classification on a small subset:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 10000 --enable-transformers --transformer-max-records 1000
```

The transformer command downloads or loads the configured Hugging Face model. Use a smaller `--transformer-max-records` while developing, then increase it when compute is available.

Run zero-shot transformer-assisted behavior classification on a small subset:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 1000 --enable-zero-shot-behavior --zero-shot-max-records 100
```

Run both optional transformer workflows together:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 1000 --enable-transformers --transformer-max-records 100 --enable-zero-shot-behavior --zero-shot-max-records 100
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
- `reports/classification_benchmark.csv`
- `reports/taxonomy_coverage.csv`
- `reports/behavioral_overlap.csv`
- `reports/unstable_classification_regions.csv`
- `reports/cognitive_outsourcing_trends.csv`
- `reports/dependency_trends.csv`
- `reports/conversational_complexity_trends.csv`
- `reports/emotional_trends.csv`

If `classification.labeled_data_file` is configured, Phase 4 will also write confusion matrices to `reports/confusion_matrices/`.

## Evaluation Notes

The taxonomy classifier is deterministic and reproducible. Transformer-assisted emotion classification is optional and records model outputs separately so rule-based and transformer signals can be compared.

When human-labeled examples become available, configure `classification.labeled_data_file` with a CSV, JSONL, JSON, or Parquet file containing `record_id` plus one or more target columns. The evaluation helper will compute accuracy, macro F1, weighted F1, and confusion matrices for matching targets.

## Methodology

Interaction mode classification uses the taxonomy in `configs/behavior_taxonomy.yml` to score tool, assistant, collaborator, companion, and therapist-surrogate behavior. The highest scoring category becomes `interaction_mode`, while the score columns preserve confidence-like evidence strength.

Emotion analysis has two layers. The deterministic layer maps prompts to project-specific emotional signals such as loneliness, vulnerability, fear, frustration, joy, affection, and dependency. The optional transformer layer runs a Hugging Face text-classification model and stores its label, score, and model name in separate columns.

Transformer-assisted behavior classification uses a zero-shot Hugging Face classifier to assign interaction mode and cognitive outsourcing labels from taxonomy descriptions. These labels are stored separately from deterministic taxonomy labels so the two approaches can be benchmarked against human labels later.

Dependency analysis combines direct dependency language, companionship signals, vulnerability language, reassurance seeking, anthropomorphic language, and self-disclosure. This supports early measurement of shifts from task-oriented use toward relational or emotionally reliant interaction.

Cognitive outsourcing classification maps prompts into writing, coding, studying, decision making, relationship advice, creative ideation, life planning, memory, and emotional regulation. Temporal trend outputs summarize how these categories change by `year_month` when timestamp metadata exists.

Conversational complexity analysis measures prompt sophistication, instruction specificity, role prompting, recursive interaction language, self-disclosure, and conversational depth. These metrics are heuristics for large-scale trend analysis, not claims about true psychological depth.

## Validation Artifacts

Phase 4 writes validation artifacts for both labeled and unlabeled settings:

- `taxonomy_coverage.csv` measures how often each behavioral dimension receives nonzero evidence.
- `behavioral_overlap.csv` measures co-occurrence among behavioral flags.
- `unstable_classification_regions.csv` identifies records where top scores are close and likely ambiguous.
- `classification_benchmark.csv` records runtime and throughput.
- `classification_evaluation.json` stores unlabeled coverage diagnostics, or labeled metrics when a labeled dataset is configured.
- `reports/confusion_matrices/` stores confusion matrices when labeled evaluation runs.

## Limitations

Rule-based scores are transparent but cannot capture all semantic nuance. Transformer emotion scores are model-dependent and should be validated against human labels before being used as final research evidence. All Phase 4 behavioral labels should be interpreted as computational indicators that support exploratory analysis and model development, not clinical, diagnostic, or definitive claims about users.

## Verification Checklist Commands

Run unit tests:

```bash
pytest
```

Run a fast Phase 4 verification sample:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 10000
```

Check that the core artifacts exist:

```cmd
dir data\processed\classification
dir reports
```

Inspect output columns:

```cmd
python -c "import pandas as pd; d=pd.read_parquet('data/processed/classification/classified_prompts.parquet'); print(d.columns.tolist()); print(d[['interaction_mode','emotion_primary','cognitive_outsourcing_type','dependency_score','anthropomorphism_score','prompt_sophistication_score']].head())"
```

Inspect validation artifacts:

```cmd
python -c "import pandas as pd; print(pd.read_csv('reports/taxonomy_coverage.csv')); print(pd.read_csv('reports/classification_benchmark.csv'))"
python -c "import pandas as pd; print(pd.read_csv('reports/behavioral_overlap.csv').head()); print(pd.read_csv('reports/unstable_classification_regions.csv').head())"
```

Inspect trend outputs:

```cmd
python -c "import pandas as pd; print(pd.read_csv('reports/emotional_trends.csv').head()); print(pd.read_csv('reports/cognitive_outsourcing_trends.csv').head())"
python -c "import pandas as pd; print(pd.read_csv('reports/dependency_trends.csv').head()); print(pd.read_csv('reports/conversational_complexity_trends.csv').head())"
```

Run optional transformer emotion verification:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 1000 --enable-transformers --transformer-max-records 100
```

Then inspect transformer columns:

```cmd
python -c "import pandas as pd; d=pd.read_parquet('data/processed/classification/classified_prompts.parquet'); print(d[['transformer_emotion_label','transformer_emotion_score','transformer_emotion_model']].dropna().head())"
```

Run optional zero-shot behavior verification:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 1000 --enable-zero-shot-behavior --zero-shot-max-records 100
```

Then inspect zero-shot behavior columns:

```cmd
python -c "import pandas as pd; d=pd.read_parquet('data/processed/classification/classified_prompts.parquet'); print(d[['transformer_interaction_mode','transformer_interaction_score','transformer_cognitive_outsourcing_type','transformer_cognitive_outsourcing_score','transformer_behavior_model']].dropna().head())"
```
