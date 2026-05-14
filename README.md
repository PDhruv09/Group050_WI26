# Human-AI Behavior Observatory

Human-AI Behavior Observatory is a redevelopment of the original COGS 108 group project into a long-term computational social science and NLP platform for studying public human interaction with conversational AI systems.

The original class materials are archived in `deprecate/`. This repository now treats that work as seed material rather than the final analytical frame.

## Research Direction

The project investigates how people use conversational AI over time, with emphasis on:

- behavioral shifts from tool use to social interaction,
- emotional and vulnerability signals,
- cognitive outsourcing patterns,
- prompt sophistication,
- semantic drift,
- user archetypes,
- conversational depth,
- behavioral network structure.

## Phase 2 Status

Phase 2 adds the first reusable data-engineering pipeline so later analysis modules can rely on a canonical prompt-level dataset.

Current infrastructure includes:

- research-oriented repository layout,
- archived COGS 108 materials,
- dependency files for `pip` and `conda`,
- project configuration,
- metadata schema draft,
- local CSV/JSON/JSONL/Parquet ingestion,
- prompt text normalization,
- timestamp normalization,
- deterministic record IDs and text hashes,
- duplicate removal,
- lightweight language tagging,
- metadata extraction,
- preprocessing registry output,
- module directories for future analysis systems.

## Repository Layout

```text
data/
  raw/
  processed/
  embeddings/
  metadata/
notebooks/
  exploratory/
  experiments/
  archived_cogs108/
src/
  preprocessing/
  embeddings/
  clustering/
  emotion_analysis/
  temporal_analysis/
  semantic_drift/
  cognitive_outsourcing/
  network_analysis/
  visualization/
  evaluation/
  dashboard/
configs/
docs/
dashboard/
website/
reports/
papers/
figures/
experiments/
models/
tests/
deprecate/
```

## Setup

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using `conda`:

```bash
conda env create -f environment.yml
conda activate human-ai-behavior-observatory
```

## Preprocessing

Validate the project structure without processing records:

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml --validate-only
```

Run the Phase 2 preprocessing pipeline on a local dataset:

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml --input data/raw/example.csv --output data/processed/master_dataset.parquet --source-dataset example
```

Supported input formats are `.csv`, `.json`, `.jsonl`, `.ndjson`, and `.parquet`.

The processed output includes canonical identifiers, normalized text, timestamp fields, prompt length, word count, language tag, question/code indicators, and placeholder score columns for later model-driven annotations.

## Data Policy

Large raw, processed, and embedding files should stay out of Git unless explicitly curated for a small reproducible sample. Use the `data/` folders for local development artifacts and document external sources in `data/metadata/`.
