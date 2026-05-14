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
- full WildChat acquisition from Hugging Face,
- local CSV/JSON/JSONL/Parquet ingestion,
- prompt text normalization,
- timestamp normalization,
- temporal metadata fields,
- deterministic record IDs and text hashes,
- duplicate removal,
- lightweight language tagging,
- configurable language filtering,
- quality flags for short and long prompts,
- deterministic train/validation/test splits,
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

Run complete Phase 2 on full WildChat:

```bash
python -m src.preprocessing.run_phase2_wildchat
```

This acquires the full Hugging Face `allenai/WildChat` `train` split, extracts prompt-level user messages into `data/raw/`, preprocesses them, and writes master/train/validation/test outputs.

For a quick development sample:

```bash
python -m src.data_acquisition.wildchat --sample-size 10000 --output data/raw/wildchat_prompts_raw.jsonl
```

Or run acquisition and preprocessing together on a sample:

```bash
python -m src.preprocessing.run_phase2_wildchat --sample-size 10000
```

Validate the project structure without processing records:

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml --validate-only
```

Run the Phase 2 preprocessing pipeline on a local dataset:

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml --input data/raw/wildchat_prompts_raw.jsonl --output data/processed/master_dataset.parquet --source-dataset wildchat
```

Supported input formats are `.csv`, `.json`, `.jsonl`, `.ndjson`, and `.parquet`.

The processed output includes canonical identifiers, normalized text, timestamp fields, prompt length, word count, language tag, question/code indicators, and placeholder score columns for later model-driven annotations.

Phase 2 preserves all timestamps and all languages by default. Language filtering is configured as a toggle in `configs/project.yaml`, which keeps the dataset dashboard-ready for later filtering.

## Data Policy

Large raw, processed, and embedding files should stay out of Git unless explicitly curated for a small reproducible sample. Use the `data/` folders for local development artifacts and document external sources in `data/metadata/`.
