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

## Phase 1 Status

Phase 1 reconstructs the repository so later work can happen in a reproducible, modular way.

Current infrastructure includes:

- research-oriented repository layout,
- archived COGS 108 materials,
- dependency files for `pip` and `conda`,
- project configuration,
- metadata schema draft,
- preprocessing pipeline entrypoint,
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

The Phase 1 preprocessing command validates the project structure and prepares a clean handoff point for Phase 2 data engineering.

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml
```

Phase 2 will extend this command with dataset-specific cleaning, normalization, deduplication, language filtering, metadata extraction, and embedding generation.

## Data Policy

Large raw, processed, and embedding files should stay out of Git unless explicitly curated for a small reproducible sample. Use the `data/` folders for local development artifacts and document external sources in `data/metadata/`.

