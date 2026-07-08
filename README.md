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

## Phase 6 Status

Phase 6 adds an interactive dashboard on top of the Phase 2 canonical dataset, Phase 3 semantic outputs, Phase 4 behavioral classifications, and Phase 5 advanced analytics.

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
- SentenceTransformer embedding generation,
- semantic search utilities,
- semantic drift centroid generation,
- BERTopic baseline runner,
- embedding benchmarking and reproducibility checks,
- standalone UMAP/HDBSCAN clustering,
- topic validation and summaries,
- semantic visualizations,
- behavioral taxonomy,
- interaction mode classification,
- cognitive outsourcing classification,
- emotional signal classification,
- optional transformer-assisted emotion classification,
- optional zero-shot transformer behavioral classification,
- companionship, vulnerability, and dependency scores,
- anthropomorphism, reassurance-seeking, and self-disclosure indicators,
- prompt sophistication and conversational depth metrics,
- temporal behavioral trend reports,
- taxonomy coverage, overlap, unstable-region, and benchmark artifacts,
- classification summaries and evaluation scaffolding,
- Phase 5 temporal evolution analysis,
- interaction and emotion transition matrices,
- conversation archetype discovery,
- behavioral network analysis,
- event-window analysis,
- statistical trend testing,
- Phase 5 interactive figures and markdown report,
- Streamlit dashboard,
- dashboard filters and prompt explorer,
- dashboard diagnostics manifest,
- interactive behavioral trend, transition, archetype, and network views,
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
  classification/
  advanced_analytics/
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
The default acquisition mode matches the original notebook: `load_dataset("allenai/WildChat", split="train")`.

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

## Embeddings and Semantics

Generate prompt embeddings:

```bash
python -m src.embeddings.generate_embeddings --config configs/project.yaml
```

Search embeddings:

```bash
python -m src.embeddings.semantic_search "emotional support and loneliness" --config configs/project.yaml --top-k 10
```

Compute semantic drift centroids:

```bash
python -m src.semantic_drift.run_semantic_drift --config configs/project.yaml
```

Run the BERTopic baseline:

```bash
python -m src.clustering.bertopic_baseline --config configs/project.yaml
```

Run semantic clustering:

```bash
python -m src.clustering.semantic_clustering --config configs/project.yaml
```

Evaluate topics and generate summaries:

```bash
python -m src.clustering.topic_evaluation --config configs/project.yaml
```

Benchmark embedding models:

```bash
python -m src.embeddings.benchmark --config configs/project.yaml
```

Generate semantic visualizations:

```bash
python -m src.visualization.semantic_plots --config configs/project.yaml --kind all
```

## Behavioral Classification

Run the Phase 4 behavioral classifier:

```bash
python -m src.classification.run_classification --config configs/project.yaml
```

Run a smaller development sample:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 10000
```

Run optional transformer-assisted emotion classification:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 10000 --enable-transformers --transformer-max-records 1000
```

Run optional zero-shot transformer-assisted behavior classification:

```bash
python -m src.classification.run_classification --config configs/project.yaml --max-records 1000 --enable-zero-shot-behavior --zero-shot-max-records 100
```

The classifier writes a behavioral dataset, classification summary, manifest, benchmark, trend reports, taxonomy coverage diagnostics, overlap diagnostics, unstable-region diagnostics, and evaluation outputs using the taxonomy in `configs/behavior_taxonomy.yml`.

## Advanced Analytics

Run Phase 5 on a development sample:

```bash
python -m src.advanced_analytics.run_phase5_analytics --config configs/project.yaml --max-records 10000
```

Run Phase 5 on the full classified dataset:

```bash
python -m src.advanced_analytics.run_phase5_analytics --config configs/project.yaml
```

Phase 5 writes temporal trends, transition matrices, archetype summaries, behavioral network outputs, event-window analysis, statistical tests, interactive figures, a manifest, and a markdown report.

## Dashboard

Generate dashboard readiness metadata:

```bash
python -m src.dashboard.run_dashboard_manifest --config configs/project.yaml
```

Run the Phase 6 dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides filters, KPIs, behavioral trends, transition heatmaps, archetype summaries, network views, event-window outputs, prompt exploration, and artifact diagnostics.

## Data Policy

Large raw, processed, and embedding files should stay out of Git unless explicitly curated for a small reproducible sample. Use the `data/` folders for local development artifacts and document external sources in `data/metadata/`.
