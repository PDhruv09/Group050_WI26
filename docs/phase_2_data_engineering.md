# Phase 2 Data Engineering

## Goal

Create the first reusable preprocessing and storage layer for public human-AI interaction datasets.

## Pipeline Scope

The Phase 2 pipeline supports:

- CSV, JSON, JSONL, NDJSON, and Parquet input files,
- canonical prompt text selection from configurable candidate columns,
- timestamp normalization to UTC,
- deterministic record IDs,
- SHA-256 prompt hashes,
- prompt length and word count extraction,
- lightweight language tagging,
- duplicate removal,
- simple prompt complexity proxy,
- placeholder metadata fields for later classifiers,
- Parquet, CSV, JSON, and JSONL output,
- JSON registry output for preprocessing provenance.

## Command

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml --input data/raw/example.csv --output data/processed/master_dataset.parquet --source-dataset example
```

## Output Columns

The canonical dataset includes:

- `record_id`
- `conversation_id`
- `turn_index`
- `prompt_text`
- `timestamp`
- `source_dataset`
- `raw_text_hash`
- `prompt_length`
- `prompt_word_count`
- `language`
- `has_question`
- `contains_code`
- `complexity_score`
- `disclosure_score`
- `dependency_score`
- `interaction_type`
- `semantic_cluster`
- `emotion_scores`
- `cleaned_at`

## Next Phase Handoff

Phase 3 can now consume `data/processed/master_dataset.parquet` for embedding generation, semantic search, BERTopic experiments, and semantic drift analysis.

