# Phase 2 Data Engineering

## Goal

Create the first reusable preprocessing and storage layer for public human-AI interaction datasets.

## Pipeline Scope

The Phase 2 pipeline supports:

- full WildChat prompt acquisition from Hugging Face,
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
- deterministic train/validation/test split files.

## Command

Run complete Phase 2 on the full WildChat dataset:

```bash
python -m src.preprocessing.run_phase2_wildchat
```

The default acquisition mode matches the original notebook and loads the Hugging Face split eagerly rather than using streaming.

Run the complete workflow on a development sample:

```bash
python -m src.preprocessing.run_phase2_wildchat --sample-size 10000
```

Acquire only the prompt-level WildChat raw file:

```bash
python -m src.data_acquisition.wildchat --sample-size 10000 --output data/raw/wildchat_prompts_raw.jsonl
```

Omit `--sample-size` to acquire all available prompt rows.

Process the acquired raw file:

```bash
python -m src.preprocessing.run_preprocessing --config configs/project.yaml --input data/raw/wildchat_prompts_raw.jsonl --output data/processed/master_dataset.parquet --source-dataset wildchat
```

## Output Columns

The canonical dataset includes:

- `record_id`
- `conversation_id`
- `turn_index`
- `prompt_text`
- `timestamp`
- `date`
- `year`
- `month`
- `year_month`
- `source_dataset`
- `raw_text_hash`
- `prompt_length`
- `prompt_word_count`
- `language`
- `language_original`
- `language_detected`
- `data_split`
- `is_short_prompt`
- `is_length_outlier`
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

Split outputs are written to:

- `data/processed/train.parquet`
- `data/processed/validation.parquet`
- `data/processed/test.parquet`
