# Phase 3 Embedding and Semantic Infrastructure

## Goal

Replace fragile bag-of-words topic modeling with reusable embedding, semantic search, topic modeling, and semantic drift infrastructure.

## Workflow

Generate prompt embeddings:

```bash
python -m src.embeddings.generate_embeddings --config configs/project.yaml
```

Search generated embeddings:

```bash
python -m src.embeddings.semantic_search "lonely and need someone to talk to" --config configs/project.yaml --top-k 10
```

Compute semantic drift centroids:

```bash
python -m src.semantic_drift.run_semantic_drift --config configs/project.yaml
```

Run BERTopic baseline after embeddings exist:

```bash
python -m src.clustering.bertopic_baseline --config configs/project.yaml
```

## Outputs

- `data/embeddings/prompt_embeddings.npy`
- `data/embeddings/prompt_embedding_metadata.parquet`
- `data/embeddings/embedding_manifest.json`
- `data/embeddings/semantic_drift_centroids.parquet`
- `data/processed/topics/bertopic_topic_info.csv`
- `data/processed/topics/bertopic_assignments.parquet`

## Design Notes

- Embeddings are generated from `prompt_text`.
- Embedding metadata is saved row-for-row with the NumPy matrix.
- Semantic search uses cosine similarity.
- Semantic drift starts with group centroids by `year_month`.
- BERTopic uses precomputed embeddings so topic modeling and embedding generation are separable.

