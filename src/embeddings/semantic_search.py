"""Semantic search over generated prompt embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.embeddings.generate_embeddings import encode_texts
from src.embeddings.io import load_embeddings


def cosine_top_k(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Return row indices and scores for the top-k cosine similarities."""
    if query_embedding.ndim == 2:
        query_embedding = query_embedding[0]

    query_norm = np.linalg.norm(query_embedding)
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    denominators = embedding_norms * query_norm
    similarities = np.divide(
        embeddings @ query_embedding,
        denominators,
        out=np.zeros(len(embeddings), dtype=float),
        where=denominators != 0,
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(index), float(similarities[index])) for index in top_indices]


def search_embeddings(config: dict, query: str, top_k: int | None = None) -> pd.DataFrame:
    """Embed a query and search prompt embeddings."""
    embedding_config = config["embeddings"]
    search_config = config.get("semantic_search", {})
    limit = top_k or int(search_config.get("top_k", 10))

    embeddings = load_embeddings(Path(embedding_config["output_file"]))
    metadata = pd.read_parquet(embedding_config["metadata_file"])
    query_embedding = encode_texts(
        [query],
        model_name=embedding_config["model_name"],
        batch_size=1,
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
    )
    matches = cosine_top_k(query_embedding, embeddings, limit)
    rows = []
    for index, score in matches:
        row = metadata.iloc[index].to_dict()
        row["similarity"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search generated prompt embeddings.")
    parser.add_argument("query")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    results = search_embeddings(config, args.query, args.top_k)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

