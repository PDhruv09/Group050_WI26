"""Benchmark and compare embedding model runs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.embeddings.generate_embeddings import encode_texts, load_embedding_source


def sample_texts(frame: pd.DataFrame, text_column: str, sample_size: int | None) -> list[str]:
    """Return a deterministic text sample for benchmarking."""
    if sample_size is not None and len(frame) > sample_size:
        frame = frame.sample(sample_size, random_state=42)
    return frame[text_column].astype(str).tolist()


def benchmark_model(
    texts: list[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> dict:
    """Benchmark one embedding model on a list of texts."""
    start = time.perf_counter()
    embeddings = encode_texts(texts, model_name, batch_size, normalize_embeddings)
    elapsed = time.perf_counter() - start
    records_per_second = len(texts) / elapsed if elapsed else 0.0
    return {
        "model_name": model_name,
        "num_records": len(texts),
        "batch_size": batch_size,
        "seconds": elapsed,
        "records_per_second": records_per_second,
        "embedding_dimensions": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else None,
    }


def compare_embedding_models(config: dict) -> pd.DataFrame:
    """Run configured embedding model benchmarks."""
    embedding_config = config["embeddings"]
    frame = load_embedding_source(
        Path(embedding_config["input_file"]),
        embedding_config["text_column"],
        embedding_config["id_column"],
    )
    texts = sample_texts(
        frame,
        embedding_config["text_column"],
        embedding_config.get("benchmark_sample_size"),
    )
    rows = [
        benchmark_model(
            texts,
            model_name,
            int(embedding_config.get("batch_size", 64)),
            bool(embedding_config.get("normalize_embeddings", True)),
        )
        for model_name in embedding_config.get("comparison_models", [embedding_config["model_name"]])
    ]
    output_file = Path(embedding_config["comparison_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_file, index=False)
    return result


def validate_embedding_reproducibility(config: dict, sample_size: int = 128) -> dict:
    """Encode the same sample twice and measure maximum absolute difference."""
    embedding_config = config["embeddings"]
    frame = load_embedding_source(
        Path(embedding_config["input_file"]),
        embedding_config["text_column"],
        embedding_config["id_column"],
    )
    texts = sample_texts(frame, embedding_config["text_column"], sample_size)
    first = encode_texts(
        texts,
        embedding_config["model_name"],
        int(embedding_config.get("batch_size", 64)),
        bool(embedding_config.get("normalize_embeddings", True)),
    )
    second = encode_texts(
        texts,
        embedding_config["model_name"],
        int(embedding_config.get("batch_size", 64)),
        bool(embedding_config.get("normalize_embeddings", True)),
    )
    max_abs_diff = float(np.max(np.abs(first - second)))
    payload = {
        "model_name": embedding_config["model_name"],
        "sample_size": len(texts),
        "max_absolute_difference": max_abs_diff,
        "is_reproducible": max_abs_diff < 1e-6,
    }
    output_file = Path(embedding_config["reproducibility_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 3 embedding models.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--reproducibility", action="store_true")
    parser.add_argument("--sample-size", type=int, default=128)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.reproducibility:
        result = validate_embedding_reproducibility(config, args.sample_size)
        print(result)
    else:
        result = compare_embedding_models(config)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
