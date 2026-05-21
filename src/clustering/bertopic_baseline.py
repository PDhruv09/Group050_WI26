"""BERTopic baseline runner for Phase 3 semantic infrastructure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def log_step(message: str) -> None:
    """Print a flushed progress message for long-running BERTopic steps."""
    print(message, flush=True)


def run_bertopic_baseline(config: dict) -> dict:
    """Run BERTopic using precomputed embeddings."""
    log_step("Starting BERTopic baseline workflow.")
    try:
        from bertopic import BERTopic
    except ImportError as error:
        raise ImportError("BERTopic is required. Install dependencies with: pip install -r requirements.txt") from error

    topic_config = config["topic_modeling"]
    log_step(f"Loading prompt text from {topic_config['input_file']}.")
    texts = pd.read_parquet(topic_config["input_file"])[topic_config["text_column"]].fillna("").astype(str).tolist()
    log_step(f"Loaded {len(texts):,} documents.")

    log_step(f"Loading embeddings from {topic_config['embeddings_file']}.")
    embeddings = np.load(topic_config["embeddings_file"])
    log_step(f"Loaded embeddings with shape {embeddings.shape}.")

    output_dir = Path(topic_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_step("Fitting BERTopic model. This may take a long time on full WildChat.")
    model = BERTopic(min_topic_size=int(topic_config.get("min_topic_size", 50)))
    topics, probabilities = model.fit_transform(texts, embeddings)
    log_step("BERTopic fit complete.")

    log_step("Writing topic info.")
    topic_info = model.get_topic_info()
    topic_info.to_csv(output_dir / "bertopic_topic_info.csv", index=False)

    log_step("Writing topic assignments.")
    assignments = pd.DataFrame({"topic": topics})
    assignments.to_parquet(output_dir / "bertopic_assignments.parquet", index=False)

    log_step("Saving BERTopic model.")
    model.save(str(output_dir / "bertopic_model"))
    log_step("BERTopic artifacts saved.")

    return {
        "num_documents": len(texts),
        "num_topics": int(topic_info[topic_info["Topic"] != -1].shape[0]),
        "output_dir": str(output_dir),
        "has_probabilities": probabilities is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BERTopic baseline with generated embeddings.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    summary = run_bertopic_baseline(config)
    print("BERTopic baseline complete.")
    print(f"Documents: {summary['num_documents']}")
    print(f"Topics: {summary['num_topics']}")
    print(f"Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()
