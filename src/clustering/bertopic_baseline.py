"""BERTopic baseline runner for Phase 3 semantic infrastructure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def run_bertopic_baseline(config: dict) -> dict:
    """Run BERTopic using precomputed embeddings."""
    try:
        from bertopic import BERTopic
    except ImportError as error:
        raise ImportError("BERTopic is required. Install dependencies with: pip install -r requirements.txt") from error

    topic_config = config["topic_modeling"]
    texts = pd.read_parquet(topic_config["input_file"])[topic_config["text_column"]].fillna("").astype(str).tolist()
    embeddings = np.load(topic_config["embeddings_file"])
    output_dir = Path(topic_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = BERTopic(min_topic_size=int(topic_config.get("min_topic_size", 50)))
    topics, probabilities = model.fit_transform(texts, embeddings)
    topic_info = model.get_topic_info()
    topic_info.to_csv(output_dir / "bertopic_topic_info.csv", index=False)

    assignments = pd.DataFrame({"topic": topics})
    assignments.to_parquet(output_dir / "bertopic_assignments.parquet", index=False)
    model.save(str(output_dir / "bertopic_model"))

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
