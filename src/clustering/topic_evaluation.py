"""Topic coherence, summaries, and interpretability validation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer


def top_terms_by_topic(texts: pd.Series, topics: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Compute top count-based terms per topic for summaries and validation."""
    rows = []
    for topic_id in sorted(topic for topic in topics.dropna().unique() if int(topic) != -1):
        topic_texts = texts[topics == topic_id].fillna("").astype(str)
        if topic_texts.empty:
            continue
        min_df = 2 if len(topic_texts) >= 2 else 1
        vectorizer = CountVectorizer(stop_words="english", max_features=20000, min_df=min_df)
        matrix = vectorizer.fit_transform(topic_texts)
        counts = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        top_indices = counts.argsort()[::-1][:top_n]
        rows.append(
            {
                "topic": int(topic_id),
                "num_records": int(len(topic_texts)),
                "top_terms": ", ".join(terms[top_indices]),
            }
        )
    return pd.DataFrame(rows)


def simple_topic_diversity(topics_terms: pd.DataFrame) -> float:
    """Measure fraction of unique terms across topic summaries."""
    terms = []
    for value in topics_terms["top_terms"].dropna():
        terms.extend([term.strip() for term in value.split(",") if term.strip()])
    return len(set(terms)) / len(terms) if terms else 0.0


def count_non_outlier_topics(topics: pd.Series) -> int:
    """Count unique BERTopic topics excluding the outlier topic -1."""
    return int(topics[topics != -1].nunique())


def evaluate_topics(config: dict) -> dict:
    """Evaluate BERTopic outputs with lightweight, reproducible metrics."""
    topic_config = config["topic_modeling"]
    output_dir = Path(topic_config["output_dir"])
    assignments_file = output_dir / "bertopic_assignments.parquet"
    data = pd.read_parquet(topic_config["input_file"])
    assignments = pd.read_parquet(assignments_file)
    topics = assignments["topic"]
    texts = data[topic_config["text_column"]].iloc[: len(assignments)]

    terms = top_terms_by_topic(texts, topics, int(topic_config.get("coherence_top_n", 10)))
    terms.to_csv(output_dir / "topic_top_terms.csv", index=False)

    diversity = simple_topic_diversity(terms)
    payload = {
        "num_documents": int(len(assignments)),
        "num_topics": count_non_outlier_topics(topics),
        "noise_fraction": float((topics == -1).mean()),
        "topic_diversity": diversity,
    }
    validation_file = Path(topic_config["topic_validation_file"])
    validation_file.parent.mkdir(parents=True, exist_ok=True)
    validation_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary_file = Path(topic_config["topic_summary_file"])
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Topic Summaries", ""]
    for row in terms.itertuples(index=False):
        lines.append(f"## Topic {row.topic}")
        lines.append(f"- Records: {row.num_records}")
        lines.append(f"- Top terms: {row.top_terms}")
        lines.append("")
    summary_file.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and summarize BERTopic outputs.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print(evaluate_topics(config))


if __name__ == "__main__":
    main()
