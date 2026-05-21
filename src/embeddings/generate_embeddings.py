"""Generate prompt embeddings from the processed master dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.embeddings.io import save_embeddings, write_embedding_manifest, write_embedding_metadata


def load_config(config_path: Path) -> dict:
    """Load YAML project config."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_embedding_source(input_file: Path, text_column: str, id_column: str) -> pd.DataFrame:
    """Load and validate the source dataset for embedding generation."""
    frame = pd.read_parquet(input_file)
    missing = [column for column in [text_column, id_column] if column not in frame.columns]
    if missing:
        raise ValueError(f"Embedding source missing required columns: {missing}")

    frame = frame[frame[text_column].notna()].copy()
    frame[text_column] = frame[text_column].astype(str)
    return frame.reset_index(drop=True)


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
):
    """Encode texts with SentenceTransformer, imported lazily to keep tests lightweight."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError(
            "sentence-transformers is required for Phase 3 embedding generation. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )


def run_embedding_pipeline(config: dict) -> dict:
    """Generate embeddings and aligned metadata files."""
    embedding_config = config["embeddings"]
    input_file = Path(embedding_config["input_file"])
    output_file = Path(embedding_config["output_file"])
    metadata_file = Path(embedding_config["metadata_file"])
    manifest_file = Path(embedding_config["manifest_file"])
    text_column = embedding_config["text_column"]
    id_column = embedding_config["id_column"]

    frame = load_embedding_source(input_file, text_column, id_column)
    embeddings = encode_texts(
        texts=frame[text_column].tolist(),
        model_name=embedding_config["model_name"],
        batch_size=int(embedding_config.get("batch_size", 64)),
        normalize_embeddings=bool(embedding_config.get("normalize_embeddings", True)),
    )

    metadata_columns = [
        column
        for column in [
            id_column,
            text_column,
            "conversation_id",
            "timestamp",
            "year_month",
            "language",
            "data_split",
            "prompt_length",
            "prompt_word_count",
        ]
        if column in frame.columns
    ]
    metadata = frame[metadata_columns].copy()

    save_embeddings(embeddings, output_file)
    write_embedding_metadata(metadata, metadata_file)
    manifest = {
        "model_name": embedding_config["model_name"],
        "input_file": str(input_file),
        "output_file": str(output_file),
        "metadata_file": str(metadata_file),
        "num_records": int(len(frame)),
        "embedding_dimensions": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else None,
        "normalized": bool(embedding_config.get("normalize_embeddings", True)),
        "text_column": text_column,
        "id_column": id_column,
    }
    write_embedding_manifest(manifest_file, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 prompt embeddings.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    manifest = run_embedding_pipeline(config)

    print("Embedding generation complete.")
    print(f"Records: {manifest['num_records']}")
    print(f"Dimensions: {manifest['embedding_dimensions']}")
    print(f"Embeddings: {manifest['output_file']}")
    print(f"Metadata: {manifest['metadata_file']}")


if __name__ == "__main__":
    main()

