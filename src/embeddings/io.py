"""Embedding storage helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_embeddings(embeddings: np.ndarray, output_file: Path) -> None:
    """Persist embedding matrix as a NumPy array."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, embeddings)


def load_embeddings(path: Path) -> np.ndarray:
    """Load an embedding matrix."""
    return np.load(path)


def write_embedding_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write embedding generation metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **manifest,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def write_embedding_metadata(frame: pd.DataFrame, path: Path) -> None:
    """Persist metadata aligned row-for-row with the embedding matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)

