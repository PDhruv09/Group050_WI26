"""Dataset loading and writing utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_INPUT_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet"}


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a local dataset into a DataFrame based on file suffix."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"Unsupported input file type '{suffix}'. Supported: {supported}")

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


def write_dataset(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame using the output suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix in {".jsonl", ".ndjson"}:
        frame.to_json(path, orient="records", lines=True)
        return
    if suffix == ".json":
        frame.to_json(path, orient="records", indent=2)
        return
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported output file type '{suffix}'.")

