"""Deterministic split assignment utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.preprocessing.io import write_dataset


def hash_to_unit_interval(value: object) -> float:
    """Map a value to a deterministic float in [0, 1)."""
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(16**16)


def assign_splits(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign train/validation/test splits while keeping conversations together."""
    split_config = config.get("splits", {})
    if not split_config.get("enabled", True):
        frame = frame.copy()
        frame["data_split"] = pd.NA
        return frame

    train_ratio = float(split_config.get("train", 0.7))
    validation_ratio = float(split_config.get("validation", 0.15))
    test_ratio = float(split_config.get("test", 0.15))
    total = train_ratio + validation_ratio + test_ratio
    if round(total, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0.")

    frame = frame.copy()
    keys = frame["conversation_id"].fillna(frame["record_id"])
    values = keys.map(hash_to_unit_interval)
    validation_cutoff = train_ratio + validation_ratio

    frame["data_split"] = "test"
    frame.loc[values < validation_cutoff, "data_split"] = "validation"
    frame.loc[values < train_ratio, "data_split"] = "train"
    return frame


def write_split_outputs(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """Write split-specific Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split_name in ["train", "validation", "test"]:
        split_path = output_dir / f"{split_name}.parquet"
        split_frame = frame[frame["data_split"] == split_name].copy()
        write_dataset(split_frame, split_path)
        outputs[split_name] = str(split_path)
    return outputs
