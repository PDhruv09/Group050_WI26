"""Regenerate train/validation/test files from an existing processed master dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.run_preprocessing import load_config
from src.preprocessing.splits import assign_splits, write_split_outputs


def regenerate_splits(config: dict, input_file: Path | None = None, output_dir: Path | None = None) -> dict[str, int]:
    """Assign deterministic splits and rewrite split Parquet files."""
    split_config = config.get("splits", {})
    source_file = input_file or Path(config["preprocessing"]["output_file"])
    target_dir = output_dir or Path(split_config.get("output_dir", source_file.parent))

    frame = pd.read_parquet(source_file)
    frame = assign_splits(frame, config)
    write_split_outputs(frame, target_dir)
    return {str(key): int(value) for key, value in frame["data_split"].value_counts().to_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate split files from the processed master dataset.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    counts = regenerate_splits(
        config,
        input_file=Path(args.input) if args.input else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    print("Split regeneration complete.")
    print(f"Split counts: {counts}")


if __name__ == "__main__":
    main()
