"""Command-line entry point for Phase 4 behavioral classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.classification.pipeline import run_classification_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 behavioral classification.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--input", default=None, help="Optional processed dataset override.")
    parser.add_argument("--output", default=None, help="Optional classified dataset override.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional record limit for development runs.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.input:
        config["classification"]["input_file"] = args.input
    if args.output:
        config["classification"]["output_file"] = args.output
    if args.max_records is not None:
        config["classification"]["max_records"] = args.max_records

    manifest = run_classification_pipeline(config)
    print(
        {
            "num_records": manifest["num_records"],
            "output_file": manifest["output_file"],
            "evaluation_file": manifest["evaluation_file"],
        }
    )


if __name__ == "__main__":
    main()
