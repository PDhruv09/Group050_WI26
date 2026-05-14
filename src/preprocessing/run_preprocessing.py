"""Preprocessing entrypoint for local human-AI interaction datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.registry import write_registry


def load_config(config_path: Path) -> dict:
    """Load the project YAML configuration."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_paths(config: dict, project_root: Path) -> list[Path]:
    """Create and return configured project paths."""
    paths = []
    for relative_path in config.get("paths", {}).values():
        path = project_root / relative_path
        path.mkdir(parents=True, exist_ok=True)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 preprocessing pipeline.")
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Path to the project configuration file.",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        default=None,
        help="Input dataset path. Overrides preprocessing.input_file in the config.",
    )
    parser.add_argument(
        "--output",
        dest="output_file",
        default=None,
        help="Output dataset path. Overrides preprocessing.output_file in the config.",
    )
    parser.add_argument(
        "--source-dataset",
        default=None,
        help="Source dataset label to store in the processed output.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate config, schema, and directories without processing data.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    config_path = project_root / args.config
    config = load_config(config_path)
    paths = validate_paths(config, project_root)

    schema_file = project_root / config["preprocessing"]["schema_file"]
    if not schema_file.exists():
        raise FileNotFoundError(f"Metadata schema not found: {schema_file}")

    input_setting = args.input_file or config["preprocessing"].get("input_file")
    output_setting = args.output_file or config["preprocessing"]["output_file"]

    if args.validate_only or not input_setting:
        print("Preprocessing infrastructure is ready.")
        print("No input dataset was provided, so no records were processed.")
        print(f"Config: {config_path}")
        print(f"Schema: {schema_file}")
        print(f"Validated paths: {len(paths)}")
        return

    input_file = project_root / input_setting
    output_file = project_root / output_setting
    if not input_file.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    summary = run_pipeline(input_file, output_file, config, args.source_dataset)
    registry_file = project_root / config["preprocessing"]["registry_file"]
    write_registry(summary, registry_file)

    print("Preprocessing complete.")
    print(f"Config: {config_path}")
    print(f"Schema: {schema_file}")
    print(f"Input rows: {summary['raw_rows']}")
    print(f"Processed rows: {summary['processed_rows']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Output: {output_file}")
    if summary["split_outputs"]:
        print(f"Train split: {summary['split_outputs']['train']}")
        print(f"Validation split: {summary['split_outputs']['validation']}")
        print(f"Test split: {summary['split_outputs']['test']}")
    print(f"Registry: {registry_file}")


if __name__ == "__main__":
    main()
