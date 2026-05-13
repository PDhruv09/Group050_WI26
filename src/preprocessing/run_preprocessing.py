"""Phase 1 preprocessing entrypoint.

This script intentionally performs structural validation only. Dataset-specific
normalization and feature extraction will be implemented in Phase 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


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
    parser = argparse.ArgumentParser(description="Validate Phase 1 preprocessing infrastructure.")
    parser.add_argument(
        "--config",
        default="configs/project.yaml",
        help="Path to the project configuration file.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    config_path = project_root / args.config
    config = load_config(config_path)
    paths = validate_paths(config, project_root)

    schema_file = project_root / config["preprocessing"]["schema_file"]
    if not schema_file.exists():
        raise FileNotFoundError(f"Metadata schema not found: {schema_file}")

    print("Phase 1 preprocessing infrastructure is ready.")
    print(f"Config: {config_path}")
    print(f"Schema: {schema_file}")
    print(f"Validated paths: {len(paths)}")


if __name__ == "__main__":
    main()

