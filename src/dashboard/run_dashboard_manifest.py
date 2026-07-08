"""Command-line dashboard readiness manifest writer."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dashboard.manifest import write_dashboard_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Write dashboard readiness manifest and sample artifacts.")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()
    output = write_dashboard_manifest(Path(args.config))
    print(output)


if __name__ == "__main__":
    main()
