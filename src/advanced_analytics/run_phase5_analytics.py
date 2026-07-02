"""Command-line runner for Phase 5 advanced analytics."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.advanced_analytics.archetypes import discover_archetypes
from src.advanced_analytics.events import write_event_window_analysis
from src.advanced_analytics.io import load_classified_dataset, write_json
from src.advanced_analytics.network import write_network_outputs
from src.advanced_analytics.reporting import write_phase5_report
from src.advanced_analytics.temporal import write_temporal_outputs
from src.advanced_analytics.visualization import write_phase5_visualizations

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def run_phase5(config: dict) -> dict:
    """Run all Phase 5 analytics outputs."""
    start = time.perf_counter()
    print("Loading Phase 4 classified dataset...")
    frame = load_classified_dataset(config)
    print(f"Loaded {len(frame):,} records for Phase 5 analytics.")

    print("Writing temporal evolution and transition analyses...")
    temporal = write_temporal_outputs(frame, config)

    print("Discovering conversation archetypes...")
    archetypes = discover_archetypes(frame, config)

    print("Building behavioral network outputs...")
    network = write_network_outputs(frame, config)

    print("Running event-window analysis...")
    events = write_event_window_analysis(frame, config)

    print("Writing Phase 5 visualizations...")
    figures = write_phase5_visualizations(config)

    manifest = {
        "phase": 5,
        "num_records": int(len(frame)),
        "elapsed_seconds": float(time.perf_counter() - start),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **temporal,
        **archetypes,
        **network,
        **events,
        "figures": figures,
    }

    print("Writing Phase 5 report and manifest...")
    report_file = write_phase5_report(config, manifest)
    manifest["report_file"] = str(report_file)
    write_json(Path(config["advanced_analytics"]["manifest_file"]), manifest)
    print("Phase 5 advanced analytics complete.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 advanced behavioral analytics.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--input", default=None, help="Optional classified dataset override.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional record limit for development runs.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.input:
        config["advanced_analytics"]["input_file"] = args.input
    if args.max_records is not None:
        config["advanced_analytics"]["max_records"] = args.max_records

    manifest = run_phase5(config)
    print(
        {
            "num_records": manifest["num_records"],
            "report_file": manifest["report_file"],
            "manifest_file": config["advanced_analytics"]["manifest_file"],
        }
    )


if __name__ == "__main__":
    main()
