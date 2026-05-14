"""Run the complete Phase 2 WildChat acquisition and preprocessing workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_acquisition.wildchat import acquire_wildchat
from src.preprocessing.run_preprocessing import load_config, validate_paths
from src.preprocessing.pipeline import run_pipeline
from src.preprocessing.registry import write_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and preprocess WildChat end to end.")
    parser.add_argument("--config", default="configs/project.yaml")
    parser.add_argument("--raw-output", default=None)
    parser.add_argument("--processed-output", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream records from Hugging Face instead of downloading the split eagerly.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Deprecated compatibility flag. Non-streaming is now the default.",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    config_path = project_root / args.config
    config = load_config(config_path)
    validate_paths(config, project_root)

    acquisition_config = config["data_acquisition"]["wildchat"]
    raw_output = project_root / (args.raw_output or acquisition_config["output_file"])
    processed_output = project_root / (args.processed_output or config["preprocessing"]["output_file"])
    sample_size = args.sample_size
    if sample_size is None:
        sample_size = acquisition_config.get("sample_size")

    acquisition_summary = acquire_wildchat(
        output_file=raw_output,
        dataset_name=acquisition_config["dataset_name"],
        split=acquisition_config["split"],
        streaming=args.streaming and not args.no_streaming and bool(acquisition_config.get("streaming", False)),
        max_conversations=args.max_conversations,
        sample_size=sample_size,
    )
    preprocessing_summary = run_pipeline(
        input_file=raw_output,
        output_file=processed_output,
        config=config,
        source_dataset="wildchat",
    )

    registry_file = project_root / config["preprocessing"]["registry_file"]
    write_registry(
        {
            "phase": 2,
            "acquisition": acquisition_summary,
            "preprocessing": preprocessing_summary,
        },
        registry_file,
    )

    print("Phase 2 WildChat workflow complete.")
    print(f"Raw prompt rows: {acquisition_summary['prompt_rows']}")
    print(f"Processed rows: {preprocessing_summary['processed_rows']}")
    print(f"Raw output: {raw_output}")
    print(f"Processed output: {processed_output}")
    print(f"Split counts: {preprocessing_summary['split_counts']}")
    print(f"Registry: {registry_file}")


if __name__ == "__main__":
    main()
