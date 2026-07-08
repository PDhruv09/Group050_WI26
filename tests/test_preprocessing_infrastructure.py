import json
from pathlib import Path

import pandas as pd

from src.preprocessing.run_preprocessing import load_config, validate_paths
from src.preprocessing.pipeline import normalize_records, run_pipeline


def test_project_config_loads() -> None:
    config = load_config(Path("configs/project.yaml"))

    assert config["project"]["slug"] == "human-ai-behavior-observatory"
    assert config["project"]["phase"] == 6
    assert "schema_file" in config["preprocessing"]
    assert "model_name" in config["embeddings"]


def test_configured_paths_exist() -> None:
    config = load_config(Path("configs/project.yaml"))
    paths = validate_paths(config, Path.cwd())

    assert paths
    assert all(path.exists() for path in paths)


def test_normalize_records_extracts_canonical_metadata() -> None:
    config = load_config(Path("configs/project.yaml"))
    frame = pd.DataFrame(
        {
            "prompt": [" How can I write a Python function? ", "print('hello')"],
            "created_at": ["2024-01-01T12:00:00Z", "not-a-date"],
            "conversation_id": ["a", "a"],
            "turn": [1, 2],
        }
    )

    normalized = normalize_records(frame, config, "sample")

    assert list(normalized["source_dataset"].unique()) == ["sample"]
    assert normalized.loc[0, "prompt_text"] == "How can I write a Python function?"
    assert normalized.loc[0, "has_question"]
    assert normalized.loc[1, "contains_code"]
    assert "raw_text_hash" in normalized.columns


def test_run_pipeline_writes_processed_dataset_and_registry_ready_summary() -> None:
    config = load_config(Path("configs/project.yaml"))
    test_dir = Path("tests/_tmp")
    test_dir.mkdir(parents=True, exist_ok=True)
    input_file = test_dir / "sample.csv"
    output_file = test_dir / "processed.csv"
    input_file.write_text(
        "prompt,created_at,conversation_id,turn\n"
        "How do I study for an exam?,2024-02-01T00:00:00Z,c1,1\n"
        "How do I study for an exam?,2024-02-01T00:00:00Z,c1,1\n"
        "okay thanks,2024-02-01T00:01:00Z,c1,2\n",
        encoding="utf-8",
    )

    summary = run_pipeline(input_file, output_file, config, "unit_test")

    assert output_file.exists()
    assert (test_dir / "splits" / "train.parquet").exists()
    assert summary["raw_rows"] == 3
    assert summary["processed_rows"] == 2
    assert "prompt_text" in summary["columns"]
    assert summary["split_outputs"]
    assert set(summary["split_counts"]).issubset({"train", "validation", "test"})

    json.dumps(summary)
