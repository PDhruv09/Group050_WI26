from pathlib import Path

import pandas as pd
import yaml

from src.classification.pipeline import add_behavioral_columns, run_classification_pipeline
from src.classification.rules import classify_prompt
from src.classification.taxonomy import load_taxonomy


TAXONOMY_FILE = Path("configs/behavior_taxonomy.yml")


def test_taxonomy_loads_required_sections() -> None:
    taxonomy = load_taxonomy(TAXONOMY_FILE)

    assert "interaction_modes" in taxonomy
    assert "cognitive_outsourcing" in taxonomy
    assert "emotional_signals" in taxonomy


def test_tool_prompt_classifies_as_coding_outsourcing() -> None:
    taxonomy = load_taxonomy(TAXONOMY_FILE)

    result = classify_prompt("Write a Python function and debug this error.", taxonomy)

    assert result["interaction_mode"] == "tool_mode"
    assert result["cognitive_outsourcing_type"] == "coding"
    assert result["is_cognitive_outsourcing"] is True


def test_lonely_prompt_sets_companionship_and_vulnerability() -> None:
    taxonomy = load_taxonomy(TAXONOMY_FILE)

    result = classify_prompt("I feel lonely and alone. I am struggling. Can you talk to me and comfort me?", taxonomy)

    assert result["interaction_mode"] == "companion_mode"
    assert result["emotion_primary"] == "loneliness"
    assert result["is_companionship"] is True
    assert result["vulnerability_score"] > 0


def test_therapy_like_prompt_classifies_as_therapist_surrogate() -> None:
    taxonomy = load_taxonomy(TAXONOMY_FILE)

    result = classify_prompt("I am having anxiety and panic after trauma. I cannot cope.", taxonomy)

    assert result["interaction_mode"] == "therapist_surrogate_mode"
    assert result["is_vulnerable"] is True
    assert result["cognitive_outsourcing_type"] == "emotional_regulation"


def test_add_behavioral_columns_updates_metadata_fields() -> None:
    taxonomy = load_taxonomy(TAXONOMY_FILE)
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "prompt_text": ["Help me decide whether to quit my job.", "Brainstorm story ideas with me."],
            "interaction_type": [pd.NA, pd.NA],
            "disclosure_score": [0.0, 0.0],
        }
    )

    result = add_behavioral_columns(
        frame,
        {"text_column": "prompt_text", "score_threshold": 0.35},
        taxonomy,
    )

    assert "interaction_mode" in result.columns
    assert "classified_at" in result.columns
    assert result.loc[0, "cognitive_outsourcing_type"] == "decision_making"
    assert result.loc[1, "interaction_mode"] == "collaborator_mode"
    assert result.loc[0, "interaction_type"] == result.loc[0, "interaction_mode"]


def test_run_classification_pipeline_writes_outputs() -> None:
    scratch_dir = Path("data/processed/_test_classification")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    input_file = scratch_dir / "processed.parquet"
    output_file = scratch_dir / "classified.parquet"
    summary_file = scratch_dir / "classification_summary.csv"
    manifest_file = scratch_dir / "classification_manifest.json"
    evaluation_file = scratch_dir / "classification_evaluation.json"
    confusion_dir = scratch_dir / "confusion_matrices"

    pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "prompt_text": [
                "Write an email for me.",
                "I feel lonely. Please keep me company.",
                "Explain this homework problem step by step.",
            ],
            "interaction_type": [pd.NA, pd.NA, pd.NA],
            "disclosure_score": [0.0, 0.0, 0.0],
        }
    ).to_parquet(input_file, index=False)

    config = {
        "classification": {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "taxonomy_file": str(TAXONOMY_FILE),
            "text_column": "prompt_text",
            "id_column": "record_id",
            "score_threshold": 0.35,
            "max_records": None,
            "summary_file": str(summary_file),
            "manifest_file": str(manifest_file),
            "labeled_data_file": None,
            "evaluation_file": str(evaluation_file),
            "confusion_matrix_dir": str(confusion_dir),
        }
    }

    manifest = run_classification_pipeline(config)
    classified = pd.read_parquet(output_file)

    assert manifest["num_records"] == 3
    assert output_file.exists()
    assert summary_file.exists()
    assert manifest_file.exists()
    assert evaluation_file.exists()
    assert classified["is_companionship"].sum() == 1


def test_project_config_contains_phase4_classification() -> None:
    with Path("configs/project.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["project"]["phase"] == 4
    assert config["classification"]["taxonomy_file"] == "configs/behavior_taxonomy.yml"
