from pathlib import Path

from src.preprocessing.run_preprocessing import load_config, validate_paths


def test_project_config_loads() -> None:
    config = load_config(Path("configs/project.yaml"))

    assert config["project"]["slug"] == "human-ai-behavior-observatory"
    assert "schema_file" in config["preprocessing"]


def test_configured_paths_exist() -> None:
    config = load_config(Path("configs/project.yaml"))
    paths = validate_paths(config, Path.cwd())

    assert paths
    assert all(path.exists() for path in paths)

