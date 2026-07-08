"""Dashboard manifest generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.dashboard.data import CLASSIFIED_DASHBOARD_COLUMNS, read_table
from src.dashboard.data import artifact_status, load_config, load_json


def build_dashboard_manifest(config_path: Path = Path("configs/project.yaml")) -> dict:
    """Build a dashboard readiness manifest."""
    config = load_config(config_path)
    statuses = artifact_status(config)
    dash = config["dashboard"]
    manifest = {
        "phase": 6,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_file": dash["app_file"],
        "available_artifacts": sum(1 for status in statuses if status.exists),
        "missing_artifacts": [status.name for status in statuses if not status.exists],
        "artifacts": [
            {
                "name": status.name,
                "path": str(status.path),
                "exists": status.exists,
                "rows": status.rows,
            }
            for status in statuses
        ],
        "phase5_manifest": load_json(Path(dash["phase5_manifest_file"])),
    }
    return manifest


def write_dashboard_sample(config: dict) -> Path | None:
    """Write a deterministic dashboard sample for responsive UI loading."""
    dash = config["dashboard"]
    source = Path(dash["classified_data_file"])
    output = Path(dash.get("sample_file", ""))
    if not source.exists() or not output:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    source_mtime = source.stat().st_mtime
    if output.exists() and output.stat().st_mtime >= source_mtime:
        return output

    sample_rows = int(dash.get("sample_rows", dash.get("max_prompt_rows", 50000)))
    random_state = int(dash.get("sample_random_state", 42))
    sample = read_table(
        source,
        max_rows=sample_rows,
        random_state=random_state,
        columns=CLASSIFIED_DASHBOARD_COLUMNS,
    )
    sample.to_parquet(output, index=False)
    return output


def write_dashboard_manifest(config_path: Path = Path("configs/project.yaml")) -> Path:
    """Write dashboard readiness manifest."""
    config = load_config(config_path)
    write_dashboard_sample(config)
    output_file = Path(config["dashboard"]["output_manifest_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    import json

    output_file.write_text(json.dumps(build_dashboard_manifest(config_path), indent=2) + "\n", encoding="utf-8")
    return output_file
