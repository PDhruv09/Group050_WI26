"""Input and output helpers for Phase 5 analytics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def load_classified_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the Phase 4 classified dataset."""
    analytics_config = config["advanced_analytics"]
    input_file = Path(analytics_config["input_file"])
    if not input_file.exists():
        raise FileNotFoundError(
            f"Phase 5 input not found: {input_file}. Run Phase 4 classification before Phase 5 analytics."
        )
    frame = pd.read_parquet(input_file)
    max_records = analytics_config.get("max_records")
    if max_records is not None:
        return frame.head(int(max_records)).copy()
    return frame


def ensure_parent(path: Path) -> None:
    """Create a file parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with a generated timestamp."""
    ensure_parent(path)
    serializable = {"generated_at": datetime.now(timezone.utc).isoformat(), **payload}
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")

