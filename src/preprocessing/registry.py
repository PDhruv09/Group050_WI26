"""Preprocessing run registry helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_registry(summary: dict, path: Path) -> None:
    """Write a JSON registry entry for the latest preprocessing run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")

