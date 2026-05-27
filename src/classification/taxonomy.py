"""Load and validate behavioral taxonomy definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "interaction_modes",
    "cognitive_outsourcing",
    "emotional_signals",
    "composite_signals",
}


def load_taxonomy(path: Path) -> dict[str, Any]:
    """Load a taxonomy YAML file and validate required sections."""
    with path.open("r", encoding="utf-8") as file:
        taxonomy = yaml.safe_load(file)
    if not isinstance(taxonomy, dict):
        raise ValueError(f"Taxonomy must be a mapping: {path}")
    validate_taxonomy(taxonomy)
    return taxonomy


def validate_taxonomy(taxonomy: dict[str, Any]) -> None:
    """Validate taxonomy sections and keyword lists."""
    missing = REQUIRED_SECTIONS - set(taxonomy)
    if missing:
        raise ValueError(f"Taxonomy is missing required sections: {', '.join(sorted(missing))}")

    for section in REQUIRED_SECTIONS:
        values = taxonomy.get(section)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"Taxonomy section '{section}' must contain label mappings.")
        for label, payload in values.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Taxonomy label '{section}.{label}' must be a mapping.")
            keywords = payload.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                raise ValueError(f"Taxonomy label '{section}.{label}' needs at least one keyword.")
