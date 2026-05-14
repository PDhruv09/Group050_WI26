# Phase 1 Reconstruction

## Goal

Rebuild the repository from a course submission into a modular research platform.

## Completed Scope

- Archived original COGS 108 materials under `deprecate/`.
- Created research-grade folder structure.
- Added dependency files for both `pip` and `conda`.
- Added project configuration.
- Drafted canonical metadata schema.
- Added a preprocessing entrypoint that validates structure and config.
- Created module directories for later pipeline work.

## Out of Scope

Phase 1 does not implement dataset-specific cleaning, embedding generation, BERTopic modeling, classifiers, dashboards, or analysis results. Those belong to later phases.

Phase 2 has now started the dataset cleaning and canonicalization layer. See `docs/phase_2_data_engineering.md`.
