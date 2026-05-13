# Data

Local data is organized by lifecycle stage.

- `raw/`: immutable source files copied or downloaded from public datasets.
- `processed/`: cleaned and normalized analysis-ready files.
- `embeddings/`: vector outputs and semantic indexes.
- `metadata/`: data dictionaries, source notes, schema exports, and provenance records.

Large data files should remain outside Git unless they are intentionally small reproducible samples.

