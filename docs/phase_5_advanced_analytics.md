# Phase 5 Advanced Analytics

Phase 5 turns classified prompt-level behavior into research-grade analytical outputs. It consumes the Phase 4 classified dataset and generates temporal trends, transition matrices, archetypes, network analysis, event-window summaries, statistical trend tests, interactive figures, and a compact markdown report.

## Run Phase 5

Run a development sample:

```bash
python -m src.advanced_analytics.run_phase5_analytics --config configs/project.yaml --max-records 10000
```

Run the full Phase 5 analytics workflow:

```bash
python -m src.advanced_analytics.run_phase5_analytics --config configs/project.yaml
```

Use a custom classified dataset:

```bash
python -m src.advanced_analytics.run_phase5_analytics --config configs/project.yaml --input data/processed/classification/classified_prompts.parquet
```

## Outputs

Phase 5 writes:

- `reports/phase5_behavior_trends.csv`
- `reports/interaction_transition_matrix.csv`
- `reports/emotion_transition_matrix.csv`
- `reports/phase5_statistical_tests.csv`
- `data/processed/advanced_analytics/archetype_assignments.parquet`
- `reports/archetype_summary.csv`
- `reports/behavior_network_edges.csv`
- `reports/behavior_network_nodes.csv`
- `data/processed/advanced_analytics/behavior_network.graphml`
- `reports/event_window_analysis.csv`
- `figures/phase5_behavior_trends.html`
- `figures/archetype_distribution.html`
- `figures/behavior_network.html`
- `reports/phase5_advanced_analytics_report.md`
- `reports/phase5_advanced_analytics_manifest.json`

## Analysis Modules

Temporal evolution:

- monthly behavioral rates,
- rolling-window behavioral trends,
- score deltas,
- linear trend statistics,
- interaction mode transitions,
- emotion transitions.

Archetype discovery:

- conversation-level behavioral aggregation,
- KMeans archetype assignment,
- interpretable archetype summaries.

Network analysis:

- behavioral signal co-occurrence graph,
- node centrality,
- weighted edges,
- GraphML export for external graph tools.

Event-window analysis:

- before/after behavioral metrics around configured events,
- per-event differences for dependency, companionship, outsourcing, and complexity measures.

Visualizations:

- behavioral trend lines,
- archetype distribution,
- strongest behavioral network edges.

## Verification Commands

Run tests:

```cmd
pytest
```

Run sample Phase 5:

```cmd
python -m src.advanced_analytics.run_phase5_analytics --config configs/project.yaml --max-records 10000
```

Check output folders:

```cmd
dir reports
dir figures
dir data\processed\advanced_analytics
```

Inspect core outputs:

```cmd
python -c "import pandas as pd; print(pd.read_csv('reports/phase5_behavior_trends.csv').head()); print(pd.read_csv('reports/phase5_statistical_tests.csv').head())"
```

```cmd
python -c "import pandas as pd; print(pd.read_csv('reports/interaction_transition_matrix.csv').head()); print(pd.read_csv('reports/emotion_transition_matrix.csv').head())"
```

```cmd
python -c "import pandas as pd; print(pd.read_csv('reports/archetype_summary.csv').head()); print(pd.read_csv('reports/behavior_network_nodes.csv').head())"
```

```cmd
python -c "import pandas as pd; print(pd.read_csv('reports/behavior_network_edges.csv').head()); print(pd.read_csv('reports/event_window_analysis.csv').head())"
```

Read the generated report:

```cmd
type reports\phase5_advanced_analytics_report.md
```

## Limitations

Phase 5 analytics are computational social-science indicators, not final causal claims. Trend statistics are descriptive, event windows do not prove causality, and archetype labels depend on the Phase 4 taxonomy and feature design. Human validation and sensitivity analysis should be added before making publication-level claims.
