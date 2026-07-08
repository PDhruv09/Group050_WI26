# Interactive Dashboard

The interactive dashboard explores the Human-AI Behavior Observatory outputs. It consumes the classified prompt dataset and advanced analytics artifacts.

The dashboard uses a research-console layout rather than the default Streamlit tab pattern:

- sidebar view navigation,
- custom observatory header,
- compact KPI tiles,
- active-filter status strip,
- dark analytical chart panels,
- dashboard-specific chart palette.

## Run Dashboard

Start the Streamlit app:

```bash
streamlit run dashboard/app.py
```

Generate a dashboard readiness manifest:

```bash
python -m src.dashboard.run_dashboard_manifest --config configs/project.yaml
```

## Dashboard Views

- Overview KPIs
- active filter status strip
- interaction mode distribution
- cognitive outsourcing distribution
- emotional signal distribution
- complexity and dependency scatter
- behavioral trend lines
- event-window differences
- interaction transition heatmap
- emotion transition heatmap
- archetype distribution and table
- behavioral network node and edge tables
- prompt explorer
- artifact diagnostics

## Inputs

Configured in `configs/project.yaml`:

- `data/processed/classification/classified_prompts.parquet`
- `data/processed/dashboard/classified_prompt_dashboard_sample.parquet`
- `reports/phase5_behavior_trends.csv`
- `reports/interaction_transition_matrix.csv`
- `reports/emotion_transition_matrix.csv`
- `reports/archetype_summary.csv`
- `data/processed/advanced_analytics/archetype_assignments.parquet`
- `reports/behavior_network_nodes.csv`
- `reports/behavior_network_edges.csv`
- `reports/event_window_analysis.csv`
- `reports/taxonomy_coverage.csv`
- `reports/classification_summary.csv`
- `reports/classification_benchmark.csv`
- `reports/phase5_advanced_analytics_manifest.json`

## Verification

Run tests:

```cmd
pytest
```

Write the manifest:

```cmd
python -m src.dashboard.run_dashboard_manifest --config configs/project.yaml
```

This command also creates or refreshes the dashboard sample file used for responsive prompt exploration.

Inspect dashboard readiness:

```cmd
python -c "import json; d=json.load(open('reports/dashboard_manifest.json')); print(d['phase']); print(d['available_artifacts']); print(d['missing_artifacts'])"
```

Launch the dashboard:

```cmd
streamlit run dashboard/app.py
```

## Notes

The dashboard is a local research interface. Large prompt datasets are loaded with a configurable row limit from the sidebar to keep interaction responsive. The manifest command creates a deterministic random dashboard sample, and the app uses that sample for normal exploration instead of repeatedly loading the full classified prompt file. This prevents the default view from being biased toward the earliest timestamp and reduces reload lag.
