# Interactive Dashboard

Run the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

Generate the dashboard readiness manifest:

```bash
python -m src.dashboard.run_dashboard_manifest --config configs/project.yaml
```

The dashboard expects classification outputs and advanced analytics outputs to exist locally.

The UI uses a custom research-console layout with sidebar navigation, compact KPI tiles, dark chart panels, and artifact diagnostics.

The prompt dataset preview uses deterministic random sampling when a row limit is active, so the default dashboard view is not limited to the earliest month.

Run the manifest command after regenerating Phase 4 outputs so the fast dashboard sample is refreshed.

Placeholder for the future Streamlit exploration platform.

