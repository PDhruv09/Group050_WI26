from pathlib import Path

import pandas as pd
import yaml

from src.dashboard.charts import behavior_trends_figure, category_bar, event_window_chart, transition_heatmap
from src.dashboard.data import compute_kpis, filter_classified_data, prompt_explorer_columns, read_table, top_counts
from src.dashboard.manifest import build_dashboard_manifest, write_dashboard_sample


def sample_dashboard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "conversation_id": ["c1", "c1", "c2"],
            "prompt_text": ["write code", "I feel lonely", "help me decide"],
            "language": ["en", "en", "es"],
            "year_month": ["2023-04", "2023-04", "2023-05"],
            "interaction_mode": ["tool_mode", "companion_mode", "assistant_mode"],
            "emotion_primary": ["none", "loneliness", "confusion"],
            "cognitive_outsourcing_type": ["coding", "none", "decision_making"],
            "is_companionship": [False, True, False],
            "is_dependency_signal": [False, True, False],
            "is_cognitive_outsourcing": [True, False, True],
            "dependency_score": [0.0, 0.8, 0.1],
            "prompt_sophistication_score": [0.2, 0.5, 0.7],
        }
    )


def test_dashboard_filter_and_kpis() -> None:
    frame = sample_dashboard_frame()

    filtered = filter_classified_data(frame, languages=["en"], interaction_modes=["companion_mode"], min_dependency=0.5)
    kpis = compute_kpis(filtered)

    assert len(filtered) == 1
    assert kpis["records"] == 1
    assert kpis["companionship_rate"] == 1.0
    assert kpis["dependency_rate"] == 1.0


def test_dashboard_top_counts_and_prompt_columns() -> None:
    frame = sample_dashboard_frame()

    counts = top_counts(frame, "interaction_mode")
    columns = prompt_explorer_columns(frame)

    assert counts.loc[0, "count"] == 1
    assert "prompt_text" in columns
    assert "record_id" in columns


def test_dashboard_charts_return_figures() -> None:
    frame = sample_dashboard_frame()
    trends = pd.DataFrame(
        {
            "year_month": ["2023-04", "2023-05"],
            "is_companionship_rate": [0.5, 0.1],
            "is_dependency_signal_rate": [0.5, 0.0],
        }
    )
    transitions = pd.DataFrame(
        {
            "from_state": ["tool_mode", "companion_mode"],
            "to_state": ["assistant_mode", "assistant_mode"],
            "probability": [0.4, 0.6],
        }
    )
    events = pd.DataFrame(
        {
            "event": ["gpt4_release", "chatgpt_ios_release"],
            "event_date": ["2023-03-14", "2023-05-18"],
            "metric": ["dependency_score", "dependency_score"],
            "before_value": [0.1, 0.2],
            "after_value": [0.2, 0.1],
            "difference": [0.1, -0.1],
        }
    )

    trend_figure = behavior_trends_figure(trends)
    event_figure = event_window_chart(events)
    assert trend_figure.data
    assert trend_figure.layout.xaxis.type == "category"
    assert category_bar(frame, "interaction_mode", "Interaction").data
    assert transition_heatmap(transitions, "Transitions").data
    assert event_figure.layout.legend.orientation == "v"
    assert event_figure.layout.legend.x > 1


def test_dashboard_config_and_manifest_shape() -> None:
    with Path("configs/project.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    manifest = build_dashboard_manifest(Path("configs/project.yaml"))

    assert config["project"]["phase"] == 6
    assert config["dashboard"]["app_file"] == "dashboard/app.py"
    assert manifest["phase"] == 6
    assert "artifacts" in manifest


def test_dashboard_limited_read_uses_deterministic_sample() -> None:
    scratch = Path("data/processed/_test_dashboard")
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "sample.parquet"
    pd.DataFrame({"value": range(20)}).to_parquet(path, index=False)

    first = read_table(path, max_rows=5, random_state=42)
    second = read_table(path, max_rows=5, random_state=42)

    assert first["value"].tolist() == second["value"].tolist()
    assert first["value"].tolist() != [0, 1, 2, 3, 4]


def test_dashboard_sample_writer() -> None:
    scratch = Path("data/processed/_test_dashboard_sample")
    scratch.mkdir(parents=True, exist_ok=True)
    source = scratch / "classified.parquet"
    sample = scratch / "sample.parquet"
    pd.DataFrame(
        {
            "record_id": [str(index) for index in range(20)],
            "conversation_id": ["c"] * 20,
            "prompt_text": ["text"] * 20,
            "year_month": ["2023-04"] * 10 + ["2023-05"] * 10,
            "language": ["en"] * 20,
            "interaction_mode": ["assistant_mode"] * 20,
            "emotion_primary": ["none"] * 20,
            "cognitive_outsourcing_type": ["none"] * 20,
            "is_companionship": [False] * 20,
            "is_dependency_signal": [False] * 20,
            "is_cognitive_outsourcing": [False] * 20,
            "dependency_score": [0.0] * 20,
            "prompt_sophistication_score": [0.1] * 20,
        }
    ).to_parquet(source, index=False)
    config = {
        "dashboard": {
            "classified_data_file": str(source),
            "sample_file": str(sample),
            "sample_rows": 5,
            "sample_random_state": 42,
        }
    }

    output = write_dashboard_sample(config)
    written = pd.read_parquet(output)

    assert output == sample
    assert len(written) == 5
