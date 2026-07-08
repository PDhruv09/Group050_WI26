"""Streamlit dashboard for the Human-AI Behavior Observatory."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.charts import (
    archetype_distribution,
    behavior_trends_figure,
    category_bar,
    complexity_scatter,
    event_window_chart,
    network_edges_bar,
    transition_heatmap,
)
from src.dashboard.data import (
    artifact_status,
    compute_kpis,
    filter_classified_data,
    load_config,
    load_dashboard_tables,
    prompt_explorer_columns,
    top_counts,
    unique_sorted,
)


VIEW_OPTIONS = [
    "Observatory",
    "Temporal Signals",
    "Transitions",
    "Archetypes",
    "Network",
    "Prompt Explorer",
    "Diagnostics",
]


@dataclass(frozen=True)
class FilterState:
    """Current dashboard filter selections."""

    rows_loaded: int
    languages: list[str]
    months: list[str]
    interactions: list[str]
    emotions: list[str]
    outsourcing: list[str]
    keyword: str
    min_dependency: float
    min_complexity: float


def inject_styles() -> None:
    """Apply a distinct research-console visual identity."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1117;
            --panel: #101820;
            --panel-2: #14212b;
            --line: rgba(226, 235, 241, 0.14);
            --text: #e6edf3;
            --muted: #94a3ad;
            --teal: #2fb8ac;
            --gold: #e5b451;
            --coral: #ef6f6c;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(18, 35, 46, 0.96) 0%, rgba(11, 17, 23, 1) 42%),
                #0b1117;
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: #0d151c;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label {
            color: var(--text);
        }
        .block-container {
            padding-top: 1.35rem;
            max-width: 1500px;
        }
        div[data-testid="stSidebarContent"] {
            padding-top: 1.3rem;
        }
        .control-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 760;
            margin-bottom: 2px;
        }
        .control-caption {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 18px;
        }
        .observatory-hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(47, 184, 172, 0.14), rgba(229, 180, 81, 0.08) 46%, rgba(239, 111, 108, 0.08));
            padding: 22px 26px;
            margin-bottom: 14px;
            position: relative;
        }
        .observatory-kicker {
            color: var(--teal);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .observatory-title {
            color: var(--text);
            font-size: 2rem;
            font-weight: 750;
            line-height: 1.15;
            margin: 0;
        }
        .observatory-subtitle {
            color: var(--muted);
            max-width: 880px;
            margin-top: 10px;
            font-size: 0.98rem;
        }
        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }
        .hero-chip {
            border: 1px solid var(--line);
            border-radius: 6px;
            color: var(--text);
            background: rgba(16, 24, 32, 0.7);
            padding: 6px 10px;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 18px;
        }
        .status-cell {
            border: 1px solid var(--line);
            border-left: 3px solid var(--teal);
            border-radius: 8px;
            background: rgba(16, 24, 32, 0.78);
            padding: 12px 13px;
        }
        .status-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 720;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .status-value {
            color: var(--text);
            margin-top: 5px;
            font-size: 0.98rem;
            font-weight: 730;
            overflow-wrap: anywhere;
        }
        .section-label {
            color: var(--gold);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin: 12px 0 8px;
        }
        .explain-panel {
            border: 1px solid var(--line);
            border-left: 3px solid var(--gold);
            border-radius: 8px;
            background: rgba(20, 33, 43, 0.72);
            padding: 12px 14px;
            margin: 8px 0 14px;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .explain-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
            gap: 10px;
            margin: 8px 0 14px;
        }
        .explain-card {
            border: 1px solid var(--line);
            border-left: 3px solid var(--gold);
            border-radius: 8px;
            background: rgba(20, 33, 43, 0.72);
            padding: 12px 14px;
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .explain-card strong {
            color: var(--text);
        }
        .explain-panel strong {
            color: var(--text);
        }
        .view-title {
            color: var(--text);
            font-size: 1.3rem;
            font-weight: 760;
            margin: 8px 0 2px;
        }
        .view-caption {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 14px;
        }
        .metric-tile {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(16, 24, 32, 0.92);
            padding: 14px 14px 12px;
            min-height: 94px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: var(--text);
            font-size: 1.42rem;
            font-weight: 760;
            margin-top: 8px;
        }
        .metric-accent {
            height: 3px;
            width: 38px;
            background: var(--teal);
            border-radius: 99px;
            margin-top: 12px;
        }
        [data-testid="stMetric"] {
            background: rgba(16, 24, 32, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px 6px 0 0;
            color: var(--muted);
            padding: 10px 14px;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text);
            border-bottom: 2px solid var(--teal);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        .stPlotlyChart {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 10px 10px 18px;
        }
        div[role="radiogroup"] label {
            border: 1px solid var(--line);
            border-radius: 6px;
            background: rgba(16, 24, 32, 0.58);
            padding: 8px 10px;
            margin-bottom: 6px;
        }
        @media (max-width: 900px) {
            .status-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .explain-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the dashboard header."""
    st.markdown(
        """
        <div class="observatory-hero">
            <div class="observatory-kicker">Computational Social Science Console</div>
            <h1 class="observatory-title">Human-AI Behavior Observatory</h1>
            <div class="observatory-subtitle">
                Explore how public conversations with AI shift across interaction modes, emotional signals,
                cognitive outsourcing, dependency indicators, archetypes, and temporal events.
            </div>
            <div class="hero-meta">
                <span class="hero-chip">Interactive Research Dashboard</span>
                <span class="hero-chip">WildChat Behavioral Layer</span>
                <span class="hero-chip">Local Research Console</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    """Render a compact section label."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def explanation(text: str) -> None:
    """Render compact interpretation guidance."""
    st.markdown(f'<div class="explain-panel">{text}</div>', unsafe_allow_html=True)


def explanation_grid(left: str, right: str) -> None:
    """Render two-part dashboard explanation."""
    st.markdown(
        f"""
        <div class="explain-grid">
            <div class="explain-card">{left}</div>
            <div class="explain-card">{right}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def active_control_summary(state: FilterState) -> str:
    """Summarize active controls for chart/table explanations."""
    controls = [f"row sample: <strong>{state.rows_loaded:,}</strong>"]
    if state.keyword:
        controls.append(f"prompt search: <strong>{escape(state.keyword)}</strong>")
    if state.min_dependency > 0:
        controls.append(f"dependency >= <strong>{state.min_dependency:.2f}</strong>")
    if state.min_complexity > 0:
        controls.append(f"complexity >= <strong>{state.min_complexity:.2f}</strong>")
    if state.months:
        controls.append(f"months: <strong>{format_list(state.months)}</strong>")
    if state.languages:
        controls.append(f"languages: <strong>{format_list(state.languages)}</strong>")
    if state.interactions:
        controls.append(f"interactions: <strong>{format_list(state.interactions)}</strong>")
    if state.emotions:
        controls.append(f"emotions: <strong>{format_list(state.emotions)}</strong>")
    if state.outsourcing:
        controls.append(f"outsourcing: <strong>{format_list(state.outsourcing)}</strong>")
    return "; ".join(controls)


def component_explanation(title: str, what: str, control_effect: str, state: FilterState | None = None) -> None:
    """Render detailed explanation for one dashboard component."""
    active_controls = f"<br><strong>Active controls:</strong> {active_control_summary(state)}" if state else ""
    explanation(
        f"<strong>{title}</strong><br>"
        f"{what}<br>"
        f"<strong>How controls change it:</strong> {control_effect}"
        f"{active_controls}"
    )


def view_heading(title: str, caption: str) -> None:
    """Render a page-level heading."""
    st.markdown(f'<div class="view-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="view-caption">{caption}</div>', unsafe_allow_html=True)


def most_common(frame: pd.DataFrame, column: str) -> str:
    """Return the most common value for a status cell."""
    if frame.empty or column not in frame.columns:
        return "Unavailable"
    counts = frame[column].dropna().astype(str).value_counts()
    return "Unavailable" if counts.empty else str(counts.index[0])


def render_status_strip(filtered: pd.DataFrame) -> None:
    """Render contextual status cells for the active filter state."""
    if filtered.empty:
        values = [("Time Range", "No records"), ("Top Mode", "No records"), ("Top Emotion", "No records"), ("Top Outsourcing", "No records")]
    else:
        if "year_month" in filtered.columns and filtered["year_month"].notna().any():
            months = filtered["year_month"].dropna().astype(str)
            time_range = f"{months.min()} to {months.max()}"
        else:
            time_range = "No timestamp"
        values = [
            ("Time Range", time_range),
            ("Top Mode", most_common(filtered, "interaction_mode")),
            ("Top Emotion", most_common(filtered, "emotion_primary")),
            ("Top Outsourcing", most_common(filtered, "cognitive_outsourcing_type")),
        ]
    cells = "".join(
        (
            '<div class="status-cell">'
            f'<div class="status-label">{label}</div>'
            f'<div class="status-value">{value}</div>'
            "</div>"
        )
        for label, value in values
    )
    st.markdown(f'<div class="status-strip">{cells}</div>', unsafe_allow_html=True)


def format_list(values: list[str], empty_label: str = "all") -> str:
    """Format selected filter values for explanatory copy."""
    if not values:
        return empty_label
    if len(values) <= 3:
        return ", ".join(escape(str(value)) for value in values)
    return f"{', '.join(escape(str(value)) for value in values[:3])}, +{len(values) - 3} more"


def filter_narrative(state: FilterState, filtered: pd.DataFrame) -> str:
    """Describe how the active controls affect the current view."""
    clauses = [
        f"The dashboard is reading up to <strong>{state.rows_loaded:,}</strong> sampled prompt records for responsive exploration.",
        f"The active filters return <strong>{len(filtered):,}</strong> records.",
    ]
    if state.months:
        clauses.append(f"Time is limited to <strong>{format_list(state.months)}</strong>.")
    if state.languages:
        clauses.append(f"Language is limited to <strong>{format_list(state.languages)}</strong>.")
    if state.interactions:
        clauses.append(f"Interaction mode is limited to <strong>{format_list(state.interactions)}</strong>.")
    if state.emotions:
        clauses.append(f"Emotion is limited to <strong>{format_list(state.emotions)}</strong>.")
    if state.outsourcing:
        clauses.append(f"Cognitive outsourcing is limited to <strong>{format_list(state.outsourcing)}</strong>.")
    if state.keyword:
        clauses.append(f"Prompt text must contain <strong>{escape(state.keyword)}</strong>.")
    if state.min_dependency > 0:
        clauses.append(f"Dependency score must be at least <strong>{state.min_dependency:.2f}</strong>.")
    if state.min_complexity > 0:
        clauses.append(f"Prompt sophistication must be at least <strong>{state.min_complexity:.2f}</strong>.")
    if len(clauses) == 2:
        clauses.append("No categorical or text filters are active.")
    return " ".join(clauses)


def metric_definitions() -> str:
    """Return dashboard metric definitions."""
    return (
        "<strong>Metric guide:</strong> Companionship marks prompts with social or relational language. "
        "Dependency marks prompts with reliance, reassurance, or emotional support signals. "
        "Outsourcing marks prompts where users delegate cognitive work such as writing, coding, studying, or decisions. "
        "Complexity summarizes prompt length, specificity, role prompting, and recursive instructions."
    )


def empty_filter_state(rows_loaded: int) -> FilterState:
    """Return an empty filter state for diagnostic-only rendering."""
    return FilterState(
        rows_loaded=rows_loaded,
        languages=[],
        months=[],
        interactions=[],
        emotions=[],
        outsourcing=[],
        keyword="",
        min_dependency=0.0,
        min_complexity=0.0,
    )


@st.cache_data(show_spinner=False)
def cached_config() -> dict:
    """Load dashboard config with Streamlit caching."""
    return load_config(Path("configs/project.yaml"))


@st.cache_data(show_spinner=False)
def cached_tables(prompt_rows: int, artifact_fingerprint: tuple[tuple[str, float | None], ...]) -> dict[str, pd.DataFrame]:
    """Load dashboard tables with Streamlit caching."""
    return load_dashboard_tables(cached_config(), prompt_rows=prompt_rows)


def artifact_fingerprint(config: dict) -> tuple[tuple[str, float | None], ...]:
    """Return artifact mtimes so Streamlit reloads data when files are regenerated."""
    fingerprint = []
    for status in artifact_status(config):
        mtime = status.path.stat().st_mtime if status.path.exists() else None
        fingerprint.append((status.name, mtime))
    return tuple(fingerprint)


def metric_row(kpis: dict[str, float]) -> None:
    """Render top-level dashboard KPIs."""
    labels = [
        ("Records", f"{kpis['records']:,}"),
        ("Conversations", f"{kpis['conversations']:,}"),
        ("Companionship", f"{kpis['companionship_rate']:.2%}"),
        ("Dependency", f"{kpis['dependency_rate']:.2%}"),
        ("Outsourcing", f"{kpis['outsourcing_rate']:.2%}"),
        ("Complexity", f"{kpis['mean_complexity']:.3f}"),
    ]
    columns = st.columns(6)
    for column, (label, value) in zip(columns, labels, strict=False):
        column.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-accent"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sidebar_filters(classified: pd.DataFrame, rows_loaded: int) -> tuple[pd.DataFrame, FilterState]:
    """Render sidebar controls and return filtered data."""
    languages = st.sidebar.multiselect("Language", unique_sorted(classified, "language"))
    months = st.sidebar.multiselect("Month", unique_sorted(classified, "year_month"))
    interactions = st.sidebar.multiselect("Interaction", unique_sorted(classified, "interaction_mode"))
    emotions = st.sidebar.multiselect("Emotion", unique_sorted(classified, "emotion_primary"))
    outsourcing = st.sidebar.multiselect("Outsourcing", unique_sorted(classified, "cognitive_outsourcing_type"))
    keyword = st.sidebar.text_input("Prompt Search")
    min_dependency = st.sidebar.slider("Minimum Dependency", 0.0, 1.0, 0.0, 0.05)
    min_complexity = st.sidebar.slider("Minimum Complexity", 0.0, 1.0, 0.0, 0.05)
    state = FilterState(
        rows_loaded=rows_loaded,
        languages=languages,
        months=months,
        interactions=interactions,
        emotions=emotions,
        outsourcing=outsourcing,
        keyword=keyword,
        min_dependency=min_dependency,
        min_complexity=min_complexity,
    )
    filtered = filter_classified_data(
        classified,
        languages=languages,
        interaction_modes=interactions,
        emotions=emotions,
        outsourcing_types=outsourcing,
        year_months=months,
        keyword=keyword,
        min_dependency=min_dependency,
        min_complexity=min_complexity,
    )
    return filtered, state


def render_overview(filtered: pd.DataFrame, tables: dict[str, pd.DataFrame], state: FilterState) -> None:
    """Render overview tab."""
    view_heading("Observatory", "A compact readout of the active behavioral slice.")
    explanation_grid(
        (
            "<strong>What this view shows:</strong> the current filtered slice of prompts. "
            "The top row summarizes volume and behavioral rates; the charts show which interaction modes, "
            "outsourcing categories, and emotional signals dominate."
        ),
        f"{filter_narrative(state, filtered)} {metric_definitions()}",
    )
    section_label("Filtered Dataset")
    metric_row(compute_kpis(filtered))
    component_explanation(
        "KPI tiles",
        "These tiles summarize the currently visible prompt slice: record count, unique conversations, companionship rate, dependency rate, cognitive outsourcing rate, and mean prompt sophistication.",
        "Rows Loaded changes the sampled pool. Search and category filters narrow the records. Minimum Dependency and Minimum Complexity remove records below those thresholds, so rates can rise or fall depending on which records remain.",
        state,
    )
    render_status_strip(filtered)
    component_explanation(
        "Status strip",
        "This strip describes the filtered slice by time range and the most common interaction, emotion, and outsourcing labels.",
        "Any filter can change the most common labels. Rows Loaded can change these values because the dashboard uses a deterministic sample for speed.",
        state,
    )
    section_label("Behavioral Distributions")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(category_bar(filtered, "interaction_mode", "Interaction Modes"), width="stretch")
        component_explanation(
            "Interaction Modes chart",
            "Counts prompts by how the user appears to treat the AI: tool, assistant, collaborator, companion, or therapist-surrogate.",
            "Search, month, language, emotion, and outsourcing filters all narrow the counted prompts. Minimum Dependency often increases companion or therapist-surrogate shares. Minimum Complexity can shift the chart toward longer or more structured prompts.",
            state,
        )
    with right:
        st.plotly_chart(category_bar(filtered, "cognitive_outsourcing_type", "Cognitive Outsourcing"), width="stretch")
        component_explanation(
            "Cognitive Outsourcing chart",
            "Counts the mental task being delegated to AI, such as coding, writing, studying, decisions, life planning, or emotional regulation.",
            "Prompt Search can strongly reshape this chart because task words like code, essay, or advice map to specific categories. Dependency and complexity thresholds can reveal whether higher-scoring prompts cluster around certain outsourced tasks.",
            state,
        )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(category_bar(filtered, "emotion_primary", "Emotional Signals"), width="stretch")
        component_explanation(
            "Emotional Signals chart",
            "Counts the primary emotional signal detected in each prompt, including none when no strong emotion signal is detected.",
            "Minimum Dependency usually reduces neutral prompts and emphasizes dependency, loneliness, vulnerability, or fear. Prompt Search can isolate specific emotional language.",
            state,
        )
    with right:
        st.plotly_chart(complexity_scatter(filtered), width="stretch")
        component_explanation(
            "Prompt Sophistication vs Dependency scatter",
            "Each point is a prompt. The x-axis is prompt sophistication; the y-axis is dependency score; color shows interaction mode.",
            "Minimum Dependency removes lower points. Minimum Complexity removes left-side points. Rows Loaded changes how many points are sampled, while Prompt Search can reveal whether particular terms sit in high-dependency or high-complexity regions.",
            state,
        )
    coverage = tables.get("taxonomy_coverage", pd.DataFrame())
    if not coverage.empty:
        section_label("Taxonomy Coverage")
        st.dataframe(coverage, width="stretch", hide_index=True)
        component_explanation(
            "Taxonomy Coverage table",
            "This table reports how often each classification dimension found nonzero evidence in the classified dataset.",
            "This table comes from the generated classification artifact, so sidebar prompt filters do not change it. To change it, rerun classification with a different taxonomy, threshold, or source dataset.",
        )


def render_trends(tables: dict[str, pd.DataFrame], state: FilterState, filtered: pd.DataFrame) -> None:
    """Render trends tab."""
    view_heading("Temporal Signals", "Track how behavioral rates and event-window changes move across time.")
    explanation_grid(
        (
            "<strong>How to read this:</strong> trend lines are monthly behavioral rates from the advanced analytics output. "
            "Event windows compare average values after an event against average values before it; positive bars indicate increases."
        ),
        f"{filter_narrative(state, filtered)} These trend charts use the prepared monthly analytics tables, while the status cards reflect your active prompt sample.",
    )
    section_label("Temporal Signal Movement")
    st.plotly_chart(behavior_trends_figure(tables.get("behavior_trends", pd.DataFrame())), width="stretch")
    component_explanation(
        "Behavioral Signal Trends chart",
        "This chart shows monthly behavioral rates from the analytics artifact: companionship, vulnerability, dependency, cognitive outsourcing, and self-disclosure.",
        "Rows Loaded and prompt-level filters do not alter this chart directly because it is based on the full prepared monthly report. To change it, rerun the analytics pipeline after changing the classified dataset or analysis settings.",
        state,
    )
    events = tables.get("event_windows", pd.DataFrame())
    section_label("Event Windows")
    st.plotly_chart(event_window_chart(events), width="stretch")
    component_explanation(
        "Event Window Differences chart",
        "This chart compares each metric after an event window against before it. A positive bar means the after-window average is higher; a negative bar means it is lower.",
        "Sidebar filters do not change this chart directly because it uses the prepared event-window report. Event definitions, before/after window size, and regenerated analytics outputs determine what appears here.",
        state,
    )
    if not events.empty:
        st.dataframe(events, width="stretch", hide_index=True)
        component_explanation(
            "Event Window table",
            "This table shows the exact before value, after value, and difference behind the event-window chart.",
            "It changes when the analytics report is regenerated with different event dates, window lengths, or source data.",
            state,
        )


def render_transitions(tables: dict[str, pd.DataFrame], state: FilterState, filtered: pd.DataFrame) -> None:
    """Render transition matrix tab."""
    view_heading("Transitions", "Inspect how conversations move between interaction and emotion states.")
    explanation_grid(
        (
            "<strong>How to read this:</strong> each heatmap cell is the probability of moving from one state to the next adjacent turn in the same conversation. "
            "Higher values indicate more common behavioral paths."
        ),
        f"{filter_narrative(state, filtered)} Transition matrices are built from conversation order, so they explain movement between turns rather than simple category frequency.",
    )
    section_label("State Transitions")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            transition_heatmap(tables.get("interaction_transition", pd.DataFrame()), "Interaction Transitions"),
            width="stretch",
        )
        component_explanation(
            "Interaction Transitions heatmap",
            "Each cell estimates how often one interaction mode is followed by another in the next turn of the same conversation.",
            "Sidebar filters do not change this prepared matrix directly. It changes when analytics are regenerated from a different classified dataset or transition definition.",
            state,
        )
    with right:
        st.plotly_chart(
            transition_heatmap(tables.get("emotion_transition", pd.DataFrame()), "Emotion Transitions"),
            width="stretch",
        )
        component_explanation(
            "Emotion Transitions heatmap",
            "Each cell estimates how often one primary emotional signal is followed by another in the next turn of the same conversation.",
            "It is generated from ordered conversation turns, so rerunning analytics after changing classification labels or source records changes the matrix.",
            state,
        )
    st.dataframe(tables.get("interaction_transition", pd.DataFrame()), width="stretch", hide_index=True)
    component_explanation(
        "Interaction Transition table",
        "This table provides the transition counts and probabilities used in the heatmap.",
        "It is useful when you need exact transition values rather than visual intensity.",
        state,
    )


def render_archetypes(tables: dict[str, pd.DataFrame], state: FilterState, filtered: pd.DataFrame) -> None:
    """Render archetype tab."""
    archetypes = tables.get("archetype_summary", pd.DataFrame())
    view_heading("Archetypes", "Conversation-level behavioral clusters and their defining traits.")
    explanation_grid(
        (
            "<strong>What this means:</strong> archetypes are clusters of conversations based on average behavioral signals, complexity, and prompt patterns. "
            "They are exploratory groups, not fixed user identities."
        ),
        f"{filter_narrative(state, filtered)} Archetype summaries are conversation-level aggregates, so they may not change with prompt-level filters until the analytics artifacts are regenerated.",
    )
    section_label("Conversation Archetypes")
    st.plotly_chart(archetype_distribution(archetypes), width="stretch")
    component_explanation(
        "Archetype Distribution chart",
        "This chart shows conversation clusters discovered from behavioral rates, dependency, complexity, and related prompt-level features.",
        "Sidebar filters do not re-cluster conversations live. To change archetypes, rerun analytics with different features, cluster count, source data, or filtering before analytics.",
        state,
    )
    st.dataframe(archetypes, width="stretch", hide_index=True)
    component_explanation(
        "Archetype Summary table",
        "This table lists each cluster with average behavioral traits and an interpretable archetype name.",
        "Use it to compare what makes clusters different. It changes when the archetype discovery pipeline is rerun.",
        state,
    )


def render_network(tables: dict[str, pd.DataFrame], state: FilterState, filtered: pd.DataFrame) -> None:
    """Render behavior network tab."""
    edges = tables.get("network_edges", pd.DataFrame())
    nodes = tables.get("network_nodes", pd.DataFrame())
    view_heading("Network", "Co-occurrence structure among behavioral signals.")
    explanation_grid(
        (
            "<strong>How to read this:</strong> network edges count how often two behavioral signals appear together in the same prompt. "
            "High-weight edges point to signals that frequently co-occur."
        ),
        f"{filter_narrative(state, filtered)} The network is useful for seeing which behaviors travel together, such as tool use with coding or vulnerability with companionship.",
    )
    section_label("Behavioral Co-Occurrence Network")
    st.plotly_chart(network_edges_bar(edges), width="stretch")
    component_explanation(
        "Behavior Network chart",
        "This chart shows the strongest co-occurrences between behavioral signals, such as an interaction mode appearing with an outsourcing type.",
        "It uses a prepared network artifact, so dashboard filters do not recompute it live. Regenerate analytics to rebuild the network from different records.",
        state,
    )
    left, right = st.columns(2)
    with left:
        st.dataframe(
            nodes.sort_values("degree_centrality", ascending=False) if not nodes.empty else nodes,
            width="stretch",
            hide_index=True,
        )
        component_explanation(
            "Network Nodes table",
            "This table shows each behavioral signal as a graph node, including frequency and centrality.",
            "Centrality reflects the prepared graph structure. Filtering prompt rows in the sidebar does not alter these graph metrics until analytics are regenerated.",
            state,
        )
    with right:
        st.dataframe(
            edges.sort_values("weight", ascending=False) if not edges.empty else edges,
            width="stretch",
            hide_index=True,
        )
        component_explanation(
            "Network Edges table",
            "This table shows which signal pairs co-occur most often and the weight of each connection.",
            "It is generated from the analytics artifact, so sidebar controls explain the active sample but do not recompute the graph live.",
            state,
        )


def render_prompts(filtered: pd.DataFrame, state: FilterState) -> None:
    """Render prompt explorer tab."""
    view_heading("Prompt Explorer", "Inspect prompt-level records under the active filter state.")
    explanation_grid(
        (
            "<strong>What this table contains:</strong> prompt-level examples from the active filters with behavioral labels and scores. "
            "Use it to sanity-check categories behind the aggregate charts."
        ),
        f"{filter_narrative(state, filtered)} The table shows the first 1,000 matching prompts from the active sampled data to keep browsing responsive.",
    )
    section_label("Prompt Explorer")
    columns = prompt_explorer_columns(filtered)
    st.dataframe(filtered[columns].head(1000) if columns else filtered.head(1000), width="stretch", hide_index=True)
    component_explanation(
        "Prompt Explorer table",
        "This table shows prompt-level records after all active sidebar controls are applied.",
        "Rows Loaded changes the sampled pool. Search filters prompt text. Minimum Dependency and Minimum Complexity remove records below those scores. The table displays up to 1,000 matching rows for responsiveness.",
        state,
    )
    st.dataframe(top_counts(filtered, "language"), width="stretch", hide_index=True)
    component_explanation(
        "Language Counts table",
        "This small table summarizes languages within the currently filtered prompt sample.",
        "Language filters can reduce it to selected languages; prompt search and score thresholds can also change the language mix of matching prompts.",
        state,
    )


def render_diagnostics(config: dict, tables: dict[str, pd.DataFrame], state: FilterState, filtered: pd.DataFrame) -> None:
    """Render diagnostics tab."""
    view_heading("Diagnostics", "Check dashboard artifact readiness and pipeline benchmarks.")
    explanation_grid(
        (
            "<strong>Why this matters:</strong> these artifacts are the files powering the dashboard. "
            "Missing files usually mean the classification, analytics, or dashboard manifest command needs to be rerun."
        ),
        f"{filter_narrative(state, filtered)} Diagnostics are not research findings; they are a readiness check for local files and generated outputs.",
    )
    section_label("Artifact Readiness")
    statuses = pd.DataFrame(
        [
            {"name": status.name, "path": str(status.path), "exists": status.exists, "rows": status.rows}
            for status in artifact_status(config)
        ]
    )
    st.dataframe(statuses, width="stretch", hide_index=True)
    component_explanation(
        "Artifact Readiness table",
        "This table checks whether the dashboard input files exist and, when possible, how many rows they contain.",
        "Sidebar filters do not change artifact readiness. Missing files usually mean a pipeline command needs to be rerun.",
        state,
    )
    benchmark = tables.get("classification_benchmark", pd.DataFrame())
    if not benchmark.empty:
        st.dataframe(benchmark, width="stretch", hide_index=True)
        component_explanation(
            "Classification Benchmark table",
            "This table records runtime and throughput from the classification pipeline.",
            "It changes only when classification is rerun, not when dashboard filters change.",
            state,
        )


def main() -> None:
    """Run the Streamlit dashboard."""
    config = cached_config()
    dash = config.get("dashboard", {})
    st.set_page_config(
        page_title=dash.get("page_title", "Human-AI Behavior Observatory"),
        page_icon=dash.get("page_icon", "HA"),
        layout="wide",
    )
    inject_styles()
    render_header()
    default_rows = int(dash.get("default_sample_rows", 10000))
    st.sidebar.markdown(
        """
        <div class="control-title">Observatory Controls</div>
        <div class="control-caption">Choose a view, then narrow the local artifact slice.</div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Refresh Artifacts", width="stretch"):
        st.cache_data.clear()
    active_view = st.sidebar.radio("View", VIEW_OPTIONS, label_visibility="collapsed")
    st.sidebar.markdown("---")
    prompt_rows = int(
        st.sidebar.number_input("Rows Loaded", min_value=1000, max_value=1000000, value=default_rows, step=1000)
    )
    tables = cached_tables(prompt_rows, artifact_fingerprint(config))
    classified = tables.get("classified", pd.DataFrame())

    if classified.empty:
        st.error("Dashboard data is not available.")
        render_diagnostics(config, tables, empty_filter_state(prompt_rows), classified)
        return

    filtered, filter_state = sidebar_filters(classified, prompt_rows)

    if active_view == "Observatory":
        render_overview(filtered, tables, filter_state)
    elif active_view == "Temporal Signals":
        render_trends(tables, filter_state, filtered)
    elif active_view == "Transitions":
        render_transitions(tables, filter_state, filtered)
    elif active_view == "Archetypes":
        render_archetypes(tables, filter_state, filtered)
    elif active_view == "Network":
        render_network(tables, filter_state, filtered)
    elif active_view == "Prompt Explorer":
        render_prompts(filtered, filter_state)
    else:
        render_diagnostics(config, tables, filter_state, filtered)


if __name__ == "__main__":
    main()
