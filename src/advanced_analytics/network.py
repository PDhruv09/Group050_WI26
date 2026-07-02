"""Behavioral network analysis for Phase 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SIGNAL_COLUMNS = [
    "interaction_mode",
    "emotion_primary",
    "cognitive_outsourcing_type",
    "is_companionship",
    "is_vulnerable",
    "is_dependency_signal",
    "is_cognitive_outsourcing",
    "is_reassurance_seeking",
    "is_anthropomorphic",
    "is_self_disclosure",
]


def row_signals(row: pd.Series) -> list[str]:
    """Extract graph node labels from one classified prompt."""
    signals = []
    for column in SIGNAL_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        if isinstance(value, bool):
            if value:
                signals.append(column)
        elif pd.notna(value) and str(value) not in {"none", "<NA>"}:
            signals.append(f"{column}:{value}")
    return sorted(set(signals))


def build_behavior_network(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Build a co-occurrence network among behavioral signals."""
    try:
        import networkx as nx
    except ImportError as error:
        raise ImportError("networkx is required for Phase 5 network analysis.") from error

    edge_counts: dict[tuple[str, str], int] = {}
    node_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        signals = row_signals(row)
        for signal in signals:
            node_counts[signal] = node_counts.get(signal, 0) + 1
        for index, left in enumerate(signals):
            for right in signals[index + 1 :]:
                key = tuple(sorted((left, right)))
                edge_counts[key] = edge_counts.get(key, 0) + 1

    graph = nx.Graph()
    for node, count in node_counts.items():
        graph.add_node(node, count=int(count))
    for (source, target), weight in edge_counts.items():
        graph.add_edge(source, target, weight=int(weight))

    centrality = nx.degree_centrality(graph) if graph.number_of_nodes() else {}
    node_rows = [
        {"node": node, "count": int(data["count"]), "degree_centrality": float(centrality.get(node, 0.0))}
        for node, data in graph.nodes(data=True)
    ]
    edge_rows = [
        {"source": source, "target": target, "weight": int(data["weight"])}
        for source, target, data in graph.edges(data=True)
    ]
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), graph


def write_network_outputs(frame: pd.DataFrame, config: dict) -> dict[str, int]:
    """Write behavior network node, edge, and graph artifacts."""
    try:
        import networkx as nx
    except ImportError as error:
        raise ImportError("networkx is required for Phase 5 network analysis.") from error

    network_config = config["advanced_analytics"]["network"]
    nodes, edges, graph = build_behavior_network(frame)

    node_file = Path(network_config["node_file"])
    node_file.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(node_file, index=False)

    edge_file = Path(network_config["edge_file"])
    edge_file.parent.mkdir(parents=True, exist_ok=True)
    edges.to_csv(edge_file, index=False)

    graph_file = Path(network_config["graph_file"])
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, graph_file)
    return {"network_nodes": int(len(nodes)), "network_edges": int(len(edges))}

