"""Tree visualization page — interactive deliberation tree."""

from __future__ import annotations

from typing import Any

import streamlit as st

from delibera.web.components.node_card import render_node_card
from delibera.web.components.tree_graph import build_agraph_data, build_graphviz_dot


def render(
    replayed: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    """Render the tree visualization page.

    Args:
        replayed: ReplayedRun as dict.
        events: Trace events.
    """
    st.title("🌳 Deliberation Tree")

    nodes = replayed.get("nodes", {})
    if not nodes:
        st.info("No tree data available for this run.")
        return

    # Legend
    col1, col2, col3, col4 = st.columns(4)
    col1.color_picker("Root", "#607D8B", disabled=True, key="legend_root")
    col1.caption("Root")
    col2.color_picker("Survivor", "#4CAF50", disabled=True, key="legend_survivor")
    col2.caption("Survivor")
    col3.color_picker("Pruned", "#F44336", disabled=True, key="legend_pruned")
    col3.caption("Pruned")
    col4.color_picker("Merged", "#FF9800", disabled=True, key="legend_merged")
    col4.caption("Merged")

    st.divider()

    # Visualization mode selection
    viz_mode = st.radio(
        "Visualization",
        ["Graphviz (static)", "Force Graph (interactive)"],
        horizontal=True,
        key="viz_mode",
    )

    if viz_mode == "Force Graph (interactive)":
        try:
            _render_agraph(replayed)
        except Exception as e:
            st.warning(f"Interactive graph unavailable: {e}")
            _render_graphviz(replayed)
    else:
        _render_graphviz(replayed)

    st.divider()

    # Node detail section
    st.subheader("Node Details")

    node_list = []
    for nid, node in nodes.items():
        label = node.get("label", node.get("kind", "?"))
        status = node.get("status", "active")
        kind = node.get("kind", "option")
        node_list.append((f"[{kind}] {label} ({status})", nid))

    if node_list:
        selected_label = st.selectbox(
            "Select a node to inspect",
            [item[0] for item in node_list],
        )
        if selected_label:
            selected_nid = next(nid for display, nid in node_list if display == selected_label)
            node_data = nodes[selected_nid]
            enriched = _enrich_node_with_scores(node_data, events)
            render_node_card(enriched)


def _render_agraph(replayed: dict[str, Any]) -> None:
    """Render tree using streamlit-agraph."""
    from streamlit_agraph import Config, Edge, Node, agraph

    nodes_data, edges_data = build_agraph_data(replayed)

    ag_nodes = [
        Node(
            id=n["id"],
            label=n["label"],
            size=n["size"],
            color=n["color"],
            title=n.get("title", ""),
        )
        for n in nodes_data
    ]
    ag_edges = [Edge(source=e["source"], target=e["target"]) for e in edges_data]

    config = Config(
        width=800,
        height=500,
        directed=True,
        physics=True,
        hierarchical=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
    )

    agraph(nodes=ag_nodes, edges=ag_edges, config=config)


def _render_graphviz(replayed: dict[str, Any]) -> None:
    """Render tree using Graphviz."""
    dot = build_graphviz_dot(replayed)
    st.graphviz_chart(dot, width="stretch")


def _enrich_node_with_scores(
    node: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add score data from trace events to node dict."""
    enriched = dict(node)
    node_id = node.get("node_id", "")

    for event in events:
        if event.get("event_type") == "score_computed":
            payload = event.get("payload", {})
            if payload.get("node_id") == node_id:
                enriched["score"] = payload.get("score")
                enriched["score_breakdown"] = payload.get("breakdown", {})
                break

    # Get evidence count
    evidence_count = 0
    for event in events:
        if (
            event.get("event_type") == "evidence_added"
            and event.get("payload", {}).get("node_id") == node_id
        ):
            evidence_count += 1
    enriched["evidence_count"] = evidence_count

    # Get objections
    blocking = 0
    nonblocking = 0
    for event in events:
        if event.get("event_type") == "objection_added":
            payload = event.get("payload", {})
            if payload.get("node_id") == node_id:
                if payload.get("blocking"):
                    blocking += 1
                else:
                    nonblocking += 1
    enriched["objections"] = {
        "blocking_open": blocking,
        "nonblocking_open": nonblocking,
        "blocking_accepted": 0,
        "nonblocking_accepted": 0,
    }

    return enriched
