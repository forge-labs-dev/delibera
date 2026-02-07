"""Scores page — score comparison charts and breakdown table."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    _artifact: dict[str, Any],
) -> None:
    """Render the scores page."""
    st.title("📊 Score Analysis")

    ranking = summary.get("decision_explanation", {}).get("ranking", [])

    # If no ranking in summary, build from score_computed events
    if not ranking:
        ranking = _build_ranking_from_events(events)

    if not ranking:
        st.info("No scoring data available for this run.")
        return

    # Bar chart
    st.subheader("Option Scores")
    _render_bar_chart(ranking)

    st.divider()

    # Radar chart (only if metrics available)
    has_metrics = any(item.get("metrics") for item in ranking)
    if has_metrics:
        st.subheader("Metric Profiles")
        _render_radar_chart(ranking)
        st.divider()

    # Score breakdown table
    st.subheader("Score Breakdown")
    _render_breakdown_table(ranking)

    # Weights
    weights = summary.get("decision_explanation", {}).get("weights_used", {})
    if weights:
        st.divider()
        st.subheader("Score Weights")
        cols = st.columns(min(len(weights), 6))
        for i, (metric, weight) in enumerate(sorted(weights.items())):
            cols[i % len(cols)].metric(metric, f"{weight:+.1f}")


def _build_ranking_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build ranking data from score_computed trace events."""
    ranking: list[dict[str, Any]] = []

    # Get node labels from node_created events
    node_labels: dict[str, str] = {}
    node_statuses: dict[str, str] = {}
    for event in events:
        if event.get("event_type") == "node_created":
            payload = event.get("payload", {})
            nid = payload.get("node_id", "")
            label = payload.get("label", "")
            kind = payload.get("kind", "")
            if nid and kind == "option":
                node_labels[nid] = label

    # Get pruned nodes
    for event in events:
        if event.get("event_type") == "prune":
            payload = event.get("payload", {})
            for nid in payload.get("pruned_ids", []):
                node_statuses[nid] = "pruned"
            for nid in payload.get("survivor_ids", []):
                node_statuses[nid] = "survivor"

    # Get scores
    for event in events:
        if event.get("event_type") == "score_computed":
            payload = event.get("payload", {})
            node_id = payload.get("node_id", "")
            if node_id in node_labels:
                ranking.append(
                    {
                        "label": node_labels[node_id],
                        "score": payload.get("score", 0),
                        "status": node_statuses.get(node_id, "active"),
                        "metrics": payload.get("breakdown", {}),
                    }
                )

    return sorted(ranking, key=lambda r: r["score"], reverse=True)


def _render_bar_chart(ranking: list[dict[str, Any]]) -> None:
    """Render horizontal bar chart of scores."""
    labels = []
    score_values = []
    colors = []

    for item in ranking:
        label = item.get("label", "?")
        labels.append(label[:40] + "..." if len(label) > 40 else label)
        score_values.append(item.get("score", 0))
        colors.append("#4CAF50" if item.get("status") == "survivor" else "#F44336")

    fig = go.Figure(
        go.Bar(
            x=score_values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.2f}" for s in score_values],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=max(200, len(labels) * 80),
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Weighted Score",
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")


def _render_radar_chart(ranking: list[dict[str, Any]]) -> None:
    """Render radar/spider chart comparing metric profiles."""
    metric_keys = [
        "evidence_coverage",
        "evidence_count",
        "weak_claims",
        "unsupported_claims",
        "blocking_objections",
        "nonblocking_objections",
    ]
    metric_labels = [
        "Evidence Coverage",
        "Evidence Count",
        "Weak Claims",
        "Unsupported Claims",
        "Blocking Objections",
        "Nonblocking Objections",
    ]

    fig = go.Figure()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    for i, item in enumerate(ranking):
        metrics = item.get("metrics", {})
        if not metrics:
            continue
        values = [abs(metrics.get(k, 0)) for k in metric_keys]
        values.append(values[0])
        theta_labels = [*metric_labels, metric_labels[0]]

        label = item.get("label", "?")
        display_label = label[:30] + "..." if len(label) > 30 else label

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=theta_labels,
                fill="toself",
                name=display_label,
                line_color=colors[i % len(colors)],
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True}},
        showlegend=True,
        height=500,
        margin={"l": 80, "r": 80, "t": 40, "b": 40},
    )
    st.plotly_chart(fig, width="stretch")


def _render_breakdown_table(ranking: list[dict[str, Any]]) -> None:
    """Render score breakdown as a table."""
    rows = []
    for item in ranking:
        label = item.get("label", "?")
        metrics = item.get("metrics", {})
        row: dict[str, Any] = {
            "Option": label[:50],
            "Score": item.get("score", 0),
            "Status": item.get("status", ""),
        }
        if metrics:
            row["Evid. Coverage"] = metrics.get("evidence_coverage", 0)
            row["Evid. Count"] = metrics.get("evidence_count", 0)
            row["Weak"] = metrics.get("weak_claims", 0)
            row["Unsupported"] = metrics.get("unsupported_claims", 0)
            row["Blocking Obj."] = metrics.get("blocking_objections", 0)
            row["NB Obj."] = metrics.get("nonblocking_objections", 0)
        rows.append(row)

    if rows:
        st.dataframe(rows, width="stretch")
