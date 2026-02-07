"""Timeline page — event timeline and filterable log table."""

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import streamlit as st

# Event type categories for grouping and coloring
EVENT_CATEGORIES: dict[str, tuple[str, str]] = {
    # event_type: (category, color)
    "run_start": ("System", "#607D8B"),
    "run_end": ("System", "#607D8B"),
    "final_artifact_written": ("System", "#607D8B"),
    "node_created": ("Tree", "#2196F3"),
    "expand": ("Tree", "#2196F3"),
    "prune": ("Tree", "#F44336"),
    "reduce": ("Tree", "#FF9800"),
    "llm_call_requested": ("Agent", "#4CAF50"),
    "llm_call_succeeded": ("Agent", "#4CAF50"),
    "work_output": ("Agent", "#4CAF50"),
    "evidence_added": ("Epistemics", "#9C27B0"),
    "claim_validation_report": ("Epistemics", "#9C27B0"),
    "objection_added": ("Epistemics", "#9C27B0"),
    "objection_resolved": ("Epistemics", "#9C27B0"),
    "score_computed": ("Epistemics", "#9C27B0"),
    "convergence_checked": ("System", "#607D8B"),
    "tool_call_executed": ("Agent", "#4CAF50"),
}


def render(events: list[dict[str, Any]]) -> None:
    """Render the timeline page.

    Args:
        events: Trace events.
    """
    st.title("⏱️ Event Timeline")

    if not events:
        st.info("No events in this run.")
        return

    # Summary
    st.metric("Total Events", len(events))

    # Time span
    first_ts = events[0].get("timestamp", "")
    last_ts = events[-1].get("timestamp", "")
    if first_ts and last_ts:
        try:
            t0 = datetime.fromisoformat(first_ts)
            t1 = datetime.fromisoformat(last_ts)
            duration = (t1 - t0).total_seconds()
            st.caption(f"Duration: {duration:.1f} seconds")
        except ValueError:
            pass

    st.divider()

    # Timeline chart
    st.subheader("Event Timeline")
    _render_timeline_chart(events)

    st.divider()

    # LLM call durations
    _render_llm_durations(events)

    st.divider()

    # Filterable event log
    st.subheader("Event Log")
    _render_event_log(events)


def _render_timeline_chart(events: list[dict[str, Any]]) -> None:
    """Render event timeline as a scatter chart."""
    timestamps: list[float] = []
    categories: list[str] = []
    labels: list[str] = []
    colors: list[str] = []
    hover_texts: list[str] = []

    base_time = None

    for event in events:
        ts_str = event.get("timestamp", "")
        event_type = event.get("event_type", "unknown")
        payload = event.get("payload", {})

        try:
            ts = datetime.fromisoformat(ts_str)
            if base_time is None:
                base_time = ts
            elapsed = (ts - base_time).total_seconds()
        except (ValueError, TypeError):
            continue

        cat, color = EVENT_CATEGORIES.get(event_type, ("Other", "#795548"))

        timestamps.append(elapsed)
        categories.append(cat)
        labels.append(event_type)
        colors.append(color)

        # Build hover text
        hover_parts = [f"Type: {event_type}", f"Time: +{elapsed:.2f}s"]
        if "node_id" in payload:
            hover_parts.append(f"Node: {payload['node_id'][:8]}")
        if "role" in payload:
            hover_parts.append(f"Role: {payload['role']}")
        if "step" in payload:
            hover_parts.append(f"Step: {payload['step']}")
        hover_texts.append("<br>".join(hover_parts))

    if not timestamps:
        st.info("No parseable timestamps in events.")
        return

    # Group by category for legend
    unique_cats = list(dict.fromkeys(categories))
    fig = go.Figure()

    for cat in unique_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        cat_color = colors[mask[0]] if mask else "#795548"
        fig.add_trace(
            go.Scatter(
                x=[timestamps[i] for i in mask],
                y=[labels[i] for i in mask],
                mode="markers",
                marker={"size": 10, "color": cat_color},
                name=cat,
                text=[hover_texts[i] for i in mask],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        xaxis_title="Time (seconds from start)",
        height=max(400, len(set(labels)) * 30),
        margin={"l": 20, "r": 20, "t": 20, "b": 40},
        showlegend=True,
    )
    st.plotly_chart(fig, width="stretch")


def _render_llm_durations(events: list[dict[str, Any]]) -> None:
    """Render LLM call durations as a bar chart."""
    requested: dict[tuple[str, str], datetime] = {}
    durations: list[dict[str, Any]] = []

    for event in events:
        event_type = event.get("event_type", "")
        payload = event.get("payload", {})
        ts_str = event.get("timestamp", "")

        if event_type == "llm_call_requested":
            key = (payload.get("role", ""), payload.get("node_id", ""))
            with contextlib.suppress(ValueError):
                requested[key] = datetime.fromisoformat(ts_str)

        elif event_type == "llm_call_succeeded":
            key = (payload.get("role", ""), payload.get("node_id", ""))
            if key in requested:
                try:
                    end = datetime.fromisoformat(ts_str)
                    start = requested[key]
                    duration = (end - start).total_seconds()
                    durations.append(
                        {
                            "role": key[0],
                            "node": key[1][:8] if key[1] else "?",
                            "duration": duration,
                        }
                    )
                    del requested[key]
                except ValueError:
                    pass

    if not durations:
        return

    st.subheader("🤖 Gemini Call Durations")

    dur_labels = [f"{d['role']} ({d['node']})" for d in durations]
    values = [d["duration"] for d in durations]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=dur_labels,
            orientation="h",
            marker_color="#4CAF50",
            text=[f"{v:.1f}s" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis_title="Duration (seconds)",
        height=max(200, len(durations) * 35),
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Total LLM time: {sum(values):.1f}s across {len(durations)} calls")


def _render_event_log(events: list[dict[str, Any]]) -> None:
    """Render filterable event log table."""
    all_types = sorted({e.get("event_type", "") for e in events})

    selected_types = st.multiselect(
        "Filter by event type",
        all_types,
        default=all_types,
        key="event_filter",
    )

    rows = []
    for event in events:
        event_type = event.get("event_type", "")
        if event_type not in selected_types:
            continue
        payload = event.get("payload", {})

        # Build summary
        summary_parts: list[str] = []
        if "node_id" in payload:
            summary_parts.append(f"node={payload['node_id'][:8]}")
        if "role" in payload:
            summary_parts.append(f"role={payload['role']}")
        if "step" in payload:
            summary_parts.append(f"step={payload['step']}")
        if "score" in payload:
            summary_parts.append(f"score={payload['score']:.2f}")
        if "question" in payload:
            summary_parts.append(f"q={payload['question'][:40]}")

        rows.append(
            {
                "Time": event.get("timestamp", "")[-12:],
                "Type": event_type,
                "Summary": ", ".join(summary_parts) or "-",
            }
        )

    if rows:
        st.dataframe(rows, width="stretch", height=400)
    else:
        st.info("No events match the selected filters.")
