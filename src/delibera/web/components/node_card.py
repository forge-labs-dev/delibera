"""Reusable node summary card component."""

from __future__ import annotations

from typing import Any

import streamlit as st

STATUS_COLORS = {
    "supported": "#4CAF50",
    "weak": "#FF9800",
    "unsupported": "#F44336",
}

KIND_ICONS = {
    "root": "🌳",
    "option": "🔀",
    "merged": "🏆",
    "plan": "📋",
    "risk": "⚠️",
}


def render_node_card(node: dict[str, Any], show_details: bool = True) -> None:
    """Render a node summary card.

    Args:
        node: Node dict from summary selected_path or replayed nodes.
        show_details: Whether to show expanded details.
    """
    kind = node.get("kind", "option")
    status = node.get("status", "active")
    label = node.get("label", kind)
    icon = KIND_ICONS.get(kind, "📌")

    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown(f"### {icon} {label}")
            st.caption(f"Kind: {kind} | Depth: {node.get('depth', 0)} | Status: {status}")
        with col2:
            score = node.get("score")
            if score is not None:
                st.metric("Score", f"{score:.2f}")
        with col3:
            st.metric("Evidence", node.get("evidence_count", 0))

        if show_details:
            claim_counts = node.get("claim_counts", {})
            if claim_counts:
                c1, c2, c3 = st.columns(3)
                c1.metric("Supported", claim_counts.get("supported", 0))
                c2.metric("Weak", claim_counts.get("weak", 0))
                c3.metric("Unsupported", claim_counts.get("unsupported", 0))

            objections = node.get("objections", {})
            if objections:
                blocking = objections.get("blocking_open", 0)
                nonblocking = objections.get("nonblocking_open", 0)
                if blocking or nonblocking:
                    st.caption(f"Objections: {blocking} blocking, {nonblocking} nonblocking")

            breakdown = node.get("score_breakdown", {})
            if breakdown:
                with st.expander("Score Breakdown"):
                    for metric, value in sorted(breakdown.items()):
                        st.text(f"  {metric}: {value:.3f}")
