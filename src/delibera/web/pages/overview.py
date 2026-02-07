"""Overview page — landing page with question, recommendation, stats, and key claims."""

from __future__ import annotations

from typing import Any

import streamlit as st

STATUS_BADGE = {
    "supported": "🟢",
    "weak": "🟡",
    "unsupported": "🔴",
}

TYPE_BADGE = {
    "fact": "📊",
    "inference": "🔗",
    "value": "💎",
    "plan": "📋",
}


def render(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    _artifact: dict[str, Any],
) -> None:
    """Render the overview page.

    Args:
        summary: RunSummary as dict.
        events: Trace events.
        artifact: Artifact dict.
    """
    # Question
    st.title("⚖️ Deliberation Overview")
    st.markdown(f"### {summary.get('question', 'No question')}")

    # Metric cards
    llm_calls = len([e for e in events if e.get("event_type") == "llm_call_succeeded"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{summary.get('confidence', 0):.1f}")
    col2.metric("Total Nodes", summary.get("total_nodes", 0))
    col3.metric("Evidence Items", summary.get("total_evidence", 0))
    col4.metric("LLM Calls", llm_calls)

    st.divider()

    # Final recommendation
    recommendation = summary.get("recommendation", "")
    if recommendation:
        st.subheader("Final Recommendation")
        st.markdown(f"> {recommendation}")
    else:
        st.info("No recommendation generated for this run.")

    st.divider()

    # Decision explanation / ranking
    decision = summary.get("decision_explanation", {})
    ranking = decision.get("ranking", [])
    if ranking:
        st.subheader("Option Ranking")
        for item in ranking:
            label = item.get("label", "?")
            score = item.get("score", 0)
            status = item.get("status", "")
            metrics = item.get("metrics", {})

            is_survivor = status == "survivor"
            icon = "✅" if is_survivor else "❌"

            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 1, 1])
                with c1:
                    st.markdown(f"{icon} **{label}**")
                with c2:
                    st.metric("Score", f"{score:.2f}")
                with c3:
                    st.caption(status.upper())

                # Metrics row
                if metrics:
                    mc = st.columns(6)
                    mc[0].caption(f"Evid. Coverage: {metrics.get('evidence_coverage', 0):.1f}")
                    mc[1].caption(f"Evid. Count: {metrics.get('evidence_count', 0):.0f}")
                    mc[2].caption(f"Weak: {metrics.get('weak_claims', 0):.0f}")
                    mc[3].caption(f"Unsupported: {metrics.get('unsupported_claims', 0):.0f}")
                    mc[4].caption(f"Blocking: {metrics.get('blocking_objections', 0):.0f}")
                    mc[5].caption(f"Nonblocking: {metrics.get('nonblocking_objections', 0):.0f}")

    st.divider()

    # Key claims
    key_claims = summary.get("key_claims", [])
    if key_claims:
        st.subheader("Key Claims")
        for claim in key_claims:
            status_icon = STATUS_BADGE.get(claim.get("status", ""), "⚪")
            type_icon = TYPE_BADGE.get(claim.get("claim_type", ""), "📌")
            with st.expander(f"{status_icon} {type_icon} {claim.get('text', '')}"):
                st.markdown(f"**Type**: {claim.get('claim_type', 'unknown')}")
                st.markdown(f"**Status**: {claim.get('status', 'unknown')}")
                citations = claim.get("citations", [])
                if citations:
                    st.markdown("**Evidence:**")
                    for cit in citations:
                        source = cit.get("source", "")
                        excerpt = cit.get("excerpt", "")
                        st.markdown(f"- **Source**: {source}")
                        if excerpt:
                            st.caption(excerpt)

    st.divider()

    # Protocol + convergence info
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Protocol")
        proto = summary.get("protocol", {})
        st.markdown(f"**Name**: {proto.get('name', 'unknown')}")
        st.markdown(f"**Version**: {proto.get('version', 'unknown')}")
        st.markdown(f"**Source**: {proto.get('source', 'unknown')}")

    with col_right:
        st.subheader("Convergence")
        refinement = summary.get("refinement", {})
        st.markdown(f"**Stop Reason**: {refinement.get('stop_reason', 'unknown')}")
        st.markdown(
            f"**Rounds**: {refinement.get('rounds_run', 0)} / {refinement.get('max_rounds', 0)}"
        )

    # Pruning history
    prune_events = summary.get("prune_events", [])
    if prune_events:
        st.divider()
        st.subheader("Pruning History")
        for i, pe in enumerate(prune_events):
            with st.expander(f"Prune Step {i + 1} — {pe.get('reason', '')}"):
                st.markdown(f"**Kept**: {', '.join(pe.get('kept', []))}")
                st.markdown(f"**Pruned**: {', '.join(pe.get('pruned', []))}")
