"""Evidence page — claims, evidence, and objections in tabs."""

from __future__ import annotations

from typing import Any

import streamlit as st

STATUS_BADGE = {
    "supported": "🟢 Supported",
    "weak": "🟡 Weak",
    "unsupported": "🔴 Unsupported",
    "unvalidated": "⚪ Unvalidated",
}

TYPE_BADGE = {
    "fact": "📊 Fact",
    "inference": "🔗 Inference",
    "value": "💎 Value",
    "plan": "📋 Plan",
}


def render(
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    artifact: dict[str, Any],
) -> None:
    """Render the evidence page."""
    st.title("🔍 Evidence & Claims")

    # Summary metrics from artifact or events
    claim_summary = artifact.get("claim_check_summary", {})
    evidence_events = [e for e in events if e.get("event_type") == "evidence_added"]
    objection_events = [e for e in events if e.get("event_type") == "objection_added"]
    validation_events = [e for e in events if e.get("event_type") == "claim_validation_report"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Evidence Items", len(evidence_events))
    supported_default = _count_from_validations(validation_events, "supported")
    weak_default = _count_from_validations(validation_events, "weak")
    col2.metric("Supported Claims", claim_summary.get("supported", supported_default))
    col3.metric("Weak Claims", claim_summary.get("weak", weak_default))
    col4.metric("Objections Raised", len(objection_events))

    st.divider()

    tab_claims, tab_evidence, tab_objections = st.tabs(["Claims", "Evidence", "Objections"])

    with tab_claims:
        _render_claims(summary, artifact, validation_events)

    with tab_evidence:
        _render_evidence(events, evidence_events)

    with tab_objections:
        _render_objections(events, artifact, objection_events)


def _count_from_validations(validation_events: list[dict[str, Any]], field: str) -> int:
    """Sum a field across all claim_validation_report events."""
    total = 0
    for event in validation_events:
        total += event.get("payload", {}).get(field, 0)
    return total


def _render_claims(
    summary: dict[str, Any],
    artifact: dict[str, Any],
    validation_events: list[dict[str, Any]],
) -> None:
    """Render claims tab."""
    key_claims = summary.get("key_claims", [])

    if not key_claims:
        key_claims = artifact.get("key_claims", [])

    if key_claims:
        for claim in key_claims:
            claim_type = claim.get("claim_type", claim.get("type", "unknown"))
            status = claim.get("status", "unknown")
            text = claim.get("text", "")

            status_badge = STATUS_BADGE.get(status, f"⚪ {status}")
            type_badge = TYPE_BADGE.get(claim_type, f"📌 {claim_type}")

            with st.expander(f"{status_badge} | {type_badge} — {text}"):
                st.markdown(f"**Full claim**: {text}")
                st.markdown(f"**Type**: {type_badge}")
                st.markdown(f"**Status**: {status_badge}")

                citations = claim.get("citations", [])
                if citations:
                    st.markdown("---")
                    st.markdown("**Supporting Evidence:**")
                    for cit in citations:
                        source = cit.get("source", "unknown")
                        excerpt = cit.get("excerpt", "")
                        is_web = source.startswith("http") or "google" in source.lower()
                        source_badge = "🌐 Gemini Search" if is_web else "📄 Local"
                        st.markdown(f"- {source_badge} **{source}**")
                        if excerpt:
                            st.caption(excerpt)
        return

    # Fallback: show claim validation results from trace events
    if validation_events:
        st.markdown("*Claim validation results from trace events:*")
        for event in validation_events:
            payload = event.get("payload", {})
            node_id = payload.get("node_id", "unknown")[:8]
            supported = payload.get("supported", 0)
            weak = payload.get("weak", 0)
            unsupported = payload.get("unsupported", 0)

            with st.container(border=True):
                st.markdown(f"**Node {node_id}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 Supported", supported)
                c2.metric("🟡 Weak", weak)
                c3.metric("🔴 Unsupported", unsupported)

                details = payload.get("details", [])
                if details:
                    with st.expander(f"View {len(details)} claim details"):
                        for d in details:
                            status = d.get("status", "unknown")
                            text = d.get("text", d.get("claim_id", "?"))
                            badge = STATUS_BADGE.get(status, f"⚪ {status}")
                            st.markdown(f"- {badge}: {text}")
    else:
        st.info("No claim data available. Run with `--use-all-llm` for detailed claims.")


def _render_evidence(
    events: list[dict[str, Any]],
    evidence_events: list[dict[str, Any]],
) -> None:
    """Render evidence tab from trace events."""
    if not evidence_events:
        # Show work_output events from researcher as fallback
        research_events = [
            e
            for e in events
            if e.get("event_type") == "work_output"
            and e.get("payload", {}).get("role") == "researcher"
        ]
        if research_events:
            st.markdown("*Research outputs (no retrieval results):*")
            for event in research_events:
                payload = event.get("payload", {})
                node_id = payload.get("node_id", "unknown")[:8]
                output = payload.get("output", {})
                queries = output.get("queries", [])
                with st.container(border=True):
                    st.markdown(f"**Node {node_id}** — Researcher")
                    if queries:
                        st.markdown("Search queries generated:")
                        for q in queries:
                            st.markdown(f"- `{q}`")
        else:
            st.info("No evidence collected. Run with `--use-all-llm` for web evidence retrieval.")
        return

    st.metric("Total Evidence Items", len(evidence_events))
    st.divider()

    # Group by node_id
    by_node: dict[str, list[dict[str, Any]]] = {}
    for event in evidence_events:
        payload = event.get("payload", {})
        node_id = payload.get("node_id", "unknown")
        if node_id not in by_node:
            by_node[node_id] = []
        by_node[node_id].append(payload)

    for node_id, evidences in by_node.items():
        with st.expander(
            f"Node {node_id[:8]}... — {len(evidences)} evidence items",
            expanded=True,
        ):
            for ev in evidences:
                evidence_data = ev.get("evidence", {})
                source = evidence_data.get("source", "unknown")
                excerpt = evidence_data.get("excerpt", "")

                is_web = source.startswith("http") or "google" in source.lower()
                source_badge = "🌐 Gemini Search" if is_web else "📄 Local"

                with st.container(border=True):
                    st.markdown(f"{source_badge} **{source}**")
                    if excerpt:
                        st.caption(excerpt)


def _render_objections(
    _events: list[dict[str, Any]],
    artifact: dict[str, Any],
    objection_events: list[dict[str, Any]],
) -> None:
    """Render objections tab."""
    # Try artifact first (structured)
    objections_data = artifact.get("objections", {})
    blocking_open = objections_data.get("blocking_open", [])
    blocking_accepted = objections_data.get("blocking_accepted", [])
    nonblocking_open = objections_data.get("nonblocking_open", [])
    nonblocking_accepted = objections_data.get("nonblocking_accepted", [])

    has_artifact_data = (
        len(blocking_open)
        + len(blocking_accepted)
        + len(nonblocking_open)
        + len(nonblocking_accepted)
    ) > 0

    if has_artifact_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Blocking Open", len(blocking_open))
        c2.metric("🟢 Blocking Accepted", len(blocking_accepted))
        c3.metric("🟡 Nonblocking Open", len(nonblocking_open))
        c4.metric("🟢 NB Accepted", len(nonblocking_accepted))
        st.divider()

        if blocking_open:
            st.subheader("🔴 Blocking Objections (Open)")
            for obj in blocking_open:
                with st.container(border=True):
                    st.markdown(f"**Target**: {obj.get('target', 'unknown')[:12]}")
                    st.markdown(obj.get("rationale", ""))

        if nonblocking_open:
            st.subheader("🟡 Nonblocking Objections (Open)")
            for obj in nonblocking_open:
                with st.container(border=True):
                    st.markdown(f"**Target**: {obj.get('target', 'unknown')[:12]}")
                    st.markdown(obj.get("rationale", ""))
        return

    # Fallback: use trace events
    if not objection_events:
        st.info("No objections raised in this run.")
        return

    # Count by type
    blocking_count = sum(1 for e in objection_events if e.get("payload", {}).get("blocking"))
    nonblocking_count = len(objection_events) - blocking_count

    c1, c2 = st.columns(2)
    c1.metric("🔴 Blocking", blocking_count)
    c2.metric("🟡 Nonblocking", nonblocking_count)
    st.divider()

    # Group by node
    by_node: dict[str, list[dict[str, Any]]] = {}
    for event in objection_events:
        payload = event.get("payload", {})
        node_id = payload.get("node_id", "unknown")
        if node_id not in by_node:
            by_node[node_id] = []
        by_node[node_id].append(payload)

    for node_id, objections in by_node.items():
        with st.expander(f"Node {node_id[:8]}... — {len(objections)} objections"):
            for obj in objections:
                blocking = obj.get("blocking", False)
                severity = "🔴 Blocking" if blocking else "🟡 Nonblocking"
                rationale = obj.get("rationale", "No rationale provided")
                with st.container(border=True):
                    st.markdown(f"{severity}")
                    st.markdown(rationale)
