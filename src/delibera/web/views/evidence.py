"""Evidence page — claims, evidence, and objections in tabs."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

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


def _extract_domain(url: str) -> str:
    """Extract a readable domain name from a URL."""
    if "vertexaisearch.cloud.google.com/grounding-api-redirect" in url:
        return "Google Search"
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.removeprefix("www.")
        return domain or "Web Source"
    except Exception:
        return "Web Source"


def _format_source(
    source: str,
    title: str = "",
    summary: str = "",
) -> str:
    """Format an evidence source for display as markdown.

    Priority: summary (LLM-generated) > title (page title) > domain name.
    """
    if not source.startswith("http"):
        return f"📄 {source}"
    domain = _extract_domain(source)
    display_name = summary or title or domain
    return f"🌐 [{display_name}]({source}) — *{domain}*"


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
                        st.markdown(f"- {_format_source(source)}")
                        if excerpt:
                            st.caption(excerpt)
        return

    # Fallback: show claim validation results from trace events
    if validation_events:
        st.markdown("*Claim validation results from trace events:*")

        # Use the latest validation event per claim_id (later rounds override earlier ones)
        claims_by_id: dict[str, dict[str, Any]] = {}
        for event in validation_events:
            for d in event.get("payload", {}).get("details", []):
                cid = d.get("claim_id", "")
                if cid:
                    claims_by_id[cid] = d

        unique_claims = list(claims_by_id.values())
        total_supported = sum(1 for d in unique_claims if d.get("status") == "supported")
        total_weak = sum(1 for d in unique_claims if d.get("status") == "weak")
        total_unsupported = sum(1 for d in unique_claims if d.get("status") == "unsupported")

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Supported", total_supported)
        c2.metric("🟡 Weak", total_weak)
        c3.metric("🔴 Unsupported", total_unsupported)

        if unique_claims:
            with st.expander(f"View {len(unique_claims)} claim details"):
                for d in unique_claims:
                    status = d.get("status", "unknown")
                    text = d.get("text", d.get("claim_id", "?"))
                    badge = STATUS_BADGE.get(status, f"⚪ {status}")
                    st.markdown(f"- {badge}: {text}")
    else:
        st.info("No claim data available. Run with `--use-all-llm` for detailed claims.")


def _build_evidence_enrichment(
    events: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build a source → {excerpt, title, summary, query} lookup from work_output events.

    This provides fallback data for older runs where evidence_added events
    don't include excerpt, title, or summary.
    """
    enrichment: dict[str, dict[str, str]] = {}
    for event in events:
        if (
            event.get("event_type") == "work_output"
            and event.get("payload", {}).get("role") == "researcher"
        ):
            output = event.get("payload", {}).get("output", {})
            for item in output.get("evidence", []):
                source = item.get("source", "")
                if source and source not in enrichment:
                    enrichment[source] = {
                        "excerpt": item.get("excerpt", ""),
                        "title": item.get("title", ""),
                        "summary": item.get("summary", ""),
                        "query": item.get("query", ""),
                    }
    return enrichment


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

    # Build enrichment map from work_output events (fallback for old runs)
    enrichment = _build_evidence_enrichment(events)

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
                title = evidence_data.get("title", "")
                summary = evidence_data.get("summary", "")

                # Enrich from work_output if missing
                if source in enrichment:
                    if not excerpt:
                        excerpt = enrichment[source].get("excerpt", "")
                    if not title:
                        title = enrichment[source].get("title", "")
                    if not summary:
                        summary = enrichment[source].get("summary", "")

                with st.container(border=True):
                    st.markdown(_format_source(source, title, summary))
                    if excerpt:
                        st.caption(excerpt)


def _resolve_target(target: str, claim_map: dict[str, str]) -> str:
    """Resolve an objection target to a human-readable label.

    Args:
        target: Either "artifact" or a claim_id.
        claim_map: Mapping from claim_id to claim text.
    """
    if not target or target == "artifact":
        return "Overall Proposal"
    text = claim_map.get(target, "")
    if text:
        display = text[:80] + "..." if len(text) > 80 else text
        return f"Claim: {display}"
    return f"Claim {target}"


def _build_claim_map(
    artifact: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, str]:
    """Build a claim_id → text lookup from artifact and validation events."""
    claim_map: dict[str, str] = {}
    # Source 1: artifact key_claims (top 5)
    for claim in artifact.get("key_claims", []):
        cid = claim.get("claim_id", "")
        text = claim.get("text", "")
        if cid and text:
            claim_map[cid] = text
    # Source 2: claim_validation_report event details (all validated claims)
    for event in events:
        if event.get("event_type") == "claim_validation_report":
            for detail in event.get("payload", {}).get("details", []):
                cid = detail.get("claim_id", "")
                text = detail.get("text", "")
                if cid and text and cid not in claim_map:
                    claim_map[cid] = text
    return claim_map


def _render_objections(
    events: list[dict[str, Any]],
    artifact: dict[str, Any],
    objection_events: list[dict[str, Any]],
) -> None:
    """Render objections tab."""
    claim_map = _build_claim_map(artifact, events)

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
                target_label = _resolve_target(obj.get("target", ""), claim_map)
                with st.container(border=True):
                    st.markdown(f"**Targets**: {target_label}")
                    st.markdown(obj.get("rationale", ""))

        if nonblocking_open:
            st.subheader("🟡 Nonblocking Objections (Open)")
            for obj in nonblocking_open:
                target_label = _resolve_target(obj.get("target", ""), claim_map)
                with st.container(border=True):
                    st.markdown(f"**Targets**: {target_label}")
                    st.markdown(obj.get("rationale", ""))
        return

    # Fallback: use trace events
    if not objection_events:
        st.info("No objections raised in this run.")
        return

    # Build node label map from events
    node_labels: dict[str, str] = {}
    for event in events:
        if event.get("event_type") == "node_created":
            payload = event.get("payload", {})
            nid = payload.get("node_id", "")
            label = payload.get("label", "")
            if nid and label:
                node_labels[nid] = label

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
        node_display = node_labels.get(node_id, node_id[:8] + "...")
        with st.expander(f"{node_display} — {len(objections)} objections"):
            for obj in objections:
                obj_data = obj.get("objection", obj)
                blocking = (
                    obj_data.get("blocking", False) or obj_data.get("severity", "") == "blocking"
                )
                severity = "🔴 Blocking" if blocking else "🟡 Nonblocking"
                rationale = obj_data.get("rationale", "No rationale provided")
                target = obj_data.get("target", "")
                target_label = _resolve_target(target, claim_map)
                with st.container(border=True):
                    st.markdown(f"{severity} — {target_label}")
                    st.markdown(rationale)
