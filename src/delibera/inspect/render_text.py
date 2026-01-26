"""Render RunSummary to human-readable terminal text.

This renderer produces plain text output suitable for terminal display.
No external dependencies are required.
"""

from typing import Any

from delibera.inspect.summarize import NodeSummary, RunSummary


def render_text(summary: RunSummary) -> str:
    """Render a RunSummary to plain text.

    Args:
        summary: The RunSummary to render.

    Returns:
        Human-readable text representation.
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append("DELIBERATION RUN SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Run identification
    lines.append(f"Run ID: {summary.run_id}")
    lines.append(f"Created: {summary.created_at}")
    lines.append(f"Question: {summary.question}")
    lines.append("")

    # Protocol info
    if summary.protocol.name:
        lines.append("--- Protocol ---")
        lines.append(f"Name: {summary.protocol.name}")
        if summary.protocol.version:
            lines.append(f"Version: {summary.protocol.version}")
        if summary.protocol.source:
            lines.append(f"Source: {summary.protocol.source}")
        lines.append("")

    # Final Recommendation
    lines.append("--- Final Recommendation ---")
    if summary.recommendation:
        lines.append(summary.recommendation)
    else:
        lines.append("(No recommendation available)")
    lines.append("")

    if summary.confidence > 0:
        lines.append(f"Confidence: {summary.confidence:.1%}")
        lines.append("")

    # Decision Explanation
    if summary.decision_explanation:
        lines.append("--- Decision Explanation ---")
        _render_decision_explanation(lines, summary.decision_explanation)
        lines.append("")

    # Refinement/Convergence
    if summary.refinement.stop_reason:
        lines.append("--- Convergence ---")
        lines.append(f"Stop reason: {summary.refinement.stop_reason}")
        if summary.refinement.rounds_run > 0:
            lines.append(
                f"Rounds: {summary.refinement.rounds_run} / {summary.refinement.max_rounds}"
            )
        lines.append("")

    # Selected Path
    lines.append("--- Selected Path ---")
    if summary.selected_path:
        for node in summary.selected_path:
            _render_node_summary(lines, node)
    else:
        lines.append("(No path information available)")
    lines.append("")

    # Key Claims with Citations
    if summary.key_claims:
        lines.append("--- Key Claims ---")
        for i, claim in enumerate(summary.key_claims, 1):
            lines.append(f"{i}. [{claim.claim_type.upper()}] {claim.text}")
            lines.append(f"   Status: {claim.status}")
            if claim.citations:
                lines.append("   Citations:")
                for cit in claim.citations:
                    lines.append(f"     - {cit.source}")
                    if cit.excerpt:
                        # Truncate excerpt for display
                        if len(cit.excerpt) > 100:
                            excerpt = cit.excerpt[:100] + "..."
                        else:
                            excerpt = cit.excerpt
                        lines.append(f'       "{excerpt}"')
            lines.append("")

    # Pruning History
    if summary.prune_events:
        lines.append("--- Pruning History ---")
        for i, prune in enumerate(summary.prune_events, 1):
            lines.append(f"Prune #{i}:")
            lines.append(f"  Kept: {', '.join(prune.kept) or '(none)'}")
            lines.append(f"  Pruned: {', '.join(prune.pruned) or '(none)'}")
            if prune.reason:
                lines.append(f"  Reason: {prune.reason}")
            lines.append("")

    # Statistics
    lines.append("--- Statistics ---")
    lines.append(f"Total nodes: {summary.total_nodes}")
    lines.append(f"Tool calls: {summary.total_tool_calls}")
    lines.append(f"Evidence items: {summary.total_evidence}")
    lines.append("")

    # Errors and Warnings
    if summary.errors:
        lines.append("--- Errors ---")
        for error in summary.errors:
            lines.append(f"  ERROR: {error}")
        lines.append("")

    if summary.warnings:
        lines.append("--- Warnings ---")
        for warning in summary.warnings:
            lines.append(f"  WARNING: {warning}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def _render_node_summary(lines: list[str], node: NodeSummary) -> None:
    """Render a single node summary."""
    indent = "  " * node.depth

    # Node header
    label = node.label or f"({node.kind})"
    lines.append(f"{indent}[{node.kind.upper()}] {label}")

    # Score
    if node.score is not None:
        lines.append(f"{indent}  Score: {node.score:.2f}")
        if node.score_breakdown:
            breakdown_parts = [f"{k}={v:.2f}" for k, v in sorted(node.score_breakdown.items())]
            lines.append(f"{indent}    Breakdown: {', '.join(breakdown_parts)}")

    # Claims
    if node.claim_counts:
        supported = node.claim_counts.get("supported", 0)
        weak = node.claim_counts.get("weak", 0)
        unsupported = node.claim_counts.get("unsupported", 0)
        lines.append(
            f"{indent}  Claims: {supported} supported, {weak} weak, {unsupported} unsupported"
        )

    # Evidence
    if node.evidence_count > 0:
        lines.append(f"{indent}  Evidence: {node.evidence_count} items")
        if node.evidence_sources:
            for source in node.evidence_sources[:3]:  # Show top 3
                lines.append(f"{indent}    - {source}")
            if len(node.evidence_sources) > 3:
                lines.append(f"{indent}    ... and {len(node.evidence_sources) - 3} more")

    # Objections
    obj = node.objections
    total_blocking = obj.blocking_open + obj.blocking_accepted
    total_nonblocking = obj.nonblocking_open + obj.nonblocking_accepted
    if total_blocking > 0 or total_nonblocking > 0:
        lines.append(
            f"{indent}  Objections: "
            f"{obj.blocking_accepted}/{total_blocking} blocking accepted, "
            f"{obj.nonblocking_accepted}/{total_nonblocking} nonblocking accepted"
        )

    lines.append("")


def _render_decision_explanation(lines: list[str], explanation: dict[str, Any]) -> None:
    """Render decision explanation section."""
    # Ranking
    ranking = explanation.get("ranking", [])
    if ranking:
        lines.append("Ranking:")
        for entry in ranking:
            label = entry.get("label", "Unknown")
            score = entry.get("score", 0.0)
            status = entry.get("status", "")
            lines.append(f"  {score:.2f} - {label} ({status})")

    # Weights used
    weights = explanation.get("weights_used", {})
    if weights:
        lines.append("Weights:")
        for key, value in sorted(weights.items()):
            lines.append(f"  {key}: {value:.2f}")
