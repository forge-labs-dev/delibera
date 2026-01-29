"""Build structured RunSummary from replay and trace data.

This module constructs a comprehensive summary of a deliberation run
using the replayed tree state, trace events, and final artifact.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from delibera.trace.reader import load_artifact, load_trace
from delibera.trace.replay import ReplayedRun, replay_run


@dataclass
class Citation:
    """A citation linking a claim to evidence."""

    evidence_id: str
    source: str
    excerpt: str  # Truncated to max 200 chars


@dataclass
class KeyClaim:
    """A key claim from the final artifact with citations."""

    claim_id: str
    claim_type: str  # fact, inference, plan
    text: str
    status: str  # supported, weak, unsupported
    citations: list[Citation] = field(default_factory=list)


@dataclass
class ObjectionSummary:
    """Summary of objections for a node."""

    blocking_open: int = 0
    blocking_accepted: int = 0
    nonblocking_open: int = 0
    nonblocking_accepted: int = 0


@dataclass
class NodeSummary:
    """Summary of a single node on the selected path."""

    node_id: str
    label: str
    kind: str  # root, option, merged
    depth: int
    score: float | None = None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    claim_counts: dict[str, int] = field(default_factory=dict)  # supported/weak/unsupported
    evidence_sources: list[str] = field(default_factory=list)  # Top N sources
    evidence_count: int = 0
    objections: ObjectionSummary = field(default_factory=ObjectionSummary)


@dataclass
class PruneEvent:
    """Summary of a single prune decision."""

    candidates: list[str]  # node_ids considered
    kept: list[str]  # node_ids kept (survivors)
    pruned: list[str]  # node_ids pruned
    reason: str = ""  # Summary reason for pruning


@dataclass
class RefinementInfo:
    """Information about refinement rounds."""

    rounds_run: int = 0
    max_rounds: int = 0
    stop_reason: str = ""  # converged, max_rounds, etc.


@dataclass
class ProtocolInfo:
    """Information about the protocol used."""

    name: str = ""
    version: str = ""
    source: str = ""  # yaml:path, builtin, etc.


@dataclass
class RunSummary:
    """Complete summary of a deliberation run.

    This is the main data structure used by renderers to produce
    human-readable output.
    """

    # Run identification
    run_id: str
    question: str
    created_at: str

    # Protocol info
    protocol: ProtocolInfo = field(default_factory=ProtocolInfo)

    # Final outcome
    recommendation: str = ""
    confidence: float = 0.0
    decision_explanation: dict[str, Any] = field(default_factory=dict)

    # Convergence info
    refinement: RefinementInfo = field(default_factory=RefinementInfo)

    # Selected path (root -> leaf in order)
    selected_path: list[NodeSummary] = field(default_factory=list)

    # All key claims with citations (from artifact)
    key_claims: list[KeyClaim] = field(default_factory=list)

    # Pruning history
    prune_events: list[PruneEvent] = field(default_factory=list)

    # Aggregate stats
    total_nodes: int = 0
    total_tool_calls: int = 0
    total_evidence: int = 0

    # Errors/warnings from reconstruction
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Limits for deterministic output
MAX_EVIDENCE_SOURCES = 5
MAX_KEY_CLAIMS = 5
MAX_EXCERPT_LENGTH = 500


def build_run_summary(run_dir: Path) -> RunSummary:
    """Build a RunSummary from a run directory.

    Args:
        run_dir: Path to run directory containing trace.jsonl and artifact.json.

    Returns:
        RunSummary with all available information.

    Raises:
        FileNotFoundError: If trace.jsonl does not exist.
    """
    trace_path = run_dir / "trace.jsonl"
    artifact_path = run_dir / "artifact.json"

    # Load trace events
    events = load_trace(trace_path)

    # Replay to get tree structure
    replayed = replay_run(
        events,
        artifact_path=artifact_path if artifact_path.exists() else None,
    )

    # Load artifact for final truth
    artifact: dict[str, Any] = {}
    if artifact_path.exists():
        artifact = load_artifact(artifact_path)

    # Build summary
    summary = RunSummary(
        run_id=replayed.run_id,
        question=replayed.question,
        created_at=replayed.created_at,
    )

    # Copy errors/warnings from replay
    summary.errors.extend(replayed.errors)
    summary.warnings.extend(replayed.warnings)

    # Extract info from events
    _extract_protocol_info(summary, events)
    _extract_refinement_info(summary, events)
    _extract_prune_events(summary, events, replayed)
    _extract_tool_stats(summary, events)
    _extract_evidence_stats(summary, events)

    # Build selected path
    _build_selected_path(summary, replayed, events)

    # Extract from artifact
    _extract_from_artifact(summary, artifact)

    # Set total nodes
    summary.total_nodes = len(replayed.nodes)

    return summary


def _extract_protocol_info(summary: RunSummary, events: list[dict[str, Any]]) -> None:
    """Extract protocol information from run_start event."""
    for event in events:
        if event.get("event_type") == "run_start":
            payload = event.get("payload", {})
            summary.protocol.name = payload.get("protocol_name", "")
            summary.protocol.version = payload.get("protocol_version", "")
            summary.protocol.source = payload.get("protocol_source", "")
            break


def _extract_refinement_info(summary: RunSummary, events: list[dict[str, Any]]) -> None:
    """Extract refinement/convergence information from events."""
    # Look for convergence_checked or run_end events
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == "convergence_checked":
            summary.refinement.rounds_run = payload.get("rounds_run", 0)
            summary.refinement.max_rounds = payload.get("max_rounds", 0)
            if payload.get("converged"):
                summary.refinement.stop_reason = "converged"

        elif event_type == "run_end":
            status = payload.get("status", "")
            if status and not summary.refinement.stop_reason:
                summary.refinement.stop_reason = status


def _extract_prune_events(
    summary: RunSummary,
    events: list[dict[str, Any]],
    replayed: ReplayedRun,
) -> None:
    """Extract pruning history from prune events."""
    for event in events:
        if event.get("event_type") == "prune":
            payload = event.get("payload", {})

            survivor_ids = payload.get("survivor_ids", [])
            pruned_ids = payload.get("pruned_ids", [])

            # Get labels for readability
            def get_label(node_id: str) -> str:
                if node_id in replayed.nodes:
                    return replayed.nodes[node_id].label or node_id
                return node_id

            prune_event = PruneEvent(
                candidates=sorted(survivor_ids + pruned_ids),  # All candidates
                kept=[get_label(nid) for nid in sorted(survivor_ids)],
                pruned=[get_label(nid) for nid in sorted(pruned_ids)],
                reason=payload.get("reason", "epistemic_score"),
            )
            summary.prune_events.append(prune_event)


def _extract_tool_stats(summary: RunSummary, events: list[dict[str, Any]]) -> None:
    """Count tool calls from trace."""
    count = 0
    for event in events:
        if event.get("event_type") == "tool_call_executed":
            count += 1
    summary.total_tool_calls = count


def _extract_evidence_stats(summary: RunSummary, events: list[dict[str, Any]]) -> None:
    """Count evidence from trace."""
    count = 0
    for event in events:
        if event.get("event_type") == "evidence_added":
            count += 1
    summary.total_evidence = count


def _build_selected_path(
    summary: RunSummary,
    replayed: ReplayedRun,
    events: list[dict[str, Any]],
) -> None:
    """Build the selected path from root to merged/final node."""
    # Build score lookup from score_computed events
    node_scores: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") == "score_computed":
            payload = event.get("payload", {})
            node_id = payload.get("node_id", "")
            if node_id:
                node_scores[node_id] = {
                    "score": payload.get("score", 0.0),
                    "breakdown": payload.get("breakdown", {}),
                }

    # Build evidence sources lookup per node
    node_evidence: dict[str, list[str]] = {}
    for event in events:
        if event.get("event_type") == "evidence_added":
            payload = event.get("payload", {})
            node_id = payload.get("node_id", "")
            evidence = payload.get("evidence", {})
            source = evidence.get("source", "")
            if node_id and source:
                if node_id not in node_evidence:
                    node_evidence[node_id] = []
                if source not in node_evidence[node_id]:
                    node_evidence[node_id].append(source)

    # Build objections lookup per node
    node_objections: dict[str, ObjectionSummary] = {}
    for event in events:
        if event.get("event_type") == "objection_added":
            payload = event.get("payload", {})
            node_id = payload.get("node_id", "")
            blocking = payload.get("blocking", False)
            if node_id:
                if node_id not in node_objections:
                    node_objections[node_id] = ObjectionSummary()
                if blocking:
                    node_objections[node_id].blocking_open += 1
                else:
                    node_objections[node_id].nonblocking_open += 1

        elif event.get("event_type") == "objection_resolved":
            payload = event.get("payload", {})
            node_id = payload.get("node_id", "")
            blocking = payload.get("blocking", False)
            accepted = payload.get("accepted", False)
            if node_id and node_id in node_objections:
                if blocking:
                    node_objections[node_id].blocking_open -= 1
                    if accepted:
                        node_objections[node_id].blocking_accepted += 1
                else:
                    node_objections[node_id].nonblocking_open -= 1
                    if accepted:
                        node_objections[node_id].nonblocking_accepted += 1

    # Find path from root to merged (or last active option)
    path_node_ids: list[str] = []

    # Start with root
    if replayed.root_id:
        path_node_ids.append(replayed.root_id)

    # Add survivor options that were merged
    survivor_ids = [
        nid
        for nid, node in replayed.nodes.items()
        if node.kind == "option" and node.status == "merged"
    ]
    # Sort by depth, then by node_id for determinism
    survivor_ids.sort(key=lambda nid: (replayed.nodes[nid].depth, nid))
    path_node_ids.extend(survivor_ids)

    # Add merged node if exists
    if replayed.merged_id and replayed.merged_id not in path_node_ids:
        path_node_ids.append(replayed.merged_id)

    # Build NodeSummary for each node on path
    for node_id in path_node_ids:
        if node_id not in replayed.nodes:
            continue

        node = replayed.nodes[node_id]

        # Get score info
        score_info = node_scores.get(node_id, {})

        # Get evidence sources (limited)
        sources = node_evidence.get(node_id, [])[:MAX_EVIDENCE_SOURCES]

        # Get claim counts
        claim_counts: dict[str, int] = {}
        if node.claim_summary:
            claim_counts = dict(node.claim_summary)

        node_summary = NodeSummary(
            node_id=node_id,
            label=node.label,
            kind=node.kind,
            depth=node.depth,
            score=score_info.get("score"),
            score_breakdown=score_info.get("breakdown", {}),
            claim_counts=claim_counts,
            evidence_sources=sources,
            evidence_count=len(node_evidence.get(node_id, [])),
            objections=node_objections.get(node_id, ObjectionSummary()),
        )
        summary.selected_path.append(node_summary)


def _extract_from_artifact(summary: RunSummary, artifact: dict[str, Any]) -> None:
    """Extract final outcome information from artifact.json."""
    summary.recommendation = artifact.get("recommendation", "")
    summary.confidence = artifact.get("confidence", 0.0)
    summary.decision_explanation = artifact.get("decision_explanation", {})

    # Extract key_claims with citations
    raw_claims = artifact.get("key_claims", [])
    for raw in raw_claims[:MAX_KEY_CLAIMS]:
        citations = []
        for cit in raw.get("citations", []):
            excerpt = cit.get("excerpt", "")
            if len(excerpt) > MAX_EXCERPT_LENGTH:
                excerpt = excerpt[:MAX_EXCERPT_LENGTH] + "..."
            citations.append(
                Citation(
                    evidence_id=cit.get("evidence_id", ""),
                    source=cit.get("source", ""),
                    excerpt=excerpt,
                )
            )

        key_claim = KeyClaim(
            claim_id=raw.get("claim_id", ""),
            claim_type=raw.get("type", ""),
            text=raw.get("text", ""),
            status=raw.get("status", ""),
            citations=citations,
        )
        summary.key_claims.append(key_claim)
