"""Trace replay for reconstructing deliberation runs.

Reconstructs the full run state from trace events without calling agents.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from delibera.trace.reader import load_artifact, load_trace
from delibera.trace.validate import validate_artifact_derivable, validate_trace


@dataclass
class ReplayedNodeState:
    """Reconstructed state for a single node."""

    node_id: str
    parent_id: str | None
    kind: Literal["root", "option", "merged"]
    depth: int
    label: str = ""
    artifact: dict[str, Any] = field(default_factory=dict)
    claim_summary: dict[str, int] | None = None
    status: Literal["active", "pruned", "merged"] = "active"


@dataclass
class ReplayedRun:
    """Complete reconstructed run from trace."""

    run_id: str
    question: str
    created_at: str
    nodes: dict[str, ReplayedNodeState] = field(default_factory=dict)
    root_id: str = ""
    merged_id: str | None = None
    final_artifact: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True if replay completed without errors."""
        return len(self.errors) == 0


def replay_run(
    events: list[dict[str, Any]],
    artifact_path: Path | None = None,
) -> ReplayedRun:
    """Reconstruct run state from trace events.

    This function does NOT call any agents - it uses only trace data.

    Args:
        events: List of trace event dictionaries.
        artifact_path: Optional path to artifact.json. If not provided,
            will try to derive from final_artifact_written event.

    Returns:
        ReplayedRun with reconstructed state.
    """
    # Initialize with defaults
    replayed = ReplayedRun(
        run_id="",
        question="",
        created_at="",
    )

    # Validate trace first
    validation_errors = validate_trace(events)
    if validation_errors:
        replayed.errors.extend(validation_errors)
        # Continue with partial reconstruction if possible

    # Process events in order
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == "run_start":
            _handle_run_start(replayed, event, payload)

        elif event_type == "node_created":
            _handle_node_created(replayed, payload)

        elif event_type == "work_output":
            _handle_work_output(replayed, payload)

        elif event_type == "claim_validation_report":
            _handle_claim_validation(replayed, payload)

        elif event_type == "prune":
            _handle_prune(replayed, payload)

        elif event_type == "reduce":
            _handle_reduce(replayed, payload)

        elif event_type == "final_artifact_written":
            _handle_final_artifact_written(replayed, payload, artifact_path)

    # Derive final artifact from merged node if not loaded
    if not replayed.final_artifact and replayed.merged_id:
        replayed.final_artifact = _derive_artifact(replayed)
        replayed.warnings.append(
            "Final artifact derived from merged node (artifact.json not loaded)"
        )

    return replayed


def _handle_run_start(
    replayed: ReplayedRun,
    event: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """Process run_start event."""
    replayed.run_id = event.get("run_id", "")
    replayed.question = payload.get("question", "")
    replayed.created_at = payload.get("created_at", "")


def _handle_node_created(
    replayed: ReplayedRun,
    payload: dict[str, Any],
) -> None:
    """Process node_created event."""
    node_id = payload.get("node_id", "")
    if not node_id:
        return

    node = ReplayedNodeState(
        node_id=node_id,
        parent_id=payload.get("parent_id"),
        kind=payload.get("kind", "option"),
        depth=payload.get("depth", 0),
        label=payload.get("label", ""),
    )
    replayed.nodes[node_id] = node

    if node.kind == "root":
        replayed.root_id = node_id
    elif node.kind == "merged":
        replayed.merged_id = node_id


def _handle_work_output(
    replayed: ReplayedRun,
    payload: dict[str, Any],
) -> None:
    """Process work_output event."""
    node_id = payload.get("node_id", "")
    step = payload.get("step", "")
    output = payload.get("output", {})

    if node_id not in replayed.nodes:
        replayed.warnings.append(f"work_output for unknown node: {node_id}")
        return

    # Only PROPOSE outputs contain node artifacts
    if step == "PROPOSE":
        # The proposer output IS the artifact
        replayed.nodes[node_id].artifact = output


def _handle_claim_validation(
    replayed: ReplayedRun,
    payload: dict[str, Any],
) -> None:
    """Process claim_validation_report event."""
    node_id = payload.get("node_id", "")

    if node_id not in replayed.nodes:
        replayed.warnings.append(f"claim_validation_report for unknown node: {node_id}")
        return

    replayed.nodes[node_id].claim_summary = {
        "supported": payload.get("supported", 0),
        "weak": payload.get("weak", 0),
        "unsupported": payload.get("unsupported", 0),
    }


def _handle_prune(
    replayed: ReplayedRun,
    payload: dict[str, Any],
) -> None:
    """Process prune event."""
    pruned_ids = payload.get("pruned_ids", [])

    for node_id in pruned_ids:
        if node_id in replayed.nodes:
            replayed.nodes[node_id].status = "pruned"


def _handle_reduce(
    replayed: ReplayedRun,
    payload: dict[str, Any],
) -> None:
    """Process reduce event."""
    survivor_ids = payload.get("survivor_ids", [])
    merged_node_id = payload.get("merged_node_id", "")
    merged_artifact = payload.get("merged_artifact", {})

    # Mark survivors as merged
    for node_id in survivor_ids:
        if node_id in replayed.nodes:
            replayed.nodes[node_id].status = "merged"

    # Update merged node artifact
    if merged_node_id in replayed.nodes:
        replayed.nodes[merged_node_id].artifact = merged_artifact
        replayed.merged_id = merged_node_id


def _handle_final_artifact_written(
    replayed: ReplayedRun,
    payload: dict[str, Any],
    artifact_path: Path | None,
) -> None:
    """Process final_artifact_written event and load artifact."""
    if artifact_path and artifact_path.exists():
        try:
            replayed.final_artifact = load_artifact(artifact_path)
        except Exception as e:
            replayed.warnings.append(f"Failed to load artifact.json: {e}")
    else:
        # Try to derive path from event
        event_path = payload.get("artifact_path", "")
        if event_path:
            path = Path(event_path)
            if path.exists():
                try:
                    replayed.final_artifact = load_artifact(path)
                except Exception as e:
                    replayed.warnings.append(f"Failed to load artifact from event path: {e}")


def _derive_artifact(replayed: ReplayedRun) -> dict[str, Any]:
    """Derive final artifact from replay state when artifact.json not available."""
    artifact: dict[str, Any] = {
        "question": replayed.question,
    }

    # Get merged node
    if replayed.merged_id and replayed.merged_id in replayed.nodes:
        merged_node = replayed.nodes[replayed.merged_id]
        artifact["recommendation"] = merged_node.artifact.get("proposal", "")
        artifact["summary"] = merged_node.artifact.get("summary", "")

    # Get options considered from option nodes
    options = [node.label for node in replayed.nodes.values() if node.kind == "option"]
    artifact["options_considered"] = options

    # Get survivors and pruned
    survivors = [
        node.label
        for node in replayed.nodes.values()
        if node.kind == "option" and node.status == "merged"
    ]
    pruned = [
        node.label
        for node in replayed.nodes.values()
        if node.kind == "option" and node.status == "pruned"
    ]
    artifact["survivors"] = survivors
    artifact["pruned"] = pruned

    # Aggregate claim summary from survivor nodes
    total_supported = 0
    total_weak = 0
    total_unsupported = 0

    for node in replayed.nodes.values():
        if node.kind == "option" and node.status == "merged" and node.claim_summary:
            total_supported += node.claim_summary.get("supported", 0)
            total_weak += node.claim_summary.get("weak", 0)
            total_unsupported += node.claim_summary.get("unsupported", 0)

    artifact["claim_check_summary"] = {
        "supported": total_supported,
        "weak": total_weak,
        "unsupported": total_unsupported,
    }

    # Generate open_questions
    open_questions: list[str] = []
    if total_weak > 0 or total_unsupported > 0:
        open_questions.append("Add evidence to support inference claims.")
    artifact["open_questions"] = open_questions

    return artifact


def replay_from_directory(run_dir: Path) -> ReplayedRun:
    """Convenience function to replay from a run directory.

    Args:
        run_dir: Path to the run directory containing trace.jsonl and artifact.json.

    Returns:
        ReplayedRun with reconstructed state.

    Raises:
        FileNotFoundError: If trace.jsonl does not exist.
    """
    trace_path = run_dir / "trace.jsonl"
    artifact_path = run_dir / "artifact.json"

    events = load_trace(trace_path)

    return replay_run(
        events,
        artifact_path=artifact_path if artifact_path.exists() else None,
    )


def verify_replay(replayed: ReplayedRun, stored_artifact: dict[str, Any]) -> list[str]:
    """Verify replay is consistent with stored artifact.

    Args:
        replayed: The replayed run state.
        stored_artifact: The artifact loaded from disk.

    Returns:
        List of inconsistency messages. Empty means consistent.
    """
    # Derive artifact from replay for comparison
    derived = _derive_artifact(replayed)

    return validate_artifact_derivable(derived, stored_artifact)
