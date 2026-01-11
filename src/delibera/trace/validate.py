"""Trace validation for replay.

Validates that a trace contains all required events and is structurally complete.
"""

from typing import Any

# Required event types for a complete trace
REQUIRED_EVENT_TYPES = {
    "run_start",
    "node_created",
    "work_output",
    "expand",
    "claim_validation_report",
    "prune",
    "reduce",
    "final_artifact_written",
    "run_end",
}


def validate_trace(events: list[dict[str, Any]]) -> list[str]:
    """Validate a trace for completeness and structural integrity.

    Args:
        events: List of trace event dictionaries.

    Returns:
        List of error messages. Empty list means trace is valid.
    """
    errors: list[str] = []

    if not events:
        errors.append("Trace is empty")
        return errors

    # Check run_start is first
    if events[0].get("event_type") != "run_start":
        errors.append("First event must be run_start")

    # Check run_end is last
    if events[-1].get("event_type") != "run_end":
        errors.append("Last event must be run_end")

    # Check run_end status
    last_event = events[-1]
    if last_event.get("event_type") == "run_end":
        status = last_event.get("payload", {}).get("status")
        if status == "failed":
            error_msg = last_event.get("payload", {}).get("error", "unknown error")
            errors.append(f"Run failed with error: {error_msg}")

    # Collect event types and check for required ones
    event_types = {e.get("event_type") for e in events}
    missing_types = REQUIRED_EVENT_TYPES - event_types
    if missing_types:
        errors.append(f"Missing required event types: {sorted(missing_types)}")

    # Check run_id consistency
    run_ids = {e.get("run_id") for e in events if e.get("run_id")}
    if len(run_ids) > 1:
        errors.append(f"Inconsistent run_ids found: {sorted(str(r) for r in run_ids)}")

    # Validate node structure
    node_errors = _validate_node_structure(events)
    errors.extend(node_errors)

    return errors


def _validate_node_structure(events: list[dict[str, Any]]) -> list[str]:
    """Validate node creation and parent-child relationships.

    Args:
        events: List of trace event dictionaries.

    Returns:
        List of error messages related to node structure.
    """
    errors: list[str] = []
    created_nodes: set[str] = set()
    root_nodes: list[str] = []

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == "node_created":
            node_id = payload.get("node_id")
            parent_id = payload.get("parent_id")
            kind = payload.get("kind")

            if not node_id:
                errors.append("node_created event missing node_id")
                continue

            if node_id in created_nodes:
                errors.append(f"Duplicate node_id: {node_id}")
            created_nodes.add(node_id)

            if kind == "root":
                root_nodes.append(node_id)
            elif parent_id and parent_id not in created_nodes:
                errors.append(f"Node {node_id} references unknown parent {parent_id}")

    # Check exactly one root
    if len(root_nodes) == 0:
        errors.append("No root node found")
    elif len(root_nodes) > 1:
        errors.append(f"Multiple root nodes found: {root_nodes}")

    return errors


def validate_artifact_derivable(
    replayed_artifact: dict[str, Any],
    stored_artifact: dict[str, Any],
) -> list[str]:
    """Check that stored artifact is consistent with replayed state.

    Args:
        replayed_artifact: Artifact derived from replay.
        stored_artifact: Artifact loaded from artifact.json.

    Returns:
        List of inconsistency messages. Empty means consistent.
    """
    errors: list[str] = []

    # Check question matches
    replayed_q = replayed_artifact.get("question", "")
    stored_q = stored_artifact.get("question", "")
    if replayed_q != stored_q:
        errors.append(f"Question mismatch: replayed={replayed_q!r}, stored={stored_q!r}")

    # Check recommendation is non-empty
    if not stored_artifact.get("recommendation"):
        errors.append("Stored artifact has empty recommendation")

    # Check required keys exist in stored artifact
    required_keys = [
        "question",
        "recommendation",
        "options_considered",
        "claim_check_summary",
        "open_questions",
    ]
    for key in required_keys:
        if key not in stored_artifact:
            errors.append(f"Stored artifact missing required key: {key}")

    # Check claim_check_summary structure
    summary = stored_artifact.get("claim_check_summary", {})
    for field in ["supported", "weak", "unsupported"]:
        if field not in summary:
            errors.append(f"claim_check_summary missing field: {field}")

    return errors
