"""Metric extraction from completed Delibera runs.

Extracts standardized metrics from artifact.json and trace.jsonl.
"""

from pathlib import Path
from typing import Any

from delibera.trace.reader import load_artifact, load_trace


def extract_metrics(run_dir: Path) -> dict[str, int | float | bool]:
    """Extract standardized metrics from a completed run.

    Reads artifact.json and trace.jsonl to compute all metrics.

    Args:
        run_dir: Path to the run directory.

    Returns:
        Dictionary of metric names to values.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    artifact_path = run_dir / "artifact.json"
    trace_path = run_dir / "trace.jsonl"

    # Load artifact
    artifact = load_artifact(artifact_path)

    # Load trace events
    events = load_trace(trace_path)

    # Extract metrics from both sources
    metrics: dict[str, int | float | bool] = {}

    # From artifact
    _extract_artifact_metrics(artifact, metrics)

    # From trace
    _extract_trace_metrics(events, metrics)

    return metrics


def _extract_artifact_metrics(
    artifact: dict[str, Any],
    metrics: dict[str, int | float | bool],
) -> None:
    """Extract metrics from artifact.json.

    Args:
        artifact: Parsed artifact dictionary.
        metrics: Output dictionary to populate.
    """
    # Claim check summary
    claim_summary = artifact.get("claim_check_summary", {})
    metrics["supported_claims_count"] = claim_summary.get("supported", 0)
    metrics["weak_claims_count"] = claim_summary.get("weak", 0)
    metrics["unsupported_claims_count"] = claim_summary.get("unsupported", 0)

    # Key claims with citations
    key_claims = artifact.get("key_claims", [])
    metrics["key_claims_count"] = len(key_claims)

    # Check if run was aborted (no recommendation)
    recommendation = artifact.get("recommendation", "")
    metrics["run_aborted"] = not bool(recommendation)


def _extract_trace_metrics(
    events: list[dict[str, Any]],
    metrics: dict[str, int | float | bool],
) -> None:
    """Extract metrics from trace.jsonl.

    Args:
        events: List of trace event dictionaries.
        metrics: Output dictionary to populate.
    """
    # Initialize counters
    evidence_count = 0
    blocking_open = 0
    blocking_accepted = 0
    nonblocking_open = 0
    tool_calls = 0

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == "evidence_added":
            evidence_count += 1

        elif event_type == "objection_added":
            objection = payload.get("objection", {})
            severity = objection.get("severity", "nonblocking")
            status = objection.get("status", "open")

            if severity == "blocking":
                if status == "open":
                    blocking_open += 1
                elif status == "accepted":
                    blocking_accepted += 1
            else:
                if status == "open":
                    nonblocking_open += 1

        elif event_type == "objection_resolved":
            # An objection that was open is now resolved
            objection = payload.get("objection", {})
            severity = objection.get("severity", "nonblocking")
            old_status = payload.get("old_status", "open")

            # Decrement open count if it was open before
            if old_status == "open":
                if severity == "blocking":
                    blocking_open = max(0, blocking_open - 1)
                else:
                    nonblocking_open = max(0, nonblocking_open - 1)

        elif event_type == "tool_call_executed":
            tool_calls += 1

        elif event_type == "run_end":
            # Check run status
            status = payload.get("status", "completed")
            if status == "aborted":
                metrics["run_aborted"] = True

    # Set final counts
    metrics["evidence_count"] = evidence_count
    metrics["blocking_objections_open"] = blocking_open
    metrics["blocking_objections_accepted"] = blocking_accepted
    metrics["nonblocking_objections_open"] = nonblocking_open
    metrics["total_tool_calls"] = tool_calls


def get_metric_names() -> list[str]:
    """Return list of all metric names computed by extract_metrics.

    Returns:
        Sorted list of metric names.
    """
    return sorted(
        [
            "supported_claims_count",
            "weak_claims_count",
            "unsupported_claims_count",
            "key_claims_count",
            "evidence_count",
            "blocking_objections_open",
            "blocking_objections_accepted",
            "nonblocking_objections_open",
            "total_tool_calls",
            "run_aborted",
        ]
    )
