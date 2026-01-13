"""Gate trigger predicates.

Provides helper functions to determine when gates should fire.
For v1, gates always fire when enabled (simple implementation).
"""

from typing import Any

# Default threshold for near-tie detection in tradeoff gate
DEFAULT_TIE_THRESHOLD = 0.15


def needs_scope_gate(
    plan_output: dict[str, Any],  # noqa: ARG001
    gates_enabled: bool = True,
) -> bool:
    """Determine if scope gate should be triggered.

    For v1, always triggers if gates are enabled.

    Args:
        plan_output: Output from the planner (contains branches).
        gates_enabled: Whether gates are enabled for this run.

    Returns:
        True if scope gate should fire.
    """
    # V1: Always trigger scope gate if enabled
    # Future: Could check plan_output for ambiguous branches, low confidence, etc.
    return gates_enabled


def needs_tradeoff_gate(
    scores: list[tuple[str, float]],
    gates_enabled: bool = True,
    tie_threshold: float = DEFAULT_TIE_THRESHOLD,
) -> bool:
    """Determine if tradeoff gate should be triggered.

    Triggers when at least 2 candidates exist and the top two
    scores are within tie_threshold of each other.

    Args:
        scores: List of (node_id, score) tuples, sorted by score descending.
        gates_enabled: Whether gates are enabled for this run.
        tie_threshold: Score difference below which a tie is detected.

    Returns:
        True if tradeoff gate should fire.
    """
    if not gates_enabled:
        return False

    # Need at least 2 candidates to have a tie
    if len(scores) < 2:
        return False

    # Scores should be sorted descending by score
    # Check difference between top two
    _, score1 = scores[0]
    _, score2 = scores[1]
    score_diff = abs(score1 - score2)

    return score_diff <= tie_threshold


def needs_final_signoff_gate(
    final_artifact: dict[str, Any],  # noqa: ARG001
    gates_enabled: bool = True,
) -> bool:
    """Determine if final sign-off gate should be triggered.

    For v1, always triggers if gates are enabled.

    Args:
        final_artifact: The constructed final artifact.
        gates_enabled: Whether gates are enabled for this run.

    Returns:
        True if final sign-off gate should fire.
    """
    # V1: Always trigger final gate if enabled
    # Future: Could check final_artifact for unresolved objections, low confidence, etc.
    return gates_enabled
