"""Gate trigger predicates.

Provides helper functions to determine when gates should fire.
For v1, gates always fire when enabled (simple implementation).
"""

from typing import Any


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
