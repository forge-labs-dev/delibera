"""Gate response application.

Applies validated gate responses to update run state/constraints.
"""

from typing import Any

from delibera.gates.models import (
    AllowedAction,
    GateAborted,
    GateResponse,
    GateType,
)


def apply_scope_response(
    response: GateResponse,
    branch_labels: list[str],
) -> list[str]:
    """Apply scope gate response to branch labels.

    Args:
        response: The validated gate response.
        branch_labels: The original branch labels from planner.

    Returns:
        Updated branch labels after applying response.

    Raises:
        GateAborted: If user chose to abort.
    """
    if response.action == AllowedAction.ABORT:
        raise GateAborted(GateType.SCOPE, "Run aborted at scope gate")

    if response.action == AllowedAction.APPROVE:
        return branch_labels

    if response.action == AllowedAction.VETO_BRANCHES:
        vetoed = set(response.parameters.get("branches", []))
        remaining = [label for label in branch_labels if label not in vetoed]

        # Ensure at least one branch remains
        if not remaining:
            # If all branches vetoed, keep original (validation should prevent this)
            return branch_labels

        return remaining

    # Unknown action - return unchanged
    return branch_labels


def apply_final_signoff_response(response: GateResponse) -> bool:
    """Apply final sign-off gate response.

    Args:
        response: The validated gate response.

    Returns:
        True if approved, False otherwise.

    Raises:
        GateAborted: If user chose to abort.
    """
    if response.action == AllowedAction.ABORT:
        raise GateAborted(GateType.FINAL_SIGNOFF, "Run aborted at final sign-off gate")

    if response.action == AllowedAction.APPROVE:
        return True

    # REQUEST_MORE_ANALYSIS not implemented in v1
    # Treat as approval for now (should be caught by validation)
    return True


def get_response_changes(
    gate_type: GateType,  # noqa: ARG001
    response: GateResponse,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the constraint changes resulting from a gate response.

    Used for trace logging to show what changed.

    Args:
        gate_type: The type of gate.
        response: The gate response.
        context: Optional context (e.g., original branches for veto).

    Returns:
        Dictionary describing the changes.
    """
    changes: dict[str, Any] = {
        "action": response.action.value,
    }

    if response.action == AllowedAction.VETO_BRANCHES:
        vetoed = response.parameters.get("branches", [])
        changes["vetoed_branches"] = vetoed

        if context and "original_branches" in context:
            original = context["original_branches"]
            remaining = [b for b in original if b not in set(vetoed)]
            changes["remaining_branches"] = remaining

    return changes
