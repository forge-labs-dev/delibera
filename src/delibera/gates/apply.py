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
from delibera.scoring.weights import ScoreWeights


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


def apply_tradeoff_response(
    response: GateResponse,
    current_weights: ScoreWeights,
) -> ScoreWeights:
    """Apply tradeoff gate response to update scoring weights.

    Args:
        response: The validated gate response.
        current_weights: The current scoring weights.

    Returns:
        Updated ScoreWeights (may be same as current if approve_default).

    Raises:
        GateAborted: If user chose to abort.
    """
    if response.action == AllowedAction.ABORT:
        raise GateAborted(GateType.TRADEOFF, "Run aborted at tradeoff gate")

    if response.action == AllowedAction.APPROVE_DEFAULT:
        return current_weights

    if response.action == AllowedAction.SET_WEIGHTS:
        # Merge user-provided weights with current weights
        weight_overrides = response.parameters.get("weights", {})
        weights_dict = current_weights.to_dict()
        weights_dict.update(weight_overrides)
        return ScoreWeights.from_dict(weights_dict)

    # Unknown action - return unchanged
    return current_weights


def apply_final_signoff_response(
    response: GateResponse,
    has_blocking_objections: bool = False,
) -> tuple[bool, list[str]]:
    """Apply final sign-off gate response.

    Args:
        response: The validated gate response.
        has_blocking_objections: Whether there are open blocking objections.

    Returns:
        Tuple of (approved, accepted_objection_ids).
        - approved: True if run should proceed to finalize.
        - accepted_objection_ids: List of objection IDs that were accepted.

    Raises:
        GateAborted: If user chose to abort or tried to approve with blockers.
    """
    if response.action == AllowedAction.ABORT:
        raise GateAborted(GateType.FINAL_SIGNOFF, "Run aborted at final sign-off gate")

    if response.action == AllowedAction.APPROVE:
        if has_blocking_objections:
            raise GateAborted(
                GateType.FINAL_SIGNOFF,
                "Cannot approve with open blocking objections. Use 'accept_objections' first.",
            )
        return True, []

    if response.action == AllowedAction.ACCEPT_OBJECTIONS:
        objection_ids = response.parameters.get("objection_ids", [])
        # Return the accepted IDs - caller will mark them as accepted
        return False, objection_ids

    # REQUEST_MORE_ANALYSIS not implemented in v1
    # Treat as approval for now (should be caught by validation)
    return True, []


def get_response_changes(
    response: GateResponse,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get the constraint changes resulting from a gate response.

    Used for trace logging to show what changed.

    Args:
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

    if response.action == AllowedAction.ACCEPT_OBJECTIONS:
        accepted_ids = response.parameters.get("objection_ids", [])
        changes["accepted_objection_ids"] = accepted_ids

    return changes
