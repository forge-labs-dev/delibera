"""Gate data models for user interaction.

Defines the structured types for user gates:
- GateType: enum of supported gate types
- GateSummary: structured summary presented to user
- GateResponse: structured user response
- AllowedAction: actions available at each gate
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GateType(Enum):
    """Supported gate types for v1.

    scope: After PLAN, before EXPAND - clarify scope and branches
    final_signoff: After convergence, before artifact write - approve final output
    """

    SCOPE = "scope"
    FINAL_SIGNOFF = "final_signoff"


class AllowedAction(Enum):
    """Actions available at gates.

    Not all actions are available at all gate types.
    """

    APPROVE = "approve"
    ABORT = "abort"
    VETO_BRANCHES = "veto_branches"
    REQUEST_MORE_ANALYSIS = "request_more_analysis"  # Not implemented in v1


# Map gate types to their allowed actions
GATE_ALLOWED_ACTIONS: dict[GateType, list[AllowedAction]] = {
    GateType.SCOPE: [
        AllowedAction.APPROVE,
        AllowedAction.VETO_BRANCHES,
        AllowedAction.ABORT,
    ],
    GateType.FINAL_SIGNOFF: [
        AllowedAction.APPROVE,
        AllowedAction.ABORT,
        # REQUEST_MORE_ANALYSIS not implemented in v1
    ],
}


@dataclass
class GateSummary:
    """Structured summary presented to user at a gate.

    The summary varies by gate type but always includes
    the gate_type and allowed_actions.
    """

    gate_type: GateType
    allowed_actions: list[AllowedAction]

    # Scope gate fields
    interpreted_question: str | None = None
    proposed_branches: list[str] = field(default_factory=list)

    # Final sign-off gate fields
    recommendation: str | None = None
    claim_check_summary: dict[str, int] | None = None
    open_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trace logging."""
        result: dict[str, Any] = {
            "gate_type": self.gate_type.value,
            "allowed_actions": [a.value for a in self.allowed_actions],
        }

        if self.interpreted_question is not None:
            result["interpreted_question"] = self.interpreted_question
        if self.proposed_branches:
            result["proposed_branches"] = self.proposed_branches
        if self.recommendation is not None:
            result["recommendation"] = self.recommendation
        if self.claim_check_summary is not None:
            result["claim_check_summary"] = self.claim_check_summary
        if self.open_questions:
            result["open_questions"] = self.open_questions

        return result


@dataclass
class GateResponse:
    """Structured user response to a gate.

    action: The action taken (must be in gate's allowed_actions)
    parameters: Action-specific parameters (e.g., branches to veto)
    """

    action: AllowedAction
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trace logging."""
        result: dict[str, Any] = {
            "action": self.action.value,
        }
        if self.parameters:
            result["parameters"] = self.parameters
        return result


@dataclass
class GateEvent:
    """A gate event for tracing.

    Captures the gate trigger and response for the trace log.
    """

    gate_type: GateType
    summary: GateSummary
    node_id: str | None = None  # Applicable for some gates

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trace payload."""
        result: dict[str, Any] = {
            "gate_type": self.gate_type.value,
            "summary": self.summary.to_dict(),
        }
        if self.node_id:
            result["node_id"] = self.node_id
        return result


class GateError(Exception):
    """Raised when gate processing fails."""

    pass


class GateAborted(Exception):
    """Raised when user aborts through a gate."""

    def __init__(self, gate_type: GateType, message: str = "Run aborted by user"):
        self.gate_type = gate_type
        self.message = message
        super().__init__(message)


def validate_response(gate_type: GateType, response: GateResponse) -> list[str]:
    """Validate a gate response.

    Args:
        gate_type: The type of gate being responded to.
        response: The user's response.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    # Check action is allowed for this gate type
    allowed = GATE_ALLOWED_ACTIONS.get(gate_type, [])
    if response.action not in allowed:
        errors.append(
            f"Action '{response.action.value}' not allowed for gate '{gate_type.value}'. "
            f"Allowed: {[a.value for a in allowed]}"
        )

    # Validate action-specific parameters
    if response.action == AllowedAction.VETO_BRANCHES:
        branches = response.parameters.get("branches", [])
        if not isinstance(branches, list):
            errors.append("veto_branches requires 'branches' parameter as a list")
        elif not branches:
            errors.append("veto_branches requires at least one branch to veto")

    if response.action == AllowedAction.REQUEST_MORE_ANALYSIS:
        errors.append("request_more_analysis is not implemented in v1")

    return errors
