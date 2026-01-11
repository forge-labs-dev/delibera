"""Gates subsystem for Delibera.

Provides structured user interaction through gates.
"""

from delibera.gates.apply import (
    apply_final_signoff_response,
    apply_scope_response,
    get_response_changes,
)
from delibera.gates.handler import (
    AutoApproveGateHandler,
    CLIGateHandler,
    GateHandler,
    ScriptedGateHandler,
)
from delibera.gates.models import (
    GATE_ALLOWED_ACTIONS,
    AllowedAction,
    GateAborted,
    GateError,
    GateResponse,
    GateSummary,
    GateType,
    validate_response,
)
from delibera.gates.predicates import needs_final_signoff_gate, needs_scope_gate

__all__ = [
    # Models
    "GateType",
    "AllowedAction",
    "GateSummary",
    "GateResponse",
    "GateError",
    "GateAborted",
    "GATE_ALLOWED_ACTIONS",
    "validate_response",
    # Handlers
    "GateHandler",
    "CLIGateHandler",
    "AutoApproveGateHandler",
    "ScriptedGateHandler",
    # Predicates
    "needs_scope_gate",
    "needs_final_signoff_gate",
    # Apply
    "apply_scope_response",
    "apply_final_signoff_response",
    "get_response_changes",
]
