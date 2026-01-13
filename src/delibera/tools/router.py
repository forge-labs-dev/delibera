"""Tool router for mediating all tool calls.

The ToolRouter is the single point of access for tool execution.
All tool calls flow through: Agent -> Engine -> ToolRouter -> Tool

The router:
- Validates requests against ToolSpec
- Consults PolicyEngine
- Executes or denies the call
- Emits trace events
"""

from collections.abc import Callable
from typing import Any

from delibera.tools.policy import PolicyContext, PolicyDecision, PolicyEngine
from delibera.tools.registry import ToolRegistry
from delibera.tools.spec import ToolDenied, ToolExecutionError

# Type alias for trace event emitter callback
TraceEmitter = Callable[[str, dict[str, Any]], None]


class ToolRouter:
    """Routes tool calls through policy and execution.

    The router is the gatekeeper for all tool access.
    Agents never bypass the router.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy_engine: PolicyEngine,
        trace_emitter: TraceEmitter | None = None,
    ) -> None:
        """Initialize the tool router.

        Args:
            registry: The tool registry containing available tools.
            policy_engine: The policy engine for access control.
            trace_emitter: Optional callback for emitting trace events.
                Signature: (event_type: str, payload: dict) -> None
        """
        self._registry = registry
        self._policy = policy_engine
        self._trace_emitter = trace_emitter

    def call(
        self,
        role: str,
        step: str,
        tool_name: str,
        tool_input: dict[str, Any],
        node_id: str | None = None,
        policy_context: PolicyContext | None = None,
    ) -> dict[str, Any]:
        """Execute a tool call through policy evaluation.

        Args:
            role: The agent role making the request.
            step: The current protocol step.
            tool_name: The name of the tool to call.
            tool_input: The input to pass to the tool.
            node_id: Optional node ID for scoped tool calls.
            policy_context: Optional policy context with allowed evidence sources.

        Returns:
            The tool output dictionary.

        Raises:
            ToolDenied: If the policy denies the tool call.
            KeyError: If the tool is not registered.
            ToolExecutionError: If the tool execution fails.
        """
        # Emit tool_call_requested
        self._emit_requested(role, step, tool_name, tool_input, node_id)

        # Look up the tool
        if not self._registry.has(tool_name):
            reason = f"Tool '{tool_name}' is not registered"
            self._emit_denied(role, step, tool_name, tool_input, node_id, reason, None)
            raise KeyError(reason)

        tool = self._registry.get(tool_name)

        # Evaluate policy (with evidence-local support)
        decision = self._policy.evaluate(
            role=role,
            step=step,
            tool_name=tool_name,
            is_discovery=tool.is_discovery,
            risk_level=tool.risk_level,
            tool_input=tool_input,
            context=policy_context,
        )

        if not decision.allow:
            self._emit_denied(role, step, tool_name, tool_input, node_id, decision.reason, decision)
            raise ToolDenied(tool_name, decision.reason)

        # Validate input
        try:
            tool.validate_input(tool_input)
        except ValueError as e:
            reason = f"Input validation failed: {e}"
            self._emit_denied(role, step, tool_name, tool_input, node_id, reason, decision)
            raise ToolExecutionError(tool_name, reason) from e

        # Execute tool
        try:
            output = tool.execute(tool_input)
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(tool_name, str(e)) from e

        # Record call for budget tracking
        self._policy.record_call(tool_name)

        # Emit tool_call_executed
        self._emit_executed(role, step, tool_name, tool_input, node_id, output, decision)

        return output

    def _emit_requested(
        self,
        role: str,
        step: str,
        tool_name: str,
        tool_input: dict[str, Any],
        node_id: str | None,
    ) -> None:
        """Emit tool_call_requested trace event."""
        if self._trace_emitter is None:
            return

        payload: dict[str, Any] = {
            "role": role,
            "step": step,
            "tool_name": tool_name,
            "input": tool_input,
        }
        if node_id:
            payload["node_id"] = node_id

        self._trace_emitter("tool_call_requested", payload)

    def _emit_executed(
        self,
        role: str,
        step: str,
        tool_name: str,
        tool_input: dict[str, Any],
        node_id: str | None,
        output: dict[str, Any],
        decision: PolicyDecision,
    ) -> None:
        """Emit tool_call_executed trace event."""
        if self._trace_emitter is None:
            return

        payload: dict[str, Any] = {
            "role": role,
            "step": step,
            "tool_name": tool_name,
            "input": tool_input,
            "output": output,
            "policy_decision": decision.to_dict(),
        }
        if node_id:
            payload["node_id"] = node_id

        self._trace_emitter("tool_call_executed", payload)

    def _emit_denied(
        self,
        role: str,
        step: str,
        tool_name: str,
        tool_input: dict[str, Any],
        node_id: str | None,
        reason: str,
        decision: PolicyDecision | None,
    ) -> None:
        """Emit tool_call_denied trace event."""
        if self._trace_emitter is None:
            return

        payload: dict[str, Any] = {
            "role": role,
            "step": step,
            "tool_name": tool_name,
            "input": tool_input,
            "reason": reason,
        }
        if node_id:
            payload["node_id"] = node_id
        if decision:
            payload["policy_decision"] = decision.to_dict()

        self._trace_emitter("tool_call_denied", payload)
