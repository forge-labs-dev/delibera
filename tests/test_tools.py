"""Tests for the tools subsystem."""

import json
from pathlib import Path
from typing import Any

import pytest

from delibera.tools import (
    GlobalPolicy,
    PolicyDecision,
    PolicyEngine,
    RiskLevel,
    RolePolicy,
    StepPolicyOverride,
    ToolDenied,
    ToolExecutionError,
    ToolRegistry,
    ToolRouter,
    create_default_policy_engine,
    create_default_registry,
)
from delibera.tools.builtin.calculator import CalculatorTool


class TestCalculatorTool:
    """Tests for the calculator tool."""

    def test_simple_addition(self):
        """Test simple addition."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "1+2"})
        assert result == {"result": 3}

    def test_simple_subtraction(self):
        """Test simple subtraction."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "10-3"})
        assert result == {"result": 7}

    def test_simple_multiplication(self):
        """Test simple multiplication."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "4*5"})
        assert result == {"result": 20}

    def test_simple_division(self):
        """Test simple division."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "15/3"})
        assert result == {"result": 5.0}

    def test_operator_precedence(self):
        """Test operator precedence (multiplication before addition)."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "1+2*3"})
        assert result == {"result": 7}

    def test_parentheses(self):
        """Test parentheses override precedence."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "(1+2)*3"})
        assert result == {"result": 9}

    def test_power_operator(self):
        """Test power operator."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "2**3"})
        assert result == {"result": 8}

    def test_modulo_operator(self):
        """Test modulo operator."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "10%3"})
        assert result == {"result": 1}

    def test_negative_numbers(self):
        """Test negative numbers."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "-5+3"})
        assert result == {"result": -2}

    def test_float_numbers(self):
        """Test floating point numbers."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "0.75+0.05"})
        assert result == {"result": 0.8}

    def test_complex_expression(self):
        """Test complex nested expression."""
        calc = CalculatorTool()
        result = calc.execute({"expression": "((10+5)*2)/3"})
        assert result == {"result": 10.0}

    def test_division_by_zero_raises(self):
        """Test division by zero raises error."""
        calc = CalculatorTool()
        with pytest.raises(ToolExecutionError) as exc_info:
            calc.execute({"expression": "1/0"})
        assert "Division by zero" in str(exc_info.value)

    def test_invalid_expression_raises(self):
        """Test invalid expression raises error."""
        calc = CalculatorTool()
        with pytest.raises(ToolExecutionError):
            calc.execute({"expression": "1+"})

    def test_function_call_rejected(self):
        """Test function calls are rejected."""
        calc = CalculatorTool()
        with pytest.raises(ToolExecutionError):
            calc.execute({"expression": "eval('1+1')"})

    def test_variable_rejected(self):
        """Test variable references are rejected."""
        calc = CalculatorTool()
        with pytest.raises(ToolExecutionError):
            calc.execute({"expression": "x+1"})

    def test_missing_expression_raises(self):
        """Test missing expression field raises error."""
        calc = CalculatorTool()
        with pytest.raises(ValueError) as exc_info:
            calc.validate_input({})
        assert "expression" in str(exc_info.value)

    def test_empty_expression_raises(self):
        """Test empty expression raises error."""
        calc = CalculatorTool()
        with pytest.raises(ValueError):
            calc.validate_input({"expression": ""})

    def test_tool_properties(self):
        """Test tool has correct properties."""
        calc = CalculatorTool()
        assert calc.name == "calculator"
        assert calc.risk_level == RiskLevel.LOW
        assert calc.is_discovery is False


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_register_and_get(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        retrieved = registry.get("calculator")
        assert retrieved.name == "calculator"

    def test_duplicate_registration_raises(self):
        """Test duplicate registration raises error."""
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)

        with pytest.raises(ValueError) as exc_info:
            registry.register(calc)
        assert "already registered" in str(exc_info.value)

    def test_get_unregistered_raises(self):
        """Test getting unregistered tool raises error."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_has_tool(self):
        """Test checking if tool is registered."""
        registry = ToolRegistry()
        assert registry.has("calculator") is False

        registry.register(CalculatorTool())
        assert registry.has("calculator") is True

    def test_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()
        assert registry.list_tools() == []

        registry.register(CalculatorTool())
        assert "calculator" in registry.list_tools()

    def test_create_default_registry(self):
        """Test creating default registry includes calculator."""
        registry = create_default_registry()
        assert registry.has("calculator")


class TestPolicyEngine:
    """Tests for the policy engine."""

    def test_global_policy_empty_allows_all(self):
        """Test empty enabled_tools allows all tools."""
        engine = PolicyEngine(global_policy=GlobalPolicy())
        decision = engine.evaluate("any_role", "any_step", "calculator")
        assert decision.allow is True

    def test_global_policy_allowlist(self):
        """Test enabled_tools allowlist."""
        engine = PolicyEngine(global_policy=GlobalPolicy(enabled_tools={"calculator"}))

        allowed = engine.evaluate("any", "any", "calculator")
        assert allowed.allow is True

        denied = engine.evaluate("any", "any", "web.search")
        assert denied.allow is False
        assert "not in enabled_tools" in denied.reason

    def test_role_policy(self):
        """Test role-based policy."""
        engine = PolicyEngine(role_policy=RolePolicy(role_tools={"researcher": {"web.search"}}))

        # Role with restrictions
        denied = engine.evaluate("researcher", "any", "calculator")
        assert denied.allow is False
        assert "not in allowed tools" in denied.reason

        allowed = engine.evaluate("researcher", "any", "web.search")
        assert allowed.allow is True

        # Role without restrictions
        unrestricted = engine.evaluate("planner", "any", "calculator")
        assert unrestricted.allow is True

    def test_step_override_claim_check(self):
        """Test CLAIM_CHECK step denies discovery tools."""
        engine = PolicyEngine(
            step_override=StepPolicyOverride(evidence_local_steps={"CLAIM_CHECK"})
        )

        # Discovery tool denied in CLAIM_CHECK
        denied = engine.evaluate(
            "any", "CLAIM_CHECK", "web.search", is_discovery=True, risk_level=RiskLevel.MEDIUM
        )
        assert denied.allow is False
        assert "discovery" in denied.reason

        # High-risk tool denied in CLAIM_CHECK
        denied_high = engine.evaluate(
            "any", "CLAIM_CHECK", "code.exec", is_discovery=False, risk_level=RiskLevel.HIGH
        )
        assert denied_high.allow is False
        assert "high-risk" in denied_high.reason

        # Low-risk non-discovery tool allowed
        allowed = engine.evaluate(
            "any", "CLAIM_CHECK", "calculator", is_discovery=False, risk_level=RiskLevel.LOW
        )
        assert allowed.allow is True

    def test_budget_constraint(self):
        """Test budget constraint enforcement."""
        engine = PolicyEngine(global_policy=GlobalPolicy(max_calls=2))

        # First two calls allowed
        engine.record_call("calculator")
        engine.record_call("calculator")

        # Third call denied
        decision = engine.evaluate("any", "any", "calculator")
        assert decision.allow is False
        assert "max calls" in decision.reason

    def test_budget_tracking(self):
        """Test budget state tracking."""
        engine = PolicyEngine(global_policy=GlobalPolicy(max_calls=10))

        engine.record_call("calculator")
        engine.record_call("calculator")
        engine.record_call("other_tool")

        info = engine.get_budget_info()
        assert info["total_calls"] == 3
        assert info["calls_by_tool"]["calculator"] == 2
        assert info["calls_by_tool"]["other_tool"] == 1

    def test_create_default_policy_engine(self):
        """Test default policy engine allows calculator."""
        engine = create_default_policy_engine()
        decision = engine.evaluate("proposer", "PROPOSE", "calculator")
        assert decision.allow is True


class TestToolRouter:
    """Tests for the tool router."""

    def test_router_executes_allowed_tool(self):
        """Test router executes tool when policy allows."""
        registry = create_default_registry()
        policy = create_default_policy_engine()
        router = ToolRouter(registry, policy)

        result = router.call("proposer", "PROPOSE", "calculator", {"expression": "1+1"})
        assert result == {"result": 2}

    def test_router_denies_when_tool_not_registered(self):
        """Test router raises when tool is not registered."""
        registry = ToolRegistry()  # Empty registry
        policy = PolicyEngine()
        router = ToolRouter(registry, policy)

        with pytest.raises(KeyError) as exc_info:
            router.call("any", "any", "calculator", {"expression": "1+1"})
        assert "not registered" in str(exc_info.value)

    def test_router_denies_when_policy_denies(self):
        """Test router raises ToolDenied when policy denies."""
        registry = create_default_registry()
        policy = PolicyEngine(
            global_policy=GlobalPolicy(enabled_tools=set())  # No tools allowed
        )
        # Need to override empty set behavior
        policy.global_policy.enabled_tools = {"web.search"}  # Only web.search allowed
        router = ToolRouter(registry, policy)

        with pytest.raises(ToolDenied) as exc_info:
            router.call("any", "any", "calculator", {"expression": "1+1"})
        assert exc_info.value.tool_name == "calculator"
        assert "not in enabled_tools" in exc_info.value.reason

    def test_router_emits_trace_events(self):
        """Test router emits trace events."""
        registry = create_default_registry()
        policy = create_default_policy_engine()

        events: list[tuple[str, dict[str, Any]]] = []

        def emitter(event_type: str, payload: dict[str, Any]):
            events.append((event_type, payload))

        router = ToolRouter(registry, policy, trace_emitter=emitter)

        router.call("proposer", "PROPOSE", "calculator", {"expression": "2+2"}, node_id="node1")

        assert len(events) == 2

        # First event: requested
        assert events[0][0] == "tool_call_requested"
        assert events[0][1]["tool_name"] == "calculator"
        assert events[0][1]["role"] == "proposer"
        assert events[0][1]["step"] == "PROPOSE"
        assert events[0][1]["node_id"] == "node1"

        # Second event: executed
        assert events[1][0] == "tool_call_executed"
        assert events[1][1]["tool_name"] == "calculator"
        assert events[1][1]["output"] == {"result": 4}
        assert events[1][1]["policy_decision"]["allow"] is True

    def test_router_emits_denied_event(self):
        """Test router emits denied event on policy denial."""
        registry = create_default_registry()
        policy = PolicyEngine(global_policy=GlobalPolicy(enabled_tools={"web.search"}))

        events: list[tuple[str, dict[str, Any]]] = []

        def emitter(event_type: str, payload: dict[str, Any]):
            events.append((event_type, payload))

        router = ToolRouter(registry, policy, trace_emitter=emitter)

        with pytest.raises(ToolDenied):
            router.call("any", "any", "calculator", {"expression": "1+1"})

        # Should have requested + denied
        assert len(events) == 2
        assert events[0][0] == "tool_call_requested"
        assert events[1][0] == "tool_call_denied"
        assert "not in enabled_tools" in events[1][1]["reason"]

    def test_router_budget_not_incremented_on_validation_failure(self):
        """Test that budget is not incremented when input validation fails."""
        registry = create_default_registry()
        policy = create_default_policy_engine()
        router = ToolRouter(registry, policy)

        # Successful call should increment budget
        router.call("any", "any", "calculator", {"expression": "1+1"})
        assert policy.budget_state.total_calls == 1

        # Validation failure should NOT increment budget
        with pytest.raises(ToolExecutionError):
            router.call("any", "any", "calculator", {"expression": ""})  # Empty expression

        # Budget should still be 1
        assert policy.budget_state.total_calls == 1


class TestToolRouterIntegration:
    """Integration tests for tool router with engine."""

    def test_proposer_uses_calculator(self, tmp_path: Path):
        """Test that proposer stub uses calculator via tool callback."""
        from delibera.engine.orchestrator import Engine

        engine = Engine(runs_dir=tmp_path)
        run_dir = engine.run("Should we use Python for this project?")

        # Read trace
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        # Check for tool_call events
        tool_events = [e for e in events if e["event_type"].startswith("tool_call")]

        # Should have tool_call_requested and tool_call_executed for each branch
        # (3 branches x 2 tools: calculator in PROPOSE, docs.read in RESEARCH = 6 total)
        requested = [e for e in tool_events if e["event_type"] == "tool_call_requested"]
        executed = [e for e in tool_events if e["event_type"] == "tool_call_executed"]

        assert len(requested) == 6  # 3 branches x 2 tools
        assert len(executed) == 6  # All should succeed

        # Verify calculator calls from PROPOSE
        calculator_requests = [e for e in requested if e["payload"]["tool_name"] == "calculator"]
        assert len(calculator_requests) == 3
        for event in calculator_requests:
            assert event["payload"]["step"] == "PROPOSE"
            assert event["payload"]["role"] == "proposer"

        # Verify docs.read calls from RESEARCH
        docs_requests = [e for e in requested if e["payload"]["tool_name"] == "docs.read"]
        assert len(docs_requests) == 3
        for event in docs_requests:
            assert event["payload"]["step"] == "RESEARCH"
            assert event["payload"]["role"] == "researcher"

        # Verify outputs
        calculator_executed = [e for e in executed if e["payload"]["tool_name"] == "calculator"]
        for event in calculator_executed:
            assert "result" in event["payload"]["output"]

        docs_executed = [e for e in executed if e["payload"]["tool_name"] == "docs.read"]
        for event in docs_executed:
            assert "text" in event["payload"]["output"]

    def test_tool_call_denied_logged(self, tmp_path: Path):
        """Test that denied tool calls are logged in trace."""
        from delibera.engine.orchestrator import Engine
        from delibera.tools import GlobalPolicy

        # Create policy that denies calculator and docs.read
        policy = PolicyEngine(global_policy=GlobalPolicy(enabled_tools={"web.search"}))

        engine = Engine(
            runs_dir=tmp_path,
            policy_engine=policy,
        )

        # Run should still succeed (proposer and researcher fall back)
        run_dir = engine.run("Test question")

        # Read trace
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        # Check for denied events (3 calculator + 3 docs.read = 6)
        denied_events = [e for e in events if e["event_type"] == "tool_call_denied"]
        assert len(denied_events) == 6  # 3 branches x 2 tools

        calculator_denied = [e for e in denied_events if e["payload"]["tool_name"] == "calculator"]
        assert len(calculator_denied) == 3
        for event in calculator_denied:
            assert "not in enabled_tools" in event["payload"]["reason"]

        docs_denied = [e for e in denied_events if e["payload"]["tool_name"] == "docs.read"]
        assert len(docs_denied) == 3
        for event in docs_denied:
            assert "not in enabled_tools" in event["payload"]["reason"]

    def test_claim_check_step_allows_calculator(self, tmp_path: Path):
        """Test that calculator is allowed during CLAIM_CHECK step."""
        from delibera.engine.orchestrator import Engine

        engine = Engine(runs_dir=tmp_path)

        # Calculator should be allowed in CLAIM_CHECK since it's low-risk and not discovery
        decision = engine._policy_engine.evaluate(
            role="validator",
            step="CLAIM_CHECK",
            tool_name="calculator",
            is_discovery=False,
            risk_level=RiskLevel.LOW,
        )
        assert decision.allow is True


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_to_dict(self):
        """Test converting decision to dictionary."""
        decision = PolicyDecision(allow=True, reason="test reason")
        d = decision.to_dict()
        assert d == {"allow": True, "reason": "test reason"}
