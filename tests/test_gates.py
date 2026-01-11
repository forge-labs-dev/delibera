"""Tests for the gates subsystem."""

import json
from pathlib import Path

import pytest

from delibera.engine.orchestrator import Engine
from delibera.gates import (
    AllowedAction,
    AutoApproveGateHandler,
    GateAborted,
    GateResponse,
    GateSummary,
    GateType,
    ScriptedGateHandler,
    apply_scope_response,
    validate_response,
)


class TestGateModels:
    """Tests for gate data models."""

    def test_gate_summary_to_dict_scope(self):
        """Test GateSummary serialization for scope gate."""
        summary = GateSummary(
            gate_type=GateType.SCOPE,
            allowed_actions=[AllowedAction.APPROVE, AllowedAction.VETO_BRANCHES],
            interpreted_question="Should we use Python?",
            proposed_branches=["Option A", "Option B", "Option C"],
        )
        d = summary.to_dict()

        assert d["gate_type"] == "scope"
        assert "approve" in d["allowed_actions"]
        assert d["interpreted_question"] == "Should we use Python?"
        assert d["proposed_branches"] == ["Option A", "Option B", "Option C"]

    def test_gate_summary_to_dict_final_signoff(self):
        """Test GateSummary serialization for final sign-off gate."""
        summary = GateSummary(
            gate_type=GateType.FINAL_SIGNOFF,
            allowed_actions=[AllowedAction.APPROVE, AllowedAction.ABORT],
            recommendation="Use Python",
            claim_check_summary={"supported": 3, "weak": 1, "unsupported": 0},
            open_questions=["Consider alternatives?"],
        )
        d = summary.to_dict()

        assert d["gate_type"] == "final_signoff"
        assert d["recommendation"] == "Use Python"
        assert d["claim_check_summary"]["supported"] == 3

    def test_gate_response_to_dict(self):
        """Test GateResponse serialization."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option C"]},
        )
        d = response.to_dict()

        assert d["action"] == "veto_branches"
        assert d["parameters"]["branches"] == ["Option C"]

    def test_validate_response_approve_valid(self):
        """Test validation of approve response."""
        response = GateResponse(action=AllowedAction.APPROVE)
        errors = validate_response(GateType.SCOPE, response)
        assert errors == []

    def test_validate_response_abort_valid(self):
        """Test validation of abort response."""
        response = GateResponse(action=AllowedAction.ABORT)
        errors = validate_response(GateType.SCOPE, response)
        assert errors == []

    def test_validate_response_veto_branches_valid(self):
        """Test validation of veto_branches response."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option C"]},
        )
        errors = validate_response(GateType.SCOPE, response)
        assert errors == []

    def test_validate_response_veto_branches_missing_branches(self):
        """Test validation rejects veto_branches without branches."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={},
        )
        errors = validate_response(GateType.SCOPE, response)
        assert len(errors) == 1
        assert "at least one branch" in errors[0]

    def test_validate_response_veto_not_allowed_for_final_signoff(self):
        """Test validation rejects veto_branches for final sign-off."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option C"]},
        )
        errors = validate_response(GateType.FINAL_SIGNOFF, response)
        assert len(errors) == 1
        assert "not allowed" in errors[0]


class TestApplyScopeResponse:
    """Tests for scope response application."""

    def test_approve_keeps_all_branches(self):
        """Test approve keeps all branches."""
        response = GateResponse(action=AllowedAction.APPROVE)
        branches = ["Option A", "Option B", "Option C"]
        result = apply_scope_response(response, branches)
        assert result == branches

    def test_veto_removes_specified_branches(self):
        """Test veto removes specified branches."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option C"]},
        )
        branches = ["Option A", "Option B", "Option C"]
        result = apply_scope_response(response, branches)
        assert result == ["Option A", "Option B"]

    def test_veto_multiple_branches(self):
        """Test veto removes multiple branches."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option A", "Option C"]},
        )
        branches = ["Option A", "Option B", "Option C"]
        result = apply_scope_response(response, branches)
        assert result == ["Option B"]

    def test_veto_all_keeps_original(self):
        """Test vetoing all branches keeps original."""
        response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option A", "Option B", "Option C"]},
        )
        branches = ["Option A", "Option B", "Option C"]
        result = apply_scope_response(response, branches)
        assert result == branches  # All vetoed, keep original

    def test_abort_raises_exception(self):
        """Test abort raises GateAborted."""
        response = GateResponse(action=AllowedAction.ABORT)
        branches = ["Option A", "Option B", "Option C"]
        with pytest.raises(GateAborted) as exc_info:
            apply_scope_response(response, branches)
        assert exc_info.value.gate_type == GateType.SCOPE


class TestGateHandlers:
    """Tests for gate handlers."""

    def test_auto_approve_handler_returns_approve(self):
        """Test AutoApproveGateHandler always approves."""
        handler = AutoApproveGateHandler()
        summary = GateSummary(
            gate_type=GateType.SCOPE,
            allowed_actions=[AllowedAction.APPROVE, AllowedAction.ABORT],
        )
        response = handler.handle(summary)
        assert response.action == AllowedAction.APPROVE

    def test_test_handler_returns_predetermined_response(self):
        """Test ScriptedGateHandler returns predetermined responses."""
        veto_response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": ["Option C"]},
        )
        handler = ScriptedGateHandler(responses={GateType.SCOPE: veto_response})
        summary = GateSummary(
            gate_type=GateType.SCOPE,
            allowed_actions=[AllowedAction.APPROVE, AllowedAction.VETO_BRANCHES],
        )
        response = handler.handle(summary)
        assert response.action == AllowedAction.VETO_BRANCHES
        assert response.parameters["branches"] == ["Option C"]

    def test_test_handler_tracks_triggered_gates(self):
        """Test ScriptedGateHandler tracks which gates were triggered."""
        handler = ScriptedGateHandler()

        summary1 = GateSummary(
            gate_type=GateType.SCOPE,
            allowed_actions=[AllowedAction.APPROVE],
        )
        summary2 = GateSummary(
            gate_type=GateType.FINAL_SIGNOFF,
            allowed_actions=[AllowedAction.APPROVE],
        )

        handler.handle(summary1)
        handler.handle(summary2)

        assert len(handler.triggered_gates) == 2
        assert handler.triggered_gates[0].gate_type == GateType.SCOPE
        assert handler.triggered_gates[1].gate_type == GateType.FINAL_SIGNOFF

    def test_test_handler_defaults_to_approve(self):
        """Test ScriptedGateHandler defaults to approve for unspecified gates."""
        handler = ScriptedGateHandler(responses={})
        summary = GateSummary(
            gate_type=GateType.FINAL_SIGNOFF,
            allowed_actions=[AllowedAction.APPROVE],
        )
        response = handler.handle(summary)
        assert response.action == AllowedAction.APPROVE


class TestEngineWithGates:
    """Integration tests for Engine with gates."""

    def test_gates_auto_approve_run_completes(self, tmp_path: Path):
        """Test run completes with auto-approve gates."""
        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=AutoApproveGateHandler(),
            gates_enabled=True,
        )
        run_dir = engine.run("Should we use Python?")

        # Check run completed
        artifact_path = run_dir / "artifact.json"
        assert artifact_path.exists()

        # Check trace has gate events
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        gate_triggered = [e for e in events if e["event_type"] == "gate_triggered"]
        gate_response = [e for e in events if e["event_type"] == "gate_response_applied"]

        # Should have 2 gates: scope and final_signoff
        assert len(gate_triggered) == 2
        assert len(gate_response) == 2

        # Verify gate types
        gate_types = [e["payload"]["gate_type"] for e in gate_triggered]
        assert "scope" in gate_types
        assert "final_signoff" in gate_types

    def test_gates_disabled_no_gate_events(self, tmp_path: Path):
        """Test run with gates disabled has no gate events."""
        engine = Engine(
            runs_dir=tmp_path,
            gates_enabled=False,
        )
        run_dir = engine.run("Should we use Python?")

        # Check run completed
        artifact_path = run_dir / "artifact.json"
        assert artifact_path.exists()

        # Check trace has no gate events
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        gate_triggered = [e for e in events if e["event_type"] == "gate_triggered"]
        assert len(gate_triggered) == 0

    def test_scope_gate_veto_reduces_branches(self, tmp_path: Path):
        """Test vetoing a branch at scope gate reduces children."""
        # Create handler that vetoes Option C
        # Note: branch label matches stub output format
        option_c_label = "Option C: Conservative approach to 'Should we use Python?...'"
        veto_response = GateResponse(
            action=AllowedAction.VETO_BRANCHES,
            parameters={"branches": [option_c_label]},
        )
        handler = ScriptedGateHandler(responses={GateType.SCOPE: veto_response})

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        run_dir = engine.run("Should we use Python?")

        # Check trace
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        # Find expand event
        expand_events = [e for e in events if e["event_type"] == "expand"]
        assert len(expand_events) == 1

        # Should only have 2 children (Option A and Option B)
        labels = expand_events[0]["payload"]["labels"]
        assert len(labels) == 2
        assert all("Option C" not in label for label in labels)

        # Check gate response was logged
        gate_response = [e for e in events if e["event_type"] == "gate_response_applied"]
        assert len(gate_response) == 2  # scope + final_signoff

        scope_response = [e for e in gate_response if e["payload"]["gate_type"] == "scope"][0]
        assert scope_response["payload"]["response"]["action"] == "veto_branches"

    def test_final_gate_abort_no_artifact(self, tmp_path: Path):
        """Test aborting at final sign-off gate prevents artifact write."""
        # Create handler that aborts at final sign-off
        handler = ScriptedGateHandler(
            responses={
                GateType.SCOPE: GateResponse(action=AllowedAction.APPROVE),
                GateType.FINAL_SIGNOFF: GateResponse(action=AllowedAction.ABORT),
            }
        )

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )

        with pytest.raises(GateAborted) as exc_info:
            engine.run("Should we use Python?")

        assert exc_info.value.gate_type == GateType.FINAL_SIGNOFF

        # Find the run directory (there should be one)
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Artifact should NOT exist
        artifact_path = run_dir / "artifact.json"
        assert not artifact_path.exists()

        # Trace should exist with aborted status
        trace_path = run_dir / "trace.jsonl"
        assert trace_path.exists()

        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        run_end = [e for e in events if e["event_type"] == "run_end"]
        assert len(run_end) == 1
        assert run_end[0]["payload"]["status"] == "aborted"
        assert run_end[0]["payload"]["gate_type"] == "final_signoff"

    def test_scope_gate_abort_early(self, tmp_path: Path):
        """Test aborting at scope gate stops run early."""
        handler = ScriptedGateHandler(
            responses={
                GateType.SCOPE: GateResponse(action=AllowedAction.ABORT),
            }
        )

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )

        with pytest.raises(GateAborted) as exc_info:
            engine.run("Should we use Python?")

        assert exc_info.value.gate_type == GateType.SCOPE

        # Find the run directory
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Trace should have aborted status
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        # Should NOT have expand event (aborted before expansion)
        expand_events = [e for e in events if e["event_type"] == "expand"]
        assert len(expand_events) == 0

        run_end = [e for e in events if e["event_type"] == "run_end"]
        assert run_end[0]["payload"]["status"] == "aborted"

    def test_gate_handler_injection(self, tmp_path: Path):
        """Test that custom gate handler can be injected."""
        handler = ScriptedGateHandler()

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        engine.run("Test question")

        # Verify handler saw both gates
        assert len(handler.triggered_gates) == 2
        gate_types = [g.gate_type for g in handler.triggered_gates]
        assert GateType.SCOPE in gate_types
        assert GateType.FINAL_SIGNOFF in gate_types


class TestGateSummaryContent:
    """Tests for gate summary content."""

    def test_scope_gate_summary_has_question_and_branches(self, tmp_path: Path):
        """Test scope gate summary includes question and branches."""
        handler = ScriptedGateHandler()

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        engine.run("Should we adopt microservices?")

        # Find scope gate summary
        scope_summary = [g for g in handler.triggered_gates if g.gate_type == GateType.SCOPE][0]

        assert scope_summary.interpreted_question == "Should we adopt microservices?"
        assert len(scope_summary.proposed_branches) == 3
        assert scope_summary.allowed_actions == [
            AllowedAction.APPROVE,
            AllowedAction.VETO_BRANCHES,
            AllowedAction.ABORT,
        ]

    def test_final_signoff_gate_summary_has_recommendation(self, tmp_path: Path):
        """Test final sign-off gate summary includes recommendation."""
        handler = ScriptedGateHandler()

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        engine.run("Should we adopt microservices?")

        # Find final sign-off gate summary
        final_summary = [
            g for g in handler.triggered_gates if g.gate_type == GateType.FINAL_SIGNOFF
        ][0]

        assert final_summary.recommendation is not None
        assert len(final_summary.recommendation) > 0
        assert final_summary.claim_check_summary is not None
        assert final_summary.allowed_actions == [
            AllowedAction.APPROVE,
            AllowedAction.ABORT,
        ]
