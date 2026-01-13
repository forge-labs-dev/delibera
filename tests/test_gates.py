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
from delibera.gates.handler import GateHandler


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

        # Should have 3 gates: scope, tradeoff, and final_signoff
        # (tradeoff fires because stub agents produce identical scores)
        assert len(gate_triggered) == 3
        assert len(gate_response) == 3

        # Verify gate types
        gate_types = [e["payload"]["gate_type"] for e in gate_triggered]
        assert "scope" in gate_types
        assert "tradeoff" in gate_types
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
        assert len(gate_response) == 3  # scope + tradeoff + final_signoff

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

        # Verify handler saw all three gates (scope, tradeoff, final_signoff)
        assert len(handler.triggered_gates) == 3
        gate_types = [g.gate_type for g in handler.triggered_gates]
        assert GateType.SCOPE in gate_types
        assert GateType.TRADEOFF in gate_types
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
            AllowedAction.ACCEPT_OBJECTIONS,
            AllowedAction.ABORT,
        ]


class TestObjectionsAndConvergence:
    """Tests for objection handling and convergence rule."""

    def test_redteam_adds_objections_to_trace(self, tmp_path: Path):
        """Test that RedTeam step produces objection_added events in trace."""
        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=AutoApproveGateHandler(),
            gates_enabled=False,  # Disable gates to avoid blocking objection check
        )

        run_dir = engine.run("Should we use Python?")

        # Check trace for objection events
        trace_path = run_dir / "trace.jsonl"
        events = []
        with trace_path.open() as f:
            for line in f:
                events.append(json.loads(line))

        objection_events = [e for e in events if e["event_type"] == "objection_added"]

        # RedTeamStub should have added objections (at least one per node)
        assert len(objection_events) >= 3  # 3 branches

        # Each objection event should have required fields (nested in payload.objection)
        for event in objection_events:
            assert "node_id" in event["payload"]
            assert "objection" in event["payload"]
            objection = event["payload"]["objection"]
            assert "objection_id" in objection
            assert "target" in objection
            assert "severity" in objection
            assert "rationale" in objection
            assert "owner" in objection

    def test_auto_approve_completes_without_blocking_objections(self, tmp_path: Path):
        """Test that auto-approve completes when no blocking objections exist.

        With the default stubs (which have evidence), inference claims are
        marked as supported, so RedTeamStub creates only nonblocking objections.
        Auto-approve should complete successfully.
        """
        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=AutoApproveGateHandler(),
            gates_enabled=True,
        )

        # Stubs with evidence produce no blocking objections
        run_dir = engine.run("Should we use Python?")

        # Verify run completed
        artifact_path = run_dir / "artifact.json"
        assert artifact_path.exists()

        # Read artifact and verify no blocking objections
        with artifact_path.open() as f:
            artifact = json.load(f)

        assert len(artifact["objections"]["blocking_open"]) == 0

    def test_accept_objections_flow(self, tmp_path: Path):
        """Test the accept_objections action properly marks objections as accepted.

        Since stubs with evidence don't produce blocking objections, this test
        verifies the accept_objections logic using the apply_final_signoff_response
        function directly.
        """
        from delibera.gates.apply import apply_final_signoff_response

        # Test accept_objections returns proper IDs
        response = GateResponse(
            action=AllowedAction.ACCEPT_OBJECTIONS,
            parameters={"objection_ids": ["o1", "o2"]},
        )

        approved, accepted_ids = apply_final_signoff_response(
            response, has_blocking_objections=True
        )

        # Should not be approved yet (need to approve after accepting)
        assert approved is False
        # Should return the accepted IDs
        assert accepted_ids == ["o1", "o2"]

    def test_final_signoff_with_handler_that_accepts_then_approves(self, tmp_path: Path):
        """Test handler that accepts objections then approves.

        Since stubs don't produce blocking objections, this handler will
        just approve directly (no blockers to accept). We verify the handler
        is called and run completes.
        """

        class AcceptThenApproveHandler(GateHandler):
            def __init__(self):
                self._final_signoff_count = 0
                self.triggered_summaries: list[GateSummary] = []

            def handle(self, summary: GateSummary) -> GateResponse:
                self.triggered_summaries.append(summary)

                if summary.gate_type == GateType.FINAL_SIGNOFF:
                    self._final_signoff_count += 1

                    # First time: accept all blocking objections (if any)
                    if self._final_signoff_count == 1 and summary.open_blocking_objections:
                        objection_ids = [
                            obj["objection_id"] for obj in summary.open_blocking_objections
                        ]
                        return GateResponse(
                            action=AllowedAction.ACCEPT_OBJECTIONS,
                            parameters={"objection_ids": objection_ids},
                        )

                    # Second time (or no blockers): approve
                    return GateResponse(action=AllowedAction.APPROVE)

                # Other gates: approve
                return GateResponse(action=AllowedAction.APPROVE)

        handler = AcceptThenApproveHandler()
        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )

        run_dir = engine.run("Should we use Python?")

        # Run should succeed
        artifact_path = run_dir / "artifact.json"
        assert artifact_path.exists()

        # Final signoff handler should have been called
        final_summaries = [
            s for s in handler.triggered_summaries if s.gate_type == GateType.FINAL_SIGNOFF
        ]
        assert len(final_summaries) >= 1

    def test_artifact_includes_objections_summary(self, tmp_path: Path):
        """Test that final artifact includes objections summary."""
        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=AutoApproveGateHandler(),
            gates_enabled=False,  # Disable gates to avoid blocking
        )

        run_dir = engine.run("Should we use Python?")

        artifact_path = run_dir / "artifact.json"
        with artifact_path.open() as f:
            artifact = json.load(f)

        # Check objections summary is present
        assert "objections" in artifact
        obj_summary = artifact["objections"]

        # Check structure matches _build_objections_summary
        assert "blocking_open" in obj_summary
        assert "blocking_accepted" in obj_summary
        assert "nonblocking_open" in obj_summary

        # Each list should contain dicts with required fields
        assert isinstance(obj_summary["blocking_open"], list)
        assert isinstance(obj_summary["blocking_accepted"], list)
        assert isinstance(obj_summary["nonblocking_open"], list)

    def test_prune_prefers_fewer_blocking_objections(self):
        """Test that prune prioritizes nodes with fewer blocking objections."""
        from delibera.engine.operators import prune
        from delibera.engine.tree import DeliberationTree
        from delibera.epistemics.ledger import Ledger
        from delibera.epistemics.models import (
            Claim,
            ClaimStatus,
            ClaimType,
            Objection,
            ObjectionSeverity,
            ObjectionStatus,
        )

        tree = DeliberationTree()
        # Create root node first
        root = tree.create_root("Test question")

        # Create two children with different blocking objection counts
        node_a = tree.add_child(root.node_id, label="Option A", kind="option")
        node_b = tree.add_child(root.node_id, label="Option B", kind="option")

        # Node A: 1 blocking objection, higher score
        node_a.artifact = {"score": 10.0}
        node_a.ledger = Ledger(
            claims=[Claim("c1", "claim", ClaimType.PLAN, 0.8, "p", ClaimStatus.SUPPORTED)],
            objections=[
                Objection("o1", "c1", ObjectionSeverity.BLOCKING, ObjectionStatus.OPEN, "r", "t")
            ],
        )

        # Node B: 0 blocking objections, lower score
        node_b.artifact = {"score": 5.0}
        node_b.ledger = Ledger(
            claims=[Claim("c2", "claim", ClaimType.PLAN, 0.8, "p", ClaimStatus.SUPPORTED)],
            objections=[],  # No objections
        )

        # Prune to keep 1
        survivors, pruned = prune(tree, [node_a.node_id, node_b.node_id], keep_count=1)

        # Node B should survive (fewer blocking objections beats higher score)
        assert survivors == [node_b.node_id]
        assert pruned == [node_a.node_id]

    def test_validate_response_accept_objections_valid(self):
        """Test validation of accept_objections response."""
        response = GateResponse(
            action=AllowedAction.ACCEPT_OBJECTIONS,
            parameters={"objection_ids": ["o1", "o2"]},
        )
        errors = validate_response(GateType.FINAL_SIGNOFF, response)
        assert errors == []

    def test_validate_response_accept_objections_missing_ids(self):
        """Test validation rejects accept_objections without IDs."""
        response = GateResponse(
            action=AllowedAction.ACCEPT_OBJECTIONS,
            parameters={"objection_ids": []},
        )
        errors = validate_response(GateType.FINAL_SIGNOFF, response)
        assert len(errors) == 1
        assert "at least one objection" in errors[0]

    def test_validate_response_accept_objections_not_allowed_for_scope(self):
        """Test validation rejects accept_objections for scope gate."""
        response = GateResponse(
            action=AllowedAction.ACCEPT_OBJECTIONS,
            parameters={"objection_ids": ["o1"]},
        )
        errors = validate_response(GateType.SCOPE, response)
        assert len(errors) == 1
        assert "not allowed" in errors[0]
