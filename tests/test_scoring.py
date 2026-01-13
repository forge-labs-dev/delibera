"""Tests for the scoring subsystem."""

from pathlib import Path

import pytest

from delibera.engine.orchestrator import Engine
from delibera.engine.tree import Node
from delibera.epistemics.ledger import Ledger
from delibera.epistemics.models import Claim, ClaimStatus, ClaimType, Evidence
from delibera.gates import AllowedAction, GateResponse, GateType, ScriptedGateHandler
from delibera.scoring import (
    ScoreWeights,
    compute_score,
    create_default_weights,
    score_node,
)
from delibera.scoring.metrics import EVIDENCE_COUNT_CAP, compute_metrics


class TestScoreWeights:
    """Tests for ScoreWeights dataclass."""

    def test_default_weights(self) -> None:
        """Test that default weights are created correctly."""
        weights = create_default_weights()

        assert weights.evidence_coverage == 1.0
        assert weights.evidence_count == 0.2
        assert weights.weak_claims == -0.3
        assert weights.unsupported_claims == -1.0
        assert weights.blocking_objections == -2.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        weights = ScoreWeights(
            evidence_coverage=2.0,
            evidence_count=0.5,
            weak_claims=-0.5,
            unsupported_claims=-2.0,
            blocking_objections=-3.0,
        )

        d = weights.to_dict()

        assert d["evidence_coverage"] == 2.0
        assert d["evidence_count"] == 0.5
        assert d["weak_claims"] == -0.5
        assert d["unsupported_claims"] == -2.0
        assert d["blocking_objections"] == -3.0

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        weights = ScoreWeights.from_dict(
            {
                "evidence_coverage": 1.5,
                "weak_claims": -0.8,
            }
        )

        # Provided values
        assert weights.evidence_coverage == 1.5
        assert weights.weak_claims == -0.8

        # Default values for unprovided
        assert weights.evidence_count == 0.2
        assert weights.unsupported_claims == -1.0
        assert weights.blocking_objections == -2.0

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test that unknown keys are ignored."""
        weights = ScoreWeights.from_dict(
            {
                "evidence_coverage": 2.0,
                "unknown_key": 5.0,
            }
        )

        assert weights.evidence_coverage == 2.0
        assert not hasattr(weights, "unknown_key")


class TestComputeMetrics:
    """Tests for metric computation."""

    def _create_node_with_ledger(self) -> Node:
        """Create a node with a ledger for testing."""
        node = Node(
            node_id="test_node",
            parent_id=None,
            kind="option",
            depth=1,
        )
        node.ledger = Ledger()
        return node

    def test_empty_ledger(self) -> None:
        """Test metrics for empty ledger."""
        node = self._create_node_with_ledger()

        metrics = compute_metrics(node)

        # No fact claims -> evidence_coverage defaults to 0.5
        assert metrics["evidence_coverage"] == 0.5
        assert metrics["evidence_count"] == 0.0
        assert metrics["weak_claims"] == 0.0
        assert metrics["unsupported_claims"] == 0.0
        assert metrics["blocking_objections"] == 0.0

    def test_evidence_coverage_all_supported(self) -> None:
        """Test evidence coverage with all supported fact claims."""
        node = self._create_node_with_ledger()

        # Add two supported fact claims
        node.ledger.claims = [
            Claim(
                claim_id="c1",
                text="Fact 1",
                claim_type=ClaimType.FACT,
                confidence=0.9,
                owner="test",
                status=ClaimStatus.SUPPORTED,
            ),
            Claim(
                claim_id="c2",
                text="Fact 2",
                claim_type=ClaimType.FACT,
                confidence=0.9,
                owner="test",
                status=ClaimStatus.SUPPORTED,
            ),
        ]

        metrics = compute_metrics(node)
        assert metrics["evidence_coverage"] == 1.0

    def test_evidence_coverage_partial(self) -> None:
        """Test evidence coverage with some unsupported fact claims."""
        node = self._create_node_with_ledger()

        node.ledger.claims = [
            Claim(
                claim_id="c1",
                text="Fact 1",
                claim_type=ClaimType.FACT,
                confidence=0.9,
                owner="test",
                status=ClaimStatus.SUPPORTED,
            ),
            Claim(
                claim_id="c2",
                text="Fact 2",
                claim_type=ClaimType.FACT,
                confidence=0.9,
                owner="test",
                status=ClaimStatus.UNSUPPORTED,
            ),
        ]

        metrics = compute_metrics(node)
        assert metrics["evidence_coverage"] == 0.5

    def test_evidence_count_capped(self) -> None:
        """Test evidence count is capped at EVIDENCE_COUNT_CAP."""
        node = self._create_node_with_ledger()

        # Add more evidence than the cap
        for i in range(EVIDENCE_COUNT_CAP + 3):
            node.ledger.evidence.append(
                Evidence(
                    evidence_id=f"ev_{i}",
                    source=f"source_{i}",
                    excerpt=f"excerpt_{i}",
                    provenance={"node_id": "test_node", "step": "TEST"},
                )
            )

        metrics = compute_metrics(node)
        assert metrics["evidence_count"] == float(EVIDENCE_COUNT_CAP)

    def test_weak_and_unsupported_counts(self) -> None:
        """Test weak and unsupported claim counts."""
        node = self._create_node_with_ledger()

        node.ledger.claims = [
            Claim(
                claim_id="c1",
                text="Claim 1",
                claim_type=ClaimType.INFERENCE,
                confidence=0.9,
                owner="test",
                status=ClaimStatus.SUPPORTED,
            ),
            Claim(
                claim_id="c2",
                text="Claim 2",
                claim_type=ClaimType.INFERENCE,
                confidence=0.7,
                owner="test",
                status=ClaimStatus.WEAK,
            ),
            Claim(
                claim_id="c3",
                text="Claim 3",
                claim_type=ClaimType.INFERENCE,
                confidence=0.6,
                owner="test",
                status=ClaimStatus.WEAK,
            ),
            Claim(
                claim_id="c4",
                text="Claim 4",
                claim_type=ClaimType.FACT,
                confidence=0.5,
                owner="test",
                status=ClaimStatus.UNSUPPORTED,
            ),
        ]

        metrics = compute_metrics(node)
        assert metrics["weak_claims"] == 2.0
        assert metrics["unsupported_claims"] == 1.0


class TestComputeScore:
    """Tests for score computation."""

    def test_score_with_default_weights(self) -> None:
        """Test score computation with default weights."""
        metrics = {
            "evidence_coverage": 1.0,
            "evidence_count": 2.0,
            "weak_claims": 0.0,
            "unsupported_claims": 0.0,
            "blocking_objections": 0.0,
        }
        weights = create_default_weights()

        score = compute_score(metrics, weights)

        # 1.0 * 1.0 + 2.0 * 0.2 + 0 * -0.3 + 0 * -1.0 + 0 * -2.0 = 1.4
        assert score == pytest.approx(1.4)

    def test_score_with_penalties(self) -> None:
        """Test score computation with penalties for bad claims."""
        metrics = {
            "evidence_coverage": 0.5,
            "evidence_count": 1.0,
            "weak_claims": 2.0,
            "unsupported_claims": 1.0,
            "blocking_objections": 1.0,
        }
        weights = create_default_weights()

        score = compute_score(metrics, weights)

        # 0.5 * 1.0 + 1.0 * 0.2 + 2.0 * -0.3 + 1.0 * -1.0 + 1.0 * -2.0
        # = 0.5 + 0.2 - 0.6 - 1.0 - 2.0 = -2.9
        assert score == pytest.approx(-2.9)

    def test_score_with_custom_weights(self) -> None:
        """Test score computation with custom weights."""
        metrics = {
            "evidence_coverage": 1.0,
            "evidence_count": 2.0,
            "weak_claims": 1.0,
            "unsupported_claims": 0.0,
            "blocking_objections": 0.0,
        }
        weights = ScoreWeights(
            evidence_coverage=2.0,
            evidence_count=0.5,
            weak_claims=-1.0,
            unsupported_claims=-2.0,
            blocking_objections=-3.0,
        )

        score = compute_score(metrics, weights)

        # 1.0 * 2.0 + 2.0 * 0.5 + 1.0 * -1.0 + 0 + 0 = 2.0
        assert score == pytest.approx(2.0)


class TestScoreNode:
    """Tests for score_node function."""

    def _create_node_with_ledger(self) -> Node:
        """Create a node with a ledger for testing."""
        node = Node(
            node_id="test_node",
            parent_id=None,
            kind="option",
            depth=1,
        )
        node.ledger = Ledger()
        return node

    def test_score_node_returns_result(self) -> None:
        """Test score_node returns ScoreResult with all fields."""
        node = self._create_node_with_ledger()

        result = score_node(node)

        assert isinstance(result.score, float)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.weights, ScoreWeights)

    def test_score_node_uses_default_weights(self) -> None:
        """Test score_node uses default weights when none provided."""
        node = self._create_node_with_ledger()

        result = score_node(node)

        assert result.weights == create_default_weights()

    def test_score_node_uses_custom_weights(self) -> None:
        """Test score_node uses provided weights."""
        node = self._create_node_with_ledger()
        custom_weights = ScoreWeights(evidence_coverage=3.0)

        result = score_node(node, custom_weights)

        assert result.weights.evidence_coverage == 3.0


class TestTradeoffGate:
    """Tests for tradeoff gate functionality."""

    def test_tradeoff_gate_triggers_on_tie(self, tmp_path: Path) -> None:
        """Test that tradeoff gate triggers when scores are tied."""
        handler = ScriptedGateHandler()

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        engine.run("Test question")

        # Find tradeoff gate
        tradeoff_gates = [g for g in handler.triggered_gates if g.gate_type == GateType.TRADEOFF]
        assert len(tradeoff_gates) == 1

        tradeoff = tradeoff_gates[0]
        # Stub agents produce identical scores, so score_difference should be 0
        assert tradeoff.score_difference == 0.0
        assert len(tradeoff.top_candidates) > 0
        assert tradeoff.current_weights is not None

    def test_tradeoff_gate_skipped_with_presets(self, tmp_path: Path) -> None:
        """Test that tradeoff gate is skipped when weights are preset."""
        handler = ScriptedGateHandler()

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
            initial_weights=ScoreWeights(evidence_coverage=2.0),
        )
        engine.run("Test question")

        # Should have scope and final_signoff, but not tradeoff
        gate_types = [g.gate_type for g in handler.triggered_gates]
        assert GateType.SCOPE in gate_types
        assert GateType.FINAL_SIGNOFF in gate_types
        assert GateType.TRADEOFF not in gate_types

    def test_tradeoff_gate_set_weights(self, tmp_path: Path) -> None:
        """Test that set_weights action updates scoring."""
        # Create handler that sets custom weights
        set_weights_response = GateResponse(
            action=AllowedAction.SET_WEIGHTS,
            parameters={"weights": {"evidence_coverage": 5.0}},
        )
        handler = ScriptedGateHandler(responses={GateType.TRADEOFF: set_weights_response})

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        run_dir = engine.run("Test question")

        # Check trace has updated weights
        import json

        trace_path = run_dir / "trace.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]

        gate_response_events = [
            e
            for e in events
            if e["event_type"] == "gate_response_applied"
            and e["payload"]["gate_type"] == "tradeoff"
        ]
        assert len(gate_response_events) == 1
        response_event = gate_response_events[0]
        assert response_event["payload"]["response"]["action"] == "set_weights"

    def test_tradeoff_gate_approve_default(self, tmp_path: Path) -> None:
        """Test that approve_default keeps current weights."""
        approve_default_response = GateResponse(action=AllowedAction.APPROVE_DEFAULT)
        handler = ScriptedGateHandler(responses={GateType.TRADEOFF: approve_default_response})

        engine = Engine(
            runs_dir=tmp_path,
            gate_handler=handler,
            gates_enabled=True,
        )
        run_dir = engine.run("Test question")

        # Verify run completed successfully
        artifact_path = run_dir / "artifact.json"
        assert artifact_path.exists()


class TestArtifactDecisionExplanation:
    """Tests for decision explanation in final artifact."""

    def test_artifact_has_decision_explanation(self, tmp_path: Path) -> None:
        """Test that artifact contains decision_explanation field."""
        engine = Engine(runs_dir=tmp_path, gates_enabled=False)
        run_dir = engine.run("Test question")

        import json

        artifact_path = run_dir / "artifact.json"
        artifact = json.loads(artifact_path.read_text())

        assert "decision_explanation" in artifact
        explanation = artifact["decision_explanation"]

        assert "summary" in explanation
        assert "ranking" in explanation
        assert "selection_criteria" in explanation

    def test_decision_explanation_ranking_structure(self, tmp_path: Path) -> None:
        """Test structure of ranking in decision explanation."""
        engine = Engine(runs_dir=tmp_path, gates_enabled=False)
        run_dir = engine.run("Test question")

        import json

        artifact_path = run_dir / "artifact.json"
        artifact = json.loads(artifact_path.read_text())

        ranking = artifact["decision_explanation"]["ranking"]
        assert len(ranking) > 0

        for option in ranking:
            assert "label" in option
            assert "score" in option
            assert "metrics" in option
            assert "epistemic" in option
            assert "status" in option
            assert option["status"] in ["survivor", "pruned"]


class TestScoreComputed:
    """Tests for score_computed trace events."""

    def test_trace_has_score_computed_events(self, tmp_path: Path) -> None:
        """Test that trace has score_computed events for each branch."""
        engine = Engine(runs_dir=tmp_path, gates_enabled=False)
        run_dir = engine.run("Test question")

        import json

        trace_path = run_dir / "trace.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]

        score_events = [e for e in events if e["event_type"] == "score_computed"]

        # Should have score events for each branch (3 with stub planner)
        assert len(score_events) == 3

        for event in score_events:
            payload = event["payload"]
            assert "node_id" in payload
            assert "score" in payload
            assert "metrics" in payload
            assert "weights" in payload

    def test_prune_event_includes_weights(self, tmp_path: Path) -> None:
        """Test that prune event includes weights used."""
        engine = Engine(runs_dir=tmp_path, gates_enabled=False)
        run_dir = engine.run("Test question")

        import json

        trace_path = run_dir / "trace.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]

        prune_events = [e for e in events if e["event_type"] == "prune"]
        assert len(prune_events) == 1

        payload = prune_events[0]["payload"]
        assert "weights_used" in payload
        assert "scores" in payload
