"""Tests for the refinement loop functionality."""

import tempfile
from pathlib import Path

import pytest

from delibera.agents.stub import RefinerStub
from delibera.engine.orchestrator import Engine
from delibera.protocol import ProtocolSpec, load_protocol_from_yaml
from delibera.protocol.spec import (
    ConvergenceSpec,
    ExpandSpec,
    PruneSpec,
    ReduceSpec,
    StepSpec,
)


class TestConvergenceSpec:
    """Tests for ConvergenceSpec validation."""

    def test_default_values(self) -> None:
        """Test ConvergenceSpec has correct defaults."""
        spec = ConvergenceSpec()
        assert spec.max_rounds == 0
        assert spec.weak_threshold == 5
        assert spec.score_epsilon == 0.01

    def test_custom_values(self) -> None:
        """Test ConvergenceSpec with custom values."""
        spec = ConvergenceSpec(
            max_rounds=5,
            weak_threshold=10,
            score_epsilon=0.05,
        )
        assert spec.max_rounds == 5
        assert spec.weak_threshold == 10
        assert spec.score_epsilon == 0.05

    def test_negative_max_rounds_raises(self) -> None:
        """Test that negative max_rounds raises ValueError."""
        with pytest.raises(ValueError, match="max_rounds must be >= 0"):
            ConvergenceSpec(max_rounds=-1)

    def test_negative_weak_threshold_raises(self) -> None:
        """Test that negative weak_threshold raises ValueError."""
        with pytest.raises(ValueError, match="weak_threshold must be >= 0"):
            ConvergenceSpec(weak_threshold=-1)

    def test_negative_score_epsilon_raises(self) -> None:
        """Test that negative score_epsilon raises ValueError."""
        with pytest.raises(ValueError, match="score_epsilon must be >= 0"):
            ConvergenceSpec(score_epsilon=-0.01)


class TestRefinerStub:
    """Tests for RefinerStub agent."""

    def test_execute_basic(self) -> None:
        """Test RefinerStub produces correct output structure."""
        refiner = RefinerStub()
        context = {
            "node_id": "test_node",
            "artifact": {"summary": "Original summary", "score": 0.5},
            "claims": [],
            "round": 1,
        }
        output = refiner.execute(context)

        assert "artifact" in output
        assert "refinement_notes" in output
        assert "weak_addressed" in output
        assert "unsupported_addressed" in output
        assert output["role"] == "refiner"
        assert output["step"] == "REFINE"

    def test_execute_with_weak_claims(self) -> None:
        """Test RefinerStub addresses weak claims."""
        refiner = RefinerStub()
        context = {
            "node_id": "test_node",
            "artifact": {"summary": "Test", "score": 0.5},
            "claims": [
                {"claim_id": "c1", "status": "weak"},
                {"claim_id": "c2", "status": "weak"},
                {"claim_id": "c3", "status": "weak"},  # Third weak claim
            ],
            "round": 1,
        }
        output = refiner.execute(context)

        # Should address up to 2 weak claims
        assert output["weak_addressed"] == 2
        assert len(output["refinement_notes"]) >= 2

    def test_execute_with_unsupported_claims(self) -> None:
        """Test RefinerStub addresses unsupported claims."""
        refiner = RefinerStub()
        context = {
            "node_id": "test_node",
            "artifact": {"summary": "Test", "score": 0.5},
            "claims": [
                {"claim_id": "c1", "status": "unsupported"},
                {"claim_id": "c2", "status": "unsupported"},
            ],
            "round": 1,
        }
        output = refiner.execute(context)

        # Should address up to 1 unsupported claim per round
        assert output["unsupported_addressed"] == 1

    def test_score_improvement_diminishes(self) -> None:
        """Test that score improvement diminishes with each round."""
        refiner = RefinerStub()
        original_score = 0.5

        # Round 1: improvement = 0.05 / 1 = 0.05
        output1 = refiner.execute({"artifact": {"score": original_score}, "claims": [], "round": 1})
        score_after_r1 = output1["artifact"]["score"]
        assert score_after_r1 == pytest.approx(0.55, abs=0.01)

        # Round 2: improvement = 0.05 / 2 = 0.025
        output2 = refiner.execute({"artifact": {"score": score_after_r1}, "claims": [], "round": 2})
        score_after_r2 = output2["artifact"]["score"]
        assert score_after_r2 == pytest.approx(0.575, abs=0.01)

    def test_artifact_marked_with_round(self) -> None:
        """Test that refined artifact includes round metadata."""
        refiner = RefinerStub()
        output = refiner.execute(
            {
                "artifact": {"summary": "Test"},
                "claims": [],
                "round": 3,
            }
        )
        assert output["artifact"]["refinement_round"] == 3
        assert "Refined in round 3" in output["artifact"]["summary"]


class TestProtocolWithRefinement:
    """Tests for protocol spec with refinement loop."""

    def _create_refine_protocol(self) -> ProtocolSpec:
        """Create a protocol with refinement enabled."""
        return ProtocolSpec(
            name="test_refine_protocol",
            protocol_version="v1",
            max_depth=1,
            expand_rules=[
                ExpandSpec(
                    id="expand_options",
                    at_step_id="plan",
                    child_kind="option",
                    max_children=3,
                    depth=1,
                    source="planner_output",
                ),
            ],
            branch_pipeline=[
                StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
                StepSpec(id="research", kind="work", step_name="RESEARCH", role="researcher"),
                StepSpec(id="validate", kind="validate", step_name="CLAIM_CHECK"),
                StepSpec(id="redteam", kind="work", step_name="REDTEAM", role="redteam"),
            ],
            prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
            reduce=ReduceSpec(rule="merge_artifacts"),
            convergence=ConvergenceSpec(
                max_rounds=3,
                weak_threshold=5,
                score_epsilon=0.01,
            ),
            refine_loop=[
                StepSpec(id="refine", kind="work", step_name="REFINE", role="refiner"),
                StepSpec(id="revalidate", kind="validate", step_name="CLAIM_CHECK"),
            ],
        )

    def test_protocol_interpreter_has_refine_loop(self) -> None:
        """Test interpreter correctly identifies refine loop."""
        from delibera.protocol.interpreter import ProtocolInterpreter

        protocol = self._create_refine_protocol()
        interpreter = ProtocolInterpreter(protocol)

        assert interpreter.has_refine_loop() is True
        assert len(interpreter.get_refine_loop_steps()) == 2

    def test_protocol_interpreter_no_refine_loop(self) -> None:
        """Test interpreter correctly identifies no refine loop."""
        from delibera.protocol.interpreter import ProtocolInterpreter

        protocol = self._create_refine_protocol()
        protocol.convergence = ConvergenceSpec(max_rounds=0)  # Disable refinement
        interpreter = ProtocolInterpreter(protocol)

        assert interpreter.has_refine_loop() is False

    def test_protocol_interpreter_empty_refine_loop(self) -> None:
        """Test interpreter with empty refine_loop."""
        from delibera.protocol.interpreter import ProtocolInterpreter

        protocol = self._create_refine_protocol()
        protocol.refine_loop = []
        protocol.convergence = ConvergenceSpec(max_rounds=3)
        interpreter = ProtocolInterpreter(protocol)

        assert interpreter.has_refine_loop() is False


class TestYamlProtocolWithRefinement:
    """Tests for YAML protocol loading with refinement."""

    def test_load_protocol_with_convergence(self, tmp_path: Path) -> None:
        """Test loading YAML with convergence settings."""
        yaml_content = """
name: test_refine
protocol_version: v1
max_depth: 1
expand_rules:
  - id: expand_options
    at_step_id: plan
    child_kind: option
    max_children: 3
    depth: 1
    source: planner_output
branch_pipeline:
  - id: propose
    kind: work
    step_name: PROPOSE
    role: proposer
refine_loop:
  - id: refine
    kind: work
    step_name: REFINE
    role: refiner
prune:
  rule: epistemic_then_score
  keep_k: 2
reduce:
  rule: merge_artifacts
convergence:
  max_rounds: 5
  weak_threshold: 10
  score_epsilon: 0.02
"""
        yaml_file = tmp_path / "protocol.yaml"
        yaml_file.write_text(yaml_content)

        protocol = load_protocol_from_yaml(yaml_file)

        assert protocol.convergence.max_rounds == 5
        assert protocol.convergence.weak_threshold == 10
        assert protocol.convergence.score_epsilon == pytest.approx(0.02)
        assert len(protocol.refine_loop) == 1

    def test_load_protocol_default_convergence(self, tmp_path: Path) -> None:
        """Test loading YAML with default convergence settings."""
        yaml_content = """
name: test_default
protocol_version: v1
max_depth: 1
expand_rules:
  - id: expand_options
    at_step_id: plan
    child_kind: option
    max_children: 3
    depth: 1
    source: planner_output
branch_pipeline:
  - id: propose
    kind: work
    step_name: PROPOSE
    role: proposer
prune:
  rule: epistemic_then_score
  keep_k: 2
reduce:
  rule: merge_artifacts
"""
        yaml_file = tmp_path / "protocol.yaml"
        yaml_file.write_text(yaml_content)

        protocol = load_protocol_from_yaml(yaml_file)

        # Should have default values
        assert protocol.convergence.max_rounds == 0
        assert protocol.convergence.weak_threshold == 5
        assert protocol.convergence.score_epsilon == 0.01


class TestEngineRefinement:
    """Tests for engine refinement loop execution."""

    def _create_refine_protocol(self, max_rounds: int = 2) -> ProtocolSpec:
        """Create a protocol with refinement enabled."""
        return ProtocolSpec(
            name="test_refine_protocol",
            protocol_version="v1",
            max_depth=1,
            expand_rules=[
                ExpandSpec(
                    id="expand_options",
                    at_step_id="plan",
                    child_kind="option",
                    max_children=3,
                    depth=1,
                    source="planner_output",
                ),
            ],
            branch_pipeline=[
                StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
                StepSpec(id="research", kind="work", step_name="RESEARCH", role="researcher"),
                StepSpec(id="validate", kind="validate", step_name="CLAIM_CHECK"),
                StepSpec(id="redteam", kind="work", step_name="REDTEAM", role="redteam"),
            ],
            prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
            reduce=ReduceSpec(rule="merge_artifacts"),
            convergence=ConvergenceSpec(
                max_rounds=max_rounds,
                weak_threshold=10,  # High threshold to allow refinement
                score_epsilon=0.001,  # Low epsilon to allow refinement
            ),
            refine_loop=[
                StepSpec(id="refine", kind="work", step_name="REFINE", role="refiner"),
                StepSpec(id="revalidate", kind="validate", step_name="CLAIM_CHECK"),
            ],
            gates_enabled=False,  # Disable gates for testing
        )

    def test_engine_no_refinement(self) -> None:
        """Test engine without refinement produces no refinement metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(
                runs_dir=runs_dir,
                gates_enabled=False,
            )

            run_dir = engine.run("Should we adopt uv?")

            # Load artifact
            import json

            artifact = json.loads((run_dir / "artifact.json").read_text())

            # Should not have refinement section (default protocol has max_rounds=0)
            assert "refinement" not in artifact

    def test_engine_with_refinement(self) -> None:
        """Test engine with refinement produces refinement metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            protocol = self._create_refine_protocol(max_rounds=2)

            engine = Engine(
                runs_dir=runs_dir,
                gates_enabled=False,
                protocol=protocol,
            )

            run_dir = engine.run("Should we adopt uv?")

            # Load artifact
            import json

            artifact = json.loads((run_dir / "artifact.json").read_text())

            # Should have refinement section
            assert "refinement" in artifact
            assert "total_rounds" in artifact["refinement"]
            assert "converged" in artifact["refinement"]
            assert "stop_reason" in artifact["refinement"]
            assert "history" in artifact["refinement"]

    def test_engine_refinement_trace_events(self) -> None:
        """Test engine emits refinement trace events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            protocol = self._create_refine_protocol(max_rounds=2)

            engine = Engine(
                runs_dir=runs_dir,
                gates_enabled=False,
                protocol=protocol,
            )

            run_dir = engine.run("Should we adopt uv?")

            # Load trace
            import json

            trace_lines = (run_dir / "trace.jsonl").read_text().strip().split("\n")
            events = [json.loads(line) for line in trace_lines]

            # Find refinement events
            event_types = [e["event_type"] for e in events]

            # Should have at least one round started and completed
            assert "refinement_round_started" in event_types
            assert "refinement_round_completed" in event_types

            # Count refinement rounds
            round_started = sum(1 for t in event_types if t == "refinement_round_started")
            assert round_started >= 1

    def test_engine_refinement_bounded_by_max_rounds(self) -> None:
        """Test engine respects max_rounds limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            protocol = self._create_refine_protocol(max_rounds=3)

            engine = Engine(
                runs_dir=runs_dir,
                gates_enabled=False,
                protocol=protocol,
            )

            run_dir = engine.run("Should we adopt uv?")

            # Load artifact
            import json

            artifact = json.loads((run_dir / "artifact.json").read_text())

            # Should not exceed max_rounds
            assert artifact["refinement"]["total_rounds"] <= 3


class TestConvergencePredicates:
    """Tests for convergence predicate checking."""

    def test_convergence_all_predicates_pass(self) -> None:
        """Test convergence when all predicates are satisfied."""
        from delibera.epistemics.ledger import Ledger
        from delibera.epistemics.models import Claim, ClaimStatus, ClaimType

        # Create a mock node with no issues
        class MockNode:
            def __init__(self) -> None:
                self.ledger = Ledger()
                self.ledger.claims = [
                    Claim(
                        claim_id="c1",
                        text="Test claim",
                        claim_type=ClaimType.FACT,
                        confidence=0.9,
                        owner="test",
                        status=ClaimStatus.SUPPORTED,
                    )
                ]
                self.artifact = {"score": 0.8}

        mock_node = MockNode()

        # Create engine and check convergence
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), gates_enabled=False)

            converged, reason = engine._check_convergence(
                merged_node=mock_node,
                weak_threshold=5,
                score_epsilon=0.02,  # Epsilon is 0.02
                previous_score=0.79,  # Delta is 0.01, which is < 0.02
                current_score=0.8,
            )

            assert converged is True
            assert reason == "all_predicates_satisfied"

    def test_convergence_blocking_objections_remain(self) -> None:
        """Test convergence fails with open blocking objections."""
        from delibera.epistemics.ledger import Ledger
        from delibera.epistemics.models import (
            Objection,
            ObjectionSeverity,
            ObjectionStatus,
        )

        class MockNode:
            def __init__(self) -> None:
                self.ledger = Ledger()
                self.ledger.objections = [
                    Objection(
                        objection_id="obj1",
                        target="artifact",
                        severity=ObjectionSeverity.BLOCKING,
                        status=ObjectionStatus.OPEN,
                        rationale="Test objection",
                        owner="test",
                    )
                ]
                self.artifact = {"score": 0.8}

        mock_node = MockNode()

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), gates_enabled=False)

            converged, reason = engine._check_convergence(
                merged_node=mock_node,
                weak_threshold=5,
                score_epsilon=0.01,
                previous_score=0.8,
                current_score=0.8,
            )

            assert converged is False
            assert reason == "blocking_objections_remain"

    def test_convergence_score_still_improving(self) -> None:
        """Test convergence fails when score is still improving."""
        from delibera.epistemics.ledger import Ledger

        class MockNode:
            def __init__(self) -> None:
                self.ledger = Ledger()
                self.artifact = {"score": 0.8}

        mock_node = MockNode()

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), gates_enabled=False)

            converged, reason = engine._check_convergence(
                merged_node=mock_node,
                weak_threshold=5,
                score_epsilon=0.01,
                previous_score=0.5,  # Large delta
                current_score=0.8,
            )

            assert converged is False
            assert reason == "score_still_improving"
