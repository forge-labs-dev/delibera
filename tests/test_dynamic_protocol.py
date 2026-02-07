"""Tests for dynamic protocol features (Phase D).

Verifies dominance threshold early termination, conditional expansion,
and YAML loader support for new fields.
"""

import json
import tempfile
from pathlib import Path

from delibera.engine.orchestrator import Engine
from delibera.protocol.defaults import create_simple_protocol
from delibera.protocol.spec import (
    ConvergenceSpec,
    ExpandSpec,
    ProtocolSpec,
    PruneSpec,
    ReduceSpec,
    StepSpec,
)


def _make_protocol(
    dominance_threshold: float | None = None,
    expand_condition: str | None = None,
    max_depth: int = 1,
) -> ProtocolSpec:
    """Helper to build a protocol for testing."""
    expand_rules = [
        ExpandSpec(
            id="expand_options",
            at_step_id="plan",
            child_kind="option",
            max_children=3,
            depth=1,
            source="planner_output",
        ),
    ]

    if max_depth >= 2:
        expand_rules.append(
            ExpandSpec(
                id="expand_subplans",
                at_step_id="propose",
                child_kind="plan",
                max_children=2,
                depth=2,
                source="agent_output",
                condition=expand_condition,
            ),
        )

    return ProtocolSpec(
        name="test_dynamic",
        protocol_version="v1",
        max_depth=max_depth,
        expand_rules=expand_rules,
        branch_pipeline=[
            StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
            StepSpec(id="research", kind="work", step_name="RESEARCH", role="researcher"),
            StepSpec(id="validate", kind="validate", step_name="CLAIM_CHECK"),
            StepSpec(id="redteam", kind="work", step_name="REDTEAM", role="redteam"),
        ],
        prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
        reduce=ReduceSpec(rule="merge_artifacts"),
        convergence=ConvergenceSpec(
            max_rounds=0,
            dominance_threshold=dominance_threshold,
        ),
    )


class TestDominanceThreshold:
    """Test early termination via dominance detection."""

    def test_dominance_threshold_prunes_to_one(self) -> None:
        """With a very low dominance threshold, the top option should dominate."""
        # score_node() computes weighted scores from ledger metrics.
        # With stubs, scores are close but not identical.
        # Use threshold=1.0 (any difference triggers dominance).
        protocol = _make_protocol(dominance_threshold=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            artifact = json.loads((run_dir / "artifact.json").read_text())
            # Only 1 survivor due to dominance (threshold 1.0 means any top > second)
            assert len(artifact["survivors"]) == 1

            # Trace should have early_termination event
            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]
            early_events = [e for e in events if e["event_type"] == "early_termination"]
            assert len(early_events) == 1
            assert early_events[0]["payload"]["reason"] == "dominance"

    def test_no_dominance_keeps_k(self) -> None:
        """Without dominance threshold, normal pruning keeps keep_k."""
        protocol = _make_protocol(dominance_threshold=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert len(artifact["survivors"]) == 2  # keep_k=2

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]
            early_events = [e for e in events if e["event_type"] == "early_termination"]
            assert len(early_events) == 0

    def test_high_threshold_no_dominance(self) -> None:
        """With a very high threshold, no dominance is detected."""
        # Ratio A/B = 1.33, threshold 5.0 → no dominance
        protocol = _make_protocol(dominance_threshold=5.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert len(artifact["survivors"]) == 2  # Normal pruning


class TestConditionalExpansion:
    """Test conditional expansion based on node metrics."""

    def test_expansion_skipped_on_condition(self) -> None:
        """Expansion should be skipped when condition is not met.

        With 'evidence_count > 100', no stub node will have 100+ evidence items,
        so expansion should be skipped for all nodes.
        """
        protocol = _make_protocol(max_depth=2, expand_condition="evidence_count > 100")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            # Should have expansion_skipped events
            skipped = [e for e in events if e["event_type"] == "expansion_skipped"]
            assert len(skipped) > 0

            # Should NOT have depth-2 nodes (since expansion was skipped)
            node_created = [e for e in events if e["event_type"] == "node_created"]
            depths = {e["payload"]["depth"] for e in node_created}
            assert 2 not in depths

    def test_expansion_proceeds_when_condition_met(self) -> None:
        """Expansion should proceed when condition IS met.

        With 'evidence_count >= 0', all nodes satisfy the condition.
        """
        protocol = _make_protocol(max_depth=2, expand_condition="evidence_count >= 0")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            # Should have depth-2 nodes (expansion proceeded)
            node_created = [e for e in events if e["event_type"] == "node_created"]
            depths = {e["payload"]["depth"] for e in node_created}
            assert 2 in depths

            # No expansion_skipped events
            skipped = [e for e in events if e["event_type"] == "expansion_skipped"]
            assert len(skipped) == 0

    def test_no_condition_always_expands(self) -> None:
        """Without condition, expansion always proceeds (backward compat)."""
        protocol = _make_protocol(max_depth=2, expand_condition=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            node_created = [e for e in events if e["event_type"] == "node_created"]
            depths = {e["payload"]["depth"] for e in node_created}
            assert 2 in depths


class TestYamlLoaderNewFields:
    """Test that YAML loader correctly parses new fields."""

    def test_yaml_loads_dominance_threshold(self) -> None:
        """dominance_threshold should parse from YAML."""
        from delibera.protocol.loader import load_protocol_from_yaml

        yaml_content = """
name: test_protocol
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
convergence:
  max_rounds: 0
  dominance_threshold: 1.5
"""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            spec = load_protocol_from_yaml(Path(f.name))

        assert spec.convergence.dominance_threshold == 1.5

    def test_yaml_loads_condition(self) -> None:
        """condition should parse from YAML."""
        from delibera.protocol.loader import load_protocol_from_yaml

        yaml_content = """
name: test_protocol
protocol_version: v1
max_depth: 2
expand_rules:
  - id: expand_options
    at_step_id: plan
    child_kind: option
    max_children: 3
    depth: 1
    source: planner_output
  - id: expand_subplans
    at_step_id: propose
    child_kind: plan
    max_children: 2
    depth: 2
    source: agent_output
    condition: "unsupported_claims > 0"
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
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            spec = load_protocol_from_yaml(Path(f.name))

        assert spec.expand_rules[1].condition == "unsupported_claims > 0"

    def test_yaml_no_new_fields_backward_compat(self) -> None:
        """Protocols without new fields should still parse correctly."""
        from delibera.protocol.loader import load_protocol_from_yaml

        yaml_content = """
name: test_protocol
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
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            spec = load_protocol_from_yaml(Path(f.name))

        assert spec.convergence.dominance_threshold is None
        assert spec.expand_rules[0].condition is None


class TestInterpreterCondition:
    """Test the ProtocolInterpreter.evaluate_expand_condition method."""

    def test_evaluate_no_condition_returns_true(self) -> None:
        """No condition should always return True."""
        from delibera.engine.tree import DeliberationTree
        from delibera.protocol.interpreter import ProtocolInterpreter

        protocol = create_simple_protocol()
        interp = ProtocolInterpreter(protocol)

        tree = DeliberationTree()
        root = tree.create_root("test")
        child = tree.add_child(root.node_id, "Option A", kind="option")

        rule = ExpandSpec(
            id="test",
            at_step_id="propose",
            child_kind="plan",
            max_children=2,
            depth=2,
            source="agent_output",
            condition=None,
        )
        assert interp.evaluate_expand_condition(rule, child) is True

    def test_evaluate_condition_true(self) -> None:
        """Condition that is satisfied should return True."""
        from delibera.engine.tree import DeliberationTree
        from delibera.protocol.interpreter import ProtocolInterpreter

        protocol = create_simple_protocol()
        interp = ProtocolInterpreter(protocol)

        tree = DeliberationTree()
        root = tree.create_root("test")
        child = tree.add_child(root.node_id, "Option A", kind="option")
        # evidence_count is 0, so "evidence_count == 0" is true
        rule = ExpandSpec(
            id="test",
            at_step_id="propose",
            child_kind="plan",
            max_children=2,
            depth=2,
            source="agent_output",
            condition="evidence_count == 0",
        )
        assert interp.evaluate_expand_condition(rule, child) is True

    def test_evaluate_condition_false(self) -> None:
        """Condition that is not satisfied should return False."""
        from delibera.engine.tree import DeliberationTree
        from delibera.protocol.interpreter import ProtocolInterpreter

        protocol = create_simple_protocol()
        interp = ProtocolInterpreter(protocol)

        tree = DeliberationTree()
        root = tree.create_root("test")
        child = tree.add_child(root.node_id, "Option A", kind="option")
        # evidence_count is 0, so "evidence_count > 5" is false
        rule = ExpandSpec(
            id="test",
            at_step_id="propose",
            child_kind="plan",
            max_children=2,
            depth=2,
            source="agent_output",
            condition="evidence_count > 5",
        )
        assert interp.evaluate_expand_condition(rule, child) is False
