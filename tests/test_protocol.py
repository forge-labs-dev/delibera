"""Tests for protocol specification and interpretation."""

import tempfile
from pathlib import Path

import pytest

from delibera.engine import Engine
from delibera.protocol import (
    DEFAULT_PROTOCOL,
    ConvergenceSpec,
    ExpandSpec,
    ProtocolInterpreter,
    ProtocolSpec,
    PruneSpec,
    ReduceSpec,
    StepSpec,
    create_simple_protocol,
    create_tree_protocol_v1,
    validate_protocol,
)


class TestProtocolSpec:
    """Tests for protocol specification dataclasses."""

    def test_step_spec_work_requires_role(self) -> None:
        """Test that work steps require a role."""
        with pytest.raises(ValueError, match="requires a role"):
            StepSpec(id="test", kind="work", step_name="TEST", role=None)

    def test_step_spec_validate_does_not_require_role(self) -> None:
        """Test that validate steps don't require a role."""
        step = StepSpec(id="test", kind="validate", step_name="TEST", role=None)
        assert step.role is None

    def test_expand_spec_max_children_positive(self) -> None:
        """Test that max_children must be positive."""
        with pytest.raises(ValueError, match="max_children must be > 0"):
            ExpandSpec(
                at_step_id="plan",
                child_kind="option",
                max_children=0,
                depth=1,
                source="planner_output",
            )

    def test_expand_spec_depth_positive(self) -> None:
        """Test that depth must be positive."""
        with pytest.raises(ValueError, match="depth must be > 0"):
            ExpandSpec(
                at_step_id="plan",
                child_kind="option",
                max_children=3,
                depth=0,
                source="planner_output",
            )

    def test_prune_spec_keep_k_positive(self) -> None:
        """Test that keep_k must be positive."""
        with pytest.raises(ValueError, match="keep_k must be > 0"):
            PruneSpec(rule="epistemic_then_score", keep_k=0)

    def test_convergence_spec_max_rounds_non_negative(self) -> None:
        """Test that max_rounds must be non-negative."""
        with pytest.raises(ValueError, match="max_rounds must be >= 0"):
            ConvergenceSpec(max_rounds=-1)

    def test_protocol_spec_max_depth_positive(self) -> None:
        """Test that max_depth must be positive."""
        with pytest.raises(ValueError, match="max_depth must be > 0"):
            ProtocolSpec(
                name="test",
                max_depth=0,
                expand_rules=[],
                branch_pipeline=[
                    StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer")
                ],
                prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
                reduce=ReduceSpec(rule="merge_artifacts"),
            )

    def test_protocol_spec_name_required(self) -> None:
        """Test that protocol name cannot be empty."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ProtocolSpec(
                name="",
                max_depth=1,
                expand_rules=[],
                branch_pipeline=[
                    StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer")
                ],
                prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
                reduce=ReduceSpec(rule="merge_artifacts"),
            )

    def test_protocol_spec_expand_depth_validation(self) -> None:
        """Test that expand rules cannot exceed max_depth."""
        with pytest.raises(ValueError, match="exceeding max_depth"):
            ProtocolSpec(
                name="test",
                max_depth=1,
                expand_rules=[
                    ExpandSpec(
                        at_step_id="plan",
                        child_kind="option",
                        max_children=3,
                        depth=2,  # Exceeds max_depth of 1
                        source="planner_output",
                    ),
                ],
                branch_pipeline=[
                    StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer")
                ],
                prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
                reduce=ReduceSpec(rule="merge_artifacts"),
            )


class TestProtocolValidation:
    """Tests for validate_protocol function."""

    def test_validate_protocol_valid(self) -> None:
        """Test that valid protocols pass validation."""
        errors = validate_protocol(DEFAULT_PROTOCOL)
        assert errors == []

    def test_validate_protocol_unknown_step_reference(self) -> None:
        """Test that expand rules referencing unknown steps are caught."""
        spec = ProtocolSpec(
            name="test",
            max_depth=1,
            expand_rules=[
                ExpandSpec(
                    at_step_id="unknown_step",  # This step doesn't exist
                    child_kind="option",
                    max_children=3,
                    depth=1,
                    source="planner_output",
                ),
            ],
            branch_pipeline=[
                StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer")
            ],
            prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
            reduce=ReduceSpec(rule="merge_artifacts"),
        )
        errors = validate_protocol(spec)
        assert len(errors) == 1
        assert "unknown step" in errors[0].lower()

    def test_validate_protocol_duplicate_step_ids(self) -> None:
        """Test that duplicate step IDs are caught."""
        spec = ProtocolSpec(
            name="test",
            max_depth=1,
            expand_rules=[],
            branch_pipeline=[
                StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
                StepSpec(id="propose", kind="validate", step_name="VALIDATE"),  # Duplicate
            ],
            prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
            reduce=ReduceSpec(rule="merge_artifacts"),
        )
        errors = validate_protocol(spec)
        assert len(errors) == 1
        assert "duplicate" in errors[0].lower()

    def test_validate_protocol_empty_branch_pipeline(self) -> None:
        """Test that empty branch_pipeline is caught."""
        spec = ProtocolSpec(
            name="test",
            max_depth=1,
            expand_rules=[],
            branch_pipeline=[],  # Empty
            prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
            reduce=ReduceSpec(rule="merge_artifacts"),
        )
        errors = validate_protocol(spec)
        assert len(errors) == 1
        assert "empty" in errors[0].lower()


class TestProtocolDefaults:
    """Tests for default protocol configurations."""

    def test_simple_protocol_structure(self) -> None:
        """Test that simple_protocol has expected structure."""
        spec = create_simple_protocol()
        assert spec.name == "simple_protocol"
        assert spec.max_depth == 1
        assert len(spec.expand_rules) == 1
        assert spec.expand_rules[0].at_step_id == "plan"
        assert spec.expand_rules[0].max_children == 3
        assert len(spec.branch_pipeline) == 2  # propose, validate
        assert spec.prune.keep_k == 2

    def test_tree_protocol_v1_structure(self) -> None:
        """Test that tree_protocol_v1 has expected structure."""
        spec = create_tree_protocol_v1()
        assert spec.name == "tree_protocol_v1"
        assert spec.max_depth == 2
        assert len(spec.expand_rules) == 2  # Level 1 and Level 2
        assert spec.expand_rules[0].depth == 1
        assert spec.expand_rules[1].depth == 2

    def test_default_protocol_is_simple(self) -> None:
        """Test that DEFAULT_PROTOCOL is the simple protocol."""
        assert DEFAULT_PROTOCOL.name == "simple_protocol"


class TestProtocolInterpreter:
    """Tests for the protocol interpreter."""

    def test_interpreter_get_prune_spec(self) -> None:
        """Test getting prune specification from interpreter."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        keep_k, rule = interpreter.get_prune_spec()
        assert keep_k == 2
        assert rule == "epistemic_then_score"

    def test_interpreter_get_reduce_rule(self) -> None:
        """Test getting reduce rule from interpreter."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        rule = interpreter.get_reduce_rule()
        assert rule == "merge_artifacts"

    def test_interpreter_max_depth(self) -> None:
        """Test getting max depth from interpreter."""
        spec = create_tree_protocol_v1()
        interpreter = ProtocolInterpreter(spec)
        assert interpreter.max_depth() == 2

    def test_interpreter_should_expand_after_plan(self) -> None:
        """Test that expansion happens after PLAN step at depth 0."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        expand_rule = interpreter.should_expand_after_step("plan", depth=0)
        assert expand_rule is not None
        assert expand_rule.child_kind == "option"

    def test_interpreter_no_expand_at_wrong_depth(self) -> None:
        """Test that expansion doesn't happen at wrong depth."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        expand_rule = interpreter.should_expand_after_step("plan", depth=1)
        assert expand_rule is None

    def test_interpreter_get_labels_from_planner_output(self) -> None:
        """Test extracting labels from planner output."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        expand_rule = spec.expand_rules[0]
        output = {"branches": ["Option A", "Option B", "Option C"]}
        labels = interpreter.get_expand_labels_from_output(output, expand_rule)
        assert labels == ["Option A", "Option B", "Option C"]

    def test_interpreter_get_labels_from_agent_output(self) -> None:
        """Test extracting labels from agent output (sub_branches)."""
        spec = create_tree_protocol_v1()
        interpreter = ProtocolInterpreter(spec)
        # Second expand rule uses agent_output source
        expand_rule = spec.expand_rules[1]
        output = {"sub_branches": ["Subplan A.1", "Subplan A.2"]}
        labels = interpreter.get_expand_labels_from_output(output, expand_rule)
        assert labels == ["Subplan A.1", "Subplan A.2"]

    def test_interpreter_limits_labels_to_max_children(self) -> None:
        """Test that labels are limited to max_children."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        expand_rule = spec.expand_rules[0]  # max_children=3
        output = {"branches": ["A", "B", "C", "D", "E"]}  # 5 branches
        labels = interpreter.get_expand_labels_from_output(output, expand_rule)
        assert len(labels) == 3  # Limited to max_children

    def test_interpreter_get_branch_pipeline_step(self) -> None:
        """Test getting steps from branch pipeline."""
        spec = create_simple_protocol()
        interpreter = ProtocolInterpreter(spec)
        step = interpreter.get_branch_pipeline_step(0)
        assert step is not None
        assert step.id == "propose"
        step = interpreter.get_branch_pipeline_step(1)
        assert step is not None
        assert step.id == "validate"
        step = interpreter.get_branch_pipeline_step(2)
        assert step is None  # Out of bounds


class TestDefaultProtocolRuns:
    """Tests for running with default protocol."""

    def test_default_protocol_runs(self) -> None:
        """Test that default protocol produces artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Verify outputs exist
            assert (run_dir / "artifact.json").exists()
            assert (run_dir / "trace.jsonl").exists()

    def test_custom_protocol_prune_k(self) -> None:
        """Test that custom keep_k value is used by engine."""
        # Create a protocol with keep_k=1 instead of 2
        custom_protocol = ProtocolSpec(
            name="keep_one",
            max_depth=1,
            expand_rules=[
                ExpandSpec(
                    at_step_id="plan",
                    child_kind="option",
                    max_children=3,
                    depth=1,
                    source="planner_output",
                ),
            ],
            branch_pipeline=[
                StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
                StepSpec(id="validate", kind="validate", step_name="VALIDATE"),
            ],
            prune=PruneSpec(rule="epistemic_then_score", keep_k=1),  # Keep only 1
            reduce=ReduceSpec(rule="merge_artifacts"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir, protocol=custom_protocol)

            run_dir = engine.run("Test question")

            # Check trace for prune event
            import json

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            prune_events = [e for e in events if e["event_type"] == "prune"]

            assert len(prune_events) == 1
            # With keep_k=1, only 1 survivor, 2 pruned
            assert len(prune_events[0]["payload"]["survivor_ids"]) == 1
            assert len(prune_events[0]["payload"]["pruned_ids"]) == 2
