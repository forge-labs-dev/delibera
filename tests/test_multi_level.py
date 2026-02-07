"""Tests for multi-level tree expansion (Phase B).

Verifies that the engine correctly processes trees with max_depth > 1,
recursing into sub-levels after the appropriate pipeline step.
"""

import json
import tempfile
from pathlib import Path

import pytest

from delibera.engine.operators import expand
from delibera.engine.orchestrator import Engine
from delibera.engine.tree import DeliberationTree
from delibera.protocol.defaults import create_simple_protocol, create_tree_protocol_v1


class TestExpandWithKind:
    """Test that operators.expand() creates correct node kinds."""

    def test_expand_default_kind_is_option(self) -> None:
        """Default kind should be 'option'."""
        tree = DeliberationTree()
        root = tree.create_root("question")
        children = expand(tree, root.node_id, ["A", "B"])
        assert all(c.kind == "option" for c in children)

    def test_expand_with_plan_kind(self) -> None:
        """Passing kind='plan' creates plan nodes."""
        tree = DeliberationTree()
        root = tree.create_root("question")
        children = expand(tree, root.node_id, ["Sub 1", "Sub 2"], kind="plan")
        assert all(c.kind == "plan" for c in children)

    def test_expand_with_risk_kind(self) -> None:
        """Passing kind='risk' creates risk nodes."""
        tree = DeliberationTree()
        root = tree.create_root("question")
        children = expand(tree, root.node_id, ["Risk 1"], kind="risk")
        assert children[0].kind == "risk"


class TestProposerStubSubBranches:
    """Test that ProposerStub includes sub_branches in output."""

    def test_proposer_stub_includes_sub_branches(self) -> None:
        """ProposerStub output should contain sub_branches list."""
        from delibera.agents.stub import ProposerStub

        stub = ProposerStub()
        output = stub.execute({"label": "Option A: something", "question": "q"})
        assert "sub_branches" in output
        assert isinstance(output["sub_branches"], list)
        assert len(output["sub_branches"]) == 2
        assert "Subplan A.1" in output["sub_branches"][0]
        assert "Subplan A.2" in output["sub_branches"][1]

    def test_proposer_stub_sub_branches_match_option(self) -> None:
        """Sub-branches should use the correct option letter."""
        from delibera.agents.stub import ProposerStub

        stub = ProposerStub()
        output_b = stub.execute({"label": "Option B: incremental", "question": "q"})
        assert "Subplan B.1" in output_b["sub_branches"][0]

        output_c = stub.execute({"label": "Option C: wait", "question": "q"})
        assert "Subplan C.1" in output_c["sub_branches"][0]


class TestSimpleProtocolUnchanged:
    """Test that simple_protocol (max_depth=1) behaves identically."""

    def test_simple_protocol_produces_same_results(self) -> None:
        """With max_depth=1, no sub-expansion should occur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_simple_protocol())
            run_dir = engine.run("Should we adopt uv?")

            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert len(artifact["options_considered"]) == 3
            assert len(artifact["survivors"]) == 2
            assert len(artifact["pruned"]) == 1

    def test_simple_protocol_no_sub_expansion_events(self) -> None:
        """Simple protocol should not emit sub_expansion_triggered events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_simple_protocol())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]
            event_types = [e["event_type"] for e in events]
            assert "sub_expansion_triggered" not in event_types


class TestTreeProtocolV1MultiLevel:
    """Test the tree_protocol_v1 with 2-level expansion."""

    def test_tree_protocol_v1_runs_end_to_end(self) -> None:
        """tree_protocol_v1 should complete without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            assert run_dir.exists()
            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert "question" in artifact
            assert "recommendation" in artifact

    def test_tree_protocol_v1_creates_depth2_nodes(self) -> None:
        """Trace should contain node_created events at depth 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            node_created_events = [e for e in events if e["event_type"] == "node_created"]
            depths = [e["payload"]["depth"] for e in node_created_events]

            # Should have nodes at depth 0 (root), 1 (options), 2 (subplans)
            assert 0 in depths
            assert 1 in depths
            assert 2 in depths

    def test_tree_protocol_v1_sub_expansion_events(self) -> None:
        """Trace should contain sub_expansion_triggered events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            sub_expansion_events = [
                e for e in events if e["event_type"] == "sub_expansion_triggered"
            ]
            # Each of the 3 options should trigger sub-expansion
            assert len(sub_expansion_events) == 3

            # Verify payload structure
            for event in sub_expansion_events:
                assert "parent_node_id" in event["payload"]
                assert "sub_labels" in event["payload"]
                assert "child_kind" in event["payload"]
                assert event["payload"]["child_kind"] == "plan"
                assert event["payload"]["target_depth"] == 2

    def test_depth2_nodes_have_plan_kind(self) -> None:
        """Depth-2 nodes should have kind='plan'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            depth2_nodes = [
                e
                for e in events
                if e["event_type"] == "node_created"
                and e["payload"].get("depth") == 2
                and e["payload"].get("kind") != "merged"
            ]
            assert len(depth2_nodes) > 0
            for node_event in depth2_nodes:
                assert node_event["payload"]["kind"] == "plan"

    def test_depth2_pipeline_runs_all_steps(self) -> None:
        """Depth-2 nodes should have work_output events for all pipeline steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            # Collect depth-2 node IDs
            depth2_node_ids = {
                e["payload"]["node_id"]
                for e in events
                if e["event_type"] == "node_created"
                and e["payload"].get("depth") == 2
                and e["payload"].get("kind") != "merged"
            }

            assert len(depth2_node_ids) > 0

            # Check that work_output events exist for depth-2 nodes
            work_outputs = [e for e in events if e["event_type"] == "work_output"]
            depth2_work_steps: dict[str, set[str]] = {}
            for wo in work_outputs:
                nid = wo["payload"]["node_id"]
                if nid in depth2_node_ids:
                    depth2_work_steps.setdefault(nid, set()).add(wo["payload"]["step"])

            # Each depth-2 node should have PROPOSE, RESEARCH, and REDTEAM
            for nid, steps in depth2_work_steps.items():
                assert "PROPOSE" in steps, f"Node {nid} missing PROPOSE"
                assert "RESEARCH" in steps, f"Node {nid} missing RESEARCH"
                assert "REDTEAM" in steps, f"Node {nid} missing REDTEAM"

    def test_depth2_prune_reduce_events(self) -> None:
        """Trace should have prune and reduce events at depth 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            prune_events = [e for e in events if e["event_type"] == "prune"]
            reduce_events = [e for e in events if e["event_type"] == "reduce"]

            # Should have prune/reduce at both depth 1 and depth 2
            prune_depths = {e["payload"].get("depth") for e in prune_events}
            reduce_depths = {e["payload"].get("depth") for e in reduce_events}

            assert 1 in prune_depths
            assert 2 in prune_depths
            assert 1 in reduce_depths
            assert 2 in reduce_depths

    def test_subtree_merge_enriches_parent_ledger(self) -> None:
        """After sub-tree merge, parent should have subtree_merged_into_parent events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            merge_events = [e for e in events if e["event_type"] == "subtree_merged_into_parent"]
            # Each of the 3 options should have its sub-tree merged back
            assert len(merge_events) == 3

    def test_tree_protocol_v1_valid_trace(self) -> None:
        """Every line in trace.jsonl should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=create_tree_protocol_v1())
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            for i, line in enumerate(trace_path.read_text().strip().split("\n")):
                try:
                    event = json.loads(line)
                    assert "event_type" in event
                    assert "run_id" in event
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON on line {i + 1}: {line[:100]}")


class TestMaxDepth3Protocol:
    """Test a custom 3-level protocol."""

    def test_max_depth_3_protocol(self) -> None:
        """A custom 3-level protocol should work (though no expand rule at depth 3)."""
        from delibera.protocol.spec import (
            ConvergenceSpec,
            ExpandSpec,
            ProtocolSpec,
            PruneSpec,
            ReduceSpec,
            StepSpec,
        )

        protocol = ProtocolSpec(
            name="three_level_test",
            protocol_version="v1",
            max_depth=3,
            expand_rules=[
                ExpandSpec(
                    id="expand_options",
                    at_step_id="plan",
                    child_kind="option",
                    max_children=2,
                    depth=1,
                    source="planner_output",
                ),
                ExpandSpec(
                    id="expand_subplans",
                    at_step_id="propose",
                    child_kind="plan",
                    max_children=2,
                    depth=2,
                    source="agent_output",
                ),
                # Depth 3: expand subplans into risks after propose
                ExpandSpec(
                    id="expand_risks",
                    at_step_id="propose",
                    child_kind="risk",
                    max_children=2,
                    depth=3,
                    source="agent_output",
                ),
            ],
            branch_pipeline=[
                StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
                StepSpec(id="research", kind="work", step_name="RESEARCH", role="researcher"),
                StepSpec(id="validate", kind="validate", step_name="CLAIM_CHECK"),
                StepSpec(id="redteam", kind="work", step_name="REDTEAM", role="redteam"),
            ],
            prune=PruneSpec(rule="epistemic_then_score", keep_k=1),
            reduce=ReduceSpec(rule="merge_artifacts"),
            convergence=ConvergenceSpec(max_rounds=0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(runs_dir=Path(tmpdir), protocol=protocol)
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            node_created_events = [e for e in events if e["event_type"] == "node_created"]
            depths = {e["payload"]["depth"] for e in node_created_events}

            # Should have nodes at depth 0, 1, 2, 3
            assert 0 in depths
            assert 1 in depths
            assert 2 in depths
            assert 3 in depths

            # Depth-3 nodes should have kind='risk'
            depth3_nodes = [
                e
                for e in node_created_events
                if e["payload"]["depth"] == 3 and e["payload"]["kind"] != "merged"
            ]
            for n in depth3_nodes:
                assert n["payload"]["kind"] == "risk"
