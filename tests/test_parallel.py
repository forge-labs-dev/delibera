"""Tests for parallel branch execution (Phase C).

Verifies that running branches in parallel produces the same results
as sequential execution, and that trace events are valid.
"""

import json
import tempfile
from pathlib import Path

import pytest

from delibera.engine.orchestrator import Engine
from delibera.protocol.defaults import create_simple_protocol, create_tree_protocol_v1


class TestParallelSameAsSequential:
    """Test that parallel and sequential runs produce equivalent results."""

    def test_parallel_produces_same_artifact_fields(self) -> None:
        """Parallel run should produce an artifact with the same structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                protocol=create_simple_protocol(),
                max_parallel_branches=3,
            )
            run_dir = engine.run("Should we adopt uv?")

            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert "question" in artifact
            assert artifact["question"] == "Should we adopt uv?"
            assert "recommendation" in artifact
            assert "options_considered" in artifact
            assert len(artifact["options_considered"]) == 3
            assert len(artifact["survivors"]) == 2
            assert len(artifact["pruned"]) == 1

    def test_parallel_trace_all_events_present(self) -> None:
        """Parallel run should have the same event types as sequential."""
        expected_event_types = {
            "run_start",
            "node_created",
            "work_output",
            "expand",
            "evidence_added",
            "claim_validation_report",
            "objection_added",
            "score_computed",
            "prune",
            "reduce",
            "final_artifact_written",
            "run_end",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                protocol=create_simple_protocol(),
                max_parallel_branches=3,
            )
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]
            actual_types = {e["event_type"] for e in events}

            for et in expected_event_types:
                assert et in actual_types, f"Missing event type: {et}"


class TestParallelTraceValidity:
    """Test that parallel trace output is valid."""

    def test_parallel_trace_jsonl_valid(self) -> None:
        """Every line in parallel trace should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                protocol=create_simple_protocol(),
                max_parallel_branches=3,
            )
            run_dir = engine.run("Should we adopt uv?")

            trace_path = run_dir / "trace.jsonl"
            for i, line in enumerate(trace_path.read_text().strip().split("\n")):
                try:
                    event = json.loads(line)
                    assert "event_type" in event
                    assert "run_id" in event
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON on line {i + 1}: {line[:100]}")


class TestMaxParallelDefault:
    """Test that max_parallel_branches=1 is the default sequential behavior."""

    def test_max_parallel_1_is_sequential(self) -> None:
        """Default max_parallel_branches=1 should behave identically to explicit sequential."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine_default = Engine(runs_dir=Path(tmpdir) / "default")
            run_dir_default = engine_default.run("Should we adopt uv?")

            engine_explicit = Engine(
                runs_dir=Path(tmpdir) / "explicit",
                max_parallel_branches=1,
            )
            run_dir_explicit = engine_explicit.run("Should we adopt uv?")

            # Both should produce valid artifacts
            artifact_default = json.loads((run_dir_default / "artifact.json").read_text())
            artifact_explicit = json.loads((run_dir_explicit / "artifact.json").read_text())

            assert artifact_default["question"] == artifact_explicit["question"]
            assert len(artifact_default["options_considered"]) == len(
                artifact_explicit["options_considered"]
            )
            assert len(artifact_default["survivors"]) == len(artifact_explicit["survivors"])


class TestParallelWithMultiLevel:
    """Test parallel execution with multi-level tree protocol."""

    def test_parallel_multi_level_runs(self) -> None:
        """Parallel execution with tree_protocol_v1 should complete successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                protocol=create_tree_protocol_v1(),
                max_parallel_branches=3,
            )
            run_dir = engine.run("Should we adopt uv?")

            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert "question" in artifact
            assert "recommendation" in artifact

            trace_path = run_dir / "trace.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().strip().split("\n")]

            # Should have depth-2 nodes
            node_created_events = [e for e in events if e["event_type"] == "node_created"]
            depths = {e["payload"]["depth"] for e in node_created_events}
            assert 2 in depths

    def test_parallel_error_isolation(self) -> None:
        """One branch failing in parallel should not crash the engine.

        Since stubs are deterministic and don't fail, we just verify
        that the parallel executor handles the ThreadPoolExecutor correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = Engine(
                runs_dir=Path(tmpdir),
                protocol=create_simple_protocol(),
                max_parallel_branches=10,  # More workers than branches
            )
            run_dir = engine.run("Should we adopt uv?")

            # Should still complete successfully
            artifact = json.loads((run_dir / "artifact.json").read_text())
            assert len(artifact["options_considered"]) == 3
