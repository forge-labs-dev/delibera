"""Integration tests for v0 deliberation runs."""

import json
import tempfile
from pathlib import Path

import pytest

from delibera.engine.orchestrator import Engine


class TestV0Run:
    """Test the v0 vertical slice end-to-end."""

    def test_run_creates_output_files(self) -> None:
        """Test that a run creates trace.jsonl and artifact.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Check run directory exists
            assert run_dir.exists()
            assert run_dir.is_dir()

            # Check trace file exists
            trace_path = run_dir / "trace.jsonl"
            assert trace_path.exists()

            # Check artifact file exists
            artifact_path = run_dir / "artifact.json"
            assert artifact_path.exists()

    def test_artifact_has_required_fields(self) -> None:
        """Test that artifact.json has all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")
            artifact_path = run_dir / "artifact.json"

            artifact = json.loads(artifact_path.read_text())

            # Check required fields
            assert "question" in artifact
            assert artifact["question"] == "Should we adopt uv?"
            assert "recommendation" in artifact
            assert "options_considered" in artifact
            assert "survivors" in artifact
            assert "pruned" in artifact
            assert "confidence" in artifact

            # Check we have 3 options considered
            assert len(artifact["options_considered"]) == 3

            # Check prune/reduce worked: 2 survivors, 1 pruned
            assert len(artifact["survivors"]) == 2
            assert len(artifact["pruned"]) == 1

    def test_trace_has_required_events(self) -> None:
        """Test that trace.jsonl contains all required event types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")
            trace_path = run_dir / "trace.jsonl"

            # Parse JSONL
            events = []
            for line in trace_path.read_text().splitlines():
                events.append(json.loads(line))

            # Collect event types
            event_types = {e["event_type"] for e in events}

            # Check all required event types are present
            required_events = {
                "run_start",
                "node_created",
                "work_output",
                "expand",
                "prune",
                "reduce",
                "final_artifact_written",
                "run_end",
            }
            assert required_events <= event_types, (
                f"Missing events: {required_events - event_types}"
            )

    def test_trace_events_have_required_fields(self) -> None:
        """Test that each trace event has run_id and timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")
            trace_path = run_dir / "trace.jsonl"

            for line in trace_path.read_text().splitlines():
                event = json.loads(line)
                assert "event_type" in event
                assert "run_id" in event
                assert "timestamp" in event
                assert "payload" in event

    def test_trace_events_in_order(self) -> None:
        """Test that trace events are in expected order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")
            trace_path = run_dir / "trace.jsonl"

            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            event_types = [e["event_type"] for e in events]

            # Check order constraints
            assert event_types[0] == "run_start"
            assert event_types[-1] == "run_end"

            # expand comes after plan work_output
            expand_idx = event_types.index("expand")
            plan_idx = next(
                i
                for i, e in enumerate(events)
                if e["event_type"] == "work_output" and e["payload"].get("step") == "PLAN"
            )
            assert expand_idx > plan_idx

            # prune comes after all PROPOSE work_outputs
            prune_idx = event_types.index("prune")
            propose_indices = [
                i
                for i, e in enumerate(events)
                if e["event_type"] == "work_output" and e["payload"].get("step") == "PROPOSE"
            ]
            assert all(prune_idx > idx for idx in propose_indices)

            # reduce comes after prune
            reduce_idx = event_types.index("reduce")
            assert reduce_idx > prune_idx

    def test_run_id_is_consistent(self) -> None:
        """Test that all events in a run have the same run_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")
            trace_path = run_dir / "trace.jsonl"

            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            run_ids = {e["run_id"] for e in events}

            assert len(run_ids) == 1, "All events should have the same run_id"

    def test_multiple_runs_create_separate_directories(self) -> None:
        """Test that multiple runs create separate directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir_1 = engine.run("Question 1")
            run_dir_2 = engine.run("Question 2")

            assert run_dir_1 != run_dir_2
            assert run_dir_1.exists()
            assert run_dir_2.exists()

    def test_empty_question_raises_error(self) -> None:
        """Test that empty questions are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            with pytest.raises(ValueError, match="Question cannot be empty"):
                engine.run("")

            with pytest.raises(ValueError, match="Question cannot be empty"):
                engine.run("   ")

    def test_merged_node_has_node_created_event(self) -> None:
        """Test that the merged node has a node_created event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Test question")
            trace_path = run_dir / "trace.jsonl"

            events = [json.loads(line) for line in trace_path.read_text().splitlines()]

            # Find node_created events
            node_created_events = [e for e in events if e["event_type"] == "node_created"]

            # Should have 5 node_created events:
            # 1 root + 3 options + 1 merged
            assert len(node_created_events) == 5

            # Check merged node is included
            merged_events = [e for e in node_created_events if e["payload"].get("kind") == "merged"]
            assert len(merged_events) == 1
