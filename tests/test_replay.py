"""Unit tests for the trace replay subsystem."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from delibera.trace.reader import load_artifact, load_trace
from delibera.trace.replay import (
    ReplayedNodeState,
    ReplayedRun,
    replay_from_directory,
    replay_run,
    verify_replay,
)
from delibera.trace.validate import (
    validate_artifact_derivable,
    validate_trace,
)


class TestTraceReader:
    """Tests for trace file reading."""

    def test_load_trace_parses_jsonl(self, tmp_path: Path) -> None:
        """Test that load_trace correctly parses JSONL format."""
        trace_path = tmp_path / "trace.jsonl"
        events = [
            {"event_type": "run_start", "run_id": "test123"},
            {"event_type": "run_end", "run_id": "test123"},
        ]
        trace_path.write_text("\n".join(json.dumps(e) for e in events))

        loaded = load_trace(trace_path)

        assert len(loaded) == 2
        assert loaded[0]["event_type"] == "run_start"
        assert loaded[1]["event_type"] == "run_end"

    def test_load_trace_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that load_trace skips empty lines."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text('{"event_type": "run_start"}\n\n{"event_type": "run_end"}\n')

        loaded = load_trace(trace_path)

        assert len(loaded) == 2

    def test_load_trace_raises_on_invalid_json(self, tmp_path: Path) -> None:
        """Test that load_trace raises JSONDecodeError for invalid JSON."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text('{"valid": true}\nnot json\n')

        with pytest.raises(json.JSONDecodeError):
            load_trace(trace_path)

    def test_load_artifact_parses_json(self, tmp_path: Path) -> None:
        """Test that load_artifact correctly parses JSON."""
        artifact_path = tmp_path / "artifact.json"
        artifact = {"question": "Test?", "recommendation": "Yes"}
        artifact_path.write_text(json.dumps(artifact))

        loaded = load_artifact(artifact_path)

        assert loaded["question"] == "Test?"
        assert loaded["recommendation"] == "Yes"


class TestTraceValidation:
    """Tests for trace validation."""

    def test_validate_empty_trace(self) -> None:
        """Test validation rejects empty trace."""
        errors = validate_trace([])

        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_validate_missing_run_start(self) -> None:
        """Test validation catches missing run_start."""
        events = [{"event_type": "run_end", "payload": {"status": "completed"}}]
        errors = validate_trace(events)

        assert any("run_start" in e.lower() for e in errors)

    def test_validate_missing_run_end(self) -> None:
        """Test validation catches missing run_end."""
        events = [{"event_type": "run_start", "payload": {"question": "Test?"}}]
        errors = validate_trace(events)

        assert any("run_end" in e.lower() for e in errors)

    def test_validate_inconsistent_run_ids(self) -> None:
        """Test validation catches inconsistent run IDs."""
        events = [
            {"event_type": "run_start", "run_id": "run1"},
            {"event_type": "run_end", "run_id": "run2"},
        ]
        errors = validate_trace(events)

        assert any("inconsistent" in e.lower() for e in errors)

    def test_validate_duplicate_node_id(self) -> None:
        """Test validation catches duplicate node IDs."""
        events = [
            {"event_type": "run_start", "run_id": "test"},
            {"event_type": "node_created", "payload": {"node_id": "n1", "kind": "root"}},
            {"event_type": "node_created", "payload": {"node_id": "n1", "kind": "option"}},
            {"event_type": "run_end", "run_id": "test"},
        ]
        errors = validate_trace(events)

        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_orphan_node(self) -> None:
        """Test validation catches node referencing unknown parent."""
        events = [
            {"event_type": "run_start", "run_id": "test"},
            {"event_type": "node_created", "payload": {"node_id": "n1", "kind": "root"}},
            {
                "event_type": "node_created",
                "payload": {"node_id": "n2", "kind": "option", "parent_id": "unknown"},
            },
            {"event_type": "run_end", "run_id": "test"},
        ]
        errors = validate_trace(events)

        assert any("unknown parent" in e.lower() for e in errors)

    def test_validate_multiple_roots(self) -> None:
        """Test validation catches multiple root nodes."""
        events = [
            {"event_type": "run_start", "run_id": "test"},
            {"event_type": "node_created", "payload": {"node_id": "n1", "kind": "root"}},
            {"event_type": "node_created", "payload": {"node_id": "n2", "kind": "root"}},
            {"event_type": "run_end", "run_id": "test"},
        ]
        errors = validate_trace(events)

        assert any("multiple root" in e.lower() for e in errors)


class TestArtifactValidation:
    """Tests for artifact derivability validation."""

    def test_validate_matching_artifacts(self) -> None:
        """Test validation passes for matching artifacts."""
        replayed = {
            "question": "Should we use X?",
            "recommendation": "Yes",
        }
        stored = {
            "question": "Should we use X?",
            "recommendation": "Yes",
            "options_considered": ["A", "B"],
            "claim_check_summary": {"supported": 1, "weak": 0, "unsupported": 0},
            "open_questions": [],
        }

        errors = validate_artifact_derivable(replayed, stored)

        assert len(errors) == 0

    def test_validate_question_mismatch(self) -> None:
        """Test validation catches question mismatch."""
        replayed = {"question": "Question A"}
        stored = {"question": "Question B", "recommendation": "X"}

        errors = validate_artifact_derivable(replayed, stored)

        assert any("question" in e.lower() and "mismatch" in e.lower() for e in errors)

    def test_validate_missing_required_keys(self) -> None:
        """Test validation catches missing required keys."""
        replayed = {"question": "Test?"}
        stored = {"question": "Test?"}  # Missing recommendation, options, etc.

        errors = validate_artifact_derivable(replayed, stored)

        assert any("recommendation" in e.lower() for e in errors)
        assert any("options_considered" in e.lower() for e in errors)

    def test_validate_empty_recommendation(self) -> None:
        """Test validation catches empty recommendation."""
        replayed = {"question": "Test?"}
        stored = {
            "question": "Test?",
            "recommendation": "",
            "options_considered": [],
            "claim_check_summary": {"supported": 0, "weak": 0, "unsupported": 0},
            "open_questions": [],
        }

        errors = validate_artifact_derivable(replayed, stored)

        assert any("empty recommendation" in e.lower() for e in errors)


class TestReplayRun:
    """Tests for replay_run function."""

    def _make_complete_trace(self) -> list[dict]:
        """Create a complete valid trace for testing."""
        return [
            {
                "event_type": "run_start",
                "run_id": "test_run_123",
                "payload": {"question": "Should we use uv?", "created_at": "2024-01-09T12:00:00"},
            },
            {
                "event_type": "node_created",
                "run_id": "test_run_123",
                "payload": {"node_id": "root", "kind": "root", "depth": 0, "label": "Root"},
            },
            {
                "event_type": "node_created",
                "run_id": "test_run_123",
                "payload": {
                    "node_id": "opt_a",
                    "kind": "option",
                    "depth": 1,
                    "label": "Option A",
                    "parent_id": "root",
                },
            },
            {
                "event_type": "node_created",
                "run_id": "test_run_123",
                "payload": {
                    "node_id": "opt_b",
                    "kind": "option",
                    "depth": 1,
                    "label": "Option B",
                    "parent_id": "root",
                },
            },
            {
                "event_type": "work_output",
                "run_id": "test_run_123",
                "payload": {
                    "step": "PROPOSE",
                    "node_id": "opt_a",
                    "role": "proposer",
                    "output": {"proposal": "Use uv for speed", "pros": ["Fast"]},
                },
            },
            {
                "event_type": "work_output",
                "run_id": "test_run_123",
                "payload": {
                    "step": "PROPOSE",
                    "node_id": "opt_b",
                    "role": "proposer",
                    "output": {"proposal": "Use pip for stability", "pros": ["Stable"]},
                },
            },
            {
                "event_type": "expand",
                "run_id": "test_run_123",
                "payload": {"parent_id": "root", "child_ids": ["opt_a", "opt_b"]},
            },
            {
                "event_type": "claim_validation_report",
                "run_id": "test_run_123",
                "payload": {
                    "node_id": "opt_a",
                    "supported": 1,
                    "weak": 1,
                    "unsupported": 0,
                    "details": [],
                },
            },
            {
                "event_type": "claim_validation_report",
                "run_id": "test_run_123",
                "payload": {
                    "node_id": "opt_b",
                    "supported": 1,
                    "weak": 0,
                    "unsupported": 0,
                    "details": [],
                },
            },
            {
                "event_type": "prune",
                "run_id": "test_run_123",
                "payload": {"survivor_ids": ["opt_a", "opt_b"], "pruned_ids": []},
            },
            {
                "event_type": "node_created",
                "run_id": "test_run_123",
                "payload": {
                    "node_id": "merged",
                    "kind": "merged",
                    "depth": 2,
                    "label": "Merged",
                    "parent_id": "root",
                },
            },
            {
                "event_type": "reduce",
                "run_id": "test_run_123",
                "payload": {
                    "survivor_ids": ["opt_a", "opt_b"],
                    "merged_node_id": "merged",
                    "merged_artifact": {"proposal": "Use uv with pip fallback"},
                },
            },
            {
                "event_type": "final_artifact_written",
                "run_id": "test_run_123",
                "payload": {"artifact_path": "runs/test_run_123/artifact.json"},
            },
            {
                "event_type": "run_end",
                "run_id": "test_run_123",
                "payload": {"status": "completed"},
            },
        ]

    def test_replay_extracts_run_metadata(self) -> None:
        """Test that replay correctly extracts run metadata."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert replayed.run_id == "test_run_123"
        assert replayed.question == "Should we use uv?"
        assert replayed.created_at == "2024-01-09T12:00:00"

    def test_replay_reconstructs_nodes(self) -> None:
        """Test that replay reconstructs all nodes."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert len(replayed.nodes) == 4  # root, opt_a, opt_b, merged
        assert "root" in replayed.nodes
        assert "opt_a" in replayed.nodes
        assert "opt_b" in replayed.nodes
        assert "merged" in replayed.nodes

    def test_replay_identifies_root_node(self) -> None:
        """Test that replay identifies the root node."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert replayed.root_id == "root"
        assert replayed.nodes["root"].kind == "root"

    def test_replay_identifies_merged_node(self) -> None:
        """Test that replay identifies the merged node."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert replayed.merged_id == "merged"
        assert replayed.nodes["merged"].kind == "merged"

    def test_replay_attaches_artifacts(self) -> None:
        """Test that replay attaches artifacts from work_output."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert replayed.nodes["opt_a"].artifact["proposal"] == "Use uv for speed"
        assert replayed.nodes["opt_b"].artifact["proposal"] == "Use pip for stability"
        assert replayed.nodes["merged"].artifact["proposal"] == "Use uv with pip fallback"

    def test_replay_attaches_claim_summaries(self) -> None:
        """Test that replay attaches claim validation summaries."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert replayed.nodes["opt_a"].claim_summary == {
            "supported": 1,
            "weak": 1,
            "unsupported": 0,
        }
        assert replayed.nodes["opt_b"].claim_summary == {
            "supported": 1,
            "weak": 0,
            "unsupported": 0,
        }

    def test_replay_marks_merged_status(self) -> None:
        """Test that replay marks survivor nodes as merged."""
        events = self._make_complete_trace()

        replayed = replay_run(events)

        assert replayed.nodes["opt_a"].status == "merged"
        assert replayed.nodes["opt_b"].status == "merged"

    def test_replay_marks_pruned_status(self) -> None:
        """Test that replay marks pruned nodes."""
        events = self._make_complete_trace()
        # Modify to prune opt_b
        for event in events:
            if event["event_type"] == "prune":
                event["payload"]["survivor_ids"] = ["opt_a"]
                event["payload"]["pruned_ids"] = ["opt_b"]
            elif event["event_type"] == "reduce":
                event["payload"]["survivor_ids"] = ["opt_a"]

        replayed = replay_run(events)

        assert replayed.nodes["opt_a"].status == "merged"
        assert replayed.nodes["opt_b"].status == "pruned"

    def test_replay_reports_errors_for_invalid_trace(self) -> None:
        """Test that replay reports errors for invalid trace."""
        events = [{"event_type": "something"}]  # Invalid trace

        replayed = replay_run(events)

        assert len(replayed.errors) > 0

    def test_replay_derives_artifact_when_not_loaded(self) -> None:
        """Test that replay derives artifact when artifact.json not available."""
        events = self._make_complete_trace()

        replayed = replay_run(events, artifact_path=None)

        # Should derive artifact and add warning
        assert replayed.final_artifact.get("question") == "Should we use uv?"
        assert any("derived" in w.lower() for w in replayed.warnings)

    def test_replay_loads_artifact_from_path(self, tmp_path: Path) -> None:
        """Test that replay loads artifact from provided path."""
        events = self._make_complete_trace()
        artifact_path = tmp_path / "artifact.json"
        artifact = {
            "question": "Should we use uv?",
            "recommendation": "Yes, use uv",
            "options_considered": ["Option A", "Option B"],
            "claim_check_summary": {"supported": 2, "weak": 1, "unsupported": 0},
            "open_questions": [],
        }
        artifact_path.write_text(json.dumps(artifact))

        replayed = replay_run(events, artifact_path=artifact_path)

        assert replayed.final_artifact["recommendation"] == "Yes, use uv"
        assert "derived" not in str(replayed.warnings).lower()


class TestReplayFromDirectory:
    """Tests for replay_from_directory convenience function."""

    def test_replay_from_directory(self, tmp_path: Path) -> None:
        """Test replay_from_directory loads and replays correctly."""
        # Create run directory
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Write trace
        trace = [
            {
                "event_type": "run_start",
                "run_id": "test",
                "payload": {"question": "Test?", "created_at": "2024-01-09"},
            },
            {
                "event_type": "node_created",
                "run_id": "test",
                "payload": {"node_id": "root", "kind": "root", "depth": 0},
            },
            {
                "event_type": "node_created",
                "run_id": "test",
                "payload": {"node_id": "opt", "kind": "option", "depth": 1, "parent_id": "root"},
            },
            {
                "event_type": "work_output",
                "run_id": "test",
                "payload": {"step": "PROPOSE", "node_id": "opt", "output": {"proposal": "X"}},
            },
            {
                "event_type": "expand",
                "run_id": "test",
                "payload": {"parent_id": "root", "child_ids": ["opt"]},
            },
            {
                "event_type": "claim_validation_report",
                "run_id": "test",
                "payload": {
                    "node_id": "opt",
                    "supported": 1,
                    "weak": 0,
                    "unsupported": 0,
                },
            },
            {
                "event_type": "prune",
                "run_id": "test",
                "payload": {"survivor_ids": ["opt"], "pruned_ids": []},
            },
            {
                "event_type": "node_created",
                "run_id": "test",
                "payload": {
                    "node_id": "merged",
                    "kind": "merged",
                    "depth": 2,
                    "parent_id": "root",
                },
            },
            {
                "event_type": "reduce",
                "run_id": "test",
                "payload": {
                    "survivor_ids": ["opt"],
                    "merged_node_id": "merged",
                    "merged_artifact": {},
                },
            },
            {"event_type": "final_artifact_written", "run_id": "test", "payload": {}},
            {"event_type": "run_end", "run_id": "test", "payload": {"status": "completed"}},
        ]
        trace_path = run_dir / "trace.jsonl"
        trace_path.write_text("\n".join(json.dumps(e) for e in trace))

        # Write artifact
        artifact = {
            "question": "Test?",
            "recommendation": "Do X",
            "options_considered": ["opt"],
            "claim_check_summary": {"supported": 1, "weak": 0, "unsupported": 0},
            "open_questions": [],
        }
        artifact_path = run_dir / "artifact.json"
        artifact_path.write_text(json.dumps(artifact))

        replayed = replay_from_directory(run_dir)

        assert replayed.run_id == "test"
        assert replayed.question == "Test?"
        assert replayed.final_artifact["recommendation"] == "Do X"

    def test_replay_from_directory_missing_trace(self, tmp_path: Path) -> None:
        """Test replay_from_directory raises for missing trace."""
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            replay_from_directory(run_dir)


class TestVerifyReplay:
    """Tests for verify_replay function."""

    def test_verify_replay_passes_for_consistent_artifact(self) -> None:
        """Test verify_replay passes for consistent artifacts."""
        replayed = ReplayedRun(
            run_id="test",
            question="Test question?",
            created_at="2024-01-09",
            nodes={
                "root": ReplayedNodeState("root", None, "root", 0),
                "opt": ReplayedNodeState("opt", "root", "option", 1, "Option 1", status="merged"),
            },
            root_id="root",
            merged_id=None,
        )
        stored = {
            "question": "Test question?",
            "recommendation": "Do it",
            "options_considered": ["Option 1"],
            "claim_check_summary": {"supported": 0, "weak": 0, "unsupported": 0},
            "open_questions": [],
        }

        issues = verify_replay(replayed, stored)

        assert len(issues) == 0

    def test_verify_replay_detects_question_mismatch(self) -> None:
        """Test verify_replay detects question mismatch."""
        replayed = ReplayedRun(
            run_id="test",
            question="Question A?",
            created_at="2024-01-09",
        )
        stored = {
            "question": "Question B?",
            "recommendation": "X",
            "options_considered": [],
            "claim_check_summary": {"supported": 0, "weak": 0, "unsupported": 0},
            "open_questions": [],
        }

        issues = verify_replay(replayed, stored)

        assert any("question" in i.lower() for i in issues)


class TestReplayNoAgentsCalled:
    """Tests ensuring replay does NOT call any agents."""

    def test_replay_does_not_import_agents(self) -> None:
        """Test that replay module doesn't import agent classes."""
        # Check that replay.py imports don't include agent modules
        import delibera.trace.replay as replay_module

        # Get all names imported in the module
        module_dict = dir(replay_module)

        # Should not have any agent-related names
        agent_names = ["ProposerAgent", "RedTeamAgent", "SynthesizerAgent", "Agent"]
        for name in agent_names:
            assert name not in module_dict, f"Replay module should not import {name}"

    def test_replay_with_mocked_agents_raises_no_errors(self) -> None:
        """Test that replay works even if agents would raise errors."""
        # This verifies replay doesn't instantiate or call agents

        # Create a minimal valid trace
        events = [
            {
                "event_type": "run_start",
                "run_id": "test",
                "payload": {"question": "Test?", "created_at": "2024-01-09"},
            },
            {
                "event_type": "node_created",
                "run_id": "test",
                "payload": {"node_id": "root", "kind": "root", "depth": 0},
            },
            {
                "event_type": "node_created",
                "run_id": "test",
                "payload": {"node_id": "opt", "kind": "option", "depth": 1, "parent_id": "root"},
            },
            {
                "event_type": "work_output",
                "run_id": "test",
                "payload": {"step": "PROPOSE", "node_id": "opt", "output": {"proposal": "X"}},
            },
            {
                "event_type": "expand",
                "run_id": "test",
                "payload": {"parent_id": "root", "child_ids": ["opt"]},
            },
            {
                "event_type": "claim_validation_report",
                "run_id": "test",
                "payload": {
                    "node_id": "opt",
                    "supported": 1,
                    "weak": 0,
                    "unsupported": 0,
                },
            },
            {
                "event_type": "prune",
                "run_id": "test",
                "payload": {"survivor_ids": ["opt"], "pruned_ids": []},
            },
            {
                "event_type": "node_created",
                "run_id": "test",
                "payload": {
                    "node_id": "merged",
                    "kind": "merged",
                    "depth": 2,
                    "parent_id": "root",
                },
            },
            {
                "event_type": "reduce",
                "run_id": "test",
                "payload": {
                    "survivor_ids": ["opt"],
                    "merged_node_id": "merged",
                    "merged_artifact": {},
                },
            },
            {"event_type": "final_artifact_written", "run_id": "test", "payload": {}},
            {"event_type": "run_end", "run_id": "test", "payload": {"status": "completed"}},
        ]

        # Monkeypatch agents to raise if called
        def raise_error(*args, **kwargs):
            raise RuntimeError("Agent should NOT be called during replay!")

        # Patch the agent module at the point where it would be imported
        with patch.dict("sys.modules", {
            "delibera.agents.proposer": MagicMock(ProposerAgent=raise_error),
            "delibera.agents.synthesizer": MagicMock(SynthesizerAgent=raise_error),
        }):
            # This should succeed without calling any agents
            replayed = replay_run(events)

            assert replayed.run_id == "test"
            assert replayed.question == "Test?"
            assert "opt" in replayed.nodes


class TestReplayedDataclasses:
    """Tests for ReplayedRun and ReplayedNodeState dataclasses."""

    def test_replayed_node_state_defaults(self) -> None:
        """Test ReplayedNodeState default values."""
        node = ReplayedNodeState(
            node_id="test",
            parent_id=None,
            kind="option",
            depth=1,
        )

        assert node.label == ""
        assert node.artifact == {}
        assert node.claim_summary is None
        assert node.status == "active"

    def test_replayed_run_is_valid(self) -> None:
        """Test ReplayedRun.is_valid property."""
        valid_run = ReplayedRun(
            run_id="test",
            question="Test?",
            created_at="2024-01-09",
            errors=[],
        )
        invalid_run = ReplayedRun(
            run_id="test",
            question="Test?",
            created_at="2024-01-09",
            errors=["Something went wrong"],
        )

        assert valid_run.is_valid is True
        assert invalid_run.is_valid is False
