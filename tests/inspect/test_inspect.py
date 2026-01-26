"""Tests for the inspect module and CLI commands."""

import tempfile
from pathlib import Path

import pytest

from delibera.engine.orchestrator import Engine
from delibera.inspect import build_run_summary, render_markdown, render_text


class TestInspectOutputsSections:
    """Test that inspect output contains required sections."""

    def test_inspect_has_final_recommendation(self) -> None:
        """Test that inspect output includes Final Recommendation section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            output = render_text(summary)

            assert "Final Recommendation" in output

    def test_inspect_has_selected_path(self) -> None:
        """Test that inspect output includes Selected Path section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            output = render_text(summary)

            assert "Selected Path" in output

    def test_inspect_has_claims_evidence_sections(self) -> None:
        """Test that inspect output includes Claims and Evidence info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            output = render_text(summary)

            # Should have claims info
            assert "Claims" in output or "supported" in output.lower()
            # Should have evidence info
            assert "Evidence" in output or "evidence" in output.lower()

    def test_inspect_has_pruning_section(self) -> None:
        """Test that inspect output includes Pruning section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            output = render_text(summary)

            assert "Pruning" in output

    def test_inspect_has_statistics(self) -> None:
        """Test that inspect output includes Statistics section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            output = render_text(summary)

            assert "Statistics" in output
            assert "Total nodes" in output
            assert "Tool calls" in output

    def test_inspect_shows_question(self) -> None:
        """Test that inspect output shows the original question."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            question = "Should we adopt uv?"
            run_dir = engine.run(question)

            summary = build_run_summary(run_dir)
            output = render_text(summary)

            assert question in output


class TestReportDeterministic:
    """Test that report generation is deterministic."""

    def test_report_md_deterministic(self) -> None:
        """Test that generating markdown report twice produces identical output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Generate report twice
            summary1 = build_run_summary(run_dir)
            report1 = render_markdown(summary1)

            summary2 = build_run_summary(run_dir)
            report2 = render_markdown(summary2)

            # Should be identical
            assert report1 == report2

    def test_text_output_deterministic(self) -> None:
        """Test that text rendering is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            # Render twice
            summary1 = build_run_summary(run_dir)
            text1 = render_text(summary1)

            summary2 = build_run_summary(run_dir)
            text2 = render_text(summary2)

            # Should be identical
            assert text1 == text2


class TestInspectReadOnly:
    """Test that inspect/report does not create new runs."""

    def test_inspect_does_not_create_run_directories(self) -> None:
        """Test that inspecting a run does not create any new run directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            # Create one run
            run_dir = engine.run("Should we adopt uv?")

            # Count directories before inspect
            dirs_before = list(runs_dir.iterdir())

            # Inspect the run
            summary = build_run_summary(run_dir)
            _ = render_text(summary)

            # Count directories after inspect
            dirs_after = list(runs_dir.iterdir())

            # Should be the same count
            assert len(dirs_before) == len(dirs_after)
            assert set(dirs_before) == set(dirs_after)

    def test_report_does_not_create_run_directories(self) -> None:
        """Test that generating a report does not create any new run directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            # Create one run
            run_dir = engine.run("Should we adopt uv?")

            # Count directories before report
            dirs_before = list(runs_dir.iterdir())

            # Generate report
            summary = build_run_summary(run_dir)
            _ = render_markdown(summary)

            # Count directories after report
            dirs_after = list(runs_dir.iterdir())

            # Should be the same count
            assert len(dirs_before) == len(dirs_after)
            assert set(dirs_before) == set(dirs_after)


class TestRunSummaryStructure:
    """Test RunSummary data structure."""

    def test_summary_has_run_id(self) -> None:
        """Test that summary includes run_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)

            assert summary.run_id != ""
            assert summary.run_id == run_dir.name

    def test_summary_has_question(self) -> None:
        """Test that summary includes the question."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            question = "Should we adopt uv?"
            run_dir = engine.run(question)

            summary = build_run_summary(run_dir)

            assert summary.question == question

    def test_summary_has_selected_path(self) -> None:
        """Test that summary includes selected path with nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)

            # Should have at least root node
            assert len(summary.selected_path) >= 1

            # First should be root
            assert summary.selected_path[0].kind == "root"

    def test_summary_has_prune_events(self) -> None:
        """Test that summary includes prune events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)

            # Should have at least one prune event
            assert len(summary.prune_events) >= 1

            # Prune event should have kept and pruned lists
            prune = summary.prune_events[0]
            assert len(prune.kept) >= 1  # At least 2 survivors
            assert len(prune.pruned) >= 1  # At least 1 pruned

    def test_summary_has_total_stats(self) -> None:
        """Test that summary includes aggregate statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)

            # Should have stats
            assert summary.total_nodes > 0
            assert summary.total_tool_calls >= 0
            assert summary.total_evidence >= 0


class TestMarkdownReport:
    """Test markdown report content."""

    def test_markdown_has_title(self) -> None:
        """Test that markdown report has title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            report = render_markdown(summary)

            assert "# Deliberation Report" in report

    def test_markdown_has_overview_table(self) -> None:
        """Test that markdown report has overview table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            report = render_markdown(summary)

            assert "## Overview" in report
            assert "| Field | Value |" in report
            assert "| Run ID |" in report

    def test_markdown_has_statistics_table(self) -> None:
        """Test that markdown report has statistics table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            report = render_markdown(summary)

            assert "## Statistics" in report
            assert "| Metric | Value |" in report
            assert "| Total nodes |" in report

    def test_markdown_has_footer(self) -> None:
        """Test that markdown report has footer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            summary = build_run_summary(run_dir)
            report = render_markdown(summary)

            assert "Generated by Delibera" in report


class TestInspectErrors:
    """Test error handling in inspect."""

    def test_missing_trace_raises(self) -> None:
        """Test that missing trace.jsonl raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "fake_run"
            run_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                build_run_summary(run_dir)

    def test_invalid_run_dir_raises(self) -> None:
        """Test that invalid run directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "nonexistent"

            with pytest.raises(FileNotFoundError):
                build_run_summary(run_dir)
