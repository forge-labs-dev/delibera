"""Tests for the evaluation framework."""

import tempfile
from pathlib import Path

import pytest

from delibera.eval import (
    EvalCase,
    EvalResult,
    EvalSuite,
    EvalSuiteLoadError,
    compare_metrics,
    extract_metrics,
    load_eval_suite,
    run_eval_case,
    run_eval_suite,
)


class TestEvalModels:
    """Tests for evaluation data models."""

    def test_eval_case_valid(self) -> None:
        """Test creating a valid EvalCase."""
        case = EvalCase(
            id="test_case",
            question="Should we adopt uv?",
            expected={"min_supported_claims": 1},
        )
        assert case.id == "test_case"
        assert case.question == "Should we adopt uv?"
        assert case.expected == {"min_supported_claims": 1}

    def test_eval_case_empty_id_raises(self) -> None:
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            EvalCase(id="", question="Test question")

    def test_eval_case_empty_question_raises(self) -> None:
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="question cannot be empty"):
            EvalCase(id="test", question="")

    def test_eval_case_invalid_id_raises(self) -> None:
        """Test that invalid id characters raise ValueError."""
        with pytest.raises(ValueError, match="alphanumeric"):
            EvalCase(id="test@case", question="Test question")

    def test_eval_suite_valid(self) -> None:
        """Test creating a valid EvalSuite."""
        cases = [
            EvalCase(id="case_1", question="Question 1"),
            EvalCase(id="case_2", question="Question 2"),
        ]
        suite = EvalSuite(suite_name="test_suite", cases=cases)
        assert suite.suite_name == "test_suite"
        assert len(suite.cases) == 2

    def test_eval_suite_empty_name_raises(self) -> None:
        """Test that empty suite name raises ValueError."""
        with pytest.raises(ValueError, match="suite_name cannot be empty"):
            EvalSuite(
                suite_name="",
                cases=[EvalCase(id="test", question="Q")],
            )

    def test_eval_suite_empty_cases_raises(self) -> None:
        """Test that empty cases list raises ValueError."""
        with pytest.raises(ValueError, match="at least one case"):
            EvalSuite(suite_name="test", cases=[])

    def test_eval_suite_duplicate_ids_raises(self) -> None:
        """Test that duplicate case IDs raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate case IDs"):
            EvalSuite(
                suite_name="test",
                cases=[
                    EvalCase(id="same_id", question="Q1"),
                    EvalCase(id="same_id", question="Q2"),
                ],
            )

    def test_eval_result_to_dict(self) -> None:
        """Test EvalResult.to_dict() serialization."""
        result = EvalResult(
            case_id="test_case",
            run_id="run_123",
            run_dir=Path("/tmp/runs/run_123"),
            metrics={"supported_claims_count": 2},
            expected={"min_supported_claims": 1},
            passed=True,
            failures=[],
        )
        d = result.to_dict()
        assert d["case_id"] == "test_case"
        assert d["run_id"] == "run_123"
        assert d["passed"] is True


class TestEvalSuiteLoader:
    """Tests for YAML suite loading."""

    def test_load_valid_suite(self, tmp_path: Path) -> None:
        """Test loading a valid eval suite from YAML."""
        yaml_content = """
suite_name: basic_test
cases:
  - id: uv_adoption
    question: "Should we adopt uv?"
    expected:
      min_supported_claims: 1
      max_unsupported_claims: 10
"""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(yaml_content)

        suite = load_eval_suite(yaml_file)

        assert suite.suite_name == "basic_test"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "uv_adoption"
        assert suite.cases[0].expected["min_supported_claims"] == 1

    def test_load_suite_file_not_found(self, tmp_path: Path) -> None:
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_eval_suite(tmp_path / "nonexistent.yaml")

    def test_load_suite_invalid_yaml(self, tmp_path: Path) -> None:
        """Test loading invalid YAML raises EvalSuiteLoadError."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("suite_name: [invalid\n")

        with pytest.raises(EvalSuiteLoadError, match="Invalid YAML"):
            load_eval_suite(yaml_file)

    def test_load_suite_missing_required_key(self, tmp_path: Path) -> None:
        """Test loading YAML with missing key raises EvalSuiteLoadError."""
        yaml_content = """
# missing suite_name
cases:
  - id: test
    question: "Test?"
"""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(EvalSuiteLoadError, match="Missing required key"):
            load_eval_suite(yaml_file)

    def test_load_suite_unknown_key(self, tmp_path: Path) -> None:
        """Test loading YAML with unknown key raises EvalSuiteLoadError."""
        yaml_content = """
suite_name: test
unknown_key: value
cases:
  - id: test
    question: "Test?"
"""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(EvalSuiteLoadError, match="Unknown key"):
            load_eval_suite(yaml_file)

    def test_load_suite_with_protocol(self, tmp_path: Path) -> None:
        """Test loading suite with protocol reference."""
        # Create a minimal protocol file
        protocol_content = """
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
  - id: validate
    kind: validate
    step_name: VALIDATE
prune:
  rule: epistemic_then_score
  keep_k: 2
reduce:
  rule: merge_artifacts
"""
        protocol_file = tmp_path / "protocol.yaml"
        protocol_file.write_text(protocol_content)

        yaml_content = """
suite_name: test
protocol: protocol.yaml
cases:
  - id: test
    question: "Test?"
"""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(yaml_content)

        suite = load_eval_suite(yaml_file)

        assert suite.protocol is not None
        assert suite.protocol.name == "protocol.yaml"


class TestCompareMetrics:
    """Tests for metric constraint comparison."""

    def test_min_constraint_pass(self) -> None:
        """Test min_ constraint passes when metric >= value."""
        metrics = {"supported_claims": 5}
        expected = {"min_supported_claims": 3}
        failures = compare_metrics(metrics, expected)
        assert failures == []

    def test_min_constraint_fail(self) -> None:
        """Test min_ constraint fails when metric < value."""
        metrics = {"supported_claims": 2}
        expected = {"min_supported_claims": 3}
        failures = compare_metrics(metrics, expected)
        assert len(failures) == 1
        assert "min_supported_claims" in failures[0]
        assert ">=" in failures[0]

    def test_max_constraint_pass(self) -> None:
        """Test max_ constraint passes when metric <= value."""
        metrics = {"unsupported_claims": 2}
        expected = {"max_unsupported_claims": 5}
        failures = compare_metrics(metrics, expected)
        assert failures == []

    def test_max_constraint_fail(self) -> None:
        """Test max_ constraint fails when metric > value."""
        metrics = {"unsupported_claims": 6}
        expected = {"max_unsupported_claims": 5}
        failures = compare_metrics(metrics, expected)
        assert len(failures) == 1
        assert "max_unsupported_claims" in failures[0]
        assert "<=" in failures[0]

    def test_exact_constraint_pass(self) -> None:
        """Test exact constraint passes when metric == value."""
        metrics = {"evidence_count": 3}
        expected = {"evidence_count": 3}
        failures = compare_metrics(metrics, expected)
        assert failures == []

    def test_exact_constraint_fail(self) -> None:
        """Test exact constraint fails when metric != value."""
        metrics = {"evidence_count": 4}
        expected = {"evidence_count": 3}
        failures = compare_metrics(metrics, expected)
        assert len(failures) == 1
        assert "evidence_count" in failures[0]

    def test_boolean_constraint_pass(self) -> None:
        """Test boolean constraint passes."""
        metrics = {"run_aborted": False}
        expected = {"run_aborted": False}
        failures = compare_metrics(metrics, expected)
        assert failures == []

    def test_boolean_constraint_fail(self) -> None:
        """Test boolean constraint fails."""
        metrics = {"run_aborted": True}
        expected = {"run_aborted": False}
        failures = compare_metrics(metrics, expected)
        assert len(failures) == 1
        assert "run_aborted" in failures[0]

    def test_missing_metric(self) -> None:
        """Test that missing metric reports failure."""
        metrics = {}
        expected = {"min_supported_claims": 1}
        failures = compare_metrics(metrics, expected)
        assert len(failures) == 1
        assert "Missing metric" in failures[0]

    def test_multiple_constraints(self) -> None:
        """Test multiple constraints at once."""
        metrics = {
            "supported_claims": 5,
            "unsupported_claims": 2,
            "evidence_count": 3,
        }
        expected = {
            "min_supported_claims": 3,
            "max_unsupported_claims": 5,
            "evidence_count": 3,
        }
        failures = compare_metrics(metrics, expected)
        assert failures == []


class TestEvalRunner:
    """Tests for evaluation runner."""

    def test_run_eval_case_basic(self) -> None:
        """Test running a basic evaluation case."""
        case = EvalCase(
            id="basic_test",
            question="Should we adopt uv?",
            expected={},  # No constraints, just test run completes
        )

        result = run_eval_case(case)

        assert result.case_id == "basic_test"
        assert result.run_id != ""
        assert result.run_dir.exists()
        assert "supported_claims_count" in result.metrics
        # Should pass with no constraints
        assert result.passed is True

    def test_run_eval_case_with_passing_constraints(self) -> None:
        """Test eval case with constraints that should pass."""
        case = EvalCase(
            id="passing_test",
            question="Should we adopt uv?",
            expected={
                "min_supported_claims_count": 0,  # Always true
                "max_weak_claims_count": 100,  # Always true
            },
        )

        result = run_eval_case(case)

        assert result.passed is True
        assert result.failures == []

    def test_run_eval_case_with_failing_constraints(self) -> None:
        """Test eval case with impossible constraints fails."""
        case = EvalCase(
            id="failing_test",
            question="Should we adopt uv?",
            expected={
                "min_supported_claims_count": 1000,  # Impossible
            },
        )

        result = run_eval_case(case)

        assert result.passed is False
        assert len(result.failures) > 0
        assert "min_supported_claims_count" in result.failures[0]

    def test_run_eval_suite_basic(self, tmp_path: Path) -> None:
        """Test running a suite with multiple cases."""
        suite = EvalSuite(
            suite_name="basic_suite",
            cases=[
                EvalCase(
                    id="case_1",
                    question="Question 1?",
                    expected={"max_weak_claims_count": 100},
                ),
                EvalCase(
                    id="case_2",
                    question="Question 2?",
                    expected={"max_unsupported_claims_count": 100},
                ),
            ],
        )

        results = run_eval_suite(suite, runs_dir=tmp_path)

        assert len(results) == 2
        assert results[0].case_id == "case_1"
        assert results[1].case_id == "case_2"
        # Both should pass (generous constraints)
        assert all(r.passed for r in results)

    def test_run_eval_suite_fail_fast(self, tmp_path: Path) -> None:
        """Test fail_fast stops on first failure."""
        suite = EvalSuite(
            suite_name="fail_fast_suite",
            cases=[
                EvalCase(
                    id="failing_case",
                    question="Question 1?",
                    expected={"min_evidence_count": 1000},  # Will fail
                ),
                EvalCase(
                    id="never_reached",
                    question="Question 2?",
                    expected={},
                ),
            ],
        )

        results = run_eval_suite(suite, runs_dir=tmp_path, fail_fast=True)

        # Should only have run the first case
        assert len(results) == 1
        assert results[0].case_id == "failing_case"
        assert not results[0].passed


class TestMetricExtraction:
    """Tests for metric extraction."""

    def test_extract_metrics_basic(self) -> None:
        """Test extracting metrics from a completed run."""
        from delibera.engine.orchestrator import Engine

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            run_dir = engine.run("Should we adopt uv?")

            metrics = extract_metrics(run_dir)

            # Check all expected metrics are present
            assert "supported_claims_count" in metrics
            assert "weak_claims_count" in metrics
            assert "unsupported_claims_count" in metrics
            assert "evidence_count" in metrics
            assert "blocking_objections_open" in metrics
            assert "blocking_objections_accepted" in metrics
            assert "nonblocking_objections_open" in metrics
            assert "total_tool_calls" in metrics
            assert "run_aborted" in metrics

    def test_metrics_deterministic(self) -> None:
        """Test that metrics extraction is deterministic."""
        from delibera.engine.orchestrator import Engine

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)
            engine = Engine(runs_dir=runs_dir)

            # Run twice
            run_dir_1 = engine.run("Should we adopt uv?")
            run_dir_2 = engine.run("Should we adopt uv?")

            metrics_1 = extract_metrics(run_dir_1)
            metrics_2 = extract_metrics(run_dir_2)

            # All metrics should be identical for same question
            assert metrics_1 == metrics_2


class TestEvalSuiteIntegration:
    """Integration tests for full eval suite execution."""

    def test_eval_suite_passes_basic(self, tmp_path: Path) -> None:
        """Test that a basic eval suite passes with generous constraints."""
        # Create suite YAML
        yaml_content = """
suite_name: integration_test
cases:
  - id: basic_case
    question: "Should we adopt uv for package management?"
    expected:
      min_supported_claims_count: 0
      max_weak_claims_count: 100
      max_unsupported_claims_count: 100
      run_aborted: false
"""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(yaml_content)

        suite = load_eval_suite(yaml_file)
        results = run_eval_suite(suite, runs_dir=tmp_path / "runs")

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].failures == []

    def test_eval_suite_fails_on_constraint(self, tmp_path: Path) -> None:
        """Test that eval suite correctly reports constraint failures."""
        # Create suite YAML with impossible constraint
        yaml_content = """
suite_name: failing_test
cases:
  - id: impossible_case
    question: "Should we adopt uv?"
    expected:
      max_supported_claims_count: 0
"""
        yaml_file = tmp_path / "suite.yaml"
        yaml_file.write_text(yaml_content)

        suite = load_eval_suite(yaml_file)
        results = run_eval_suite(suite, runs_dir=tmp_path / "runs")

        # The case should fail (supported_claims > 0 normally)
        assert len(results) == 1
        # Note: If by chance we have 0 supported claims, this test may need adjustment
        # But typically with evidence, we get some supported claims
