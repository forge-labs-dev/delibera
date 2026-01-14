"""Evaluation runner for executing eval suites.

Runs Delibera on evaluation cases and collects results.
"""

import tempfile
from collections.abc import Callable
from pathlib import Path

from delibera.engine.orchestrator import Engine
from delibera.eval.compare import compare_metrics
from delibera.eval.metrics import extract_metrics
from delibera.eval.models import EvalCase, EvalResult, EvalSuite
from delibera.gates import AutoApproveGateHandler
from delibera.protocol import ProtocolSpec, load_protocol_from_yaml


def run_eval_case(
    case: EvalCase,
    protocol: ProtocolSpec | None = None,
    runs_dir: Path | None = None,
) -> EvalResult:
    """Run a single evaluation case.

    Creates a fresh run, extracts metrics, and compares against expected.

    Args:
        case: The evaluation case to run.
        protocol: Optional protocol to use. Uses default if not provided.
        runs_dir: Optional directory for run outputs.

    Returns:
        EvalResult with metrics and pass/fail status.
    """
    # Use provided runs_dir or create a temporary one
    if runs_dir is None:
        # Create a temp directory that persists for result inspection
        runs_dir = Path(tempfile.mkdtemp(prefix="delibera_eval_"))

    # Create engine with auto-approve gates
    engine = Engine(
        runs_dir=runs_dir,
        gate_handler=AutoApproveGateHandler(),
        gates_enabled=True,
        protocol=protocol,
        evidence_root=case.evidence_dir,
    )

    # Run deliberation
    run_dir: Path
    error: str | None = None
    run_id = ""

    try:
        run_dir = engine.run(case.question)
        run_id = run_dir.name
    except Exception as e:
        # Capture error but don't fail yet
        error = str(e)
        # Create a placeholder result
        return EvalResult(
            case_id=case.id,
            run_id="",
            run_dir=runs_dir,
            metrics={},
            expected=case.expected,
            passed=False,
            failures=[f"Run failed: {error}"],
            error=error,
        )

    # Extract metrics
    try:
        metrics = extract_metrics(run_dir)
    except Exception as e:
        return EvalResult(
            case_id=case.id,
            run_id=run_id,
            run_dir=run_dir,
            metrics={},
            expected=case.expected,
            passed=False,
            failures=[f"Metric extraction failed: {e}"],
            error=str(e),
        )

    # Compare against expected constraints
    failures = compare_metrics(metrics, case.expected)
    passed = len(failures) == 0

    return EvalResult(
        case_id=case.id,
        run_id=run_id,
        run_dir=run_dir,
        metrics=metrics,
        expected=case.expected,
        passed=passed,
        failures=failures,
        error=None,
    )


def run_eval_suite(
    suite: EvalSuite,
    runs_dir: Path | None = None,
    fail_fast: bool = False,
    on_case_complete: Callable[[EvalResult], None] | None = None,
) -> list[EvalResult]:
    """Run all cases in an evaluation suite.

    Args:
        suite: The evaluation suite to run.
        runs_dir: Optional base directory for run outputs.
        fail_fast: If True, stop on first failure.
        on_case_complete: Optional callback called after each case completes.

    Returns:
        List of EvalResult for each case.
    """
    # Load protocol if specified
    protocol: ProtocolSpec | None = None
    if suite.protocol:
        protocol = load_protocol_from_yaml(suite.protocol)

    # Use provided runs_dir or create a temporary one
    if runs_dir is None:
        runs_dir = Path(tempfile.mkdtemp(prefix="delibera_eval_"))

    results: list[EvalResult] = []

    for case in suite.cases:
        # Each case gets its own subdirectory
        case_runs_dir = runs_dir / case.id

        result = run_eval_case(
            case=case,
            protocol=protocol,
            runs_dir=case_runs_dir,
        )
        results.append(result)

        # Call callback if provided
        if on_case_complete:
            on_case_complete(result)

        # Stop early if fail_fast and case failed
        if fail_fast and not result.passed:
            break

    return results
