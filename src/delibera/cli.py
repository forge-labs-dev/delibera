"""Command-line interface for Delibera."""

import json
import sys
from pathlib import Path

import click

from delibera.engine.orchestrator import Engine
from delibera.eval import EvalSuiteLoadError, load_eval_suite, run_eval_suite
from delibera.gates import AutoApproveGateHandler, CLIGateHandler, GateAborted, GateHandler
from delibera.gates.predicates import DEFAULT_TIE_THRESHOLD
from delibera.protocol import ProtocolLoadError, ProtocolSpec, load_protocol_from_yaml
from delibera.scoring import ScoreWeights
from delibera.trace.reader import load_artifact
from delibera.trace.replay import replay_from_directory, verify_replay


def _parse_weights(weights_str: str) -> ScoreWeights:
    """Parse weights string into ScoreWeights.

    Args:
        weights_str: Comma-separated key=value pairs.

    Returns:
        ScoreWeights with parsed values merged with defaults.

    Raises:
        click.BadParameter: If parsing fails.
    """
    weights_dict: dict[str, float] = {}
    for pair in weights_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise click.BadParameter(
                f"Invalid weight format: '{pair}'. Expected 'key=value'.",
                param_hint="--weights",
            )
        key, value = pair.split("=", 1)
        try:
            weights_dict[key.strip()] = float(value.strip())
        except ValueError as err:
            raise click.BadParameter(
                f"Invalid weight value: '{value}'. Must be a number.",
                param_hint="--weights",
            ) from err

    return ScoreWeights.from_dict(weights_dict)


@click.group()
def main() -> None:
    """Delibera - An engine for decision-grade AI deliberation."""
    pass


@main.command()
@click.option(
    "--question",
    required=True,
    help="The question or problem to deliberate on.",
)
@click.option(
    "--gates/--no-gates",
    default=True,
    help="Enable or disable user gates. Default: enabled.",
)
@click.option(
    "--auto-approve-gates",
    is_flag=True,
    default=False,
    help="Automatically approve all gates without prompting. For CI/tests.",
)
@click.option(
    "--tie-threshold",
    type=float,
    default=DEFAULT_TIE_THRESHOLD,
    help=f"Score difference threshold for tradeoff gate. Default: {DEFAULT_TIE_THRESHOLD}",
)
@click.option(
    "--weights",
    type=str,
    default=None,
    help="Scoring weights as key=value pairs (e.g., 'evidence_coverage=2.0,weak_claims=-0.5'). "
    "If provided, skips tradeoff gate.",
)
@click.option(
    "--protocol",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML protocol file. If not provided, uses builtin default.",
)
def run(
    question: str,
    gates: bool,
    auto_approve_gates: bool,
    tie_threshold: float,
    weights: str | None,
    protocol: str | None,
) -> None:
    """Run a deliberation on the given question.

    Creates a new run directory with trace.jsonl and artifact.json.

    Gates are enabled by default. Use --no-gates to disable them entirely,
    or --auto-approve-gates to automatically approve without prompting.

    Use --tie-threshold to adjust when the tradeoff gate fires (lower = more sensitive).
    Use --weights to set scoring weights directly, which skips the tradeoff gate.
    Use --protocol to specify a YAML protocol file.
    """
    # Determine gate handler
    gate_handler: GateHandler
    if not gates:
        # Gates disabled - use auto-approve (no gates will fire)
        gate_handler = AutoApproveGateHandler()
        gates_enabled = False
    elif auto_approve_gates:
        # Gates enabled but auto-approved
        gate_handler = AutoApproveGateHandler()
        gates_enabled = True
    else:
        # Interactive gates
        gate_handler = CLIGateHandler()
        gates_enabled = True

    # Parse weights if provided
    initial_weights: ScoreWeights | None = None
    if weights:
        initial_weights = _parse_weights(weights)

    # Load protocol if provided
    protocol_spec: ProtocolSpec | None = None
    protocol_source: str | None = None
    if protocol:
        protocol_path = Path(protocol)
        try:
            protocol_spec = load_protocol_from_yaml(protocol_path)
            protocol_source = f"yaml:{protocol_path}"
        except ProtocolLoadError as e:
            click.echo(f"Error loading protocol: {e}", err=True)
            sys.exit(1)
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    engine = Engine(
        gate_handler=gate_handler,
        gates_enabled=gates_enabled,
        tie_threshold=tie_threshold,
        initial_weights=initial_weights,
        protocol=protocol_spec,
        protocol_source=protocol_source,
    )

    try:
        run_dir = engine.run(question)
        click.echo(f"Run completed: {run_dir}")
        click.echo(f"Trace: {run_dir}/trace.jsonl")
        click.echo(f"Artifact: {run_dir}/artifact.json")
    except GateAborted as e:
        click.echo(f"Run aborted: {e.message}", err=True)
        click.echo(f"Gate: {e.gate_type.value}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--run-id",
    help="Run ID to replay (looks in ./runs/<run_id>/).",
)
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Path to run directory containing trace.jsonl.",
)
def replay(run_id: str | None, path: str | None) -> None:
    """Replay a deliberation from its trace.jsonl.

    Reconstructs the run state without calling agents and validates
    the trace structure and artifact derivability.

    Use either --run-id or --path to specify the run to replay.
    """
    # Determine run directory
    if path:
        run_dir = Path(path)
    elif run_id:
        run_dir = Path("runs") / run_id
    else:
        click.echo("Error: Must provide either --run-id or --path", err=True)
        sys.exit(1)

    # Check directory exists
    if not run_dir.exists():
        click.echo(f"Error: Run directory not found: {run_dir}", err=True)
        sys.exit(1)

    # Check trace file exists
    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        click.echo(f"Error: Trace file not found: {trace_path}", err=True)
        sys.exit(1)

    # Perform replay
    try:
        replayed = replay_from_directory(run_dir)
    except Exception as e:
        click.echo(f"Error: Failed to replay: {e}", err=True)
        sys.exit(1)

    # Check for replay errors
    if replayed.errors:
        click.echo(f"Replay FAILED: {replayed.run_id}", err=True)
        click.echo("Errors:", err=True)
        for error in replayed.errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    # Verify artifact if available
    artifact_path = run_dir / "artifact.json"
    if artifact_path.exists():
        try:
            stored_artifact = load_artifact(artifact_path)
            inconsistencies = verify_replay(replayed, stored_artifact)
            if inconsistencies:
                click.echo(f"Replay WARNING: {replayed.run_id}")
                click.echo("Artifact inconsistencies:")
                for issue in inconsistencies:
                    click.echo(f"  - {issue}")
        except Exception as e:
            click.echo(f"Warning: Could not verify artifact: {e}")

    # Print warnings if any
    if replayed.warnings:
        for warning in replayed.warnings:
            click.echo(f"Warning: {warning}")

    # Success output
    click.echo(f"Replay OK: {replayed.run_id}")
    click.echo(f"Question: {replayed.question}")

    recommendation = replayed.final_artifact.get("recommendation", "")
    if recommendation:
        click.echo(f"Recommendation: {recommendation}")

    click.echo(f"Artifact: {artifact_path}")


@main.command(name="eval")
@click.option(
    "--suite",
    required=True,
    type=click.Path(exists=True),
    help="Path to YAML evaluation suite file.",
)
@click.option(
    "--fail-fast",
    is_flag=True,
    default=False,
    help="Stop on first failure instead of running all cases.",
)
@click.option(
    "--save-results",
    type=click.Path(),
    default=None,
    help="Path to save JSON results file.",
)
def eval_cmd(suite: str, fail_fast: bool, save_results: str | None) -> None:
    """Run an evaluation suite against Delibera.

    Runs each case in the suite, extracts metrics, and compares
    against expected constraints. Exits with code 1 if any case fails.

    Examples:
        delibera eval --suite suites/basic.yaml
        delibera eval --suite suites/basic.yaml --fail-fast
        delibera eval --suite suites/basic.yaml --save-results results.json
    """
    suite_path = Path(suite)

    # Load the evaluation suite
    try:
        eval_suite = load_eval_suite(suite_path)
    except EvalSuiteLoadError as e:
        click.echo(f"Error loading suite: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Running eval suite: {eval_suite.suite_name}")
    click.echo(f"Cases: {len(eval_suite.cases)}")
    click.echo()

    # Track results for summary
    passed_count = 0
    failed_count = 0

    def on_case_complete(result: "EvalResult") -> None:
        """Callback to print case result as it completes."""
        nonlocal passed_count, failed_count

        if result.passed:
            click.echo(f"  PASS: {result.case_id}")
            passed_count += 1
        else:
            click.echo(f"  FAIL: {result.case_id}")
            for failure in result.failures:
                click.echo(f"        - {failure}")
            if result.error:
                click.echo(f"        Error: {result.error}")
            failed_count += 1

    # Import EvalResult type for callback
    from delibera.eval import EvalResult

    # Run the suite
    results = run_eval_suite(
        suite=eval_suite,
        fail_fast=fail_fast,
        on_case_complete=on_case_complete,
    )

    # Print summary
    click.echo()
    click.echo(f"Summary: {passed_count} passed, {failed_count} failed")

    # Save results if requested
    if save_results:
        results_path = Path(save_results)
        results_data = {
            "suite_name": eval_suite.suite_name,
            "total_cases": len(eval_suite.cases),
            "passed": passed_count,
            "failed": failed_count,
            "results": [r.to_dict() for r in results],
        }
        results_path.write_text(json.dumps(results_data, indent=2))
        click.echo(f"Results saved to: {results_path}")

    # Exit with error code if any failures
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
