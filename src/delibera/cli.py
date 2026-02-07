"""Command-line interface for Delibera."""

import json
import sys
from pathlib import Path

import click

from delibera.__version__ import __version__
from delibera.engine.orchestrator import Engine
from delibera.eval import EvalSuiteLoadError, load_eval_suite, run_eval_suite
from delibera.gates import AutoApproveGateHandler, CLIGateHandler, GateAborted, GateHandler
from delibera.gates.predicates import DEFAULT_TIE_THRESHOLD
from delibera.protocol import (
    ProtocolLoadError,
    ProtocolSpec,
    load_protocol_from_yaml,
    warnings_for_protocol,
)
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
@click.version_option(version=__version__, prog_name="delibera")
def main() -> None:
    """Delibera - An engine for decision-grade AI deliberation."""
    pass


@main.command()
def version() -> None:
    """Print the Delibera version."""
    click.echo(f"delibera {__version__}")


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
@click.option(
    "--use-llm-proposer",
    is_flag=True,
    default=False,
    help="Use LLM-backed proposer instead of stub. Requires GEMINI_API_KEY.",
)
@click.option(
    "--use-llm-planner",
    is_flag=True,
    default=False,
    help="Use LLM-backed planner instead of stub. Requires GEMINI_API_KEY.",
)
@click.option(
    "--use-llm-researcher",
    is_flag=True,
    default=False,
    help="Use LLM-backed researcher for smarter evidence queries. Requires GEMINI_API_KEY.",
)
@click.option(
    "--use-llm-redteam",
    is_flag=True,
    default=False,
    help="Use LLM-backed red-teamer for meaningful objections. Requires GEMINI_API_KEY.",
)
@click.option(
    "--use-llm-refiner",
    is_flag=True,
    default=False,
    help="Use LLM-backed refiner for intelligent proposal improvement. Requires GEMINI_API_KEY.",
)
@click.option(
    "--use-llm-validator",
    is_flag=True,
    default=False,
    help="Use LLM-backed claim validator for semantic evidence matching. Requires GEMINI_API_KEY.",
)
@click.option(
    "--use-all-llm",
    is_flag=True,
    default=False,
    help="Enable all LLM-backed agents. Requires GEMINI_API_KEY.",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["gemini"]),
    default="gemini",
    help="LLM provider to use. Default: gemini.",
)
@click.option(
    "--llm-model",
    type=str,
    default=None,
    help="LLM model name (e.g., 'gemini-2.0-flash'). If not set, uses provider default.",
)
@click.option(
    "--llm-temperature",
    type=float,
    default=0.2,
    help="LLM temperature (0.0-2.0). Default: 0.2.",
)
@click.option(
    "--llm-max-output-tokens",
    type=int,
    default=800,
    help="Maximum output tokens for LLM. Default: 800.",
)
@click.option(
    "--retrieval-method",
    type=click.Choice(["embedding", "keyword", "web", "hybrid"]),
    default="keyword",
    help="Evidence retrieval method. Default: keyword (no API needed).",
)
@click.option(
    "--evidence-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing evidence files. Default: ./evidence.",
)
@click.option(
    "--verify/--no-verify",
    default=False,
    help="Verify web search results by fetching actual URLs. Default: disabled.",
)
@click.option(
    "--max-parallel-branches",
    type=int,
    default=1,
    help="Max parallel branch execution (1=sequential). Default: 1.",
)
def run(
    question: str,
    gates: bool,
    auto_approve_gates: bool,
    tie_threshold: float,
    weights: str | None,
    protocol: str | None,
    use_llm_proposer: bool,
    use_llm_planner: bool,
    use_llm_researcher: bool,
    use_llm_redteam: bool,
    use_llm_refiner: bool,
    use_llm_validator: bool,
    use_all_llm: bool,
    llm_provider: str,
    llm_model: str | None,
    llm_temperature: float,
    llm_max_output_tokens: int,
    retrieval_method: str,
    evidence_dir: str | None,
    verify: bool,
    max_parallel_branches: int,
) -> None:
    """Run a deliberation on the given question.

    Creates a new run directory with trace.jsonl and artifact.json.

    Gates are enabled by default. Use --no-gates to disable them entirely,
    or --auto-approve-gates to automatically approve without prompting.

    Use --tie-threshold to adjust when the tradeoff gate fires (lower = more sensitive).
    Use --weights to set scoring weights directly, which skips the tradeoff gate.
    Use --protocol to specify a YAML protocol file.

    To use LLM-backed proposer, set --use-llm-proposer and ensure GEMINI_API_KEY is set.

    Evidence retrieval methods:
    - keyword: Simple keyword matching (default, no API needed)
    - embedding: Semantic search using Gemini embeddings
    - web: Real-time web search using Gemini grounding
    - hybrid: Combines embedding + web search with RRF fusion

    Use --verify to fetch and verify web search results before using them.
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
            # Show warnings for protocol configuration issues
            warnings = warnings_for_protocol(protocol_spec)
            for warning in warnings:
                click.echo(f"Warning: {warning}", err=True)
        except ProtocolLoadError as e:
            click.echo(f"Error loading protocol: {e}", err=True)
            sys.exit(1)
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    # Apply --use-all-llm convenience flag
    if use_all_llm:
        use_llm_planner = True
        use_llm_proposer = True
        use_llm_researcher = True
        use_llm_redteam = True
        use_llm_refiner = True
        use_llm_validator = True

    # Auto-upgrade retrieval method to web when LLM researcher is active
    # and user hasn't explicitly chosen a non-keyword method
    if use_llm_researcher and retrieval_method == "keyword":
        retrieval_method = "web"
        click.echo("Auto-enabled web retrieval for LLM researcher.")

    # Initialize LLM client if using any LLM-backed agent
    llm_client = None
    any_llm = (
        use_llm_proposer
        or use_llm_planner
        or use_llm_researcher
        or use_llm_redteam
        or use_llm_refiner
        or use_llm_validator
    )
    if any_llm and llm_provider == "gemini":
        from delibera.llm import GEMINI_API_KEY_ENV, GeminiClient, LLMAuthError

        try:
            llm_client = GeminiClient(
                model=llm_model or "gemini-2.0-flash",
            )
            llm_agents = []
            if use_llm_planner:
                llm_agents.append("planner")
            if use_llm_proposer:
                llm_agents.append("proposer")
            if use_llm_researcher:
                llm_agents.append("researcher")
            if use_llm_redteam:
                llm_agents.append("redteam")
            if use_llm_refiner:
                llm_agents.append("refiner")
            if use_llm_validator:
                llm_agents.append("validator")
            click.echo(
                f"LLM agents enabled: {', '.join(llm_agents)} "
                f"({llm_provider}, {llm_model or 'default'})"
            )
        except LLMAuthError as e:
            click.echo(f"Error: {e}", err=True)
            click.echo(f"Set {GEMINI_API_KEY_ENV} environment variable.", err=True)
            sys.exit(1)

    # Create retriever based on method
    retriever = None
    evidence_path = Path(evidence_dir) if evidence_dir else Path("evidence")

    if retrieval_method in ("embedding", "keyword", "hybrid") and not evidence_path.exists():
        click.echo(f"Warning: Evidence directory not found: {evidence_path}", err=True)
        click.echo("Creating empty evidence directory.", err=True)
        evidence_path.mkdir(parents=True, exist_ok=True)

    if retrieval_method != "keyword":  # keyword uses stub, no API needed
        from delibera.retrieval import create_retriever
        from delibera.retrieval.base import RetrieverError

        try:
            retriever = create_retriever(
                method=retrieval_method,  # type: ignore[arg-type]
                evidence_dir=evidence_path,
            )
            click.echo(f"Retriever enabled: {retrieval_method}")
        except RetrieverError as e:
            click.echo(f"Error initializing retriever: {e}", err=True)
            sys.exit(1)

    # Create verifier if requested
    verifier = None
    if verify:
        from delibera.retrieval import create_verifier

        verifier = create_verifier("fetch")
        click.echo("Verification enabled: fetch")

    engine = Engine(
        gate_handler=gate_handler,
        gates_enabled=gates_enabled,
        tie_threshold=tie_threshold,
        initial_weights=initial_weights,
        protocol=protocol_spec,
        protocol_source=protocol_source,
        llm_client=llm_client,
        use_llm_proposer=use_llm_proposer,
        use_llm_planner=use_llm_planner,
        use_llm_researcher=use_llm_researcher,
        use_llm_redteam=use_llm_redteam,
        use_llm_refiner=use_llm_refiner,
        use_llm_validator=use_llm_validator,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_output_tokens=llm_max_output_tokens,
        evidence_root=evidence_path,
        retriever=retriever,
        verifier=verifier,
        max_parallel_branches=max_parallel_branches,
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
    except KeyboardInterrupt:
        click.echo("\nRun interrupted by user.", err=True)
        sys.exit(130)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: Unexpected failure during run: {e}", err=True)
        click.echo("Check your configuration and API keys.", err=True)
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


@main.command()
@click.option(
    "--run-id",
    help="Run ID to inspect (looks in ./runs/<run_id>/).",
)
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Path to run directory containing trace.jsonl.",
)
def inspect(run_id: str | None, path: str | None) -> None:
    """Inspect a deliberation run and print a readable summary.

    This command loads the trace and artifact from a completed run
    and displays a human-readable summary including:

    - Final recommendation
    - Selected path through the deliberation tree
    - Key claims with citations
    - Pruning decisions
    - Statistics

    This is read-only and does not create new runs or call agents.

    Use either --run-id or --path to specify the run to inspect.
    """
    from delibera.inspect import build_run_summary, render_text

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

    # Build summary
    try:
        summary = build_run_summary(run_dir)
    except Exception as e:
        click.echo(f"Error: Failed to build summary: {e}", err=True)
        sys.exit(1)

    # Render and print
    output = render_text(summary)
    click.echo(output)

    # Exit with error if there were errors in reconstruction
    if summary.errors:
        sys.exit(1)


@main.command()
@click.option(
    "--run-id",
    help="Run ID to generate report for (looks in ./runs/<run_id>/).",
)
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Path to run directory containing trace.jsonl.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["md", "markdown"]),
    default="md",
    help="Output format. Default: md (Markdown).",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output file path for the report.",
)
def report(
    run_id: str | None,
    path: str | None,
    output_format: str,
    out: str,
) -> None:
    """Generate a report from a deliberation run.

    This command loads the trace and artifact from a completed run
    and generates a deterministic report file.

    The report includes:
    - Run overview and metadata
    - Final recommendation
    - Decision explanation
    - Selected path with node details
    - Key claims with citations
    - Pruning history
    - Statistics

    This is read-only and does not create new runs or call agents.

    Use either --run-id or --path to specify the run.
    """
    from delibera.inspect import build_run_summary, render_markdown

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

    # Build summary
    try:
        summary = build_run_summary(run_dir)
    except Exception as e:
        click.echo(f"Error: Failed to build summary: {e}", err=True)
        sys.exit(1)

    # Render based on format
    if output_format in ("md", "markdown"):
        content = render_markdown(summary)
    else:
        # Fallback (shouldn't happen due to click.Choice)
        content = render_markdown(summary)

    # Write output
    out_path = Path(out)
    try:
        out_path.write_text(content, encoding="utf-8")
    except Exception as e:
        click.echo(f"Error: Failed to write report: {e}", err=True)
        sys.exit(1)

    click.echo(f"Report written to: {out_path}")

    # Exit with error if there were errors in reconstruction
    if summary.errors:
        click.echo("Warning: Report generated with errors", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--port",
    default=8501,
    help="Port for the Streamlit server.",
)
@click.option(
    "--runs-dir",
    default="runs",
    help="Directory containing run data.",
)
def ui(port: int, runs_dir: str) -> None:
    """Launch the Delibera web UI.

    Opens a Streamlit dashboard for visualizing deliberation runs.
    Requires the [web] optional dependencies: pip install delibera[web]
    """
    import subprocess

    app_path = Path(__file__).parent / "web" / "app.py"

    if not app_path.exists():
        click.echo(f"Error: Web UI not found at {app_path}", err=True)
        sys.exit(1)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(port),
                "--",
                "--runs-dir",
                runs_dir,
            ],
            check=True,
        )
    except FileNotFoundError:
        click.echo(
            "Error: Streamlit not found. Install with: pip install delibera[web]",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
