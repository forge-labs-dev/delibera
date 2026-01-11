"""Command-line interface for Delibera."""

import sys
from pathlib import Path

import click

from delibera.engine.orchestrator import Engine
from delibera.trace.reader import load_artifact
from delibera.trace.replay import replay_from_directory, verify_replay


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
def run(question: str) -> None:
    """Run a deliberation on the given question.

    Creates a new run directory with trace.jsonl and artifact.json.
    """
    engine = Engine()
    run_dir = engine.run(question)

    click.echo(f"Run completed: {run_dir}")
    click.echo(f"Trace: {run_dir}/trace.jsonl")
    click.echo(f"Artifact: {run_dir}/artifact.json")


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


if __name__ == "__main__":
    main()
