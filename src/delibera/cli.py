"""Command-line interface for Delibera."""

import click

from delibera.engine.orchestrator import Engine


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


if __name__ == "__main__":
    main()
