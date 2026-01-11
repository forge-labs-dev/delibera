"""Gate handlers for processing user gates.

Defines the GateHandler protocol and implementations:
- CLIGateHandler: Interactive CLI-based handler
- AutoApproveGateHandler: Non-interactive handler for tests/CI
- ScriptedGateHandler: Programmable handler for unit tests
"""

from abc import ABC, abstractmethod

import click

from delibera.gates.models import (
    AllowedAction,
    GateResponse,
    GateSummary,
    GateType,
    validate_response,
)


class GateHandler(ABC):
    """Abstract base class for gate handlers.

    Gate handlers are responsible for:
    1. Presenting the gate summary to the user
    2. Collecting and validating the user response
    3. Returning a validated GateResponse
    """

    @abstractmethod
    def handle(self, summary: GateSummary) -> GateResponse:
        """Handle a gate by presenting summary and collecting response.

        Args:
            summary: The gate summary to present.

        Returns:
            The validated user response.
        """
        ...


class AutoApproveGateHandler(GateHandler):
    """Gate handler that automatically approves all gates.

    Used for tests and CI where interactive input is not available.
    """

    def handle(self, summary: GateSummary) -> GateResponse:  # noqa: ARG002
        """Automatically approve the gate.

        Args:
            summary: The gate summary (ignored).

        Returns:
            An approve response.
        """
        return GateResponse(action=AllowedAction.APPROVE)


class ScriptedGateHandler(GateHandler):
    """Programmable gate handler for unit tests.

    Allows tests to specify predetermined responses for each gate.
    """

    def __init__(self, responses: dict[GateType, GateResponse] | None = None):
        """Initialize with optional predetermined responses.

        Args:
            responses: Map of gate type to response. If a gate type
                is not in the map, approve is returned.
        """
        self._responses = responses or {}
        self._triggered_gates: list[GateSummary] = []

    def handle(self, summary: GateSummary) -> GateResponse:
        """Return predetermined response or approve.

        Args:
            summary: The gate summary.

        Returns:
            The predetermined response or approve.
        """
        self._triggered_gates.append(summary)

        response = self._responses.get(
            summary.gate_type,
            GateResponse(action=AllowedAction.APPROVE),
        )
        return response

    @property
    def triggered_gates(self) -> list[GateSummary]:
        """Get list of gates that were triggered."""
        return self._triggered_gates


class CLIGateHandler(GateHandler):
    """Interactive CLI-based gate handler.

    Presents gate summaries and collects responses via the terminal.
    """

    def handle(self, summary: GateSummary) -> GateResponse:
        """Present gate summary and collect user response interactively.

        Args:
            summary: The gate summary to present.

        Returns:
            The validated user response.
        """
        self._print_summary(summary)

        while True:
            response = self._collect_response(summary)
            errors = validate_response(summary.gate_type, response)
            if not errors:
                return response

            click.echo("Invalid response:")
            for error in errors:
                click.echo(f"  - {error}")
            click.echo("Please try again.\n")

    def _print_summary(self, summary: GateSummary) -> None:
        """Print the gate summary to the terminal."""
        click.echo("\n" + "=" * 60)
        click.echo(f"GATE: {summary.gate_type.value.upper()}")
        click.echo("=" * 60)

        if summary.gate_type == GateType.SCOPE:
            click.echo(f"\nQuestion: {summary.interpreted_question}")
            click.echo("\nProposed branches:")
            for i, branch in enumerate(summary.proposed_branches, 1):
                click.echo(f"  {i}. {branch}")

        elif summary.gate_type == GateType.FINAL_SIGNOFF:
            click.echo(f"\nRecommendation: {summary.recommendation}")

            if summary.claim_check_summary:
                click.echo("\nClaim check summary:")
                for key, value in summary.claim_check_summary.items():
                    click.echo(f"  {key}: {value}")

            if summary.open_questions:
                click.echo("\nOpen questions:")
                for q in summary.open_questions:
                    click.echo(f"  - {q}")

        click.echo("\nAllowed actions:")
        for action in summary.allowed_actions:
            click.echo(f"  - {action.value}")

        click.echo("")

    def _collect_response(self, summary: GateSummary) -> GateResponse:
        """Collect user response from terminal input."""
        click.echo("Enter your response:")

        if summary.gate_type == GateType.SCOPE:
            click.echo("  'approve' - Accept proposed branches")
            click.echo("  'veto <branch1,branch2,...>' - Remove specific branches")
            click.echo("  'abort' - Cancel the run")
        elif summary.gate_type == GateType.FINAL_SIGNOFF:
            click.echo("  'approve' - Accept and finalize")
            click.echo("  'abort' - Cancel the run")

        user_input = click.prompt(">", type=str).strip().lower()

        if user_input == "approve":
            return GateResponse(action=AllowedAction.APPROVE)
        elif user_input == "abort":
            return GateResponse(action=AllowedAction.ABORT)
        elif user_input.startswith("veto "):
            # Parse comma-separated branch labels
            branches_str = user_input[5:].strip()
            branches = [b.strip() for b in branches_str.split(",") if b.strip()]
            return GateResponse(
                action=AllowedAction.VETO_BRANCHES,
                parameters={"branches": branches},
            )
        elif user_input == "veto":
            # Veto without branches - will fail validation, prompting retry
            return GateResponse(
                action=AllowedAction.VETO_BRANCHES,
                parameters={"branches": []},
            )
        else:
            click.echo(f"Unknown command: '{user_input}'. Please try again.")
            return self._collect_response(summary)
