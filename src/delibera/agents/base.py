"""Base agent protocol for Delibera."""

from typing import Any, Protocol


class Agent(Protocol):
    """Protocol defining the agent interface.

    Agents are heuristic generators that produce structured outputs.
    They cannot modify the tree structure or control termination.
    """

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's logic.

        Args:
            context: Input context containing relevant information
                for the agent's task.

        Returns:
            A structured dictionary containing the agent's output.
        """
        ...
