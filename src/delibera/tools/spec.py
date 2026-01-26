"""Tool specification and schema definitions.

Defines the ToolSpec protocol that all tools must implement,
and the RiskLevel and ToolCapability enums for tool classification.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Protocol


class RiskLevel(Enum):
    """Risk classification for tools.

    Low: Deterministic, no external effects (e.g., calculator)
    Medium: External reads, limited side effects (e.g., web.open)
    High: Arbitrary side effects, requires explicit authorization
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolCapability(Enum):
    """Capability classification for tools.

    discovery: Tools that find/search for new information (e.g., docs.search)
    inspect: Tools that read already-known sources (e.g., docs.read)
    compute: Tools that perform computation without external data (e.g., calculator)

    Discovery tools are denied during CLAIM_CHECK (evidence-local restriction).
    """

    DISCOVERY = "discovery"
    INSPECT = "inspect"
    COMPUTE = "compute"


class ToolSpec(Protocol):
    """Protocol defining the tool interface.

    Every tool must implement this protocol to be registerable
    with the ToolRegistry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        ...

    @property
    @abstractmethod
    def risk_level(self) -> RiskLevel:
        """Risk classification of this tool."""
        ...

    @property
    def capability(self) -> ToolCapability:
        """Capability classification of this tool.

        Defaults to COMPUTE. Override in subclasses.
        """
        return ToolCapability.COMPUTE

    @property
    def is_discovery(self) -> bool:
        """Whether this tool discovers new information.

        Discovery tools are denied during CLAIM_CHECK step
        to enforce evidence-local restriction.

        This is derived from capability for backwards compatibility.
        """
        return self.capability == ToolCapability.DISCOVERY

    def validate_input(self, tool_input: dict[str, Any]) -> None:
        """Validate the input before execution.

        Args:
            tool_input: The input dictionary to validate.

        Raises:
            ValueError: If the input is invalid.
        """
        ...

    @abstractmethod
    def execute(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with the given input.

        Args:
            tool_input: The validated input dictionary.

        Returns:
            The tool output as a dictionary.

        Raises:
            ToolExecutionError: If execution fails.
        """
        ...


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")


class ToolDenied(Exception):
    """Raised when a tool call is denied by policy."""

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' denied: {reason}")
