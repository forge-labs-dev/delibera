"""Tool registry for managing available tools.

The registry is the central place where tools are registered and looked up.
"""

from typing import Any

from delibera.tools.spec import RiskLevel, ToolSpec


class ToolRegistry:
    """Registry for managing tool instances.

    Tools must be registered before they can be used by the ToolRouter.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        """Register a tool with the registry.

        Args:
            tool: The tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        """Get a tool by name.

        Args:
            name: The unique name of the tool.

        Returns:
            The tool instance.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")
        return self._tools[name]

    def has(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The unique name of the tool.

        Returns:
            True if the tool is registered, False otherwise.
        """
        return name in self._tools

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of registered tool names.
        """
        return list(self._tools.keys())

    def list_by_risk_level(self, risk_level: RiskLevel) -> list[str]:
        """List tools filtered by risk level.

        Args:
            risk_level: The risk level to filter by.

        Returns:
            List of tool names with the specified risk level.
        """
        return [name for name, tool in self._tools.items() if tool.risk_level == risk_level]

    def get_tool_info(self, name: str) -> dict[str, Any]:
        """Get information about a tool.

        Args:
            name: The unique name of the tool.

        Returns:
            Dictionary with tool metadata.

        Raises:
            KeyError: If the tool is not registered.
        """
        tool = self.get(name)
        return {
            "name": tool.name,
            "risk_level": tool.risk_level.value,
            "is_discovery": tool.is_discovery,
        }


def create_default_registry(evidence_root: Any = None) -> ToolRegistry:
    """Create a registry with default built-in tools.

    Args:
        evidence_root: Optional custom root directory for evidence files (Path or None).
            If not provided, uses the default evidence/ directory.

    Returns:
        A ToolRegistry pre-populated with built-in tools.
    """
    from delibera.tools.builtin.calculator import CalculatorTool
    from delibera.tools.builtin.docs import DocsReadTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DocsReadTool(evidence_root=evidence_root))
    return registry
