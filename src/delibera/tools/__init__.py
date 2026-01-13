"""Tools subsystem for Delibera.

Provides governed tool access with policy enforcement and tracing.
"""

from collections.abc import Callable
from typing import Any

from delibera.tools.policy import (
    BudgetState,
    GlobalPolicy,
    PolicyContext,
    PolicyDecision,
    PolicyEngine,
    RolePolicy,
    StepPolicyOverride,
    create_default_policy_engine,
)
from delibera.tools.registry import ToolRegistry, create_default_registry
from delibera.tools.router import ToolRouter
from delibera.tools.spec import RiskLevel, ToolDenied, ToolExecutionError, ToolSpec

# Type alias for tool callback provided to agents
ToolCallback = Callable[[str, dict[str, Any]], dict[str, Any]]

__all__ = [
    # Spec
    "ToolSpec",
    "RiskLevel",
    "ToolExecutionError",
    "ToolDenied",
    # Registry
    "ToolRegistry",
    "create_default_registry",
    # Policy
    "PolicyEngine",
    "PolicyContext",
    "PolicyDecision",
    "GlobalPolicy",
    "RolePolicy",
    "StepPolicyOverride",
    "BudgetState",
    "create_default_policy_engine",
    # Router
    "ToolRouter",
    # Type aliases
    "ToolCallback",
]
