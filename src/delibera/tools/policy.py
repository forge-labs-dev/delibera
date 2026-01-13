"""Policy engine for governing tool access.

Implements layered policy evaluation:
1. Global policy - enabled tools, budgets
2. Role-based policy - per-role allowed tools
3. Step-level overrides - e.g., CLAIM_CHECK restrictions
4. Budget constraints - max tool calls
"""

from dataclasses import dataclass, field
from typing import Any

from delibera.tools.spec import RiskLevel


@dataclass
class PolicyDecision:
    """Result of a policy evaluation.

    Contains whether access is allowed and the reason.
    """

    allow: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trace logging."""
        return {
            "allow": self.allow,
            "reason": self.reason,
        }


@dataclass
class GlobalPolicy:
    """Global policy applied to the entire run.

    Controls:
    - Which tools are enabled (allowlist)
    - Maximum total tool calls across the run
    """

    enabled_tools: set[str] = field(default_factory=set)
    max_calls: int | None = None  # None means unlimited

    def allows(self, tool_name: str) -> PolicyDecision:
        """Check if tool is globally enabled."""
        if not self.enabled_tools:
            # Empty set means all tools allowed
            return PolicyDecision(allow=True, reason="global: all tools allowed")

        if tool_name in self.enabled_tools:
            return PolicyDecision(allow=True, reason="global: tool in allowlist")

        return PolicyDecision(
            allow=False,
            reason=f"global: tool '{tool_name}' not in enabled_tools",
        )


@dataclass
class RolePolicy:
    """Role-based policy restricting tools per agent role.

    Maps role names to sets of allowed tool names.
    If a role is not in the mapping, all globally-enabled tools are allowed.
    """

    role_tools: dict[str, set[str]] = field(default_factory=dict)

    def allows(self, role: str, tool_name: str) -> PolicyDecision:
        """Check if role is allowed to use tool."""
        if role not in self.role_tools:
            # No restriction for this role
            return PolicyDecision(allow=True, reason=f"role: no restrictions for '{role}'")

        allowed = self.role_tools[role]
        if tool_name in allowed:
            return PolicyDecision(allow=True, reason=f"role: '{tool_name}' allowed for '{role}'")

        return PolicyDecision(
            allow=False,
            reason=f"role: '{tool_name}' not in allowed tools for '{role}'",
        )


@dataclass
class PolicyContext:
    """Context for policy evaluation.

    Provides additional information needed for evidence-local restrictions.
    """

    # Sources already in ledger that can be accessed during CLAIM_CHECK
    allowed_evidence_sources: set[str] = field(default_factory=set)


@dataclass
class StepPolicyOverride:
    """Step-level policy overrides.

    Applies additional restrictions during specific steps.
    Most important: CLAIM_CHECK enforces evidence-local access only.
    """

    # Steps where discovery tools are denied
    evidence_local_steps: set[str] = field(default_factory=lambda: {"CLAIM_CHECK"})

    # Tools subject to evidence-local source check
    evidence_local_tools: set[str] = field(default_factory=lambda: {"docs.read"})

    def allows(
        self,
        step: str,
        tool_name: str,
        is_discovery: bool,
        risk_level: RiskLevel,
        tool_input: dict[str, Any] | None = None,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        """Check if tool is allowed in current step.

        During evidence-local steps (like CLAIM_CHECK):
        - Discovery tools are denied
        - High-risk tools are denied
        - docs.read only allowed for sources already in ledger
        """
        if step not in self.evidence_local_steps:
            return PolicyDecision(allow=True, reason=f"step: no override for '{step}'")

        # Evidence-local mode: deny discovery tools
        if is_discovery:
            return PolicyDecision(
                allow=False,
                reason=f"step: discovery tools denied during '{step}'",
            )

        # Evidence-local mode: deny high-risk tools
        if risk_level == RiskLevel.HIGH:
            return PolicyDecision(
                allow=False,
                reason=f"step: high-risk tools denied during '{step}'",
            )

        # Evidence-local mode: check source restriction for docs.read
        if tool_name in self.evidence_local_tools:
            if context is None or tool_input is None:
                return PolicyDecision(
                    allow=False,
                    reason=f"step: {tool_name} requires evidence-local context during '{step}'",
                )

            path = tool_input.get("path", "")
            if path not in context.allowed_evidence_sources:
                return PolicyDecision(
                    allow=False,
                    reason=f"step: source '{path}' not in allowed evidence during '{step}'",
                )

        return PolicyDecision(
            allow=True,
            reason="step: tool allowed in evidence-local mode",
        )


@dataclass
class BudgetState:
    """Tracks budget consumption for a run."""

    total_calls: int = 0
    calls_by_tool: dict[str, int] = field(default_factory=dict)

    def record_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self.total_calls += 1
        self.calls_by_tool[tool_name] = self.calls_by_tool.get(tool_name, 0) + 1


class PolicyEngine:
    """Evaluates layered policies for tool access.

    A tool call is permitted only if ALL policy layers allow it.
    """

    def __init__(
        self,
        global_policy: GlobalPolicy | None = None,
        role_policy: RolePolicy | None = None,
        step_override: StepPolicyOverride | None = None,
    ) -> None:
        """Initialize the policy engine.

        Args:
            global_policy: Global tool restrictions.
            role_policy: Per-role tool restrictions.
            step_override: Step-level overrides.
        """
        self.global_policy = global_policy or GlobalPolicy()
        self.role_policy = role_policy or RolePolicy()
        self.step_override = step_override or StepPolicyOverride()
        self.budget_state = BudgetState()

    def evaluate(
        self,
        role: str,
        step: str,
        tool_name: str,
        is_discovery: bool = False,
        risk_level: RiskLevel = RiskLevel.LOW,
        tool_input: dict[str, Any] | None = None,
        context: PolicyContext | None = None,
    ) -> PolicyDecision:
        """Evaluate all policy layers for a tool call.

        Args:
            role: The agent role making the request.
            step: The current protocol step.
            tool_name: The name of the tool being requested.
            is_discovery: Whether the tool discovers new information.
            risk_level: The risk classification of the tool.
            tool_input: The tool input (for evidence-local checks).
            context: Policy context with allowed evidence sources.

        Returns:
            PolicyDecision indicating allow/deny with reason.
        """
        # Layer 1: Global policy
        global_decision = self.global_policy.allows(tool_name)
        if not global_decision.allow:
            return global_decision

        # Layer 2: Role policy
        role_decision = self.role_policy.allows(role, tool_name)
        if not role_decision.allow:
            return role_decision

        # Layer 3: Step override (with evidence-local support)
        step_decision = self.step_override.allows(
            step, tool_name, is_discovery, risk_level, tool_input, context
        )
        if not step_decision.allow:
            return step_decision

        # Layer 4: Budget check
        max_calls = self.global_policy.max_calls
        if max_calls is not None and self.budget_state.total_calls >= max_calls:
            return PolicyDecision(
                allow=False,
                reason=f"budget: max calls ({max_calls}) exceeded",
            )

        return PolicyDecision(allow=True, reason="all policies allow")

    def record_call(self, tool_name: str) -> None:
        """Record a successful tool call for budget tracking.

        Args:
            tool_name: The name of the tool that was called.
        """
        self.budget_state.record_call(tool_name)

    def get_budget_info(self) -> dict[str, Any]:
        """Get current budget consumption info.

        Returns:
            Dictionary with budget state.
        """
        return {
            "total_calls": self.budget_state.total_calls,
            "max_calls": self.global_policy.max_calls,
            "calls_by_tool": dict(self.budget_state.calls_by_tool),
        }


def create_default_policy_engine() -> PolicyEngine:
    """Create a policy engine with sensible defaults.

    Returns:
        PolicyEngine with all built-in tools enabled.
    """
    global_policy = GlobalPolicy(
        enabled_tools={"calculator", "docs.read"},  # Enable built-in tools
        max_calls=100,  # Default budget
    )
    return PolicyEngine(global_policy=global_policy)
