"""Protocol specification dataclasses.

Defines the declarative structure for deliberation protocols.
Protocols specify WHAT steps run and WHERE expansion happens;
the engine still applies operators.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class StepSpec:
    """Specification for a single protocol step.

    A step is a unit of work that the engine executes.
    """

    id: str
    kind: Literal["work", "validate", "score"]
    step_name: str
    role: str | None = None  # Required for "work" steps

    def __post_init__(self) -> None:
        """Validate step specification."""
        if self.kind == "work" and not self.role:
            raise ValueError(f"Step '{self.id}' of kind 'work' requires a role")


@dataclass
class ExpandSpec:
    """Specification for an expansion point in the protocol.

    Defines where and how the tree should expand.
    """

    id: str  # Unique identifier for this expand rule
    at_step_id: str  # Step after which expansion occurs
    child_kind: Literal["option", "plan", "risk"]
    max_children: int
    depth: int  # Expansion depth level (1 for options, 2 for subplans)
    source: Literal["planner_output", "agent_output"]  # Where to read branch specs
    min_quality: float | None = None  # Optional quality threshold (not used in v1)
    condition: str | None = None  # Optional condition e.g. "unsupported_claims > 0"

    def __post_init__(self) -> None:
        """Validate expand specification."""
        if not self.id:
            raise ValueError("Expand rule id cannot be empty")
        if self.max_children <= 0:
            raise ValueError(f"max_children must be > 0, got {self.max_children}")
        if self.depth <= 0:
            raise ValueError(f"depth must be > 0, got {self.depth}")


@dataclass
class PruneSpec:
    """Specification for pruning behavior."""

    rule: Literal["epistemic_then_score", "score_only"]
    keep_k: int

    def __post_init__(self) -> None:
        """Validate prune specification."""
        if self.keep_k <= 0:
            raise ValueError(f"keep_k must be > 0, got {self.keep_k}")


@dataclass
class ReduceSpec:
    """Specification for reduction behavior."""

    rule: Literal["promote_single", "merge_artifacts"]


@dataclass
class ConvergenceSpec:
    """Specification for convergence behavior.

    Convergence predicates (all must be True to converge early):
    1. no_blocking_objections: No open blocking objections remain
    2. unsupported_claims == 0: All claims are supported or weak
    3. weak_claims <= weak_threshold: Number of weak claims is acceptable
    4. score_improvement < score_epsilon: Score delta between rounds is minimal
    """

    max_rounds: int = 0  # For refine loop; 0 means no refinement
    weak_threshold: int = 5  # Maximum acceptable weak claims for convergence
    score_epsilon: float = 0.01  # Minimum score improvement to continue refining
    dominance_threshold: float | None = None  # Ratio: top/second score for early termination

    def __post_init__(self) -> None:
        """Validate convergence specification."""
        if self.max_rounds < 0:
            raise ValueError(f"max_rounds must be >= 0, got {self.max_rounds}")
        if self.weak_threshold < 0:
            raise ValueError(f"weak_threshold must be >= 0, got {self.weak_threshold}")
        if self.score_epsilon < 0:
            raise ValueError(f"score_epsilon must be >= 0, got {self.score_epsilon}")


@dataclass
class ProtocolSpec:
    """Complete specification for a deliberation protocol.

    Defines the step sequence, expansion points, and convergence rules.
    """

    name: str
    protocol_version: str  # Version identifier (e.g., "v1")
    max_depth: int
    expand_rules: list[ExpandSpec]
    branch_pipeline: list[StepSpec]  # Steps run per node after expansion
    prune: PruneSpec
    reduce: ReduceSpec
    convergence: ConvergenceSpec = field(default_factory=ConvergenceSpec)
    refine_loop: list[StepSpec] = field(default_factory=list)  # Optional refinement steps
    gates_enabled: bool = True  # Engine may override via CLI flags

    def __post_init__(self) -> None:
        """Validate protocol specification."""
        if not self.name:
            raise ValueError("Protocol name cannot be empty")
        if not self.protocol_version:
            raise ValueError("Protocol version cannot be empty")
        if self.max_depth <= 0:
            raise ValueError(f"max_depth must be > 0, got {self.max_depth}")

        # Validate expand rules don't exceed max_depth
        for expand in self.expand_rules:
            if expand.depth > self.max_depth:
                raise ValueError(
                    f"Expand rule '{expand.id}' has depth {expand.depth} "
                    f"exceeding max_depth {self.max_depth}"
                )

    def get_expand_rule_at_step(self, step_id: str, depth: int) -> ExpandSpec | None:
        """Get the expand rule that triggers after a given step at a given depth.

        Args:
            step_id: The step ID to check.
            depth: The current tree depth.

        Returns:
            ExpandSpec if an expansion should occur, None otherwise.
        """
        for rule in self.expand_rules:
            if rule.at_step_id == step_id and rule.depth == depth + 1:
                return rule
        return None

    def get_step_by_id(self, step_id: str) -> StepSpec | None:
        """Get a step specification by ID.

        Args:
            step_id: The step ID to find.

        Returns:
            StepSpec if found, None otherwise.
        """
        for step in self.branch_pipeline:
            if step.id == step_id:
                return step
        for step in self.refine_loop:
            if step.id == step_id:
                return step
        return None


def validate_protocol(spec: ProtocolSpec) -> list[str]:
    """Validate a protocol specification for correctness.

    Args:
        spec: The protocol specification to validate.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[str] = []

    # Check that expand rules reference valid steps
    all_step_ids = {s.id for s in spec.branch_pipeline} | {s.id for s in spec.refine_loop}

    for expand in spec.expand_rules:
        # at_step_id can be "plan" for root-level expansion
        if expand.at_step_id != "plan" and expand.at_step_id not in all_step_ids:
            step_ref = expand.at_step_id
            errors.append(f"Expand rule '{expand.id}' references unknown step '{step_ref}'")

    # Check for duplicate step IDs
    step_ids = [s.id for s in spec.branch_pipeline] + [s.id for s in spec.refine_loop]
    seen: set[str] = set()
    for sid in step_ids:
        if sid in seen:
            errors.append(f"Duplicate step ID: '{sid}'")
        seen.add(sid)

    # Check for duplicate expand rule IDs
    expand_ids = [e.id for e in spec.expand_rules]
    seen_expand: set[str] = set()
    for eid in expand_ids:
        if eid in seen_expand:
            errors.append(f"Duplicate expand rule ID: '{eid}'")
        seen_expand.add(eid)

    # Check that branch_pipeline is not empty
    if not spec.branch_pipeline:
        errors.append("branch_pipeline cannot be empty")

    # Check expand rule depth vs max_depth
    for expand in spec.expand_rules:
        if expand.depth > spec.max_depth:
            errors.append(
                f"Expand rule '{expand.id}' has depth {expand.depth} "
                f"but max_depth is {spec.max_depth}"
            )

    # Check refine_loop without convergence.max_rounds (warn-level, not error)
    # This is a warning, so we don't add it to errors list but the caller
    # can detect this via warnings_for_protocol()

    return errors


def warnings_for_protocol(spec: ProtocolSpec) -> list[str]:
    """Return warnings for a protocol specification.

    These are not blocking errors, but indicate potential issues.

    Args:
        spec: The protocol specification to check.

    Returns:
        List of warning messages (empty if none).
    """
    warnings: list[str] = []

    # Warn if refine_loop is defined but max_rounds is 0
    if spec.refine_loop and spec.convergence.max_rounds == 0:
        warnings.append(
            "refine_loop is defined but convergence.max_rounds is 0; "
            "refinement loop will never execute"
        )

    # Warn if keep_k > expand max_children (makes pruning pointless)
    for expand in spec.expand_rules:
        if spec.prune.keep_k >= expand.max_children:
            warnings.append(
                f"prune.keep_k ({spec.prune.keep_k}) >= "
                f"expand rule '{expand.id}' max_children ({expand.max_children}); "
                f"pruning will keep all children"
            )

    return warnings
