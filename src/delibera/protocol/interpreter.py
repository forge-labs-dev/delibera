"""Protocol interpreter for driving engine execution.

The interpreter translates a ProtocolSpec into actions for the engine.
It does NOT execute operators directly; it returns instructions for the engine.
"""

from dataclasses import dataclass
from typing import Any

from delibera.protocol.spec import ConvergenceSpec, ExpandSpec, ProtocolSpec, StepSpec


@dataclass
class ExpandAction:
    """Instruction to expand a node."""

    node_id: str
    labels: list[str]
    child_kind: str
    depth: int
    expand_rule_id: str  # For trace logging


@dataclass
class WorkAction:
    """Instruction to execute a work step on a node."""

    node_id: str
    step: StepSpec
    context: dict[str, Any]


@dataclass
class ValidateAction:
    """Instruction to validate claims on a node."""

    node_id: str
    step: StepSpec
    owner: str  # Role that produced the artifact


@dataclass
class PruneAction:
    """Instruction to prune nodes."""

    node_ids: list[str]
    keep_k: int
    rule: str


@dataclass
class ReduceAction:
    """Instruction to reduce nodes."""

    parent_id: str
    survivor_ids: list[str]
    rule: str


@dataclass
class FinalizeAction:
    """Instruction to finalize the run."""

    merged_node_id: str


# Union type for all actions
Action = ExpandAction | WorkAction | ValidateAction | PruneAction | ReduceAction | FinalizeAction


class ProtocolInterpreter:
    """Interprets a protocol specification to drive engine execution.

    The interpreter maintains state about where we are in the protocol
    and returns the next action(s) for the engine to execute.
    """

    def __init__(self, spec: ProtocolSpec) -> None:
        """Initialize the interpreter with a protocol specification.

        Args:
            spec: The protocol to interpret.
        """
        self.spec = spec

    def get_expand_labels_from_output(
        self,
        output: dict[str, Any],
        expand_rule: ExpandSpec,
    ) -> list[str]:
        """Extract branch labels from agent output based on expand rule.

        Args:
            output: The agent output dictionary.
            expand_rule: The expand rule specifying where to find labels.

        Returns:
            List of branch labels.
        """
        if expand_rule.source == "planner_output":
            branches = output.get("branches", [])
        else:  # agent_output
            branches = output.get("sub_branches", [])

        # Extract labels, handling both dict and string formats
        labels: list[str] = []
        for branch in branches:
            if isinstance(branch, dict):
                labels.append(branch.get("label", str(branch)))
            else:
                labels.append(str(branch))

        # Limit to max_children
        return labels[: expand_rule.max_children]

    def should_expand_after_step(self, step_id: str, depth: int) -> ExpandSpec | None:
        """Check if expansion should occur after a step at a given depth.

        Args:
            step_id: The step that just completed.
            depth: The current node depth.

        Returns:
            ExpandSpec if expansion should occur, None otherwise.
        """
        return self.spec.get_expand_rule_at_step(step_id, depth)

    def get_branch_pipeline_step(self, index: int) -> StepSpec | None:
        """Get a step from the branch pipeline by index.

        Args:
            index: The step index.

        Returns:
            StepSpec if valid index, None otherwise.
        """
        if 0 <= index < len(self.spec.branch_pipeline):
            return self.spec.branch_pipeline[index]
        return None

    def needs_prune_at_depth(self, depth: int) -> bool:
        """Check if pruning should occur at a given depth.

        Pruning occurs after all nodes at a depth have completed
        the branch_pipeline and there are no more expansions pending.

        Args:
            depth: The depth to check.

        Returns:
            True if pruning should occur.
        """
        # Prune at each depth after completing branch pipeline
        # Only prune if there are nodes to prune at this depth
        return depth >= 1

    def needs_reduce_at_depth(self, depth: int) -> bool:
        """Check if reduction should occur at a given depth.

        Reduction occurs after pruning at each depth level.

        Args:
            depth: The depth to check.

        Returns:
            True if reduction should occur.
        """
        return depth >= 1

    def get_prune_spec(self) -> tuple[int, str]:
        """Get pruning parameters.

        Returns:
            Tuple of (keep_k, rule).
        """
        return self.spec.prune.keep_k, self.spec.prune.rule

    def get_reduce_rule(self) -> str:
        """Get the reduce rule.

        Returns:
            The reduce rule name.
        """
        return self.spec.reduce.rule

    def max_depth(self) -> int:
        """Get the maximum tree depth.

        Returns:
            The max_depth from the protocol.
        """
        return self.spec.max_depth

    def has_refine_loop(self) -> bool:
        """Check if the protocol has a refinement loop.

        Returns:
            True if refine_loop is non-empty and max_rounds > 0.
        """
        return len(self.spec.refine_loop) > 0 and self.spec.convergence.max_rounds > 0

    def get_refine_loop_steps(self) -> list[StepSpec]:
        """Get the refinement loop steps.

        Returns:
            List of StepSpec for refinement.
        """
        return self.spec.refine_loop

    def get_convergence_spec(self) -> ConvergenceSpec:
        """Get the convergence specification.

        Returns:
            The ConvergenceSpec from the protocol.
        """
        return self.spec.convergence

    def get_refine_step_by_id(self, step_id: str) -> StepSpec | None:
        """Get a refinement step by ID.

        Args:
            step_id: The step ID to find.

        Returns:
            StepSpec if found in refine_loop, None otherwise.
        """
        for step in self.spec.refine_loop:
            if step.id == step_id:
                return step
        return None
