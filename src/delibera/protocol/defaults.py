"""Default protocol specifications.

Provides the Tree Protocol v1 as the default deliberation protocol.
"""

from delibera.protocol.spec import (
    ConvergenceSpec,
    ExpandSpec,
    ProtocolSpec,
    PruneSpec,
    ReduceSpec,
    StepSpec,
)


def create_tree_protocol_v1() -> ProtocolSpec:
    """Create the default Tree Protocol v1.

    This protocol implements a 2-level expansion:
    - Level 1: Root expands into 3 options after PLAN
    - Level 2: Each option expands into 2 subplans after PROPOSE

    Then reduces subplans back to options, prunes options, and
    reduces to final merged artifact.

    Returns:
        The Tree Protocol v1 specification.
    """
    return ProtocolSpec(
        name="tree_protocol_v1",
        max_depth=2,
        expand_rules=[
            # Level 1: After PLAN, expand root into options
            ExpandSpec(
                at_step_id="plan",
                child_kind="option",
                max_children=3,
                depth=1,
                source="planner_output",
            ),
            # Level 2: After PROPOSE, expand each option into subplans
            ExpandSpec(
                at_step_id="propose",
                child_kind="plan",
                max_children=2,
                depth=2,
                source="agent_output",
            ),
        ],
        branch_pipeline=[
            # Steps run for each node after expansion
            StepSpec(
                id="propose",
                kind="work",
                step_name="PROPOSE",
                role="proposer",
            ),
            StepSpec(
                id="research",
                kind="work",
                step_name="RESEARCH",
                role="researcher",
            ),
            StepSpec(
                id="validate",
                kind="validate",
                step_name="CLAIM_CHECK",
                role=None,
            ),
        ],
        prune=PruneSpec(
            rule="epistemic_then_score",
            keep_k=2,
        ),
        reduce=ReduceSpec(
            rule="merge_artifacts",
        ),
        convergence=ConvergenceSpec(
            max_rounds=0,  # No refinement loop in v1
        ),
        refine_loop=[],  # Empty for v1
        gates_enabled=True,
    )


def create_simple_protocol() -> ProtocolSpec:
    """Create a simple single-level protocol for testing.

    This is the v0-compatible protocol with only 1 level of expansion.

    Returns:
        A simple single-level protocol specification.
    """
    return ProtocolSpec(
        name="simple_protocol",
        max_depth=1,
        expand_rules=[
            # Single expansion after PLAN
            ExpandSpec(
                at_step_id="plan",
                child_kind="option",
                max_children=3,
                depth=1,
                source="planner_output",
            ),
        ],
        branch_pipeline=[
            StepSpec(
                id="propose",
                kind="work",
                step_name="PROPOSE",
                role="proposer",
            ),
            StepSpec(
                id="research",
                kind="work",
                step_name="RESEARCH",
                role="researcher",
            ),
            StepSpec(
                id="validate",
                kind="validate",
                step_name="CLAIM_CHECK",
                role=None,
            ),
        ],
        prune=PruneSpec(
            rule="epistemic_then_score",
            keep_k=2,
        ),
        reduce=ReduceSpec(
            rule="merge_artifacts",
        ),
        convergence=ConvergenceSpec(
            max_rounds=0,
        ),
        refine_loop=[],
        gates_enabled=True,
    )


# Default protocol used by the engine
DEFAULT_PROTOCOL = create_simple_protocol()
