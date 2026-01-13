"""Engine operators for Delibera.

Operators are applied exclusively by the engine. They modify the
deliberation tree structure and produce trace-worthy events.
"""

from typing import Any

from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree, Node
from delibera.epistemics.extract import extract_claims
from delibera.epistemics.ledger import merge_ledgers
from delibera.epistemics.models import ClaimStatus
from delibera.epistemics.validate import ClaimCheckReport, validate_claims


def expand(tree: DeliberationTree, parent_id: str, labels: list[str]) -> list[Node]:
    """Create child nodes for the given parent.

    Args:
        tree: The deliberation tree.
        parent_id: The parent node's ID.
        labels: Labels for each child branch.

    Returns:
        List of created child nodes.
    """
    children = []
    for label in labels:
        child = tree.add_child(parent_id, label=label, kind="option")
        children.append(child)
    return children


def validate(tree: DeliberationTree, node_id: str, owner: str) -> ClaimCheckReport:
    """Extract and validate claims for a node.

    This operator:
    1. Extracts claims from the node's artifact
    2. Validates claims using ledger evidence
    3. Stores claims in the node's ledger
    4. Returns a validation report

    Args:
        tree: The deliberation tree.
        node_id: The node to validate.
        owner: The role that produced the artifact.

    Returns:
        ClaimCheckReport with validation results.
    """
    node = tree.get_node(node_id)

    # Extract claims from artifact
    claims = extract_claims(node_id, node.artifact, owner)

    # Validate claims using ledger evidence
    report = validate_claims(claims, evidence=node.ledger.evidence)

    # Store claims in ledger
    node.ledger.claims = claims

    return report


def prune(
    tree: DeliberationTree,
    node_ids: list[str],
    keep_count: int = 2,
) -> tuple[list[str], list[str]]:
    """Prune weak branches using epistemic quality metrics.

    Pruning priority (deterministic):
    1. Fewer unsupported claims (ascending)
    2. Fewer weak claims (ascending)
    3. Higher artifact score (descending)

    Args:
        tree: The deliberation tree.
        node_ids: IDs of nodes to consider for pruning.
        keep_count: Number of top nodes to keep.

    Returns:
        Tuple of (survivor_ids, pruned_ids).
    """

    def compute_epistemic_key(node_id: str) -> tuple[int, int, float]:
        """Compute sorting key for epistemic-based pruning.

        Returns (unsupported_count, weak_count, -score) for sorting.
        Lower is better for counts, higher is better for score.
        """
        node = tree.get_node(node_id)

        # Count claims by status
        unsupported = sum(1 for c in node.ledger.claims if c.status == ClaimStatus.UNSUPPORTED)
        weak = sum(1 for c in node.ledger.claims if c.status == ClaimStatus.WEAK)
        score = node.artifact.get("score", 0.0)

        # Return tuple for sorting: (unsupported, weak, -score)
        # Ascending sort means fewer unsupported/weak is better,
        # and higher score (negative becomes more negative) is better
        return (unsupported, weak, -score)

    # Sort nodes by epistemic quality
    sorted_node_ids = sorted(node_ids, key=compute_epistemic_key)

    # Split into survivors and pruned
    survivor_ids = sorted_node_ids[:keep_count]
    pruned_ids = sorted_node_ids[keep_count:]

    # Update status of pruned nodes
    for node_id in pruned_ids:
        tree.set_status(node_id, "pruned")

    return survivor_ids, pruned_ids


def reduce(tree: DeliberationTree, parent_id: str, survivor_ids: list[str]) -> Node:
    """Merge surviving branches into a single node.

    Args:
        tree: The deliberation tree.
        parent_id: The parent node's ID (where to attach merged node).
        survivor_ids: IDs of nodes to merge.

    Returns:
        The merged node.
    """
    # Collect artifacts and ledgers from survivors
    survivors = [tree.get_node(nid) for nid in survivor_ids]

    # Build merged artifact
    merged_labels = [s.label for s in survivors]
    merged_pros: list[str] = []
    merged_cons: list[str] = []
    total_score = 0.0

    for s in survivors:
        merged_pros.extend(s.artifact.get("pros", []))
        merged_cons.extend(s.artifact.get("cons", []))
        total_score += s.artifact.get("score", 0.0)

    avg_score = total_score / len(survivors) if survivors else 0.0

    merged_artifact: dict[str, Any] = {
        "merged_from": merged_labels,
        "proposal": f"Merged recommendation combining: {', '.join(merged_labels)}",
        "summary": "Combined approach incorporating strengths of top options.",
        "pros": merged_pros,
        "cons": merged_cons,
        "score": avg_score,
    }

    # Create merged node
    merged_node = tree.add_child(parent_id, label="merged", kind="merged")
    tree.update_artifact(merged_node.node_id, merged_artifact)

    # Merge ledgers from survivors
    survivor_ledgers = [s.ledger for s in survivors]
    merged_node.ledger = merge_ledgers(survivor_ledgers)

    # Mark survivors as merged
    for nid in survivor_ids:
        tree.set_status(nid, "merged")

    return merged_node


def finalize(state: RunState, merged_node: Node) -> dict[str, Any]:
    """Build the final decision artifact.

    Args:
        state: The run state.
        merged_node: The final merged node.

    Returns:
        The final artifact dictionary with claim_check_summary and open_questions.
    """
    # Get all option nodes to list what was considered (exclude merged node)
    root = state.tree.get_root()
    all_children = state.tree.get_children(root.node_id)
    option_nodes = [n for n in all_children if n.kind == "option"]

    options_considered = [n.label for n in option_nodes]
    survivors = merged_node.artifact.get("merged_from", [])
    pruned = [label for label in options_considered if label not in survivors]

    # Compute claim check summary from merged ledger
    supported = sum(1 for c in merged_node.ledger.claims if c.status == ClaimStatus.SUPPORTED)
    weak = sum(1 for c in merged_node.ledger.claims if c.status == ClaimStatus.WEAK)
    unsupported = sum(1 for c in merged_node.ledger.claims if c.status == ClaimStatus.UNSUPPORTED)

    claim_check_summary = {
        "supported": supported,
        "weak": weak,
        "unsupported": unsupported,
    }

    # Generate open questions based on epistemic quality
    open_questions: list[str] = []
    if weak > 0 or unsupported > 0:
        open_questions.append("Add evidence to support inference claims.")

    artifact: dict[str, Any] = {
        "question": state.question,
        "recommendation": merged_node.artifact.get("proposal", ""),
        "summary": merged_node.artifact.get("summary", ""),
        "options_considered": options_considered,
        "survivors": survivors,
        "pruned": pruned,
        "rationale": merged_node.artifact.get("pros", []),
        "risks": merged_node.artifact.get("cons", []),
        "confidence": merged_node.artifact.get("score", 0.0),
        "claim_check_summary": claim_check_summary,
        "open_questions": open_questions,
    }

    return artifact
