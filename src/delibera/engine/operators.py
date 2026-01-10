"""Engine operators for Delibera.

Operators are applied exclusively by the engine. They modify the
deliberation tree structure and produce trace-worthy events.
"""

from typing import Any

from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree, Node


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


def prune(
    tree: DeliberationTree,
    node_ids: list[str],
    keep_count: int = 2,
) -> tuple[list[str], list[str]]:
    """Prune weak branches, keeping the top-k by score.

    Args:
        tree: The deliberation tree.
        node_ids: IDs of nodes to consider for pruning.
        keep_count: Number of top nodes to keep.

    Returns:
        Tuple of (survivor_ids, pruned_ids).
    """
    # Get nodes and their scores
    nodes_with_scores: list[tuple[str, float]] = []
    for node_id in node_ids:
        node = tree.get_node(node_id)
        score = node.artifact.get("score", 0.0)
        nodes_with_scores.append((node_id, score))

    # Sort by score descending
    nodes_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Split into survivors and pruned
    survivor_ids = [nid for nid, _ in nodes_with_scores[:keep_count]]
    pruned_ids = [nid for nid, _ in nodes_with_scores[keep_count:]]

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
    # Collect artifacts from survivors
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
        The final artifact dictionary.
    """
    # Get all option nodes to list what was considered (exclude merged node)
    root = state.tree.get_root()
    all_children = state.tree.get_children(root.node_id)
    option_nodes = [n for n in all_children if n.kind == "option"]

    options_considered = [n.label for n in option_nodes]
    survivors = merged_node.artifact.get("merged_from", [])
    pruned = [label for label in options_considered if label not in survivors]

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
    }

    return artifact
