"""Engine operators for Delibera.

Operators are applied exclusively by the engine. They modify the
deliberation tree structure and produce trace-worthy events.
"""

from typing import Any

from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree, Node
from delibera.epistemics.extract import extract_claims
from delibera.epistemics.ledger import merge_ledgers
from delibera.epistemics.models import ClaimStatus, ClaimType, ObjectionSeverity, ObjectionStatus
from delibera.epistemics.validate import ClaimCheckReport, validate_claims


def expand(
    tree: DeliberationTree,
    parent_id: str,
    labels: list[str],
    kind: str = "option",
) -> list[Node]:
    """Create child nodes for the given parent.

    Args:
        tree: The deliberation tree.
        parent_id: The parent node's ID.
        labels: Labels for each child branch.
        kind: The kind of child node to create (default "option").

    Returns:
        List of created child nodes.
    """
    children = []
    for label in labels:
        child = tree.add_child(parent_id, label=label, kind=kind)  # type: ignore[arg-type]
        children.append(child)
    return children


def validate(tree: DeliberationTree, node_id: str, owner: str) -> ClaimCheckReport:
    """Extract and validate claims for a node.

    This operator:
    1. Extracts claims from the node's artifact
    2. Validates claims using ledger evidence
    3. Stores claims in the node's ledger
    4. Links claims to supporting evidence via support_relations
    5. Returns a validation report

    Args:
        tree: The deliberation tree.
        node_id: The node to validate.
        owner: The role that produced the artifact.

    Returns:
        ClaimCheckReport with validation results and support_relations.
    """
    node = tree.get_node(node_id)

    # Extract claims from artifact
    claims = extract_claims(node_id, node.artifact, owner)

    # Validate claims using ledger evidence
    report = validate_claims(claims, evidence=node.ledger.evidence)

    # Store claims in ledger
    node.ledger.claims = claims

    # Record support relations in ledger
    for claim_id, evidence_ids in report.support_relations.items():
        for evidence_id in evidence_ids:
            node.ledger.link_support(claim_id, evidence_id)

    return report


def prune(
    tree: DeliberationTree,
    node_ids: list[str],
    keep_count: int = 2,
    weights: Any | None = None,  # ScoreWeights, import avoided for simplicity
) -> tuple[list[str], list[str]]:
    """Prune weak branches using epistemic quality + weighted score.

    Pruning priority (deterministic):
    1. Fewer open blocking objections (ascending)
    2. Fewer unsupported claims (ascending)
    3. Fewer weak claims (ascending)
    4. Higher weighted score (descending)
    5. Lexicographic tie-break by label (ascending)

    Args:
        tree: The deliberation tree.
        node_ids: IDs of nodes to consider for pruning.
        keep_count: Number of top nodes to keep.
        weights: Optional ScoreWeights for score computation (uses artifact score).

    Returns:
        Tuple of (survivor_ids, pruned_ids).
    """
    # Note: weights parameter is accepted for interface compatibility.
    # The actual score is pre-computed and stored in node.artifact["score"]
    # by the orchestrator's SCORE step.
    _ = weights  # Unused here, score already in artifact

    def compute_epistemic_key(node_id: str) -> tuple[int, int, int, float, str]:
        """Compute sorting key for epistemic-based pruning.

        Returns tuple for sorting where lower is better for counts,
        higher is better for score.
        Lexicographic label provides stable, deterministic tie-break.
        """
        node = tree.get_node(node_id)

        # Count open blocking objections (highest priority for pruning)
        blocking_count = sum(
            1
            for obj in node.ledger.objections
            if obj.severity == ObjectionSeverity.BLOCKING and obj.status == ObjectionStatus.OPEN
        )

        # Count claims by status
        unsupported = sum(1 for c in node.ledger.claims if c.status == ClaimStatus.UNSUPPORTED)
        weak = sum(1 for c in node.ledger.claims if c.status == ClaimStatus.WEAK)
        score = node.artifact.get("score", 0.0)

        # Return tuple: (blocking, unsupported, weak, -score, label)
        # Ascending sort means fewer blocking/unsupported/weak is better,
        # higher score (negative becomes more negative) is better
        return (blocking_count, unsupported, weak, -score, node.label)

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
    merged_rationale: list[str] = []
    total_score = 0.0

    # Collect recommendations from survivors (use actual recommendation text, not labels)
    survivor_recommendations: list[str] = []
    for s in survivors:
        merged_pros.extend(s.artifact.get("pros", []))
        merged_cons.extend(s.artifact.get("cons", []))
        merged_rationale.extend(s.artifact.get("rationale", []))
        total_score += s.artifact.get("score", 0.0)
        # Prefer actual recommendation text over label
        rec = s.artifact.get("recommendation", "")
        if rec:
            survivor_recommendations.append(rec)

    avg_score = total_score / len(survivors) if survivors else 0.0

    # Build merged recommendation from actual recommendations if available
    if survivor_recommendations:
        merged_recommendation = " Additionally: ".join(survivor_recommendations)
    else:
        merged_recommendation = f"Merged recommendation combining: {', '.join(merged_labels)}"

    merged_artifact: dict[str, Any] = {
        "merged_from": merged_labels,
        "proposal": merged_recommendation,
        "recommendation": merged_recommendation,
        "summary": "Combined approach incorporating strengths of top options.",
        "pros": merged_pros,
        "cons": merged_cons,
        "rationale": merged_rationale,
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
        The final artifact dictionary with claim_check_summary, open_questions,
        decision_explanation, and key_claims with citations.
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

    # Build decision explanation showing why survivors won
    decision_explanation = _build_decision_explanation(option_nodes, survivors, pruned)

    # Build key_claims with citations (up to 5 claims)
    key_claims = _build_key_claims(merged_node)

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
        "decision_explanation": decision_explanation,
        "key_claims": key_claims,
    }

    return artifact


def _build_decision_explanation(
    option_nodes: list[Node],
    survivors: list[str],
    pruned: list[str],
) -> dict[str, Any]:
    """Build explanation of why certain options were selected.

    Args:
        option_nodes: All option nodes that were considered.
        survivors: Labels of nodes that survived pruning.
        pruned: Labels of nodes that were pruned.

    Returns:
        Dictionary with scoring details for each option.
    """
    # Collect scoring info for each option
    option_scores: list[dict[str, Any]] = []
    for node in option_nodes:
        # Count open blocking objections
        blocking_count = sum(
            1
            for obj in node.ledger.objections
            if obj.severity == ObjectionSeverity.BLOCKING and obj.status == ObjectionStatus.OPEN
        )

        # Count epistemic quality
        unsupported_count = sum(
            1 for c in node.ledger.claims if c.status == ClaimStatus.UNSUPPORTED
        )
        weak_count = sum(1 for c in node.ledger.claims if c.status == ClaimStatus.WEAK)
        score = node.artifact.get("score", 0.0)
        metrics = node.artifact.get("metrics", {})

        option_scores.append(
            {
                "label": node.label,
                "score": score,
                "metrics": metrics,
                "epistemic": {
                    "blocking_objections": blocking_count,
                    "unsupported_claims": unsupported_count,
                    "weak_claims": weak_count,
                },
                "status": "survivor" if node.label in survivors else "pruned",
            }
        )

    # Sort by the same criteria as prune: blocking, unsupported, weak, -score
    option_scores.sort(
        key=lambda x: (
            x["epistemic"]["blocking_objections"],
            x["epistemic"]["unsupported_claims"],
            x["epistemic"]["weak_claims"],
            -x["score"],
        )
    )

    # Build explanation text
    if survivors and pruned:
        explanation_text = (
            f"Options {', '.join(survivors)} were selected based on epistemic quality "
            f"(fewer blocking objections, unsupported/weak claims) and weighted score. "
            f"Options {', '.join(pruned)} were pruned."
        )
    elif survivors:
        explanation_text = f"All options ({', '.join(survivors)}) were retained."
    else:
        explanation_text = "No options were evaluated."

    return {
        "summary": explanation_text,
        "ranking": option_scores,
        "selection_criteria": [
            "1. Fewer open blocking objections",
            "2. Fewer unsupported claims",
            "3. Fewer weak claims",
            "4. Higher weighted score",
            "5. Lexicographic tie-break",
        ],
    }


# Maximum excerpt length for citations
MAX_CITATION_EXCERPT_LENGTH = 240
# Maximum number of key claims in artifact
MAX_KEY_CLAIMS = 5


def _build_key_claims(merged_node: Node) -> list[dict[str, Any]]:
    """Build key_claims section with citations for artifact.

    Selects up to MAX_KEY_CLAIMS claims that have supporting evidence,
    prioritized by type (fact > inference > plan) and then by claim_id
    for deterministic ordering.

    Args:
        merged_node: The merged node with ledger containing claims and evidence.

    Returns:
        List of claim dicts with citations.
    """
    # Build evidence lookup
    evidence_map = {e.evidence_id: e for e in merged_node.ledger.evidence}

    # Priority: fact=0, inference=1, plan=2, value=3 (lower is better)
    type_priority = {
        ClaimType.FACT: 0,
        ClaimType.INFERENCE: 1,
        ClaimType.PLAN: 2,
        ClaimType.VALUE: 3,
    }

    # Filter to claims with support and sort by priority
    claims_with_support: list[tuple[int, str, Any]] = []
    for claim in merged_node.ledger.claims:
        if claim.status != ClaimStatus.SUPPORTED:
            continue
        # Get support relations
        evidence_ids = merged_node.ledger.support_relations.get(claim.claim_id, [])
        if not evidence_ids:
            continue
        priority = type_priority.get(claim.claim_type, 99)
        claims_with_support.append((priority, claim.claim_id, claim))

    # Sort by (priority, claim_id) for deterministic ordering
    claims_with_support.sort(key=lambda x: (x[0], x[1]))

    # Build key_claims list
    key_claims: list[dict[str, Any]] = []
    for _, _, claim in claims_with_support[:MAX_KEY_CLAIMS]:
        evidence_ids = merged_node.ledger.support_relations.get(claim.claim_id, [])

        # Build citations for this claim
        citations: list[dict[str, str]] = []
        for eid in evidence_ids:
            evidence = evidence_map.get(eid)
            if evidence is None:
                continue
            # Truncate excerpt if needed
            excerpt = evidence.excerpt
            if len(excerpt) > MAX_CITATION_EXCERPT_LENGTH:
                excerpt = excerpt[: MAX_CITATION_EXCERPT_LENGTH - 3] + "..."
            citations.append(
                {
                    "evidence_id": eid,
                    "source": evidence.source,
                    "excerpt": excerpt,
                }
            )

        key_claims.append(
            {
                "claim_id": claim.claim_id,
                "type": claim.claim_type.value,
                "text": claim.text,
                "status": claim.status.value,
                "citations": citations,
            }
        )

    return key_claims
