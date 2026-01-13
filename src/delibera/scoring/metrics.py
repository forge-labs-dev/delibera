"""Metric computation for scoring nodes.

Computes epistemic metrics from a node's ledger and artifact
that feed into the scoring function.
"""

from typing import Any

from delibera.engine.tree import Node
from delibera.epistemics.models import (
    ClaimStatus,
    ClaimType,
    ObjectionSeverity,
    ObjectionStatus,
)

# Cap for evidence count to prevent unbounded reward
EVIDENCE_COUNT_CAP = 5


def compute_metrics(node: Node) -> dict[str, float]:
    """Compute epistemic metrics for a node.

    Metrics computed:
    - evidence_coverage: Fraction of fact claims that are supported (0..1).
      If no fact claims exist, defaults to 0.5 (neutral).
    - evidence_count: Number of evidence items, capped at EVIDENCE_COUNT_CAP.
    - weak_claims: Count of claims with status WEAK.
    - unsupported_claims: Count of claims with status UNSUPPORTED.
    - blocking_objections: Count of open blocking objections.
    - nonblocking_objections: Count of open nonblocking objections.

    Args:
        node: The node to compute metrics for.

    Returns:
        Dictionary mapping metric names to float values.
    """
    ledger = node.ledger

    # Count claims by status
    fact_claims_total = 0
    fact_claims_supported = 0
    weak_count = 0
    unsupported_count = 0

    for claim in ledger.claims:
        if claim.claim_type == ClaimType.FACT:
            fact_claims_total += 1
            if claim.status == ClaimStatus.SUPPORTED:
                fact_claims_supported += 1

        if claim.status == ClaimStatus.WEAK:
            weak_count += 1
        elif claim.status == ClaimStatus.UNSUPPORTED:
            unsupported_count += 1

    # Compute evidence coverage (neutral 0.5 if no fact claims)
    evidence_coverage = fact_claims_supported / fact_claims_total if fact_claims_total > 0 else 0.5

    # Count evidence items (capped)
    evidence_count = min(len(ledger.evidence), EVIDENCE_COUNT_CAP)

    # Count open objections by severity
    blocking_count = sum(
        1
        for obj in ledger.objections
        if obj.severity == ObjectionSeverity.BLOCKING and obj.status == ObjectionStatus.OPEN
    )
    nonblocking_count = sum(
        1
        for obj in ledger.objections
        if obj.severity == ObjectionSeverity.NONBLOCKING and obj.status == ObjectionStatus.OPEN
    )

    return {
        "evidence_coverage": evidence_coverage,
        "evidence_count": float(evidence_count),
        "weak_claims": float(weak_count),
        "unsupported_claims": float(unsupported_count),
        "blocking_objections": float(blocking_count),
        "nonblocking_objections": float(nonblocking_count),
    }


def metrics_to_summary(metrics: dict[str, float]) -> dict[str, Any]:
    """Convert metrics to a summary for display.

    Args:
        metrics: The computed metrics.

    Returns:
        Dictionary with formatted metric values.
    """
    return {
        "evidence_coverage": f"{metrics['evidence_coverage']:.1%}",
        "evidence_count": int(metrics["evidence_count"]),
        "weak_claims": int(metrics["weak_claims"]),
        "unsupported_claims": int(metrics["unsupported_claims"]),
        "blocking_objections": int(metrics["blocking_objections"]),
        "nonblocking_objections": int(metrics.get("nonblocking_objections", 0)),
    }
