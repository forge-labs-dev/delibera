"""Epistemic ledger for tracking claims, evidence, and objections.

The ledger is the single source of truth for epistemic state within a node.
"""

from dataclasses import dataclass, field

from delibera.epistemics.models import (
    Claim,
    ClaimStatus,
    Evidence,
    Objection,
)


@dataclass
class Ledger:
    """Aggregates epistemic objects for a node.

    The ledger maintains claims, evidence, objections, and the
    support relations between claims and evidence.

    Attributes:
        claims: List of claims extracted from this node.
        evidence: List of evidence items.
        objections: List of objections raised.
        support_relations: Maps claim_id to list of supporting evidence_ids.
    """

    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    support_relations: dict[str, list[str]] = field(default_factory=dict)


def merge_ledgers(ledgers: list[Ledger]) -> Ledger:
    """Merge multiple ledgers into a single ledger.

    During REDUCE, ledgers from surviving branches are merged.
    Rules:
    - Claims with status UNSUPPORTED are dropped
    - Supported and accepted claims survive
    - Evidence provenance is preserved
    - Objections are merged (targets remain unchanged for now)

    Args:
        ledgers: List of ledgers to merge.

    Returns:
        A new merged Ledger.
    """
    merged_claims: list[Claim] = []
    merged_evidence: list[Evidence] = []
    merged_objections: list[Objection] = []
    merged_support: dict[str, list[str]] = {}

    seen_claim_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()
    seen_objection_ids: set[str] = set()

    for ledger in ledgers:
        # Merge claims (drop unsupported)
        for claim in ledger.claims:
            if claim.claim_id in seen_claim_ids:
                continue
            # Drop unsupported claims during merge
            if claim.status == ClaimStatus.UNSUPPORTED:
                continue
            merged_claims.append(claim)
            seen_claim_ids.add(claim.claim_id)

        # Merge evidence (preserve all with provenance)
        for evidence in ledger.evidence:
            if evidence.evidence_id in seen_evidence_ids:
                continue
            merged_evidence.append(evidence)
            seen_evidence_ids.add(evidence.evidence_id)

        # Merge objections
        for objection in ledger.objections:
            if objection.objection_id in seen_objection_ids:
                continue
            merged_objections.append(objection)
            seen_objection_ids.add(objection.objection_id)

        # Merge support relations
        for claim_id, evidence_ids in ledger.support_relations.items():
            if claim_id not in merged_support:
                merged_support[claim_id] = []
            for eid in evidence_ids:
                if eid not in merged_support[claim_id]:
                    merged_support[claim_id].append(eid)

    return Ledger(
        claims=merged_claims,
        evidence=merged_evidence,
        objections=merged_objections,
        support_relations=merged_support,
    )
