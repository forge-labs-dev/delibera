"""Epistemic ledger for tracking claims, evidence, and objections.

The ledger is the single source of truth for epistemic state within a node.
"""

from dataclasses import dataclass, field

from delibera.epistemics.models import (
    Claim,
    ClaimStatus,
    Evidence,
    Objection,
    ObjectionStatus,
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

    def add_objection(self, objection: Objection) -> None:
        """Add an objection to the ledger.

        Avoids duplicates by checking objection_id.

        Args:
            objection: The objection to add.
        """
        if any(o.objection_id == objection.objection_id for o in self.objections):
            return
        self.objections.append(objection)

    def link_support(self, claim_id: str, evidence_id: str) -> None:
        """Link a claim to supporting evidence.

        Adds the evidence_id to the support_relations for the claim.
        Avoids duplicates and maintains stable ordering.

        Args:
            claim_id: The claim ID to link.
            evidence_id: The evidence ID that supports the claim.
        """
        if claim_id not in self.support_relations:
            self.support_relations[claim_id] = []
        if evidence_id not in self.support_relations[claim_id]:
            self.support_relations[claim_id].append(evidence_id)

    def support_for(self, claim_id: str) -> list[Evidence]:
        """Get evidence items that support a claim.

        Args:
            claim_id: The claim ID to look up.

        Returns:
            List of Evidence items supporting this claim, in stable order.
        """
        evidence_ids = self.support_relations.get(claim_id, [])
        # Build evidence lookup
        evidence_map = {e.evidence_id: e for e in self.evidence}
        # Return evidence in the order they were linked
        return [evidence_map[eid] for eid in evidence_ids if eid in evidence_map]


def merge_objections(objections_list: list[list[Objection]]) -> list[Objection]:
    """Merge objections from multiple sources with conflict resolution.

    Deterministic merge rules:
    - Preserve all objections (de-duplicate by id)
    - If same id appears, prefer "resolved" over "open", and "accepted" over "open"
    - Do NOT auto-accept objections

    Status priority: resolved > accepted > open

    Args:
        objections_list: List of objection lists to merge.

    Returns:
        Merged list of objections with duplicates resolved.
    """
    # Build a map of objection_id -> best objection
    objection_map: dict[str, Objection] = {}

    # Status priority for conflict resolution (higher = preferred)
    status_priority = {
        ObjectionStatus.OPEN: 0,
        ObjectionStatus.ACCEPTED: 1,
        ObjectionStatus.RESOLVED: 2,
    }

    for objections in objections_list:
        for obj in objections:
            existing = objection_map.get(obj.objection_id)
            if existing is None:
                objection_map[obj.objection_id] = obj
            else:
                # Prefer the one with higher-priority status
                if status_priority[obj.status] > status_priority[existing.status]:
                    objection_map[obj.objection_id] = obj

    # Return objections in deterministic order (sorted by id)
    return sorted(objection_map.values(), key=lambda o: o.objection_id)


def merge_ledgers(ledgers: list[Ledger]) -> Ledger:
    """Merge multiple ledgers into a single ledger.

    During REDUCE, ledgers from surviving branches are merged.
    Rules:
    - Claims with status UNSUPPORTED are dropped
    - Supported and accepted claims survive
    - Evidence provenance is preserved with stable ordering (sorted by source, id)
    - Support relations are merged with unique evidence IDs per claim
    - Objections are merged with conflict resolution (prefer resolved > accepted > open)

    Args:
        ledgers: List of ledgers to merge.

    Returns:
        A new merged Ledger.
    """
    merged_claims: list[Claim] = []
    merged_evidence: list[Evidence] = []
    merged_support: dict[str, list[str]] = {}

    seen_claim_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()

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

        # Merge support relations (union evidence IDs, preserving order)
        for claim_id, evidence_ids in ledger.support_relations.items():
            if claim_id not in merged_support:
                merged_support[claim_id] = []
            for eid in evidence_ids:
                if eid not in merged_support[claim_id]:
                    merged_support[claim_id].append(eid)

    # Sort evidence by (source, id) for stable ordering
    merged_evidence.sort(key=lambda e: (e.source, e.evidence_id))

    # Sort support relation evidence IDs for each claim
    for claim_id in merged_support:
        merged_support[claim_id].sort()

    # Merge objections using the dedicated function with conflict resolution
    objections_list = [ledger.objections for ledger in ledgers]
    merged_objections = merge_objections(objections_list)

    return Ledger(
        claims=merged_claims,
        evidence=merged_evidence,
        objections=merged_objections,
        support_relations=merged_support,
    )
