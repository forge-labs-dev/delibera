"""Claim extraction from artifacts.

Extracts claims from structured agent outputs using deterministic heuristics.
"""

import hashlib
from typing import Any

from delibera.epistemics.models import Claim, ClaimType


def _generate_claim_id(node_id: str, text: str) -> str:
    """Generate a deterministic claim ID from node ID and text.

    Args:
        node_id: The node that owns this claim.
        text: The claim text.

    Returns:
        A 10-character hex string ID.
    """
    normalized = text.strip().lower()
    hash_input = f"{node_id}:{normalized}"
    return hashlib.sha1(hash_input.encode()).hexdigest()[:10]


def extract_claims(
    node_id: str,
    artifact: dict[str, Any],
    owner: str,
) -> list[Claim]:
    """Extract claims from a node's artifact.

    Extraction rules (v1):
    - artifact["proposal"] -> plan claim (confidence 0.8)
    - artifact["facts"] items -> fact claims (confidence 0.9)
    - artifact["pros"] items -> inference claims (confidence 0.6)
    - artifact["rationale"] items -> inference claims (confidence 0.6)
    - Empty strings are skipped

    Args:
        node_id: The node ID (used for claim ID generation).
        artifact: The node's artifact dict.
        owner: The role that produced this artifact.

    Returns:
        List of extracted Claim objects.
    """
    claims: list[Claim] = []

    # Extract plan claim from proposal
    proposal = artifact.get("proposal", "")
    if proposal and isinstance(proposal, str) and proposal.strip():
        claims.append(
            Claim(
                claim_id=_generate_claim_id(node_id, proposal),
                text=proposal.strip(),
                claim_type=ClaimType.PLAN,
                confidence=0.8,
                owner=owner,
            )
        )

    # Extract fact claims from facts
    facts = artifact.get("facts", [])
    if isinstance(facts, list):
        for fact in facts:
            if fact and isinstance(fact, str) and fact.strip():
                claims.append(
                    Claim(
                        claim_id=_generate_claim_id(node_id, fact),
                        text=fact.strip(),
                        claim_type=ClaimType.FACT,
                        confidence=0.9,
                        owner=owner,
                    )
                )

    # Extract inference claims from pros
    pros = artifact.get("pros", [])
    if isinstance(pros, list):
        for pro in pros:
            if pro and isinstance(pro, str) and pro.strip():
                claims.append(
                    Claim(
                        claim_id=_generate_claim_id(node_id, pro),
                        text=pro.strip(),
                        claim_type=ClaimType.INFERENCE,
                        confidence=0.6,
                        owner=owner,
                    )
                )

    # Extract inference claims from rationale (if present)
    rationale = artifact.get("rationale", [])
    if isinstance(rationale, list):
        for item in rationale:
            if item and isinstance(item, str) and item.strip():
                claims.append(
                    Claim(
                        claim_id=_generate_claim_id(node_id, item),
                        text=item.strip(),
                        claim_type=ClaimType.INFERENCE,
                        confidence=0.6,
                        owner=owner,
                    )
                )

    return claims
