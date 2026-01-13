"""Claim validation for epistemically grounded deliberation.

Validates claims based on their type and available evidence.
"""

from dataclasses import dataclass

from delibera.epistemics.models import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
)


@dataclass
class ClaimCheckReport:
    """Report from claim validation.

    Summarizes the validation results for a set of claims.

    Attributes:
        supported: Count of supported claims.
        weak: Count of weak claims.
        unsupported: Count of unsupported claims.
        details: List of per-claim status details.
    """

    supported: int
    weak: int
    unsupported: int
    details: list[dict[str, str]]


def _claim_has_evidence_support(claim: Claim, evidence: list[Evidence]) -> bool:
    """Check if any evidence excerpt contains keywords from the claim.

    This is a simple heuristic: the claim is considered supported if
    at least one evidence excerpt contains words from the claim text.

    Args:
        claim: The claim to check.
        evidence: Available evidence items.

    Returns:
        True if evidence supports the claim.
    """
    if not evidence:
        return False

    # Extract keywords from claim text (words of 4+ chars)
    claim_words = {word.lower().strip(".,;:!?") for word in claim.text.split() if len(word) >= 4}

    # Check if any evidence excerpt contains claim keywords
    for ev in evidence:
        excerpt_lower = ev.excerpt.lower()
        # Supported if at least one keyword appears in excerpt
        for word in claim_words:
            if word in excerpt_lower:
                return True

    return False


def validate_claims(
    claims: list[Claim],
    evidence: list[Evidence],
) -> ClaimCheckReport:
    """Validate claims based on type and available evidence.

    Validation rules (PR #7):
    - plan claims -> supported (no evidence required)
    - value claims -> supported (no evidence required)
    - fact claims -> supported if evidence contains claim keywords,
                     unsupported otherwise
    - inference claims -> supported if evidence exists AND no unsupported facts,
                          weak otherwise

    Updates claim.status in-place and returns a summary report.

    Args:
        claims: List of claims to validate.
        evidence: List of available evidence from ledger.

    Returns:
        ClaimCheckReport summarizing validation results.
    """
    supported = 0
    weak = 0
    unsupported = 0
    details: list[dict[str, str]] = []

    # First pass: validate fact, plan, and value claims
    fact_results: dict[str, ClaimStatus] = {}
    for claim in claims:
        if claim.claim_type == ClaimType.PLAN:
            # Plan claims are automatically supported
            claim.status = ClaimStatus.SUPPORTED
            supported += 1
        elif claim.claim_type == ClaimType.VALUE:
            # Value claims are automatically supported
            claim.status = ClaimStatus.SUPPORTED
            supported += 1
        elif claim.claim_type == ClaimType.FACT:
            # Fact claims: supported if evidence contains keywords
            if _claim_has_evidence_support(claim, evidence):
                claim.status = ClaimStatus.SUPPORTED
                supported += 1
            else:
                claim.status = ClaimStatus.UNSUPPORTED
                unsupported += 1
            fact_results[claim.claim_id] = claim.status
        # Skip inference claims for now

    # Count unsupported facts for inference validation
    unsupported_facts = sum(
        1 for status in fact_results.values() if status == ClaimStatus.UNSUPPORTED
    )

    # Second pass: validate inference claims
    for claim in claims:
        if claim.claim_type == ClaimType.INFERENCE:
            # Inference: supported if evidence exists AND no unsupported facts
            if len(evidence) > 0 and unsupported_facts == 0:
                claim.status = ClaimStatus.SUPPORTED
                supported += 1
            else:
                claim.status = ClaimStatus.WEAK
                weak += 1
        elif claim.claim_type not in (ClaimType.PLAN, ClaimType.VALUE, ClaimType.FACT):
            # Fallback for any unknown type
            claim.status = ClaimStatus.UNSUPPORTED
            unsupported += 1

    # Build details list
    for claim in claims:
        detail: dict[str, str] = {
            "claim_id": claim.claim_id,
            "status": claim.status.value,
            "type": claim.claim_type.value,
        }
        details.append(detail)

    return ClaimCheckReport(
        supported=supported,
        weak=weak,
        unsupported=unsupported,
        details=details,
    )
