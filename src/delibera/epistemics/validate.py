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


def validate_claims(
    claims: list[Claim],
    evidence: list[Evidence],
) -> ClaimCheckReport:
    """Validate claims based on type and available evidence.

    Validation rules (v1, deterministic):
    - plan claims -> supported (no evidence required)
    - value claims -> supported (no evidence required)
    - inference claims -> weak (no evidence in v1)
    - fact claims -> unsupported (unless evidence exists)

    Updates claim.status in-place and returns a summary report.

    Args:
        claims: List of claims to validate.
        evidence: List of available evidence (empty in v1).

    Returns:
        ClaimCheckReport summarizing validation results.
    """
    supported = 0
    weak = 0
    unsupported = 0
    details: list[dict[str, str]] = []

    # Build evidence lookup (unused in v1, but structure is ready)
    _ = {e.evidence_id: e for e in evidence}

    for claim in claims:
        if claim.claim_type == ClaimType.PLAN:
            # Plan claims are automatically supported
            claim.status = ClaimStatus.SUPPORTED
            supported += 1
        elif claim.claim_type == ClaimType.VALUE:
            # Value claims are automatically supported
            claim.status = ClaimStatus.SUPPORTED
            supported += 1
        elif claim.claim_type == ClaimType.INFERENCE:
            # Inference claims are weak without evidence in v1
            claim.status = ClaimStatus.WEAK
            weak += 1
        elif claim.claim_type == ClaimType.FACT:
            # Fact claims require evidence; unsupported by default in v1
            claim.status = ClaimStatus.UNSUPPORTED
            unsupported += 1
        else:
            # Fallback for any unknown type
            claim.status = ClaimStatus.UNSUPPORTED
            unsupported += 1

        details.append(
            {
                "claim_id": claim.claim_id,
                "status": claim.status.value,
                "type": claim.claim_type.value,
            }
        )

    return ClaimCheckReport(
        supported=supported,
        weak=weak,
        unsupported=unsupported,
        details=details,
    )
