"""Epistemics package for Delibera.

This package provides epistemic objects (claims, evidence, objections) and
operations (extraction, validation, ledger management) that enable
epistemically grounded deliberation.
"""

from delibera.epistemics.extract import extract_claims
from delibera.epistemics.ledger import Ledger, merge_ledgers
from delibera.epistemics.models import (
    Claim,
    ClaimStatus,
    ClaimType,
    Evidence,
    Objection,
    ObjectionSeverity,
    ObjectionStatus,
)
from delibera.epistemics.validate import ClaimCheckReport, validate_claims

__all__ = [
    "Claim",
    "ClaimCheckReport",
    "ClaimStatus",
    "ClaimType",
    "Evidence",
    "Ledger",
    "Objection",
    "ObjectionSeverity",
    "ObjectionStatus",
    "extract_claims",
    "merge_ledgers",
    "validate_claims",
]
