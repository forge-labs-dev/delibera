"""Epistemic object models for Delibera.

Defines the core epistemic objects: Claim, Evidence, and Objection.
These objects are used to track and validate reasoning throughout deliberation.
"""

from dataclasses import dataclass
from enum import Enum


class ClaimType(str, Enum):
    """Type of claim determining validation requirements.

    - FACT: Asserts something about the world; requires evidence.
    - INFERENCE: Draws a conclusion from other claims; inherits uncertainty.
    - VALUE: Expresses a preference or judgment; no evidence required.
    - PLAN: Proposes an action or steps; no evidence required.
    """

    FACT = "fact"
    INFERENCE = "inference"
    VALUE = "value"
    PLAN = "plan"


class ClaimStatus(str, Enum):
    """Validation status of a claim.

    - SUPPORTED: Claim is supported by evidence or inherently valid.
    - WEAK: Claim has indirect or partial support.
    - UNSUPPORTED: Claim lacks required evidence.
    - UNVALIDATED: Claim has not yet been validated.
    """

    SUPPORTED = "supported"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"
    UNVALIDATED = "unvalidated"


@dataclass
class Claim:
    """An atomic assertion extracted from agent outputs.

    Claims are the fundamental unit of epistemic tracking. Each claim
    has a type that determines its validation requirements.

    Attributes:
        claim_id: Unique identifier (deterministic hash-based).
        text: The claim text.
        claim_type: Type of claim (fact, inference, value, plan).
        confidence: Confidence score in [0, 1].
        owner: Role that introduced this claim.
        status: Current validation status.
    """

    claim_id: str
    text: str
    claim_type: ClaimType
    confidence: float
    owner: str
    status: ClaimStatus = ClaimStatus.UNVALIDATED


@dataclass
class Evidence:
    """Evidence supporting one or more claims.

    Evidence items are immutable once recorded and must be explicitly
    linked to claims via support relations.

    Attributes:
        evidence_id: Unique identifier.
        source: Document, URL, dataset, or file path.
        excerpt: The relevant supporting content.
        provenance: Structured provenance information (e.g., tool_call_id, node_id).
    """

    evidence_id: str
    source: str
    excerpt: str
    provenance: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for trace logging."""
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "excerpt": self.excerpt[:500],
            "provenance": self.provenance,
        }


class ObjectionSeverity(str, Enum):
    """Severity level of an objection.

    - BLOCKING: Prevents convergence unless resolved or accepted.
    - NONBLOCKING: Should be addressed but does not block convergence.
    """

    BLOCKING = "blocking"
    NONBLOCKING = "nonblocking"


class ObjectionStatus(str, Enum):
    """Status of an objection in its lifecycle.

    - OPEN: Objection has been raised and not addressed.
    - RESOLVED: Objection has been addressed through revision or evidence.
    - ACCEPTED: Objection remains valid but explicitly accepted as a tradeoff.
    """

    OPEN = "open"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


@dataclass
class Objection:
    """An objection challenging a claim or artifact.

    Objections track disagreements and concerns that must be
    addressed before convergence.

    Attributes:
        objection_id: Unique identifier (deterministic hash-based).
        target: Reference to a claim_id or "artifact".
        severity: Whether this objection blocks convergence.
        status: Current status in the objection lifecycle.
        rationale: Explanation of the objection.
        owner: Role that raised this objection.
    """

    objection_id: str
    target: str
    severity: ObjectionSeverity
    status: ObjectionStatus
    rationale: str
    owner: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for trace logging."""
        return {
            "objection_id": self.objection_id,
            "target": self.target,
            "severity": self.severity.value,
            "status": self.status.value,
            "rationale": self.rationale,
            "owner": self.owner,
        }
