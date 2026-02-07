"""Score weights for computing node scores.

Defines the ScoreWeights dataclass with default values for
weighting epistemic metrics in score computation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ScoreWeights:
    """Weights for scoring metrics.

    Each weight is multiplied by the corresponding metric value
    to produce a weighted score. Positive weights reward the metric,
    negative weights penalize it.

    Defaults are tuned for v2:
    - Strongly reward evidence coverage (supported fact claims)
    - Reward having evidence (up to diminishing returns)
    - Lightly penalize weak claims (inference without full support)
    - Penalize unsupported claims
    - Penalize blocking objections (but not so heavily they dominate)
    - Lightly penalize nonblocking objections

    With these weights, a well-evidenced proposal (coverage=1.0, count=5)
    scores +5.0 before penalties, giving evidence a realistic chance to
    counterbalance objections.

    Attributes:
        evidence_coverage: Weight for fraction of fact claims supported (0..1).
        evidence_count: Weight for number of evidence items (capped).
        weak_claims: Weight for count of weak claims (typically negative).
        unsupported_claims: Weight for count of unsupported claims (negative).
        blocking_objections: Weight for count of open blocking objections (negative).
        nonblocking_objections: Weight for count of open nonblocking objections (negative).
    """

    evidence_coverage: float = 3.0
    evidence_count: float = 0.4
    weak_claims: float = -0.3
    unsupported_claims: float = -0.5
    blocking_objections: float = -1.0
    nonblocking_objections: float = -0.2

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for trace logging and serialization."""
        return {
            "evidence_coverage": self.evidence_coverage,
            "evidence_count": self.evidence_count,
            "weak_claims": self.weak_claims,
            "unsupported_claims": self.unsupported_claims,
            "blocking_objections": self.blocking_objections,
            "nonblocking_objections": self.nonblocking_objections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreWeights":
        """Create from dictionary, using defaults for missing keys.

        Args:
            data: Dictionary with weight overrides.

        Returns:
            ScoreWeights with provided values merged with defaults.
        """
        defaults = cls()
        return cls(
            evidence_coverage=float(data.get("evidence_coverage", defaults.evidence_coverage)),
            evidence_count=float(data.get("evidence_count", defaults.evidence_count)),
            weak_claims=float(data.get("weak_claims", defaults.weak_claims)),
            unsupported_claims=float(data.get("unsupported_claims", defaults.unsupported_claims)),
            blocking_objections=float(
                data.get("blocking_objections", defaults.blocking_objections)
            ),
            nonblocking_objections=float(
                data.get("nonblocking_objections", defaults.nonblocking_objections)
            ),
        )


def create_default_weights() -> ScoreWeights:
    """Create a ScoreWeights instance with default values.

    Returns:
        ScoreWeights with default configuration.
    """
    return ScoreWeights()
