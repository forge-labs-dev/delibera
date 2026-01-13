"""Score computation for nodes.

Combines metrics and weights to produce a scalar score
and score vector for use in pruning decisions.
"""

from dataclasses import dataclass
from typing import Any

from delibera.engine.tree import Node
from delibera.scoring.metrics import compute_metrics
from delibera.scoring.weights import ScoreWeights, create_default_weights


@dataclass
class ScoreResult:
    """Result of scoring a node.

    Attributes:
        score: The computed scalar score.
        metrics: The raw metrics used in computation.
        weights: The weights used in computation.
    """

    score: float
    metrics: dict[str, float]
    weights: ScoreWeights

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trace logging."""
        return {
            "score": self.score,
            "metrics": self.metrics,
            "weights": self.weights.to_dict(),
        }


def compute_score(metrics: dict[str, float], weights: ScoreWeights) -> float:
    """Compute a scalar score from metrics and weights.

    The score is computed as a weighted sum:
        score = sum(weight_i * metric_i)

    Metrics are used directly without additional normalization.
    Counts (weak_claims, unsupported_claims, blocking_objections)
    contribute linearly to the score.

    Args:
        metrics: Dictionary of metric name to value.
        weights: The weights to apply.

    Returns:
        The computed scalar score.
    """
    score = 0.0

    # Evidence coverage (0..1) weighted
    score += weights.evidence_coverage * metrics.get("evidence_coverage", 0.0)

    # Evidence count (0..cap) weighted
    score += weights.evidence_count * metrics.get("evidence_count", 0.0)

    # Weak claims (count) weighted (typically negative weight)
    score += weights.weak_claims * metrics.get("weak_claims", 0.0)

    # Unsupported claims (count) weighted (negative weight)
    score += weights.unsupported_claims * metrics.get("unsupported_claims", 0.0)

    # Blocking objections (count) weighted (heavily negative)
    score += weights.blocking_objections * metrics.get("blocking_objections", 0.0)

    return score


def score_node(node: Node, weights: ScoreWeights | None = None) -> ScoreResult:
    """Compute score for a single node.

    Args:
        node: The node to score.
        weights: Optional weights to use. Defaults to default weights.

    Returns:
        ScoreResult with score, metrics, and weights.
    """
    if weights is None:
        weights = create_default_weights()

    metrics = compute_metrics(node)
    score = compute_score(metrics, weights)

    return ScoreResult(score=score, metrics=metrics, weights=weights)
