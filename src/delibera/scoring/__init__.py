"""Scoring subsystem for Delibera.

Provides deterministic scoring of nodes based on epistemic metrics
and configurable weights.
"""

from delibera.scoring.metrics import (
    EVIDENCE_COUNT_CAP,
    compute_metrics,
    metrics_to_summary,
)
from delibera.scoring.score import ScoreResult, compute_score, score_node
from delibera.scoring.weights import ScoreWeights, create_default_weights

__all__ = [
    # Metrics
    "compute_metrics",
    "metrics_to_summary",
    "EVIDENCE_COUNT_CAP",
    # Weights
    "ScoreWeights",
    "create_default_weights",
    # Score
    "compute_score",
    "score_node",
    "ScoreResult",
]
