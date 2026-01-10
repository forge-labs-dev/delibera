"""Stub agents for v0 testing without LLMs."""

from typing import Any


class PlannerStub:
    """Stub planner that returns deterministic branch labels.

    For v0 testing, always returns 3 options.
    """

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate branch labels for expansion.

        Args:
            context: Must contain "question" key.

        Returns:
            Dict with "branches" key containing list of labels.
        """
        question = context.get("question", "")
        return {
            "branches": [
                f"Option A: Direct approach to '{question[:30]}...'",
                f"Option B: Alternative approach to '{question[:30]}...'",
                f"Option C: Conservative approach to '{question[:30]}...'",
            ],
            "role": "planner",
        }


class ProposerStub:
    """Stub proposer that returns deterministic proposals with scores.

    Scores are deterministic based on option label for testing:
    - Option A: 0.8 (highest)
    - Option B: 0.6 (medium)
    - Option C: 0.4 (lowest, will be pruned)
    """

    # Deterministic scores for testing
    _SCORES: dict[str, float] = {
        "A": 0.8,
        "B": 0.6,
        "C": 0.4,
    }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate a proposal for a branch.

        Args:
            context: Must contain "label" key with branch label,
                and "question" key with the original question.

        Returns:
            Dict with proposal details and score.
        """
        label = context.get("label", "")
        question = context.get("question", "")

        # Extract option letter (A, B, or C) from label
        option_letter = "A"  # default
        for letter in ["A", "B", "C"]:
            if f"Option {letter}" in label:
                option_letter = letter
                break

        score = self._SCORES.get(option_letter, 0.5)

        return {
            "proposal": f"Proposal for {label}",
            "summary": f"This approach addresses '{question[:50]}...' by...",
            "pros": [f"Pro 1 for option {option_letter}", f"Pro 2 for option {option_letter}"],
            "cons": [f"Con 1 for option {option_letter}"],
            "score": score,
            "role": "proposer",
        }
