"""Run state for Delibera deliberation runs."""

from dataclasses import dataclass

from delibera.engine.tree import DeliberationTree


@dataclass
class RunState:
    """State for a single deliberation run.

    Holds the run metadata and the deliberation tree.
    """

    run_id: str
    question: str
    created_at: str  # ISO format timestamp
    tree: DeliberationTree
