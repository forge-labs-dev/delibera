"""Data models for the evaluation framework.

Defines EvalSuite, EvalCase, EvalResult, and metric types.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """A single evaluation case within a suite.

    Attributes:
        id: Unique identifier for this case within the suite.
        question: The question to deliberate on.
        evidence_dir: Optional path to evidence directory for this case.
        expected: Optional dict of metric constraints for comparison.
    """

    id: str
    question: str
    evidence_dir: Path | None = None
    expected: dict[str, int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate case configuration."""
        if not self.id:
            raise ValueError("EvalCase id cannot be empty")
        if not self.question:
            raise ValueError("EvalCase question cannot be empty")
        if self.id and not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"EvalCase id must be alphanumeric with underscores/hyphens: {self.id}"
            )


@dataclass
class EvalSuite:
    """A collection of evaluation cases.

    Attributes:
        suite_name: Name of the evaluation suite.
        cases: List of evaluation cases to run.
        protocol: Optional path to YAML protocol file.
    """

    suite_name: str
    cases: list[EvalCase]
    protocol: Path | None = None

    def __post_init__(self) -> None:
        """Validate suite configuration."""
        if not self.suite_name:
            raise ValueError("suite_name cannot be empty")
        if not self.cases:
            raise ValueError("suite must have at least one case")
        # Check for duplicate case IDs
        case_ids = [c.id for c in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Duplicate case IDs found in suite")


@dataclass
class EvalResult:
    """Result from running a single evaluation case.

    Attributes:
        case_id: The ID of the evaluated case.
        run_id: The Delibera run ID for this evaluation.
        run_dir: Path to the run directory.
        metrics: Extracted metrics from the run.
        expected: The expected metric constraints.
        passed: Whether all constraints were satisfied.
        failures: List of constraint violation messages.
        error: Error message if run failed entirely.
    """

    case_id: str
    run_id: str
    run_dir: Path
    metrics: dict[str, int | float | bool] = field(default_factory=dict)
    expected: dict[str, int | float | bool] = field(default_factory=dict)
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "metrics": self.metrics,
            "expected": self.expected,
            "passed": self.passed,
            "failures": self.failures,
            "error": self.error,
        }
