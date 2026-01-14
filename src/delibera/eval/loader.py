"""YAML loader for evaluation suites.

Loads and validates evaluation suite definitions from YAML files.
"""

from pathlib import Path
from typing import Any

import yaml

from delibera.eval.models import EvalCase, EvalSuite


class EvalSuiteLoadError(Exception):
    """Exception raised when loading an eval suite fails."""

    pass


# Required keys in suite YAML
REQUIRED_SUITE_KEYS = {"suite_name", "cases"}

# Allowed keys in suite YAML
ALLOWED_SUITE_KEYS = {"suite_name", "protocol", "cases"}

# Required keys in case dict
REQUIRED_CASE_KEYS = {"id", "question"}

# Allowed keys in case dict
ALLOWED_CASE_KEYS = {"id", "question", "evidence_dir", "expected"}


def load_eval_suite(path: Path) -> EvalSuite:
    """Load an evaluation suite from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed EvalSuite.

    Raises:
        FileNotFoundError: If the file does not exist.
        EvalSuiteLoadError: If the YAML is invalid or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Eval suite file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise EvalSuiteLoadError(f"Invalid YAML syntax: {e}") from e

    if data is None:
        raise EvalSuiteLoadError("Empty YAML file")

    if not isinstance(data, dict):
        raise EvalSuiteLoadError("Root of YAML must be a mapping")

    return _parse_suite(data, path.parent)


def _parse_suite(data: dict[str, Any], base_dir: Path) -> EvalSuite:
    """Parse a suite dictionary into an EvalSuite.

    Args:
        data: Parsed YAML dictionary.
        base_dir: Base directory for resolving relative paths.

    Returns:
        EvalSuite instance.

    Raises:
        EvalSuiteLoadError: If validation fails.
    """
    # Check required keys
    for key in REQUIRED_SUITE_KEYS:
        if key not in data:
            raise EvalSuiteLoadError(f"Missing required key: '{key}'")

    # Check for unknown keys
    unknown_keys = set(data.keys()) - ALLOWED_SUITE_KEYS
    if unknown_keys:
        raise EvalSuiteLoadError(f"Unknown key(s): {', '.join(sorted(unknown_keys))}")

    suite_name = data["suite_name"]
    if not isinstance(suite_name, str) or not suite_name:
        raise EvalSuiteLoadError("suite_name must be a non-empty string")

    # Parse protocol path
    protocol: Path | None = None
    if "protocol" in data and data["protocol"]:
        protocol_str = data["protocol"]
        if not isinstance(protocol_str, str):
            raise EvalSuiteLoadError("protocol must be a string path")
        protocol = base_dir / protocol_str
        if not protocol.exists():
            raise EvalSuiteLoadError(f"Protocol file not found: {protocol}")

    # Parse cases
    cases_data = data["cases"]
    if not isinstance(cases_data, list):
        raise EvalSuiteLoadError("cases must be a list")
    if not cases_data:
        raise EvalSuiteLoadError("cases list cannot be empty")

    cases: list[EvalCase] = []
    for i, case_data in enumerate(cases_data):
        try:
            case = _parse_case(case_data, base_dir)
            cases.append(case)
        except EvalSuiteLoadError as e:
            raise EvalSuiteLoadError(f"Invalid case at index {i}: {e}") from e

    try:
        return EvalSuite(
            suite_name=suite_name,
            cases=cases,
            protocol=protocol,
        )
    except ValueError as e:
        raise EvalSuiteLoadError(f"Suite validation failed: {e}") from e


def _parse_case(data: dict[str, Any], base_dir: Path) -> EvalCase:
    """Parse a case dictionary into an EvalCase.

    Args:
        data: Case dictionary from YAML.
        base_dir: Base directory for resolving relative paths.

    Returns:
        EvalCase instance.

    Raises:
        EvalSuiteLoadError: If validation fails.
    """
    if not isinstance(data, dict):
        raise EvalSuiteLoadError("Case must be a mapping")

    # Check required keys
    for key in REQUIRED_CASE_KEYS:
        if key not in data:
            raise EvalSuiteLoadError(f"Missing required key: '{key}'")

    # Check for unknown keys
    unknown_keys = set(data.keys()) - ALLOWED_CASE_KEYS
    if unknown_keys:
        raise EvalSuiteLoadError(f"Unknown key(s): {', '.join(sorted(unknown_keys))}")

    case_id = data["id"]
    if not isinstance(case_id, str) or not case_id:
        raise EvalSuiteLoadError("id must be a non-empty string")

    question = data["question"]
    if not isinstance(question, str) or not question:
        raise EvalSuiteLoadError("question must be a non-empty string")

    # Parse evidence_dir
    evidence_dir: Path | None = None
    if "evidence_dir" in data and data["evidence_dir"]:
        evidence_str = data["evidence_dir"]
        if not isinstance(evidence_str, str):
            raise EvalSuiteLoadError("evidence_dir must be a string path")
        evidence_dir = base_dir / evidence_str
        if not evidence_dir.exists():
            raise EvalSuiteLoadError(f"Evidence directory not found: {evidence_dir}")
        if not evidence_dir.is_dir():
            raise EvalSuiteLoadError(f"evidence_dir must be a directory: {evidence_dir}")

    # Parse expected constraints
    expected: dict[str, int | float | bool] = {}
    if "expected" in data and data["expected"]:
        expected_data = data["expected"]
        if not isinstance(expected_data, dict):
            raise EvalSuiteLoadError("expected must be a mapping")
        for key, value in expected_data.items():
            if not isinstance(key, str):
                raise EvalSuiteLoadError(f"Expected key must be string: {key}")
            if not isinstance(value, (int, float, bool)):
                raise EvalSuiteLoadError(f"Expected value for '{key}' must be int, float, or bool")
            expected[key] = value

    try:
        return EvalCase(
            id=case_id,
            question=question,
            evidence_dir=evidence_dir,
            expected=expected,
        )
    except ValueError as e:
        raise EvalSuiteLoadError(f"Case validation failed: {e}") from e
