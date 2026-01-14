"""YAML protocol loader for Delibera.

Provides functions to load ProtocolSpec from YAML files with strict
validation and clear error messages.
"""

from pathlib import Path
from typing import Any

import yaml

from delibera.protocol.spec import (
    ConvergenceSpec,
    ExpandSpec,
    ProtocolSpec,
    PruneSpec,
    ReduceSpec,
    StepSpec,
    validate_protocol,
)


class ProtocolLoadError(Exception):
    """Error raised when protocol loading or validation fails."""

    def __init__(self, message: str, path: str | None = None) -> None:
        """Initialize error with message and optional path context.

        Args:
            message: The error description.
            path: Optional path context (e.g., "expand_rules[0].id").
        """
        self.path = path
        full_message = f"At '{path}': {message}" if path else message
        super().__init__(full_message)


# Valid values for enum-like fields
VALID_STEP_KINDS = {"work", "validate", "score"}
VALID_CHILD_KINDS = {"option", "plan", "risk"}
VALID_SOURCES = {"planner_output", "agent_output"}
VALID_PRUNE_RULES = {"epistemic_then_score", "score_only"}
VALID_REDUCE_RULES = {"promote_single", "merge_artifacts"}


def load_protocol_from_yaml(path: Path) -> ProtocolSpec:
    """Load a protocol specification from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed ProtocolSpec.

    Raises:
        ProtocolLoadError: If loading or validation fails.
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ProtocolLoadError(f"Invalid YAML syntax: {e}") from e

    if not isinstance(data, dict):
        raise ProtocolLoadError("Protocol file must contain a YAML mapping (object)")

    return protocol_from_dict(data)


def protocol_from_dict(data: dict[str, Any]) -> ProtocolSpec:
    """Convert a dictionary to a ProtocolSpec.

    Performs strict validation with clear error messages.

    Args:
        data: The protocol data dictionary.

    Returns:
        The parsed ProtocolSpec.

    Raises:
        ProtocolLoadError: If validation fails.
    """
    # Check for required top-level keys
    required_keys = {
        "name",
        "protocol_version",
        "max_depth",
        "expand_rules",
        "branch_pipeline",
        "prune",
        "reduce",
    }
    _check_required_keys(data, required_keys, "")

    # Check for unknown keys (strict mode)
    known_keys = required_keys | {"gates_enabled", "refine_loop", "convergence"}
    _check_unknown_keys(data, known_keys, "")

    # Parse basic fields
    name = _require_string(data, "name", "")
    protocol_version = _require_string(data, "protocol_version", "")
    max_depth = _require_int(data, "max_depth", "")
    gates_enabled = data.get("gates_enabled", True)
    if not isinstance(gates_enabled, bool):
        raise ProtocolLoadError("Expected boolean", "gates_enabled")

    # Parse expand_rules
    expand_rules_data = data["expand_rules"]
    if not isinstance(expand_rules_data, list):
        raise ProtocolLoadError("Expected list", "expand_rules")
    expand_rules = [
        _parse_expand_spec(item, f"expand_rules[{i}]") for i, item in enumerate(expand_rules_data)
    ]

    # Parse branch_pipeline
    branch_pipeline_data = data["branch_pipeline"]
    if not isinstance(branch_pipeline_data, list):
        raise ProtocolLoadError("Expected list", "branch_pipeline")
    branch_pipeline = [
        _parse_step_spec(item, f"branch_pipeline[{i}]")
        for i, item in enumerate(branch_pipeline_data)
    ]

    # Parse refine_loop (optional)
    refine_loop_data = data.get("refine_loop", [])
    if not isinstance(refine_loop_data, list):
        raise ProtocolLoadError("Expected list", "refine_loop")
    refine_loop = [
        _parse_step_spec(item, f"refine_loop[{i}]") for i, item in enumerate(refine_loop_data)
    ]

    # Parse prune
    prune_data = data["prune"]
    if not isinstance(prune_data, dict):
        raise ProtocolLoadError("Expected mapping", "prune")
    prune = _parse_prune_spec(prune_data, "prune")

    # Parse reduce
    reduce_data = data["reduce"]
    if not isinstance(reduce_data, dict):
        raise ProtocolLoadError("Expected mapping", "reduce")
    reduce = _parse_reduce_spec(reduce_data, "reduce")

    # Parse convergence (optional)
    convergence_data = data.get("convergence", {})
    if not isinstance(convergence_data, dict):
        raise ProtocolLoadError("Expected mapping", "convergence")
    convergence = _parse_convergence_spec(convergence_data, "convergence")

    # Create the spec (this runs dataclass validation)
    try:
        spec = ProtocolSpec(
            name=name,
            protocol_version=protocol_version,
            max_depth=max_depth,
            expand_rules=expand_rules,
            branch_pipeline=branch_pipeline,
            prune=prune,
            reduce=reduce,
            convergence=convergence,
            refine_loop=refine_loop,
            gates_enabled=gates_enabled,
        )
    except ValueError as e:
        raise ProtocolLoadError(str(e)) from e

    # Run additional validation
    errors = validate_protocol(spec)
    if errors:
        raise ProtocolLoadError("Validation failed: " + "; ".join(errors))

    return spec


def _parse_step_spec(data: Any, path: str) -> StepSpec:
    """Parse a step specification from a dictionary."""
    if not isinstance(data, dict):
        raise ProtocolLoadError("Expected mapping", path)

    required_keys = {"id", "kind", "step_name"}
    _check_required_keys(data, required_keys, path)

    known_keys = required_keys | {"role"}
    _check_unknown_keys(data, known_keys, path)

    step_id = _require_string(data, "id", path)
    kind = _require_string(data, "kind", path)
    step_name = _require_string(data, "step_name", path)

    if kind not in VALID_STEP_KINDS:
        raise ProtocolLoadError(
            f"Invalid step kind '{kind}'. Expected one of: {', '.join(sorted(VALID_STEP_KINDS))}",
            f"{path}.kind",
        )

    role = data.get("role")
    if role is not None and not isinstance(role, str):
        raise ProtocolLoadError("Expected string or null", f"{path}.role")

    try:
        return StepSpec(
            id=step_id,
            kind=kind,  # type: ignore[arg-type]
            step_name=step_name,
            role=role,
        )
    except ValueError as e:
        raise ProtocolLoadError(str(e), path) from e


def _parse_expand_spec(data: Any, path: str) -> ExpandSpec:
    """Parse an expand specification from a dictionary."""
    if not isinstance(data, dict):
        raise ProtocolLoadError("Expected mapping", path)

    required_keys = {"id", "at_step_id", "child_kind", "max_children", "depth", "source"}
    _check_required_keys(data, required_keys, path)

    known_keys = required_keys | {"min_quality"}
    _check_unknown_keys(data, known_keys, path)

    expand_id = _require_string(data, "id", path)
    at_step_id = _require_string(data, "at_step_id", path)
    child_kind = _require_string(data, "child_kind", path)
    max_children = _require_int(data, "max_children", path)
    depth = _require_int(data, "depth", path)
    source = _require_string(data, "source", path)

    if child_kind not in VALID_CHILD_KINDS:
        valid = ", ".join(sorted(VALID_CHILD_KINDS))
        raise ProtocolLoadError(
            f"Invalid child_kind '{child_kind}'. Expected one of: {valid}",
            f"{path}.child_kind",
        )

    if source not in VALID_SOURCES:
        raise ProtocolLoadError(
            f"Invalid source '{source}'. Expected one of: {', '.join(sorted(VALID_SOURCES))}",
            f"{path}.source",
        )

    min_quality = data.get("min_quality")
    if min_quality is not None and not isinstance(min_quality, (int, float)):
        raise ProtocolLoadError("Expected number or null", f"{path}.min_quality")

    try:
        return ExpandSpec(
            id=expand_id,
            at_step_id=at_step_id,
            child_kind=child_kind,  # type: ignore[arg-type]
            max_children=max_children,
            depth=depth,
            source=source,  # type: ignore[arg-type]
            min_quality=float(min_quality) if min_quality is not None else None,
        )
    except ValueError as e:
        raise ProtocolLoadError(str(e), path) from e


def _parse_prune_spec(data: dict[str, Any], path: str) -> PruneSpec:
    """Parse a prune specification from a dictionary."""
    required_keys = {"rule", "keep_k"}
    _check_required_keys(data, required_keys, path)
    _check_unknown_keys(data, required_keys, path)

    rule = _require_string(data, "rule", path)
    keep_k = _require_int(data, "keep_k", path)

    if rule not in VALID_PRUNE_RULES:
        raise ProtocolLoadError(
            f"Invalid prune rule '{rule}'. Expected one of: {', '.join(sorted(VALID_PRUNE_RULES))}",
            f"{path}.rule",
        )

    try:
        return PruneSpec(
            rule=rule,  # type: ignore[arg-type]
            keep_k=keep_k,
        )
    except ValueError as e:
        raise ProtocolLoadError(str(e), path) from e


def _parse_reduce_spec(data: dict[str, Any], path: str) -> ReduceSpec:
    """Parse a reduce specification from a dictionary."""
    required_keys = {"rule"}
    _check_required_keys(data, required_keys, path)
    _check_unknown_keys(data, required_keys, path)

    rule = _require_string(data, "rule", path)

    if rule not in VALID_REDUCE_RULES:
        valid = ", ".join(sorted(VALID_REDUCE_RULES))
        raise ProtocolLoadError(
            f"Invalid reduce rule '{rule}'. Expected one of: {valid}",
            f"{path}.rule",
        )

    return ReduceSpec(rule=rule)  # type: ignore[arg-type]


def _parse_convergence_spec(data: dict[str, Any], path: str) -> ConvergenceSpec:
    """Parse a convergence specification from a dictionary."""
    known_keys = {"max_rounds", "weak_threshold", "score_epsilon"}
    _check_unknown_keys(data, known_keys, path)

    max_rounds = data.get("max_rounds", 0)
    if not isinstance(max_rounds, int):
        raise ProtocolLoadError("Expected integer", f"{path}.max_rounds")

    weak_threshold = data.get("weak_threshold", 5)
    if not isinstance(weak_threshold, int):
        raise ProtocolLoadError("Expected integer", f"{path}.weak_threshold")

    score_epsilon = data.get("score_epsilon", 0.01)
    if not isinstance(score_epsilon, (int, float)):
        raise ProtocolLoadError("Expected number", f"{path}.score_epsilon")

    try:
        return ConvergenceSpec(
            max_rounds=max_rounds,
            weak_threshold=weak_threshold,
            score_epsilon=float(score_epsilon),
        )
    except ValueError as e:
        raise ProtocolLoadError(str(e), path) from e


def _check_required_keys(data: dict[str, Any], required: set[str], path: str) -> None:
    """Check that all required keys are present."""
    missing = required - set(data.keys())
    if missing:
        path_prefix = f"At '{path}': " if path else ""
        raise ProtocolLoadError(
            f"{path_prefix}Missing required key(s): {', '.join(sorted(missing))}"
        )


def _check_unknown_keys(data: dict[str, Any], known: set[str], path: str) -> None:
    """Check for unknown keys (strict mode)."""
    unknown = set(data.keys()) - known
    if unknown:
        path_prefix = f"At '{path}': " if path else ""
        raise ProtocolLoadError(f"{path_prefix}Unknown key(s): {', '.join(sorted(unknown))}")


def _require_string(data: dict[str, Any], key: str, path: str) -> str:
    """Get a required string value."""
    value = data.get(key)
    if not isinstance(value, str):
        full_path = f"{path}.{key}" if path else key
        raise ProtocolLoadError("Expected string", full_path)
    return value


def _require_int(data: dict[str, Any], key: str, path: str) -> int:
    """Get a required integer value."""
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        full_path = f"{path}.{key}" if path else key
        raise ProtocolLoadError("Expected integer", full_path)
    return value
