"""Smoke tests for Delibera CLI and core functionality.

These tests verify basic functionality without full deliberation runs.
They are fast and suitable for pre-commit hooks.
"""

import subprocess
import sys
from pathlib import Path


def test_cli_version_command() -> None:
    """Test that `delibera version` runs and prints version."""
    result = subprocess.run(
        [sys.executable, "-m", "delibera.cli", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "delibera" in result.stdout.lower()
    assert "0." in result.stdout  # Version string like 0.1.0


def test_cli_help_command() -> None:
    """Test that `delibera --help` runs and shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "delibera.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "delibera" in result.stdout.lower()
    assert "run" in result.stdout
    assert "replay" in result.stdout


def test_cli_version_flag() -> None:
    """Test that `delibera --version` works."""
    result = subprocess.run(
        [sys.executable, "-m", "delibera.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "delibera" in result.stdout.lower()


def test_imports_work() -> None:
    """Test that core imports succeed without errors."""
    from delibera import __version__
    from delibera.engine import Engine
    from delibera.protocol import DEFAULT_PROTOCOL, ProtocolSpec

    assert __version__.__version__
    assert Engine
    assert DEFAULT_PROTOCOL
    assert ProtocolSpec


def test_protocol_creation() -> None:
    """Test that default protocol can be created and validated."""
    from delibera.protocol import (
        DEFAULT_PROTOCOL,
        create_simple_protocol,
        create_tree_protocol_v1,
        validate_protocol,
    )

    # Default protocol should be valid
    errors = validate_protocol(DEFAULT_PROTOCOL)
    assert errors == []

    # Create protocols and validate
    simple = create_simple_protocol()
    assert simple.name == "simple_protocol"
    errors = validate_protocol(simple)
    assert errors == []

    tree = create_tree_protocol_v1()
    assert tree.name == "tree_protocol_v1"
    errors = validate_protocol(tree)
    assert errors == []


def test_yaml_protocol_file_exists() -> None:
    """Test that builtin protocol YAML files exist."""
    protocols_dir = Path(__file__).parent.parent / "protocols"
    if protocols_dir.exists():
        yaml_files = list(protocols_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML protocol files found"


def test_protocol_warnings() -> None:
    """Test that protocol warnings work."""
    from delibera.protocol import (
        ConvergenceSpec,
        ExpandSpec,
        ProtocolSpec,
        PruneSpec,
        ReduceSpec,
        StepSpec,
        warnings_for_protocol,
    )

    # Create a protocol with refine_loop but max_rounds=0
    spec = ProtocolSpec(
        name="warning_test",
        protocol_version="v1",
        max_depth=1,
        expand_rules=[
            ExpandSpec(
                id="expand",
                at_step_id="plan",
                child_kind="option",
                max_children=3,
                depth=1,
                source="planner_output",
            ),
        ],
        branch_pipeline=[
            StepSpec(id="propose", kind="work", step_name="PROPOSE", role="proposer"),
        ],
        refine_loop=[
            StepSpec(id="refine", kind="work", step_name="REFINE", role="refiner"),
        ],
        prune=PruneSpec(rule="epistemic_then_score", keep_k=2),
        reduce=ReduceSpec(rule="merge_artifacts"),
        convergence=ConvergenceSpec(max_rounds=0),  # Will cause warning
    )

    warnings = warnings_for_protocol(spec)
    assert len(warnings) == 1
    assert "refine_loop" in warnings[0]
    assert "max_rounds" in warnings[0]
