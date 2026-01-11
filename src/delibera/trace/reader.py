"""Trace file reader for replay.

Loads and parses trace.jsonl files into structured event lists.
"""

import json
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Load and parse a trace.jsonl file.

    Args:
        path: Path to the trace.jsonl file.

    Returns:
        List of parsed trace event dictionaries.

    Raises:
        FileNotFoundError: If the trace file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
    """
    events: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e

    return events


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and parse an artifact.json file.

    Args:
        path: Path to the artifact.json file.

    Returns:
        Parsed artifact dictionary.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with path.open("r", encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result
