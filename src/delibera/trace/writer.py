"""Trace writer for persisting deliberation events."""

import json
import threading
from pathlib import Path
from typing import Any

from delibera.trace.events import TraceEvent


class TraceWriter:
    """Writes trace events to JSONL and artifacts to JSON.

    Thread-safe: all file writes are protected by a lock for
    concurrent branch execution.

    Each run gets its own directory under the runs root:
        runs/<run_id>/
            trace.jsonl
            artifact.json
    """

    def __init__(self, run_dir: Path) -> None:
        """Initialize the trace writer.

        Args:
            run_dir: The directory for this run's output files.
        """
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.artifact_path = self.run_dir / "artifact.json"
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        """Append a trace event to the JSONL file.

        Args:
            event: The trace event to record.
        """
        with self._lock, self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def write_artifact(self, artifact: dict[str, Any]) -> None:
        """Write the final artifact to JSON.

        Args:
            artifact: The final decision artifact.
        """
        with self.artifact_path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
