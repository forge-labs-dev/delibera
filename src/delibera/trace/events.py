"""Trace event definitions for Delibera."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event in a deliberation run.

    Trace events are append-only and form the complete audit log
    of a deliberation run.
    """

    event_type: str
    run_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert the event to a JSON-serializable dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "payload": self.payload,
        }
