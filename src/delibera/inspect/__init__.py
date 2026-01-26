"""Run inspection and report generation for Delibera.

This module provides tools for inspecting completed runs and generating
human-readable summaries and reports without re-running agents.
"""

from delibera.inspect.render_md import render_markdown
from delibera.inspect.render_text import render_text
from delibera.inspect.summarize import (
    NodeSummary,
    PruneEvent,
    RunSummary,
    build_run_summary,
)

__all__ = [
    # Summary builder
    "RunSummary",
    "NodeSummary",
    "PruneEvent",
    "build_run_summary",
    # Renderers
    "render_text",
    "render_markdown",
]
