"""Sidebar run selector component."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from delibera.web.data import RunInfo, list_runs


def render_run_selector(runs_dir: Path) -> RunInfo | None:
    """Render run selector in the sidebar and return selected run.

    Args:
        runs_dir: Path to runs directory.

    Returns:
        Selected RunInfo or None if no runs available.
    """
    runs = list_runs(runs_dir)

    if not runs:
        st.sidebar.warning("No runs found in " + str(runs_dir))
        return None

    def format_run(run: RunInfo) -> str:
        # Show date + truncated question
        date_part = run.run_id[:8]
        date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        q_preview = run.question[:50] + ("..." if len(run.question) > 50 else "")
        return f"{date_str} | {q_preview}"

    selected_idx = st.sidebar.selectbox(
        "Select Run",
        range(len(runs)),
        format_func=lambda i: format_run(runs[i]),
        key="run_selector",
    )

    if selected_idx is not None:
        selected: RunInfo = runs[selected_idx]
        st.sidebar.caption(f"Run ID: `{selected.run_id}`")
        return selected

    return None
