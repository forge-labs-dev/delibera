"""Delibera Web UI — Streamlit entry point.

Launch with: streamlit run src/delibera/web/app.py -- --runs-dir runs
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st  # noqa: E402

# Ensure the project root is on the path when run directly
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from delibera.web.components.run_selector import render_run_selector  # noqa: E402
from delibera.web.data import (  # noqa: E402
    load_events,
    load_replayed,
    load_run_artifact,
    load_summary,
)
from delibera.web.views import (  # noqa: E402
    evidence,
    new_run,
    overview,
    scores,
    timeline,
    tree,
)


def main() -> None:
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Delibera",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Parse --runs-dir from command-line args
    runs_dir = Path("runs")
    args = sys.argv[1:]
    if "--runs-dir" in args:
        idx = args.index("--runs-dir")
        if idx + 1 < len(args):
            runs_dir = Path(args[idx + 1])

    # Sidebar header
    st.sidebar.markdown("# ⚖️ Delibera")
    st.sidebar.caption("Decision-Grade AI Deliberation")
    st.sidebar.divider()

    # Page navigation — New Run is always available
    page = st.sidebar.radio(
        "View",
        ["New Run", "Overview", "Tree", "Scores", "Evidence", "Timeline"],
        key="page_nav",
    )

    st.sidebar.divider()

    # New Run page doesn't need a selected run
    if page == "New Run":
        st.sidebar.caption("Built with Delibera — structured multi-agent deliberation")
        new_run.render()
        return

    # Run selector (only shown for result pages)
    selected_run = render_run_selector(runs_dir)

    if not selected_run:
        st.title("⚖️ Delibera")
        st.info(
            "No deliberation runs found. Run a deliberation first, or use the **New Run** page."
        )
        return

    # Load data for selected run
    run_dir_str = str(selected_run.path)
    summary = load_summary(run_dir_str)
    replayed = load_replayed(run_dir_str)
    events = load_events(run_dir_str)
    artifact = load_run_artifact(run_dir_str)

    # Gemini badge
    st.sidebar.divider()
    llm_calls = len([e for e in events if e.get("event_type") == "llm_call_succeeded"])
    if llm_calls > 0:
        st.sidebar.markdown("**🤖 Powered by Google Gemini**")
        st.sidebar.caption(f"{llm_calls} Gemini API calls in this run")
    st.sidebar.caption("Built with Delibera — structured multi-agent deliberation")

    # Render selected page
    if page == "Overview":
        overview.render(summary, events, artifact)
    elif page == "Tree":
        tree.render(replayed, events)
    elif page == "Scores":
        scores.render(summary, events, artifact)
    elif page == "Evidence":
        evidence.render(summary, events, artifact)
    elif page == "Timeline":
        timeline.render(events)


if __name__ == "__main__":
    main()
