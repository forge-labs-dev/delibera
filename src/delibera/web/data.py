"""Data loading with Streamlit caching for the web UI.

Wraps existing delibera.inspect and delibera.trace functions with
@st.cache_data for efficient repeated access across page switches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from delibera.inspect.summarize import RunSummary, build_run_summary
from delibera.trace.reader import load_artifact, load_trace
from delibera.trace.replay import ReplayedRun, replay_from_directory


@dataclass
class RunInfo:
    """Minimal info about a run for the selector."""

    run_id: str
    path: Path
    question: str
    created_at: str


def list_runs(runs_dir: Path) -> list[RunInfo]:
    """Scan runs directory and return RunInfo list sorted by recency.

    Args:
        runs_dir: Path to directory containing run subdirectories.

    Returns:
        List of RunInfo, most recent first.
    """
    runs: list[RunInfo] = []
    if not runs_dir.exists():
        return runs

    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        trace_path = run_dir / "trace.jsonl"
        if not trace_path.exists():
            continue

        # Peek at first event to get question
        question = ""
        created_at = ""
        try:
            with trace_path.open("r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    event = json.loads(first_line)
                    payload = event.get("payload", {})
                    question = payload.get("question", "")
                    created_at = payload.get("created_at", "")
        except (json.JSONDecodeError, OSError):
            pass

        runs.append(
            RunInfo(
                run_id=run_dir.name,
                path=run_dir,
                question=question or "(no question)",
                created_at=created_at,
            )
        )

    return runs


@st.cache_data(ttl=300)
def load_summary(run_dir: str) -> dict[str, Any]:
    """Load and cache RunSummary as a dict (for Streamlit hashing).

    Args:
        run_dir: String path to run directory.

    Returns:
        RunSummary serialized to dict for caching.
    """
    path = Path(run_dir)
    summary = build_run_summary(path)
    return _summary_to_dict(summary)


@st.cache_data(ttl=300)
def load_replayed(run_dir: str) -> dict[str, Any]:
    """Load and cache ReplayedRun as a dict.

    Args:
        run_dir: String path to run directory.

    Returns:
        ReplayedRun serialized to dict for caching.
    """
    path = Path(run_dir)
    replayed = replay_from_directory(path)
    return _replayed_to_dict(replayed)


@st.cache_data(ttl=300)
def load_events(run_dir: str) -> list[dict[str, Any]]:
    """Load and cache trace events.

    Args:
        run_dir: String path to run directory.

    Returns:
        List of trace event dicts.
    """
    path = Path(run_dir)
    return load_trace(path / "trace.jsonl")


@st.cache_data(ttl=300)
def load_run_artifact(run_dir: str) -> dict[str, Any]:
    """Load and cache artifact.

    Args:
        run_dir: String path to run directory.

    Returns:
        Artifact dict.
    """
    path = Path(run_dir) / "artifact.json"
    if path.exists():
        return load_artifact(path)
    return {}


def get_events_by_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    """Filter events by type.

    Args:
        events: List of trace events.
        event_type: Event type to filter for.

    Returns:
        Filtered list.
    """
    return [e for e in events if e.get("event_type") == event_type]


def _summary_to_dict(summary: RunSummary) -> dict[str, Any]:
    """Convert RunSummary dataclass to a plain dict for Streamlit caching."""
    return {
        "run_id": summary.run_id,
        "question": summary.question,
        "created_at": summary.created_at,
        "protocol": {
            "name": summary.protocol.name,
            "version": summary.protocol.version,
            "source": summary.protocol.source,
        },
        "recommendation": summary.recommendation,
        "confidence": summary.confidence,
        "decision_explanation": summary.decision_explanation,
        "refinement": {
            "rounds_run": summary.refinement.rounds_run,
            "max_rounds": summary.refinement.max_rounds,
            "stop_reason": summary.refinement.stop_reason,
        },
        "selected_path": [
            {
                "node_id": n.node_id,
                "label": n.label,
                "kind": n.kind,
                "depth": n.depth,
                "score": n.score,
                "score_breakdown": n.score_breakdown,
                "claim_counts": n.claim_counts,
                "evidence_sources": n.evidence_sources,
                "evidence_count": n.evidence_count,
                "objections": {
                    "blocking_open": n.objections.blocking_open,
                    "blocking_accepted": n.objections.blocking_accepted,
                    "nonblocking_open": n.objections.nonblocking_open,
                    "nonblocking_accepted": n.objections.nonblocking_accepted,
                },
            }
            for n in summary.selected_path
        ],
        "key_claims": [
            {
                "claim_id": c.claim_id,
                "claim_type": c.claim_type,
                "text": c.text,
                "status": c.status,
                "citations": [
                    {
                        "evidence_id": cit.evidence_id,
                        "source": cit.source,
                        "excerpt": cit.excerpt,
                    }
                    for cit in c.citations
                ],
            }
            for c in summary.key_claims
        ],
        "prune_events": [
            {
                "candidates": p.candidates,
                "kept": p.kept,
                "pruned": p.pruned,
                "reason": p.reason,
            }
            for p in summary.prune_events
        ],
        "total_nodes": summary.total_nodes,
        "total_tool_calls": summary.total_tool_calls,
        "total_evidence": summary.total_evidence,
        "errors": summary.errors,
        "warnings": summary.warnings,
    }


def _replayed_to_dict(replayed: ReplayedRun) -> dict[str, Any]:
    """Convert ReplayedRun to a plain dict for Streamlit caching."""
    nodes = {}
    for nid, node in replayed.nodes.items():
        nodes[nid] = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "kind": node.kind,
            "depth": node.depth,
            "label": node.label,
            "artifact": node.artifact,
            "claim_summary": node.claim_summary,
            "status": node.status,
        }
    return {
        "run_id": replayed.run_id,
        "question": replayed.question,
        "created_at": replayed.created_at,
        "nodes": nodes,
        "root_id": replayed.root_id,
        "merged_id": replayed.merged_id,
        "final_artifact": replayed.final_artifact,
        "errors": replayed.errors,
        "warnings": replayed.warnings,
    }
