"""New Run page — configure and launch a new deliberation."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from delibera.protocol.loader import ProtocolLoadError, protocol_from_dict
from delibera.protocol.spec import warnings_for_protocol

# ---------------------------------------------------------------------------
# Protocol presets
# ---------------------------------------------------------------------------

_PIPELINE_FULL: list[dict[str, Any]] = [
    {"id": "propose", "kind": "work", "step_name": "PROPOSE", "role": "proposer"},
    {"id": "research", "kind": "work", "step_name": "RESEARCH", "role": "researcher"},
    {"id": "validate", "kind": "validate", "step_name": "CLAIM_CHECK"},
    {"id": "redteam", "kind": "work", "step_name": "REDTEAM", "role": "redteam"},
]

PROTOCOL_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "label": "Quick Analysis",
        "description": ("Single-level, 3 options, no refinement. Fastest run (~2 min)."),
        "spec": {
            "name": "quick_protocol",
            "protocol_version": "v1",
            "max_depth": 1,
            "gates_enabled": True,
            "expand_rules": [
                {
                    "id": "expand_options",
                    "at_step_id": "plan",
                    "child_kind": "option",
                    "max_children": 3,
                    "depth": 1,
                    "source": "planner_output",
                },
            ],
            "branch_pipeline": _PIPELINE_FULL,
            "refine_loop": [],
            "prune": {"rule": "epistemic_then_score", "keep_k": 2},
            "reduce": {"rule": "merge_artifacts"},
            "convergence": {"max_rounds": 0},
        },
    },
    "thorough": {
        "label": "Thorough Analysis",
        "description": (
            "Single-level with iterative refinement (up to 3 rounds). Better quality (~5 min)."
        ),
        "spec": {
            "name": "thorough_protocol",
            "protocol_version": "v1",
            "max_depth": 1,
            "gates_enabled": True,
            "expand_rules": [
                {
                    "id": "expand_options",
                    "at_step_id": "plan",
                    "child_kind": "option",
                    "max_children": 3,
                    "depth": 1,
                    "source": "planner_output",
                },
            ],
            "branch_pipeline": _PIPELINE_FULL,
            "refine_loop": [
                {
                    "id": "refine",
                    "kind": "work",
                    "step_name": "REFINE",
                    "role": "refiner",
                },
                {
                    "id": "revalidate",
                    "kind": "validate",
                    "step_name": "CLAIM_CHECK",
                },
            ],
            "prune": {"rule": "epistemic_then_score", "keep_k": 2},
            "reduce": {"rule": "merge_artifacts"},
            "convergence": {
                "max_rounds": 3,
                "weak_threshold": 5,
                "score_epsilon": 0.01,
            },
        },
    },
    "deep_tree": {
        "label": "Deep Tree",
        "description": (
            "Two-level tree: 3 options, each with 2 sub-plans. Widest exploration (~8 min)."
        ),
        "spec": {
            "name": "deep_tree_protocol",
            "protocol_version": "v1",
            "max_depth": 2,
            "gates_enabled": True,
            "expand_rules": [
                {
                    "id": "expand_options",
                    "at_step_id": "plan",
                    "child_kind": "option",
                    "max_children": 3,
                    "depth": 1,
                    "source": "planner_output",
                },
                {
                    "id": "expand_subplans",
                    "at_step_id": "propose",
                    "child_kind": "plan",
                    "max_children": 2,
                    "depth": 2,
                    "source": "agent_output",
                },
            ],
            "branch_pipeline": _PIPELINE_FULL,
            "refine_loop": [],
            "prune": {"rule": "epistemic_then_score", "keep_k": 2},
            "reduce": {"rule": "merge_artifacts"},
            "convergence": {"max_rounds": 0},
        },
    },
}

_PRESET_KEYS = list(PROTOCOL_PRESETS.keys())
_PRESET_LABELS = [PROTOCOL_PRESETS[k]["label"] for k in _PRESET_KEYS]


# ---------------------------------------------------------------------------
# Page renderer
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the new run configuration page."""
    st.title("New Deliberation")

    # Question input
    st.subheader("Question")
    question = st.text_area(
        "What should be deliberated?",
        placeholder=(
            "e.g., Our 50-person B2B SaaS startup currently uses a Django "
            "monolith. Should we migrate to microservices?"
        ),
        height=100,
        key="question_input",
    )

    st.divider()

    # Configuration columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("LLM Agents")

        use_all_llm = st.toggle(
            "Enable all LLM agents (recommended)",
            value=True,
            key="use_all_llm",
            help=(
                "Uses Gemini for all agent roles: planner, proposer, "
                "researcher, red-teamer, refiner, validator"
            ),
        )

        if not use_all_llm:
            st.caption("Select individual agents:")
            use_planner = st.checkbox("LLM Planner", key="use_planner")
            use_proposer = st.checkbox("LLM Proposer", key="use_proposer")
            use_researcher = st.checkbox("LLM Researcher", key="use_researcher")
            use_redteam = st.checkbox("LLM Red-Teamer", key="use_redteam")
            use_refiner = st.checkbox("LLM Refiner", key="use_refiner")
            use_validator = st.checkbox("LLM Validator", key="use_validator")
        else:
            use_planner = use_proposer = use_researcher = False
            use_redteam = use_refiner = use_validator = False

        st.divider()

        st.subheader("Evidence Retrieval")
        retrieval_method = st.selectbox(
            "Retrieval method",
            ["web", "hybrid", "keyword", "embedding"],
            index=0,
            key="retrieval_method",
            help="'web' uses Gemini Google Search grounding for real-time evidence",
        )

        verify = st.toggle(
            "Verify web results",
            value=False,
            key="verify",
            help="Fetch URLs to validate web search excerpts",
        )

    with col_right:
        st.subheader("Execution")

        max_parallel = st.slider(
            "Parallel branches",
            min_value=1,
            max_value=5,
            value=3,
            key="max_parallel",
            help="Number of branches to process concurrently",
        )

        st.divider()

        st.subheader("LLM Parameters")
        llm_model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"],
            index=0,
            key="llm_model",
        )

        llm_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.1,
            key="llm_temperature",
        )

        llm_max_tokens = st.number_input(
            "Max output tokens",
            min_value=200,
            max_value=4000,
            value=800,
            step=100,
            key="llm_max_tokens",
        )

        st.divider()

        st.subheader("Gates")
        gates_mode = st.selectbox(
            "Gate handling",
            ["Disabled (fastest)", "Auto-approve", "Interactive (CLI)"],
            index=0,
            key="gates_mode",
            help="Gates are human-in-the-loop checkpoints",
        )

    st.divider()

    # ------------------------------------------------------------------
    # Protocol section
    # ------------------------------------------------------------------
    st.subheader("Protocol")

    protocol_mode = st.radio(
        "Protocol mode",
        [*_PRESET_LABELS, "Custom"],
        horizontal=True,
        key="protocol_mode",
    )

    if protocol_mode == "Custom":
        spec_dict = _render_custom_builder()
    else:
        preset_key = _PRESET_KEYS[_PRESET_LABELS.index(protocol_mode)]
        preset = PROTOCOL_PRESETS[preset_key]
        st.caption(preset["description"])
        spec_dict = dict(preset["spec"])  # Copy so we can override reduce

    # Reduce rule selector (applies to all modes)
    _reduce_options = [
        "merge_artifacts",
        "promote_single",
        "ranked_alternatives",
        "llm_synthesize",
    ]
    _reduce_help = {
        "merge_artifacts": "Concatenate all survivors' recommendations.",
        "promote_single": "Pick the single highest-scoring winner.",
        "ranked_alternatives": "Present top option + ranked alternatives.",
        "llm_synthesize": "LLM synthesizes a coherent recommendation.",
    }
    current_rule = spec_dict.get("reduce", {}).get("rule", "merge_artifacts")
    current_idx = _reduce_options.index(current_rule) if current_rule in _reduce_options else 0
    selected_reduce = st.selectbox(
        "Reduce strategy",
        _reduce_options,
        index=current_idx,
        key="reduce_strategy",
        help="How to combine surviving options into a final recommendation.",
    )
    st.caption(_reduce_help.get(selected_reduce, ""))
    spec_dict["reduce"] = {"rule": selected_reduce}

    # Validate and show feedback
    _render_protocol_validation(spec_dict)

    # YAML preview
    with st.expander("Protocol YAML Preview"):
        st.code(
            yaml.dump(spec_dict, default_flow_style=False, sort_keys=False),
            language="yaml",
        )

    st.divider()

    # API key input
    st.subheader("API Key")
    has_server_key = bool(os.environ.get("GEMINI_API_KEY", ""))
    if has_server_key:
        st.success("Server-side Gemini API key is configured.")
        api_key = ""  # use env var directly; don't expose to frontend
    else:
        api_key = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            key="gemini_api_key",
            help="Required for LLM agents.",
        )
    needs_llm = use_all_llm or any(
        [
            use_planner,
            use_proposer,
            use_researcher,
            use_redteam,
            use_refiner,
            use_validator,
        ]
    )
    if needs_llm and not api_key and not has_server_key:
        st.warning("A Gemini API key is required to run LLM agents.")

    # Build command preview
    cmd_parts = _build_command(
        question=question,
        use_all_llm=use_all_llm,
        use_planner=use_planner,
        use_proposer=use_proposer,
        use_researcher=use_researcher,
        use_redteam=use_redteam,
        use_refiner=use_refiner,
        use_validator=use_validator,
        retrieval_method=retrieval_method,
        verify=verify,
        max_parallel=max_parallel,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        llm_max_tokens=llm_max_tokens,
        gates_mode=gates_mode,
    )

    st.subheader("Command Preview")
    cmd_str = " \\\n  ".join(cmd_parts)
    st.code(cmd_str, language="bash")

    # Run button
    can_run = bool(question.strip()) and (not needs_llm or bool(api_key) or has_server_key)
    col_run, col_status = st.columns([1, 3])
    with col_run:
        run_clicked = st.button(
            "Run Deliberation",
            type="primary",
            disabled=not can_run,
            key="run_button",
        )

    if run_clicked and can_run:
        _execute_run(cmd_parts, col_status, api_key=api_key, protocol_spec=spec_dict)


# ---------------------------------------------------------------------------
# Custom protocol builder
# ---------------------------------------------------------------------------


def _render_custom_builder() -> dict[str, Any]:
    """Render the custom protocol builder and return a spec dict."""
    col_tree, col_pipe = st.columns(2)

    with col_tree:
        st.markdown("**Tree Structure**")
        max_depth = st.selectbox(
            "Max depth",
            [1, 2],
            index=0,
            key="cp_max_depth",
            help="1 = single-level options, 2 = options with sub-plans",
        )

        l1_children = st.slider(
            "Options to generate",
            min_value=2,
            max_value=5,
            value=3,
            key="cp_l1_children",
        )

        l2_children = 2
        if max_depth >= 2:
            l2_children = st.slider(
                "Sub-plans per option",
                min_value=2,
                max_value=4,
                value=2,
                key="cp_l2_children",
            )

    with col_pipe:
        st.markdown("**Pipeline & Pruning**")
        st.checkbox("Propose", value=True, disabled=True, key="cp_propose")
        step_research = st.checkbox("Research", value=True, key="cp_research")
        step_validate = st.checkbox("Validate (claim check)", value=True, key="cp_validate")
        step_redteam = st.checkbox("Red-team", value=True, key="cp_redteam")

        prune_rule = st.selectbox(
            "Prune rule",
            ["epistemic_then_score", "score_only"],
            index=0,
            key="cp_prune_rule",
        )
        keep_k = st.number_input(
            "Keep top K",
            min_value=1,
            max_value=5,
            value=2,
            key="cp_keep_k",
        )
        reduce_rule = st.selectbox(
            "Reduce rule",
            [
                "merge_artifacts",
                "promote_single",
                "ranked_alternatives",
                "llm_synthesize",
            ],
            index=0,
            key="cp_reduce_rule",
        )
        _reduce_captions = {
            "merge_artifacts": "Concatenate all survivors' recommendations.",
            "promote_single": "Pick the single highest-scoring winner.",
            "ranked_alternatives": "Present top option + ranked alternatives.",
            "llm_synthesize": "LLM synthesizes a coherent recommendation.",
        }
        st.caption(_reduce_captions.get(reduce_rule, ""))

    # Refinement section
    st.markdown("**Refinement**")
    enable_refine = st.toggle(
        "Enable refinement loop",
        value=False,
        key="cp_enable_refine",
    )

    max_rounds = 0
    weak_threshold = 5
    score_epsilon = 0.01
    if enable_refine:
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            max_rounds = st.slider(
                "Max rounds",
                min_value=1,
                max_value=5,
                value=3,
                key="cp_max_rounds",
            )
        with rc2:
            weak_threshold = st.number_input(
                "Weak claim threshold",
                min_value=0,
                max_value=20,
                value=5,
                key="cp_weak_threshold",
            )
        with rc3:
            score_epsilon = st.number_input(
                "Score epsilon",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.005,
                format="%.3f",
                key="cp_score_epsilon",
            )

    return _build_custom_protocol_dict(
        max_depth=max_depth,
        l1_children=l1_children,
        l2_children=l2_children,
        steps={
            "propose": True,
            "research": step_research,
            "validate": step_validate,
            "redteam": step_redteam,
        },
        prune_rule=prune_rule,
        keep_k=keep_k,
        reduce_rule=reduce_rule,
        enable_refine=enable_refine,
        max_rounds=max_rounds,
        weak_threshold=weak_threshold,
        score_epsilon=score_epsilon,
    )


def _build_custom_protocol_dict(
    *,
    max_depth: int,
    l1_children: int,
    l2_children: int,
    steps: dict[str, bool],
    prune_rule: str,
    keep_k: int,
    reduce_rule: str,
    enable_refine: bool,
    max_rounds: int,
    weak_threshold: int,
    score_epsilon: float,
) -> dict[str, Any]:
    """Build a protocol spec dict from custom builder widget values."""
    expand_rules: list[dict[str, Any]] = [
        {
            "id": "expand_options",
            "at_step_id": "plan",
            "child_kind": "option",
            "max_children": l1_children,
            "depth": 1,
            "source": "planner_output",
        },
    ]
    if max_depth >= 2:
        expand_rules.append(
            {
                "id": "expand_subplans",
                "at_step_id": "propose",
                "child_kind": "plan",
                "max_children": l2_children,
                "depth": 2,
                "source": "agent_output",
            }
        )

    pipeline: list[dict[str, Any]] = []
    if steps.get("propose", True):
        pipeline.append(
            {
                "id": "propose",
                "kind": "work",
                "step_name": "PROPOSE",
                "role": "proposer",
            }
        )
    if steps.get("research", True):
        pipeline.append(
            {
                "id": "research",
                "kind": "work",
                "step_name": "RESEARCH",
                "role": "researcher",
            }
        )
    if steps.get("validate", True):
        pipeline.append(
            {
                "id": "validate",
                "kind": "validate",
                "step_name": "CLAIM_CHECK",
            }
        )
    if steps.get("redteam", True):
        pipeline.append(
            {
                "id": "redteam",
                "kind": "work",
                "step_name": "REDTEAM",
                "role": "redteam",
            }
        )

    refine_loop: list[dict[str, Any]] = []
    convergence: dict[str, Any] = {"max_rounds": 0}
    if enable_refine:
        refine_loop = [
            {
                "id": "refine",
                "kind": "work",
                "step_name": "REFINE",
                "role": "refiner",
            },
            {
                "id": "revalidate",
                "kind": "validate",
                "step_name": "CLAIM_CHECK",
            },
        ]
        convergence = {
            "max_rounds": max_rounds,
            "weak_threshold": weak_threshold,
            "score_epsilon": score_epsilon,
        }

    return {
        "name": "custom_protocol",
        "protocol_version": "v1",
        "max_depth": max_depth,
        "gates_enabled": True,
        "expand_rules": expand_rules,
        "branch_pipeline": pipeline,
        "refine_loop": refine_loop,
        "prune": {"rule": prune_rule, "keep_k": keep_k},
        "reduce": {"rule": reduce_rule},
        "convergence": convergence,
    }


# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------


def _render_protocol_validation(spec_dict: dict[str, Any]) -> None:
    """Validate the protocol spec and show errors/warnings."""
    try:
        spec = protocol_from_dict(spec_dict)
        warnings = warnings_for_protocol(spec)
        for w in warnings:
            st.warning(f"Protocol warning: {w}")
    except ProtocolLoadError as e:
        st.error(f"Protocol error: {e}")


# ---------------------------------------------------------------------------
# YAML save helper
# ---------------------------------------------------------------------------


def _save_protocol_yaml(spec_dict: dict[str, Any]) -> str:
    """Save a protocol dict to a YAML file and return the path."""
    protocols_dir = Path("protocols")
    protocols_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"custom_{timestamp}.yaml"
    path = protocols_dir / filename
    with path.open("w") as f:
        yaml.dump(spec_dict, f, default_flow_style=False, sort_keys=False)
    return str(path)


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def _build_command(
    question: str,
    use_all_llm: bool,
    use_planner: bool,
    use_proposer: bool,
    use_researcher: bool,
    use_redteam: bool,
    use_refiner: bool,
    use_validator: bool,
    retrieval_method: str,
    verify: bool,
    max_parallel: int,
    llm_model: str,
    llm_temperature: float,
    llm_max_tokens: int,
    gates_mode: str,
) -> list[str]:
    """Build the CLI command parts."""
    parts = ["delibera run"]
    parts.append(f'--question "{question}"')

    if use_all_llm:
        parts.append("--use-all-llm")
    else:
        if use_planner:
            parts.append("--use-llm-planner")
        if use_proposer:
            parts.append("--use-llm-proposer")
        if use_researcher:
            parts.append("--use-llm-researcher")
        if use_redteam:
            parts.append("--use-llm-redteam")
        if use_refiner:
            parts.append("--use-llm-refiner")
        if use_validator:
            parts.append("--use-llm-validator")

    if retrieval_method != "keyword":
        parts.append(f"--retrieval-method {retrieval_method}")

    if verify:
        parts.append("--verify")

    if max_parallel > 1:
        parts.append(f"--max-parallel-branches {max_parallel}")

    parts.append(f"--llm-model {llm_model}")
    parts.append(f"--llm-temperature {llm_temperature}")
    parts.append(f"--llm-max-output-tokens {llm_max_tokens}")

    if gates_mode == "Disabled (fastest)":
        parts.append("--no-gates")
    elif gates_mode == "Auto-approve":
        parts.append("--auto-approve-gates")

    # Protocol path is added at execution time
    parts.append("--protocol <protocol.yaml>")

    return parts


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------


def _execute_run(
    cmd_parts: list[str],
    status_col: Any,
    *,
    api_key: str = "",
    protocol_spec: dict[str, Any] | None = None,
) -> None:
    """Execute the deliberation run."""
    # Save protocol to YAML and update the command
    if protocol_spec is not None:
        protocol_path = _save_protocol_yaml(protocol_spec)
        cmd_parts = [
            p if p != "--protocol <protocol.yaml>" else f"--protocol {protocol_path}"
            for p in cmd_parts
        ]

    full_cmd = " ".join(cmd_parts)
    actual_cmd = full_cmd.replace("delibera run", f"{sys.executable} -m delibera.cli run", 1)

    # Pass API key via environment
    env = dict(os.environ)
    if api_key:
        env["GEMINI_API_KEY"] = api_key

    with status_col, st.status("Running deliberation...", expanded=True) as status:
        st.write("Launching deliberation engine...")
        st.code(full_cmd, language="bash")

        try:
            result = subprocess.run(
                actual_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path.cwd()),
                env=env,
            )

            if result.returncode == 0:
                status.update(label="Deliberation complete!", state="complete")
                st.success("Run completed successfully!")
                if result.stdout:
                    st.text(result.stdout)
                st.info("Switch to a results tab and refresh the run selector to see the new run.")
            else:
                status.update(label="Deliberation failed", state="error")
                st.error("Run failed!")
                if result.stderr:
                    st.code(result.stderr, language="text")
                if result.stdout:
                    st.text(result.stdout)

        except subprocess.TimeoutExpired:
            status.update(label="Timed out", state="error")
            st.error("Run timed out after 10 minutes.")
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Failed to start: {e}")
