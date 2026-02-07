"""Deliberation engine orchestrator.

The engine is the central control loop that owns the deliberation tree,
applies operators, enforces protocols, and determines convergence.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from delibera.agents.stub import (
    PlannerStub,
    ProposerStub,
    RedTeamStub,
    RefinerStub,
    ResearcherStub,
)

if TYPE_CHECKING:
    from delibera.llm.base import LLMClient
    from delibera.retrieval.base import EvidenceRetriever, EvidenceVerifier
from delibera.engine import operators
from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree
from delibera.epistemics.models import (
    ClaimStatus,
    Evidence,
    Objection,
    ObjectionSeverity,
    ObjectionStatus,
)
from delibera.epistemics.validate import ClaimCheckReport
from delibera.gates import (
    GATE_ALLOWED_ACTIONS,
    AutoApproveGateHandler,
    GateAborted,
    GateHandler,
    GateSummary,
    GateType,
    TradeoffCandidate,
    apply_final_signoff_response,
    apply_scope_response,
    apply_tradeoff_response,
    get_response_changes,
    needs_final_signoff_gate,
    needs_scope_gate,
    needs_tradeoff_gate,
)
from delibera.gates.predicates import DEFAULT_TIE_THRESHOLD
from delibera.protocol import DEFAULT_PROTOCOL, ProtocolInterpreter, ProtocolSpec
from delibera.scoring import ScoreWeights, create_default_weights, score_node
from delibera.tools import (
    PolicyEngine,
    ToolCallback,
    ToolRegistry,
    ToolRouter,
    create_default_policy_engine,
    create_default_registry,
)
from delibera.trace.events import TraceEvent
from delibera.trace.writer import TraceWriter


class Engine:
    """The deliberation engine orchestrator.

    Implements protocol-driven deliberation with deterministic stub agents.
    The protocol specifies WHAT steps run; the engine applies operators.

    Default protocol (simple_protocol):
    1. PLAN - Generate branch labels
    2. SCOPE GATE - User approves/vetoes branches (if gates enabled)
    3. EXPAND - Create child nodes
    4. PROPOSE - Generate proposals per branch (with optional tool use)
    5. VALIDATE - Extract and validate claims
    6. PRUNE - Keep top-k by epistemic quality
    7. REDUCE - Merge survivors
    8. FINAL SIGN-OFF GATE - User approves final output (if gates enabled)
    9. FINALIZE - Write artifact with claim_check_summary
    """

    def __init__(
        self,
        runs_dir: Path | None = None,
        tool_registry: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
        gate_handler: GateHandler | None = None,
        gates_enabled: bool = True,
        protocol: ProtocolSpec | None = None,
        protocol_source: str | None = None,
        tie_threshold: float = DEFAULT_TIE_THRESHOLD,
        initial_weights: ScoreWeights | None = None,
        evidence_root: Path | None = None,
        llm_client: LLMClient | None = None,
        use_llm_proposer: bool = False,
        use_llm_planner: bool = False,
        use_llm_researcher: bool = False,
        use_llm_redteam: bool = False,
        use_llm_refiner: bool = False,
        use_llm_validator: bool = False,
        llm_model: str | None = None,
        llm_temperature: float = 0.2,
        llm_max_output_tokens: int = 800,
        retriever: EvidenceRetriever | None = None,
        verifier: EvidenceVerifier | None = None,
        max_parallel_branches: int = 1,
    ) -> None:
        """Initialize the engine.

        Args:
            runs_dir: Directory for run outputs. Defaults to ./runs.
            tool_registry: Tool registry for available tools.
            policy_engine: Policy engine for tool access control.
            gate_handler: Handler for user gates. Defaults to AutoApproveGateHandler.
            gates_enabled: Whether gates are enabled. Defaults to True.
            protocol: Protocol specification to use. Defaults to DEFAULT_PROTOCOL.
            protocol_source: Source of protocol ("builtin" or "yaml:<path>").
            tie_threshold: Score difference below which tradeoff gate triggers.
            initial_weights: Initial scoring weights. If provided, tradeoff gate skipped.
            evidence_root: Root directory for evidence files. Defaults to ./evidence.
            llm_client: Optional LLM client for LLM-backed agents.
            use_llm_proposer: Whether to use LLM-backed proposer. Defaults to False.
            use_llm_planner: Whether to use LLM-backed planner. Defaults to False.
            use_llm_researcher: Whether to use LLM-backed researcher. Defaults to False.
            use_llm_redteam: Whether to use LLM-backed red-teamer. Defaults to False.
            use_llm_refiner: Whether to use LLM-backed refiner. Defaults to False.
            use_llm_validator: Whether to use LLM-backed claim validator. Defaults to False.
            llm_model: Model name for LLM (if different from client default).
            llm_temperature: Temperature for LLM calls. Defaults to 0.2.
            llm_max_output_tokens: Max output tokens for LLM. Defaults to 800.
            retriever: Optional evidence retriever for RESEARCH step.
            verifier: Optional evidence verifier for validating web results.
            max_parallel_branches: Max concurrent branches (1=sequential). Defaults to 1.
        """
        self.runs_dir = runs_dir or Path("runs")
        self._tool_registry = tool_registry or create_default_registry(evidence_root=evidence_root)
        self._policy_engine = policy_engine or create_default_policy_engine()
        self._gate_handler = gate_handler or AutoApproveGateHandler()
        self._gates_enabled = gates_enabled
        self._protocol = protocol or DEFAULT_PROTOCOL
        self._protocol_source = protocol_source or "builtin"
        self._tie_threshold = tie_threshold
        self._initial_weights = initial_weights

        # LLM configuration
        self._llm_client = llm_client
        self._use_llm_proposer = use_llm_proposer
        self._use_llm_planner = use_llm_planner
        self._use_llm_researcher = use_llm_researcher
        self._use_llm_redteam = use_llm_redteam
        self._use_llm_refiner = use_llm_refiner
        self._use_llm_validator = use_llm_validator
        self._llm_model = llm_model
        self._llm_temperature = llm_temperature
        self._llm_max_output_tokens = llm_max_output_tokens

        # Parallel execution
        self._max_parallel = max_parallel_branches

        # Evidence retriever and verifier
        self._retriever = retriever
        self._verifier = verifier

        # These are set during run execution
        self._current_run_id: str = ""
        self._current_writer: TraceWriter | None = None
        self._tool_router: ToolRouter | None = None
        self._interpreter: ProtocolInterpreter | None = None

    def run(self, question: str) -> Path:
        """Execute a full deliberation run.

        Args:
            question: The question or problem to deliberate on.

        Returns:
            Path to the run directory containing trace.jsonl and artifact.json.

        Raises:
            ValueError: If question is empty or whitespace-only.
            GateAborted: If user aborts through a gate.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        # Initialize run
        run_id = self._generate_run_id()
        run_dir = self.runs_dir / run_id
        created_at = datetime.now(UTC).isoformat()

        # Initialize tree and state
        tree = DeliberationTree()
        state = RunState(
            run_id=run_id,
            question=question,
            created_at=created_at,
            tree=tree,
        )

        # Initialize trace writer (may raise on permission errors)
        writer: TraceWriter | None = None
        try:
            writer = TraceWriter(run_dir)

            # Set run context for tool routing
            self._current_run_id = run_id
            self._current_writer = writer
            self._tool_router = ToolRouter(
                registry=self._tool_registry,
                policy_engine=self._policy_engine,
                trace_emitter=self._emit_trace_event,
            )
            self._interpreter = ProtocolInterpreter(self._protocol)

            # Emit run_start
            writer.emit(
                TraceEvent(
                    event_type="run_start",
                    run_id=run_id,
                    payload={
                        "question": question,
                        "created_at": created_at,
                        "protocol_name": self._protocol.name,
                        "protocol_version": self._protocol.protocol_version,
                        "protocol_source": self._protocol_source,
                    },
                )
            )

            # Create root node
            root = tree.create_root(question)
            writer.emit(
                TraceEvent(
                    event_type="node_created",
                    run_id=run_id,
                    payload={
                        "node_id": root.node_id,
                        "kind": root.kind,
                        "depth": root.depth,
                    },
                )
            )

            # PLAN: Generate branch labels
            plan_output = self._execute_plan(root, question, run_id, writer)

            # SCOPE GATE: Get user approval for branches
            branch_labels = plan_output["branches"]
            if needs_scope_gate(plan_output, self._gates_enabled):
                branch_labels = self._handle_scope_gate(
                    question=question,
                    branch_labels=branch_labels,
                    node_id=root.node_id,
                )

            # PROCESS TREE: Expand, run pipeline, score, prune, reduce (recursive)
            current_weights = self._initial_weights or create_default_weights()
            merged_node, current_weights = self._process_level(
                tree=tree,
                parent_id=root.node_id,
                labels=branch_labels,
                child_kind="option",
                question=question,
                run_id=run_id,
                writer=writer,
                current_weights=current_weights,
                depth=1,
            )

            # REFINE LOOP: Execute bounded refinement if protocol defines it
            refinement_metadata = self._execute_refine_loop(
                merged_node=merged_node,
                tree=tree,
                run_id=run_id,
                writer=writer,
                current_weights=current_weights,
                question=question,
            )

            # Build artifact for sign-off
            artifact = operators.finalize(state, merged_node)

            # Add refinement metadata to artifact if refinement occurred
            if refinement_metadata["total_rounds"] > 0:
                artifact["refinement"] = refinement_metadata

            # FINAL SIGN-OFF GATE: Get user approval for final output
            if needs_final_signoff_gate(artifact, self._gates_enabled):
                artifact = self._handle_final_signoff_gate(
                    artifact=artifact,
                    node_id=merged_node.node_id,
                    merged_node=merged_node,
                    tree=tree,
                )
            else:
                # Add objections summary even if gate not triggered
                artifact["objections"] = self._build_objections_summary(merged_node)

            # FINALIZE: Write artifact
            writer.write_artifact(artifact)

            writer.emit(
                TraceEvent(
                    event_type="final_artifact_written",
                    run_id=run_id,
                    payload={"artifact_path": str(writer.artifact_path)},
                )
            )

            # Emit run_end
            writer.emit(
                TraceEvent(
                    event_type="run_end",
                    run_id=run_id,
                    payload={"status": "completed"},
                )
            )

            return run_dir

        except GateAborted as e:
            # Emit aborted status
            if writer is not None:
                writer.emit(
                    TraceEvent(
                        event_type="run_end",
                        run_id=run_id,
                        payload={
                            "status": "aborted",
                            "gate_type": e.gate_type.value,
                            "message": e.message,
                        },
                    )
                )
            raise

        except Exception as e:
            # Emit run_end with failed status on any error (only if writer initialized)
            if writer is not None:
                writer.emit(
                    TraceEvent(
                        event_type="run_end",
                        run_id=run_id,
                        payload={"status": "failed", "error": str(e)},
                    )
                )
            raise

        finally:
            # Clean up run context
            self._current_run_id = ""
            self._current_writer = None
            self._tool_router = None
            self._interpreter = None

    def _handle_scope_gate(
        self,
        question: str,
        branch_labels: list[str],
        node_id: str,
    ) -> list[str]:
        """Handle the scope gate after PLAN.

        Args:
            question: The original question.
            branch_labels: Proposed branch labels from planner.
            node_id: The root node ID.

        Returns:
            Updated branch labels after user response.

        Raises:
            GateAborted: If user chooses to abort.
        """
        # Build summary
        summary = GateSummary(
            gate_type=GateType.SCOPE,
            allowed_actions=GATE_ALLOWED_ACTIONS[GateType.SCOPE],
            interpreted_question=question,
            proposed_branches=branch_labels,
        )

        # Emit gate_triggered
        self._emit_trace_event(
            "gate_triggered",
            {
                "gate_type": GateType.SCOPE.value,
                "summary": summary.to_dict(),
                "node_id": node_id,
            },
        )

        # Get response from handler
        response = self._gate_handler.handle(summary)

        # Apply response
        updated_labels = apply_scope_response(response, branch_labels)

        # Emit gate_response_applied
        changes = get_response_changes(
            response,
            context={"original_branches": branch_labels},
        )
        self._emit_trace_event(
            "gate_response_applied",
            {
                "gate_type": GateType.SCOPE.value,
                "response": response.to_dict(),
                "changes": changes,
                "node_id": node_id,
            },
        )

        return updated_labels

    def _handle_tradeoff_gate(
        self,
        children: list[Any],
        node_scores: dict[str, tuple[float, dict[str, float]]],
        current_weights: ScoreWeights,
        root_node_id: str,
    ) -> ScoreWeights:
        """Handle the tradeoff gate when near-tie detected.

        Args:
            children: List of child nodes.
            node_scores: Map of node_id to (score, metrics).
            current_weights: Current scoring weights.
            root_node_id: The root node ID.

        Returns:
            Updated ScoreWeights after user response.

        Raises:
            GateAborted: If user chooses to abort.
        """
        # Sort by score descending
        sorted_scores = sorted(
            [(nid, score) for nid, (score, _) in node_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # Build top candidates for summary
        top_candidates: list[TradeoffCandidate] = []
        node_map = {c.node_id: c for c in children}
        for nid, score in sorted_scores[:3]:  # Top 3 candidates
            node = node_map.get(nid)
            _, metrics = node_scores[nid]
            top_candidates.append(
                TradeoffCandidate(
                    label=node.label if node else nid,
                    score=score,
                    metrics=metrics,
                )
            )

        # Compute score difference between top 2
        score_difference = None
        if len(sorted_scores) >= 2:
            score_difference = abs(sorted_scores[0][1] - sorted_scores[1][1])

        # Build summary
        summary = GateSummary(
            gate_type=GateType.TRADEOFF,
            allowed_actions=GATE_ALLOWED_ACTIONS[GateType.TRADEOFF],
            top_candidates=top_candidates,
            current_weights=current_weights.to_dict(),
            score_difference=score_difference,
        )

        # Emit gate_triggered
        self._emit_trace_event(
            "gate_triggered",
            {
                "gate_type": GateType.TRADEOFF.value,
                "summary": summary.to_dict(),
                "node_id": root_node_id,
            },
        )

        # Get response from handler
        response = self._gate_handler.handle(summary)

        # Apply response (may raise GateAborted)
        updated_weights = apply_tradeoff_response(response, current_weights)

        # Emit gate_response_applied
        changes = get_response_changes(response)
        if updated_weights != current_weights:
            changes["new_weights"] = updated_weights.to_dict()

        self._emit_trace_event(
            "gate_response_applied",
            {
                "gate_type": GateType.TRADEOFF.value,
                "response": response.to_dict(),
                "changes": changes,
                "node_id": root_node_id,
            },
        )

        return updated_weights

    def _handle_final_signoff_gate(
        self,
        artifact: dict[str, Any],
        node_id: str,
        merged_node: Any,
        tree: DeliberationTree,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Handle the final sign-off gate before writing artifact.

        This method handles the convergence rule: runs cannot finalize with
        open blocking objections unless explicitly accepted.

        Auto-approve handlers will abort if blocking objections exist.

        Args:
            artifact: The final artifact to approve.
            node_id: The merged node ID.
            merged_node: The merged node (to access/update ledger).
            tree: The deliberation tree.

        Returns:
            Updated artifact with objections summary.

        Raises:
            GateAborted: If user chooses to abort or cannot proceed.
        """
        # Loop until approved or aborted (allows accept_objections -> approve)
        while True:
            # Get current open blocking objections from merged ledger
            open_blocking = [
                {"objection_id": obj.objection_id, "rationale": obj.rationale}
                for obj in merged_node.ledger.objections
                if obj.severity == ObjectionSeverity.BLOCKING and obj.status == ObjectionStatus.OPEN
            ]
            has_blocking_objections = len(open_blocking) > 0

            # Auto-approve handler must abort if blocking objections exist
            if has_blocking_objections and isinstance(self._gate_handler, AutoApproveGateHandler):
                raise GateAborted(
                    GateType.FINAL_SIGNOFF,
                    "Blocking objections require explicit acceptance; "
                    "auto-approve cannot accept objections.",
                )

            # Build summary
            summary = GateSummary(
                gate_type=GateType.FINAL_SIGNOFF,
                allowed_actions=GATE_ALLOWED_ACTIONS[GateType.FINAL_SIGNOFF],
                recommendation=artifact.get("recommendation", ""),
                claim_check_summary=artifact.get("claim_check_summary"),
                open_questions=artifact.get("open_questions", []),
                open_blocking_objections=open_blocking,
            )

            # Emit gate_triggered
            self._emit_trace_event(
                "gate_triggered",
                {
                    "gate_type": GateType.FINAL_SIGNOFF.value,
                    "summary": summary.to_dict(),
                    "node_id": node_id,
                },
            )

            # Get response from handler
            response = self._gate_handler.handle(summary)

            # Apply response (may raise GateAborted)
            approved, accepted_ids = apply_final_signoff_response(response, has_blocking_objections)

            # Emit gate_response_applied
            changes = get_response_changes(response)
            self._emit_trace_event(
                "gate_response_applied",
                {
                    "gate_type": GateType.FINAL_SIGNOFF.value,
                    "response": response.to_dict(),
                    "changes": changes,
                    "node_id": node_id,
                },
            )

            # Handle accept_objections
            if accepted_ids:
                for obj in merged_node.ledger.objections:
                    if obj.objection_id in accepted_ids:
                        obj.status = ObjectionStatus.ACCEPTED
                        # Emit status change event
                        self._emit_trace_event(
                            "objection_status_changed",
                            {
                                "node_id": node_id,
                                "objection_id": obj.objection_id,
                                "new_status": "accepted",
                            },
                        )
                # Loop again to allow approve after accepting
                continue

            if approved:
                # Update artifact with objections summary
                artifact["objections"] = self._build_objections_summary(merged_node)
                return artifact

    def _build_objections_summary(self, merged_node: Any) -> dict[str, list[dict[str, str]]]:
        """Build objections summary for artifact.json.

        Args:
            merged_node: The merged node with ledger.

        Returns:
            Dictionary with blocking_open, blocking_accepted, nonblocking_open lists.
        """
        blocking_open: list[dict[str, str]] = []
        blocking_accepted: list[dict[str, str]] = []
        nonblocking_open: list[dict[str, str]] = []

        for obj in merged_node.ledger.objections:
            obj_summary = {
                "objection_id": obj.objection_id,
                "target": obj.target,
                "rationale": obj.rationale,
            }

            if obj.severity == ObjectionSeverity.BLOCKING:
                if obj.status == ObjectionStatus.OPEN:
                    blocking_open.append(obj_summary)
                elif obj.status == ObjectionStatus.ACCEPTED:
                    blocking_accepted.append(obj_summary)
            elif (
                obj.severity == ObjectionSeverity.NONBLOCKING and obj.status == ObjectionStatus.OPEN
            ):
                nonblocking_open.append(obj_summary)

        return {
            "blocking_open": blocking_open,
            "blocking_accepted": blocking_accepted,
            "nonblocking_open": nonblocking_open,
        }

    def _generate_run_id(self) -> str:
        """Generate a unique, filesystem-safe run ID."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"{timestamp}_{short_uuid}"

    def _emit_trace_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit a trace event through the writer.

        This is used as a callback for the ToolRouter.
        """
        if self._current_writer is None:
            return
        self._current_writer.emit(
            TraceEvent(
                event_type=event_type,
                run_id=self._current_run_id,
                payload=payload,
            )
        )

    def call_tool(
        self,
        node_id: str,
        role: str,
        step: str,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool call through the router.

        This is the only way agents should access tools.

        Args:
            node_id: The node ID context for the tool call.
            role: The agent role making the request.
            step: The current protocol step.
            tool_name: The name of the tool to call.
            tool_input: The input to pass to the tool.

        Returns:
            The tool output dictionary.

        Raises:
            ToolDenied: If the policy denies the tool call.
            KeyError: If the tool is not registered.
            ToolExecutionError: If the tool execution fails.
            RuntimeError: If called outside of a run context.
        """
        if self._tool_router is None:
            raise RuntimeError("call_tool can only be called during a run")
        return self._tool_router.call(
            role=role,
            step=step,
            tool_name=tool_name,
            tool_input=tool_input,
            node_id=node_id,
        )

    def make_tool_callback(
        self,
        node_id: str,
        role: str,
        step: str,
    ) -> ToolCallback:
        """Create a tool callback for an agent.

        This creates a bound callback that agents can use to request
        tool execution without direct access to the router.

        Args:
            node_id: The node ID context for tool calls.
            role: The agent role.
            step: The current protocol step.

        Returns:
            A callback function: (tool_name, tool_input) -> output
        """

        def callback(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
            return self.call_tool(node_id, role, step, tool_name, tool_input)

        return callback

    def _research_fallback(
        self,
        question: str,
        child: Any,
        tool_callback: Any,
    ) -> dict[str, Any]:
        """Fall back to retriever or stub for research when LLM researcher fails.

        Args:
            question: The deliberation question.
            child: The child node being researched.
            tool_callback: Tool callback for the research step.

        Returns:
            Research output dict.
        """
        if self._retriever is not None:
            query = f"{question} {child.label}"
            try:
                results = self._retriever.retrieve(query, max_results=5)
                return {
                    "evidence": [{"source": r.source, "excerpt": r.excerpt} for r in results],
                    "notes": [
                        f"Found {len(results)} results via retriever (LLM fallback)",
                    ],
                }
            except Exception:
                pass
        researcher = ResearcherStub()
        return researcher.execute(
            {"label": child.label, "question": question},
            tool=tool_callback,
        )

    def _validate_claims(
        self,
        tree: DeliberationTree,
        node_id: str,
        owner: str,
        run_id: str,
        writer: TraceWriter,
    ) -> ClaimCheckReport:
        """Extract and validate claims, optionally using LLM validator.

        When use_llm_validator is enabled, uses the LLM to semantically
        assess claim-evidence support instead of keyword matching.
        Falls back to heuristic validation on LLM error.

        Args:
            tree: The deliberation tree.
            node_id: The node to validate.
            owner: The role that produced the artifact.
            run_id: The current run ID.
            writer: The trace writer.

        Returns:
            ClaimCheckReport with validation results.
        """
        if not (self._use_llm_validator and self._llm_client is not None):
            return operators.validate(tree, node_id, owner)

        from delibera.agents.llm_validator import ClaimValidatorLLM
        from delibera.epistemics.extract import extract_claims

        node = tree.get_node(node_id)

        # Extract claims (same as heuristic path)
        claims = extract_claims(node_id, node.artifact, owner)
        node.ledger.claims = claims

        # Use LLM validator
        validator = ClaimValidatorLLM(
            llm_client=self._llm_client,
            model=self._llm_model,
            temperature=0.1,
            max_output_tokens=1500,
        )

        writer.emit(
            TraceEvent(
                event_type="llm_call_requested",
                run_id=run_id,
                payload={
                    "node_id": node_id,
                    "role": "validator",
                    "step": "CLAIM_CHECK",
                    "provider": "gemini",
                    "model": self._llm_model or "default",
                },
            )
        )

        try:
            report = validator.validate(claims, node.ledger.evidence)

            writer.emit(
                TraceEvent(
                    event_type="llm_call_succeeded",
                    run_id=run_id,
                    payload={
                        "node_id": node_id,
                        "role": "validator",
                        "step": "CLAIM_CHECK",
                        "output_length": len(str(report.details)),
                        "llm_generated": True,
                    },
                )
            )

            # Record support relations in ledger
            for claim_id, evidence_ids in report.support_relations.items():
                for evidence_id in evidence_ids:
                    node.ledger.link_support(claim_id, evidence_id)

            return report

        except Exception as e:
            from delibera.llm.redaction import redact_text

            writer.emit(
                TraceEvent(
                    event_type="llm_call_failed",
                    run_id=run_id,
                    payload={
                        "node_id": node_id,
                        "role": "validator",
                        "step": "CLAIM_CHECK",
                        "error_type": type(e).__name__,
                        "error_message": redact_text(str(e))[:200],
                    },
                )
            )
            # Fall back to heuristic validation
            return operators.validate(tree, node_id, owner)

    def _merge_evidence_to_ledger(
        self,
        tree: DeliberationTree,
        node_id: str,
        evidence_items: list[dict[str, Any]],
        run_id: str,
        writer: TraceWriter,
    ) -> None:
        """Merge evidence items from agent output into node ledger.

        This is an engine-controlled state update. Agents return evidence
        in their output, and the engine merges it into the ledger.

        Args:
            tree: The deliberation tree.
            node_id: The node to add evidence to.
            evidence_items: List of evidence dicts with source and excerpt.
            run_id: The current run ID.
            writer: The trace writer.
        """
        node = tree.get_node(node_id)

        for i, item in enumerate(evidence_items):
            # Generate deterministic evidence ID
            evidence_id = f"ev_{node_id[:4]}_{i}"

            evidence = Evidence(
                evidence_id=evidence_id,
                source=item.get("source", ""),
                excerpt=item.get("excerpt", ""),
                provenance={
                    "node_id": node_id,
                    "step": "RESEARCH",
                    "role": "researcher",
                },
            )

            # Add to ledger
            node.ledger.evidence.append(evidence)

            # Emit evidence_added trace event
            writer.emit(
                TraceEvent(
                    event_type="evidence_added",
                    run_id=run_id,
                    payload={
                        "node_id": node_id,
                        "evidence": evidence.to_dict(),
                    },
                )
            )

    def _merge_objections_to_ledger(
        self,
        tree: DeliberationTree,
        node_id: str,
        objection_items: list[dict[str, Any]],
        run_id: str,
        writer: TraceWriter,
    ) -> None:
        """Merge objections from agent output into node ledger.

        This is an engine-controlled state update. Agents return objections
        in their output, and the engine merges them into the ledger.

        Args:
            tree: The deliberation tree.
            node_id: The node to add objections to.
            objection_items: List of objection dicts.
            run_id: The current run ID.
            writer: The trace writer.
        """
        node = tree.get_node(node_id)

        for item in objection_items:
            # Parse severity and status from string values
            severity_str = item.get("severity", "nonblocking")
            status_str = item.get("status", "open")

            severity = (
                ObjectionSeverity.BLOCKING
                if severity_str == "blocking"
                else ObjectionSeverity.NONBLOCKING
            )
            status = ObjectionStatus(status_str)

            objection = Objection(
                objection_id=item.get("objection_id", ""),
                target=item.get("target", "artifact"),
                severity=severity,
                status=status,
                rationale=item.get("rationale", ""),
                owner="redteam",
            )

            # Add to ledger (uses add_objection to avoid duplicates)
            node.ledger.add_objection(objection)

            # Emit objection_added trace event
            writer.emit(
                TraceEvent(
                    event_type="objection_added",
                    run_id=run_id,
                    payload={
                        "node_id": node_id,
                        "objection": objection.to_dict(),
                        "source_step": "REDTEAM",
                    },
                )
            )

    def _execute_refine_loop(
        self,
        merged_node: Any,
        tree: DeliberationTree,
        run_id: str,
        writer: TraceWriter,
        current_weights: ScoreWeights,
        question: str = "",
    ) -> dict[str, Any]:
        """Execute the bounded refinement loop on the merged node.

        Refinement continues until:
        1. max_rounds is reached
        2. All convergence predicates are satisfied:
           - no_blocking_objections: No open blocking objections
           - unsupported_claims == 0
           - weak_claims <= weak_threshold
           - score_improvement < score_epsilon

        Args:
            merged_node: The merged node to refine.
            tree: The deliberation tree.
            run_id: The current run ID.
            writer: The trace writer.
            current_weights: The current scoring weights.
            question: The deliberation question (for LLM refiner context).

        Returns:
            Refinement metadata dict with total_rounds, converged, stop_reason, history.
        """
        assert self._interpreter is not None  # Set during run()

        # Check if refinement is enabled
        if not self._interpreter.has_refine_loop():
            return {
                "total_rounds": 0,
                "converged": False,
                "stop_reason": "refinement_disabled",
                "history": [],
            }

        convergence_spec = self._interpreter.get_convergence_spec()
        max_rounds = convergence_spec.max_rounds
        weak_threshold = convergence_spec.weak_threshold
        score_epsilon = convergence_spec.score_epsilon

        history: list[dict[str, Any]] = []
        previous_score = merged_node.artifact.get("score", 0.0)
        converged = False
        stop_reason = "max_rounds_reached"

        refiner: RefinerStub | Any  # Allow LLM refiner
        if self._use_llm_refiner and self._llm_client is not None:
            from delibera.agents.llm_refiner import RefinerLLM

            refiner = RefinerLLM(
                llm_client=self._llm_client,
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_output_tokens=600,
            )
        else:
            refiner = RefinerStub()

        for round_num in range(1, max_rounds + 1):
            # Emit round started event
            writer.emit(
                TraceEvent(
                    event_type="refinement_round_started",
                    run_id=run_id,
                    payload={
                        "round": round_num,
                        "node_id": merged_node.node_id,
                        "max_rounds": max_rounds,
                    },
                )
            )

            # Check convergence predicates before refining
            converged, stop_reason = self._check_convergence(
                merged_node=merged_node,
                weak_threshold=weak_threshold,
                score_epsilon=score_epsilon,
                previous_score=previous_score,
                current_score=merged_node.artifact.get("score", 0.0),
            )

            if converged:
                # Record early convergence
                history.append(
                    {
                        "round": round_num,
                        "action": "converged_early",
                        "reason": stop_reason,
                    }
                )
                writer.emit(
                    TraceEvent(
                        event_type="refinement_round_completed",
                        run_id=run_id,
                        payload={
                            "round": round_num,
                            "node_id": merged_node.node_id,
                            "converged": True,
                            "stop_reason": stop_reason,
                        },
                    )
                )
                break

            # Build claims summary for refiner
            claims_summary = [
                {
                    "claim_id": c.claim_id,
                    "claim_type": c.claim_type.value,
                    "status": c.status.value,
                    "text": c.text,
                }
                for c in merged_node.ledger.claims
            ]

            # Build objections summary for refiner context
            objections_summary = [
                {
                    "severity": o.severity.value,
                    "rationale": o.rationale,
                    "target": o.target,
                    "status": o.status.value,
                }
                for o in merged_node.ledger.objections
                if o.status.value == "open"
            ]

            refine_context: dict[str, Any] = {
                "node_id": merged_node.node_id,
                "artifact": merged_node.artifact,
                "claims": claims_summary,
                "round": round_num,
                "question": question,
                "objections": objections_summary,
            }

            # Execute refiner with LLM trace events if applicable
            if self._use_llm_refiner and self._llm_client is not None:
                writer.emit(
                    TraceEvent(
                        event_type="llm_call_requested",
                        run_id=run_id,
                        payload={
                            "node_id": merged_node.node_id,
                            "role": "refiner",
                            "step": "REFINE",
                            "round": round_num,
                            "provider": "gemini",
                            "model": self._llm_model or "default",
                        },
                    )
                )

                try:
                    refine_output = refiner.execute(refine_context)

                    writer.emit(
                        TraceEvent(
                            event_type="llm_call_succeeded",
                            run_id=run_id,
                            payload={
                                "node_id": merged_node.node_id,
                                "role": "refiner",
                                "step": "REFINE",
                                "round": round_num,
                                "output_length": len(str(refine_output)),
                                "llm_generated": True,
                            },
                        )
                    )
                except Exception as e:
                    from delibera.llm.redaction import redact_text

                    writer.emit(
                        TraceEvent(
                            event_type="llm_call_failed",
                            run_id=run_id,
                            payload={
                                "node_id": merged_node.node_id,
                                "role": "refiner",
                                "step": "REFINE",
                                "round": round_num,
                                "error_type": type(e).__name__,
                                "error_message": redact_text(str(e))[:200],
                            },
                        )
                    )
                    # Fall back to stub
                    refine_output = RefinerStub().execute(refine_context)
            else:
                refine_output = refiner.execute(refine_context)

            # Update artifact with refined version
            refined_artifact = refine_output.get("artifact", merged_node.artifact)
            tree.update_artifact(merged_node.node_id, refined_artifact)

            # Emit work output for refine step
            writer.emit(
                TraceEvent(
                    event_type="work_output",
                    run_id=run_id,
                    payload={
                        "step": "REFINE",
                        "node_id": merged_node.node_id,
                        "role": "refiner",
                        "round": round_num,
                        "output": refine_output,
                    },
                )
            )

            # Re-validate claims after refinement
            report = self._validate_claims(tree, merged_node.node_id, "refiner", run_id, writer)
            writer.emit(
                TraceEvent(
                    event_type="claim_validation_report",
                    run_id=run_id,
                    payload={
                        "node_id": merged_node.node_id,
                        "round": round_num,
                        "supported": report.supported,
                        "weak": report.weak,
                        "unsupported": report.unsupported,
                        "details": report.details,
                        "support_relations": report.support_relations,
                    },
                )
            )

            # Rescore after refinement
            score_result = score_node(merged_node, current_weights)
            new_score = score_result.score
            score_delta = new_score - previous_score

            # Record round history
            round_record = {
                "round": round_num,
                "weak_addressed": refine_output.get("weak_addressed", 0),
                "unsupported_addressed": refine_output.get("unsupported_addressed", 0),
                "previous_score": previous_score,
                "new_score": new_score,
                "score_delta": score_delta,
                "weak_claims": report.weak,
                "unsupported_claims": report.unsupported,
            }
            history.append(round_record)

            # Emit round completed event
            writer.emit(
                TraceEvent(
                    event_type="refinement_round_completed",
                    run_id=run_id,
                    payload={
                        "round": round_num,
                        "node_id": merged_node.node_id,
                        "converged": False,
                        "score_delta": score_delta,
                        "weak_claims": report.weak,
                        "unsupported_claims": report.unsupported,
                    },
                )
            )

            previous_score = new_score

        # Final convergence check
        if not converged:
            converged, stop_reason = self._check_convergence(
                merged_node=merged_node,
                weak_threshold=weak_threshold,
                score_epsilon=score_epsilon,
                previous_score=previous_score,
                current_score=merged_node.artifact.get("score", 0.0),
            )
            if not converged:
                stop_reason = "max_rounds_reached"

        return {
            "total_rounds": len(history),
            "converged": converged,
            "stop_reason": stop_reason,
            "history": history,
        }

    def _check_convergence(
        self,
        merged_node: Any,
        weak_threshold: int,
        score_epsilon: float,
        previous_score: float,
        current_score: float,
    ) -> tuple[bool, str]:
        """Check if convergence predicates are satisfied.

        Predicates (all must be True):
        1. no_blocking_objections: No open blocking objections
        2. unsupported_claims == 0
        3. weak_claims <= weak_threshold
        4. score_improvement < score_epsilon

        Args:
            merged_node: The node to check.
            weak_threshold: Maximum acceptable weak claims.
            score_epsilon: Minimum score improvement to continue.
            previous_score: Score before last refinement.
            current_score: Score after last refinement.

        Returns:
            Tuple of (converged, reason) where reason explains which predicate failed.
        """
        # Check for open blocking objections
        blocking_open = sum(
            1
            for obj in merged_node.ledger.objections
            if obj.severity == ObjectionSeverity.BLOCKING and obj.status == ObjectionStatus.OPEN
        )
        if blocking_open > 0:
            return False, "blocking_objections_remain"

        # Count claim statuses
        unsupported = sum(
            1 for c in merged_node.ledger.claims if c.status == ClaimStatus.UNSUPPORTED
        )
        weak = sum(1 for c in merged_node.ledger.claims if c.status == ClaimStatus.WEAK)

        if unsupported > 0:
            return False, "unsupported_claims_remain"

        if weak > weak_threshold:
            return False, "weak_claims_exceed_threshold"

        # Check score improvement (if scores are close, we've converged)
        score_delta = abs(current_score - previous_score)
        if score_delta >= score_epsilon:
            return False, "score_still_improving"

        # All predicates satisfied
        return True, "all_predicates_satisfied"

    # ------------------------------------------------------------------
    # Extracted step methods (Phase A refactor)
    # ------------------------------------------------------------------

    def _execute_plan(
        self,
        root: Any,
        question: str,
        run_id: str,
        writer: TraceWriter,
    ) -> dict[str, Any]:
        """Execute the PLAN step: generate branch labels.

        Args:
            root: The root node.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer.

        Returns:
            Plan output dict with "branches" key.
        """
        if self._use_llm_planner and self._llm_client is not None:
            from delibera.agents.llm_planner import PlannerLLM
            from delibera.agents.llm_proposer import check_llm_allowed_in_step

            check_llm_allowed_in_step("work")

            planner_llm = PlannerLLM(
                llm_client=self._llm_client,
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_output_tokens=400,
            )

            writer.emit(
                TraceEvent(
                    event_type="llm_call_requested",
                    run_id=run_id,
                    payload={
                        "node_id": root.node_id,
                        "role": "planner",
                        "step": "PLAN",
                        "provider": "gemini",
                        "model": self._llm_model or "default",
                    },
                )
            )

            try:
                plan_output = planner_llm.execute({"question": question})

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_succeeded",
                        run_id=run_id,
                        payload={
                            "node_id": root.node_id,
                            "role": "planner",
                            "step": "PLAN",
                            "output_length": len(str(plan_output)),
                            "llm_generated": True,
                        },
                    )
                )
            except Exception as e:
                from delibera.llm.redaction import redact_text

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_failed",
                        run_id=run_id,
                        payload={
                            "node_id": root.node_id,
                            "role": "planner",
                            "step": "PLAN",
                            "error_type": type(e).__name__,
                            "error_message": redact_text(str(e))[:200],
                        },
                    )
                )
                plan_output = PlannerStub().execute({"question": question})
        else:
            plan_output = PlannerStub().execute({"question": question})

        writer.emit(
            TraceEvent(
                event_type="work_output",
                run_id=run_id,
                payload={
                    "step": "PLAN",
                    "node_id": root.node_id,
                    "role": "planner",
                    "output": plan_output,
                },
            )
        )

        return plan_output

    def _execute_propose_on_node(
        self,
        child: Any,
        question: str,
        run_id: str,
        writer: TraceWriter,
        tree: DeliberationTree,
    ) -> dict[str, Any]:
        """Execute the PROPOSE step for a single node.

        Args:
            child: The child node to generate a proposal for.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer.
            tree: The deliberation tree.

        Returns:
            The propose output dict.
        """
        # Create tool callback for this step
        tool_callback = self.make_tool_callback(
            node_id=child.node_id,
            role="proposer",
            step="PROPOSE",
        )

        # Emit LLM trace events if using LLM proposer
        if self._use_llm_proposer and self._llm_client is not None:
            from delibera.agents.llm_proposer import ProposerLLM, check_llm_allowed_in_step

            check_llm_allowed_in_step("work")

            proposer: ProposerStub | ProposerLLM = ProposerLLM(
                llm_client=self._llm_client,
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_output_tokens=self._llm_max_output_tokens,
            )

            # Build context with node_id for tracing
            context = {
                "label": child.label,
                "question": question,
                "node_id": child.node_id,
            }

            # Emit llm_call_requested before the call
            writer.emit(
                TraceEvent(
                    event_type="llm_call_requested",
                    run_id=run_id,
                    payload={
                        "node_id": child.node_id,
                        "role": "proposer",
                        "step": "PROPOSE",
                        "provider": "gemini",
                        "model": self._llm_model or "default",
                    },
                )
            )

            try:
                propose_output = proposer.execute(context, tool=tool_callback)

                # Emit llm_call_succeeded
                writer.emit(
                    TraceEvent(
                        event_type="llm_call_succeeded",
                        run_id=run_id,
                        payload={
                            "node_id": child.node_id,
                            "role": "proposer",
                            "step": "PROPOSE",
                            "output_length": len(str(propose_output)),
                            "llm_generated": True,
                        },
                    )
                )
            except Exception as e:
                # Emit llm_call_failed and fall back to stub
                from delibera.llm.redaction import redact_text

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_failed",
                        run_id=run_id,
                        payload={
                            "node_id": child.node_id,
                            "role": "proposer",
                            "step": "PROPOSE",
                            "error_type": type(e).__name__,
                            "error_message": redact_text(str(e))[:200],
                        },
                    )
                )
                # Fall back to stub
                propose_output = ProposerStub().execute(
                    {"label": child.label, "question": question},
                    tool=tool_callback,
                )
        else:
            proposer = ProposerStub()
            propose_output = proposer.execute(
                {"label": child.label, "question": question},
                tool=tool_callback,
            )

        tree.update_artifact(child.node_id, propose_output)

        writer.emit(
            TraceEvent(
                event_type="work_output",
                run_id=run_id,
                payload={
                    "step": "PROPOSE",
                    "node_id": child.node_id,
                    "role": "proposer",
                    "output": propose_output,
                },
            )
        )

        return propose_output

    def _execute_research_on_node(
        self,
        child: Any,
        question: str,
        run_id: str,
        writer: TraceWriter,
        tree: DeliberationTree,
    ) -> dict[str, Any]:
        """Execute the RESEARCH step for a single node.

        Args:
            child: The child node to research.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer.
            tree: The deliberation tree.

        Returns:
            The research output dict.
        """
        # Create tool callback for this step
        tool_callback = self.make_tool_callback(
            node_id=child.node_id,
            role="researcher",
            step="RESEARCH",
        )

        research_output: dict[str, Any]

        if self._use_llm_researcher and self._llm_client is not None:
            from delibera.agents.llm_researcher import ResearcherLLM

            researcher_llm = ResearcherLLM(
                llm_client=self._llm_client,
                retriever=self._retriever,
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_output_tokens=400,
            )

            context = {
                "label": child.label,
                "question": question,
                "node_id": child.node_id,
                "proposal": child.artifact.get("recommendation", ""),
            }

            writer.emit(
                TraceEvent(
                    event_type="llm_call_requested",
                    run_id=run_id,
                    payload={
                        "node_id": child.node_id,
                        "role": "researcher",
                        "step": "RESEARCH",
                        "provider": "gemini",
                        "model": self._llm_model or "default",
                    },
                )
            )

            try:
                research_output = researcher_llm.execute(context, tool=tool_callback)

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_succeeded",
                        run_id=run_id,
                        payload={
                            "node_id": child.node_id,
                            "role": "researcher",
                            "step": "RESEARCH",
                            "output_length": len(str(research_output)),
                            "llm_generated": True,
                        },
                    )
                )
            except Exception as e:
                from delibera.llm.redaction import redact_text

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_failed",
                        run_id=run_id,
                        payload={
                            "node_id": child.node_id,
                            "role": "researcher",
                            "step": "RESEARCH",
                            "error_type": type(e).__name__,
                            "error_message": redact_text(str(e))[:200],
                        },
                    )
                )
                # Fall back to retriever or stub
                research_output = self._research_fallback(question, child, tool_callback)

        elif self._retriever is not None:
            # Build query from question and label
            query = f"{question} {child.label}"
            try:
                results = self._retriever.retrieve(query, max_results=5)

                # Verify results if verifier is available
                verified_count = 0
                if self._verifier is not None:
                    verified_results = []
                    for r in results:
                        v = self._verifier.verify(r)
                        if v.verified:
                            verified_results.append(r)
                            verified_count += 1
                    # Use verified results, or fall back to all if none verified
                    if verified_results:
                        results = verified_results

                research_output = {
                    "evidence": [{"source": r.source, "excerpt": r.excerpt} for r in results],
                    "notes": [
                        f"Found {len(results)} results via retriever",
                        f"Methods: {', '.join({r.method for r in results})}",
                    ],
                }

                if self._verifier is not None:
                    research_output["notes"].append(
                        f"Verified: {verified_count}/{len(results)} passed"
                    )
            except Exception as e:
                # Fall back to stub on retriever error
                researcher = ResearcherStub()
                research_output = researcher.execute(
                    {"label": child.label, "question": question},
                    tool=tool_callback,
                )
                research_output["notes"] = research_output.get("notes", []) + [
                    f"Retriever failed: {type(e).__name__}, used stub fallback"
                ]
        else:
            researcher = ResearcherStub()
            research_output = researcher.execute(
                {"label": child.label, "question": question},
                tool=tool_callback,
            )

        writer.emit(
            TraceEvent(
                event_type="work_output",
                run_id=run_id,
                payload={
                    "step": "RESEARCH",
                    "node_id": child.node_id,
                    "role": "researcher",
                    "output": research_output,
                },
            )
        )

        # Merge evidence into node ledger (engine-controlled state update)
        evidence_items = research_output.get("evidence", [])
        self._merge_evidence_to_ledger(tree, child.node_id, evidence_items, run_id, writer)

        return research_output

    def _execute_claim_check_on_node(
        self,
        child: Any,
        run_id: str,
        writer: TraceWriter,
        tree: DeliberationTree,
    ) -> ClaimCheckReport:
        """Execute the CLAIM_CHECK step for a single node.

        Args:
            child: The child node to validate.
            run_id: The current run ID.
            writer: The trace writer.
            tree: The deliberation tree.

        Returns:
            ClaimCheckReport with validation results.
        """
        report = self._validate_claims(tree, child.node_id, "proposer", run_id, writer)
        writer.emit(
            TraceEvent(
                event_type="claim_validation_report",
                run_id=run_id,
                payload={
                    "node_id": child.node_id,
                    "supported": report.supported,
                    "weak": report.weak,
                    "unsupported": report.unsupported,
                    "details": report.details,
                    "support_relations": report.support_relations,
                },
            )
        )
        return report

    def _execute_redteam_on_node(
        self,
        child: Any,
        question: str,
        run_id: str,
        writer: TraceWriter,
        tree: DeliberationTree,
    ) -> dict[str, Any]:
        """Execute the REDTEAM step for a single node.

        Args:
            child: The child node to attack.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer.
            tree: The deliberation tree.

        Returns:
            The redteam output dict.
        """
        if self._use_llm_redteam and self._llm_client is not None:
            from delibera.agents.llm_redteam import RedTeamLLM

            redteam: RedTeamStub | RedTeamLLM = RedTeamLLM(
                llm_client=self._llm_client,
                model=self._llm_model,
                temperature=self._llm_temperature,
                max_output_tokens=600,
            )
        else:
            redteam = RedTeamStub()

        # Build claims summary for RedTeam context
        claims_summary = [
            {
                "claim_id": c.claim_id,
                "claim_type": c.claim_type.value,
                "status": c.status.value,
            }
            for c in child.ledger.claims
        ]

        redteam_context: dict[str, Any] = {
            "node_id": child.node_id,
            "artifact": child.artifact,
            "claims": claims_summary,
            "evidence_count": len(child.ledger.evidence),
            "question": question,
        }

        if self._use_llm_redteam and self._llm_client is not None:
            writer.emit(
                TraceEvent(
                    event_type="llm_call_requested",
                    run_id=run_id,
                    payload={
                        "node_id": child.node_id,
                        "role": "redteam",
                        "step": "REDTEAM",
                        "provider": "gemini",
                        "model": self._llm_model or "default",
                    },
                )
            )

            try:
                redteam_output = redteam.execute(redteam_context)

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_succeeded",
                        run_id=run_id,
                        payload={
                            "node_id": child.node_id,
                            "role": "redteam",
                            "step": "REDTEAM",
                            "output_length": len(str(redteam_output)),
                            "llm_generated": True,
                        },
                    )
                )
            except Exception as e:
                from delibera.llm.redaction import redact_text

                writer.emit(
                    TraceEvent(
                        event_type="llm_call_failed",
                        run_id=run_id,
                        payload={
                            "node_id": child.node_id,
                            "role": "redteam",
                            "step": "REDTEAM",
                            "error_type": type(e).__name__,
                            "error_message": redact_text(str(e))[:200],
                        },
                    )
                )
                # Fall back to stub
                redteam_output = RedTeamStub().execute(redteam_context)
        else:
            redteam_output = redteam.execute(redteam_context)

        writer.emit(
            TraceEvent(
                event_type="work_output",
                run_id=run_id,
                payload={
                    "step": "REDTEAM",
                    "node_id": child.node_id,
                    "role": "redteam",
                    "output": redteam_output,
                },
            )
        )

        # Merge objections into node ledger (engine-controlled state update)
        objection_items = redteam_output.get("objections", [])
        self._merge_objections_to_ledger(tree, child.node_id, objection_items, run_id, writer)

        return redteam_output

    def _execute_score_nodes(
        self,
        children: list[Any],
        run_id: str,
        writer: TraceWriter,
        weights: ScoreWeights,
    ) -> dict[str, tuple[float, dict[str, float]]]:
        """Execute the SCORE step for all children.

        Args:
            children: List of child nodes to score.
            run_id: The current run ID.
            writer: The trace writer.
            weights: Scoring weights to use.

        Returns:
            Dict mapping node_id to (score, metrics) tuples.
        """
        node_scores: dict[str, tuple[float, dict[str, float]]] = {}

        for child in children:
            score_result = score_node(child, weights)
            node_scores[child.node_id] = (score_result.score, score_result.metrics)

            # Store score in node artifact for pruning
            child.artifact["score"] = score_result.score
            child.artifact["metrics"] = score_result.metrics

            writer.emit(
                TraceEvent(
                    event_type="score_computed",
                    run_id=run_id,
                    payload={
                        "node_id": child.node_id,
                        "score": score_result.score,
                        "metrics": score_result.metrics,
                        "weights": score_result.weights.to_dict(),
                    },
                )
            )

        return node_scores

    def _execute_step_on_node(
        self,
        step: Any,
        node: Any,
        question: str,
        run_id: str,
        writer: TraceWriter,
        tree: DeliberationTree,
    ) -> dict[str, Any]:
        """Execute a single pipeline step on a single node.

        Routes based on step.id to the appropriate execution method.
        Returns the step output dict (useful for expansion label extraction).

        Args:
            step: The StepSpec to execute.
            node: The node to execute on.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer.
            tree: The deliberation tree.

        Returns:
            Step output dict.
        """
        if step.id == "propose":
            return self._execute_propose_on_node(node, question, run_id, writer, tree)
        elif step.id == "research":
            return self._execute_research_on_node(node, question, run_id, writer, tree)
        elif step.id == "validate":
            report = self._execute_claim_check_on_node(node, run_id, writer, tree)
            return {"report": report}
        elif step.id == "redteam":
            return self._execute_redteam_on_node(node, question, run_id, writer, tree)
        else:
            raise ValueError(f"Unknown step id: {step.id}")

    # ------------------------------------------------------------------
    # Multi-level tree processing (Phase B)
    # ------------------------------------------------------------------

    def _process_level(
        self,
        tree: DeliberationTree,
        parent_id: str,
        labels: list[str],
        child_kind: str,
        question: str,
        run_id: str,
        writer: TraceWriter,
        current_weights: ScoreWeights,
        depth: int,
    ) -> tuple[Any, ScoreWeights]:
        """Process one level of the deliberation tree.

        1. EXPAND parent into children
        2. For each step in branch_pipeline:
           a. Execute step on all children
           b. Check if expansion triggers after this step at this depth
           c. If triggers: recurse into sub-level for each child
           d. Merge sub-tree results back into child's ledger
        3. SCORE children
        4. TRADEOFF GATE (depth 1 only)
        5. PRUNE + REDUCE → return merged node

        Args:
            tree: The deliberation tree.
            parent_id: The parent node ID to expand from.
            labels: Branch labels for child nodes.
            child_kind: The kind of child node ("option", "plan", "risk").
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer.
            current_weights: Scoring weights.
            depth: The current depth level (1-based).

        Returns:
            Tuple of (merged_node, current_weights) — weights may be
            updated by the tradeoff gate at depth 1.
        """
        assert self._interpreter is not None

        # EXPAND
        children = operators.expand(tree, parent_id, labels, kind=child_kind)

        writer.emit(
            TraceEvent(
                event_type="expand",
                run_id=run_id,
                payload={
                    "parent_id": parent_id,
                    "child_ids": [c.node_id for c in children],
                    "labels": labels,
                    "depth": depth,
                },
            )
        )

        for child in children:
            writer.emit(
                TraceEvent(
                    event_type="node_created",
                    run_id=run_id,
                    payload={
                        "node_id": child.node_id,
                        "parent_id": child.parent_id,
                        "kind": child.kind,
                        "depth": child.depth,
                        "label": child.label,
                    },
                )
            )

        # Execute branch pipeline steps, checking for sub-expansion after each
        last_outputs: dict[str, dict[str, Any]] = {}

        for step in self._protocol.branch_pipeline:
            if self._max_parallel > 1 and len(children) > 1:
                last_outputs.update(
                    self._execute_step_parallel(step, children, question, run_id, writer, tree)
                )
            else:
                for child in children:
                    output = self._execute_step_on_node(step, child, question, run_id, writer, tree)
                    last_outputs[child.node_id] = output

            # Check for sub-expansion after this step
            expand_rule = self._interpreter.should_expand_after_step(step.id, depth)
            if expand_rule is not None and depth < self._interpreter.max_depth():
                # Collect children that need sub-expansion
                children_with_labels: list[tuple[Any, list[str]]] = []
                for child in children:
                    sub_labels = self._interpreter.get_expand_labels_from_output(
                        last_outputs.get(child.node_id, {}), expand_rule
                    )
                    if sub_labels:
                        writer.emit(
                            TraceEvent(
                                event_type="sub_expansion_triggered",
                                run_id=run_id,
                                payload={
                                    "parent_node_id": child.node_id,
                                    "expand_rule_id": expand_rule.id,
                                    "sub_labels": sub_labels,
                                    "child_kind": expand_rule.child_kind,
                                    "target_depth": depth + 1,
                                },
                            )
                        )
                        children_with_labels.append((child, sub_labels))

                # Filter by conditional expansion
                eligible: list[tuple[Any, list[str]]] = []
                for child, sub_labels in children_with_labels:
                    if self._interpreter.evaluate_expand_condition(expand_rule, child):
                        eligible.append((child, sub_labels))
                    else:
                        writer.emit(
                            TraceEvent(
                                event_type="expansion_skipped",
                                run_id=run_id,
                                payload={
                                    "node_id": child.node_id,
                                    "expand_rule_id": expand_rule.id,
                                    "condition": expand_rule.condition,
                                    "reason": "condition_not_met",
                                },
                            )
                        )

                if self._max_parallel > 1 and len(eligible) > 1:
                    self._process_sub_expansions_parallel(
                        eligible,
                        tree,
                        expand_rule,
                        question,
                        run_id,
                        writer,
                        current_weights,
                        depth,
                    )
                else:
                    for child, sub_labels in eligible:
                        sub_merged, _ = self._process_level(
                            tree=tree,
                            parent_id=child.node_id,
                            labels=sub_labels,
                            child_kind=expand_rule.child_kind,
                            question=question,
                            run_id=run_id,
                            writer=writer,
                            current_weights=current_weights,
                            depth=depth + 1,
                        )
                        self._merge_subtree_into_parent(child, sub_merged, run_id, writer)

        # SCORE
        node_scores = self._execute_score_nodes(children, run_id, writer, current_weights)

        # TRADEOFF GATE (depth 1 only)
        if depth == 1 and self._initial_weights is None:
            sorted_scores = sorted(
                [(nid, score) for nid, (score, _) in node_scores.items()],
                key=lambda x: x[1],
                reverse=True,
            )

            if needs_tradeoff_gate(sorted_scores, self._gates_enabled, self._tie_threshold):
                current_weights = self._handle_tradeoff_gate(
                    children=children,
                    node_scores=node_scores,
                    current_weights=current_weights,
                    root_node_id=parent_id,
                )

                # Recompute scores with new weights
                for child in children:
                    score_result = score_node(child, current_weights)
                    node_scores[child.node_id] = (score_result.score, score_result.metrics)
                    child.artifact["score"] = score_result.score
                    child.artifact["metrics"] = score_result.metrics

        # DOMINANCE CHECK: Override keep_k if one option clearly dominates
        keep_k, _prune_rule = self._interpreter.get_prune_spec()
        convergence = self._interpreter.get_convergence_spec()
        if convergence.dominance_threshold is not None:
            sorted_by_score = sorted(node_scores.values(), key=lambda x: x[0], reverse=True)
            if len(sorted_by_score) >= 2:
                top_score = sorted_by_score[0][0]
                second_score = sorted_by_score[1][0]
                if second_score > 0 and top_score / second_score >= convergence.dominance_threshold:
                    writer.emit(
                        TraceEvent(
                            event_type="early_termination",
                            run_id=run_id,
                            payload={
                                "reason": "dominance",
                                "top_score": top_score,
                                "second_score": second_score,
                                "ratio": top_score / second_score,
                                "threshold": convergence.dominance_threshold,
                                "depth": depth,
                            },
                        )
                    )
                    keep_k = 1

        # PRUNE
        child_ids = [c.node_id for c in children]
        survivor_ids, pruned_ids = operators.prune(
            tree, child_ids, keep_count=keep_k, weights=current_weights
        )

        writer.emit(
            TraceEvent(
                event_type="prune",
                run_id=run_id,
                payload={
                    "survivor_ids": survivor_ids,
                    "pruned_ids": pruned_ids,
                    "weights_used": current_weights.to_dict(),
                    "scores": {nid: score for nid, (score, _) in node_scores.items()},
                    "depth": depth,
                },
            )
        )

        # REDUCE
        merged_node = operators.reduce(tree, parent_id, survivor_ids)

        writer.emit(
            TraceEvent(
                event_type="node_created",
                run_id=run_id,
                payload={
                    "node_id": merged_node.node_id,
                    "parent_id": merged_node.parent_id,
                    "kind": merged_node.kind,
                    "depth": merged_node.depth,
                    "label": merged_node.label,
                },
            )
        )

        writer.emit(
            TraceEvent(
                event_type="reduce",
                run_id=run_id,
                payload={
                    "survivor_ids": survivor_ids,
                    "merged_node_id": merged_node.node_id,
                    "merged_artifact": merged_node.artifact,
                    "depth": depth,
                },
            )
        )

        return merged_node, current_weights

    def _merge_subtree_into_parent(
        self,
        parent_node: Any,
        sub_merged: Any,
        run_id: str,
        writer: TraceWriter,
    ) -> None:
        """Merge sub-tree results into parent node's ledger.

        After a sub-level completes (prune + reduce), its merged node's
        ledger (claims, evidence, objections) is merged into the parent
        node so that remaining pipeline steps at the parent level benefit
        from the sub-tree's findings.

        Args:
            parent_node: The parent node to enrich.
            sub_merged: The merged node from the sub-level.
            run_id: The current run ID.
            writer: The trace writer.
        """
        from delibera.epistemics.ledger import merge_ledgers

        parent_node.ledger = merge_ledgers([parent_node.ledger, sub_merged.ledger])
        parent_node.artifact.setdefault("sub_trees", []).append(
            {
                "sub_merged_from": sub_merged.artifact.get("merged_from", []),
                "sub_recommendation": sub_merged.artifact.get("recommendation", ""),
                "sub_score": sub_merged.artifact.get("score", 0.0),
            }
        )

        writer.emit(
            TraceEvent(
                event_type="subtree_merged_into_parent",
                run_id=run_id,
                payload={
                    "parent_node_id": parent_node.node_id,
                    "sub_merged_node_id": sub_merged.node_id,
                    "sub_merged_from": sub_merged.artifact.get("merged_from", []),
                },
            )
        )

    # ------------------------------------------------------------------
    # Parallel execution helpers (Phase C)
    # ------------------------------------------------------------------

    def _execute_step_parallel(
        self,
        step: Any,
        children: list[Any],
        question: str,
        run_id: str,
        writer: TraceWriter,
        tree: DeliberationTree,
    ) -> dict[str, dict[str, Any]]:
        """Execute a pipeline step on multiple children in parallel.

        Args:
            step: The StepSpec to execute.
            children: List of child nodes.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer (thread-safe).
            tree: The deliberation tree.

        Returns:
            Dict mapping node_id to step output.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        outputs: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self._max_parallel) as executor:
            futures = {
                executor.submit(
                    self._execute_step_on_node, step, child, question, run_id, writer, tree
                ): child
                for child in children
            }
            for future in as_completed(futures):
                child = futures[future]
                try:
                    outputs[child.node_id] = future.result()
                except Exception as e:
                    writer.emit(
                        TraceEvent(
                            event_type="step_execution_failed",
                            run_id=run_id,
                            payload={
                                "node_id": child.node_id,
                                "step": step.id,
                                "error_type": type(e).__name__,
                                "error_message": str(e)[:200],
                            },
                        )
                    )
                    outputs[child.node_id] = {}
        return outputs

    def _process_sub_expansions_parallel(
        self,
        children_with_labels: list[tuple[Any, list[str]]],
        tree: DeliberationTree,
        expand_rule: Any,
        question: str,
        run_id: str,
        writer: TraceWriter,
        current_weights: ScoreWeights,
        depth: int,
    ) -> None:
        """Process sub-expansions for multiple children in parallel.

        Args:
            children_with_labels: List of (child, sub_labels) tuples.
            tree: The deliberation tree.
            expand_rule: The expand rule that triggered sub-expansion.
            question: The deliberation question.
            run_id: The current run ID.
            writer: The trace writer (thread-safe).
            current_weights: Scoring weights.
            depth: The current depth level.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=self._max_parallel) as executor:
            futures = {
                executor.submit(
                    self._process_level,
                    tree,
                    child.node_id,
                    sub_labels,
                    expand_rule.child_kind,
                    question,
                    run_id,
                    writer,
                    current_weights,
                    depth + 1,
                ): child
                for child, sub_labels in children_with_labels
            }
            for future in as_completed(futures):
                child = futures[future]
                try:
                    sub_merged, _ = future.result()
                    self._merge_subtree_into_parent(child, sub_merged, run_id, writer)
                except Exception as e:
                    writer.emit(
                        TraceEvent(
                            event_type="sub_expansion_failed",
                            run_id=run_id,
                            payload={
                                "parent_node_id": child.node_id,
                                "error_type": type(e).__name__,
                                "error_message": str(e)[:200],
                            },
                        )
                    )
