"""Deliberation engine orchestrator.

The engine is the central control loop that owns the deliberation tree,
applies operators, enforces protocols, and determines convergence.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from delibera.agents.stub import PlannerStub, ProposerStub, RedTeamStub, ResearcherStub
from delibera.engine import operators
from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree
from delibera.epistemics.models import Evidence, Objection, ObjectionSeverity, ObjectionStatus
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

            # PLAN: Call planner stub
            planner = PlannerStub()
            plan_output = planner.execute({"question": question})
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

            # SCOPE GATE: Get user approval for branches
            branch_labels = plan_output["branches"]
            if needs_scope_gate(plan_output, self._gates_enabled):
                branch_labels = self._handle_scope_gate(
                    question=question,
                    branch_labels=branch_labels,
                    node_id=root.node_id,
                )

            # EXPAND: Create child nodes
            children = operators.expand(tree, root.node_id, branch_labels)

            writer.emit(
                TraceEvent(
                    event_type="expand",
                    run_id=run_id,
                    payload={
                        "parent_id": root.node_id,
                        "child_ids": [c.node_id for c in children],
                        "labels": branch_labels,
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

            # PROPOSE: Generate proposals for each branch
            proposer = ProposerStub()
            for child in children:
                # Create tool callback for this step
                tool_callback = self.make_tool_callback(
                    node_id=child.node_id,
                    role="proposer",
                    step="PROPOSE",
                )
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

            # RESEARCH: Collect evidence for each branch
            researcher = ResearcherStub()
            for child in children:
                # Create tool callback for this step
                tool_callback = self.make_tool_callback(
                    node_id=child.node_id,
                    role="researcher",
                    step="RESEARCH",
                )
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

            # CLAIM_CHECK: Extract and validate claims for each branch
            for child in children:
                report = operators.validate(tree, child.node_id, "proposer")
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

            # REDTEAM: Generate objections for each branch
            redteam = RedTeamStub()
            for child in children:
                # Build claims summary for RedTeam context
                claims_summary = [
                    {
                        "claim_id": c.claim_id,
                        "claim_type": c.claim_type.value,
                        "status": c.status.value,
                    }
                    for c in child.ledger.claims
                ]

                redteam_output = redteam.execute(
                    {
                        "node_id": child.node_id,
                        "artifact": child.artifact,
                        "claims": claims_summary,
                        "evidence_count": len(child.ledger.evidence),
                    }
                )

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
                self._merge_objections_to_ledger(
                    tree, child.node_id, objection_items, run_id, writer
                )

            # SCORE: Compute scores for each branch
            current_weights = self._initial_weights or create_default_weights()
            node_scores: dict[str, tuple[float, dict[str, float]]] = {}

            for child in children:
                score_result = score_node(child, current_weights)
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

            # TRADEOFF GATE: Trigger if near-tie detected (unless weights pre-set)
            sorted_scores = sorted(
                [(nid, score) for nid, (score, _) in node_scores.items()],
                key=lambda x: x[1],
                reverse=True,
            )

            # Only trigger tradeoff gate if weights weren't pre-set via CLI
            if self._initial_weights is None and needs_tradeoff_gate(
                sorted_scores, self._gates_enabled, self._tie_threshold
            ):
                current_weights = self._handle_tradeoff_gate(
                    children=children,
                    node_scores=node_scores,
                    current_weights=current_weights,
                    root_node_id=root.node_id,
                )

                # Recompute scores with new weights
                for child in children:
                    score_result = score_node(child, current_weights)
                    node_scores[child.node_id] = (score_result.score, score_result.metrics)
                    child.artifact["score"] = score_result.score
                    child.artifact["metrics"] = score_result.metrics

            # PRUNE: Keep top-k by epistemic quality + score (from protocol)
            child_ids = [c.node_id for c in children]
            keep_k, _prune_rule = self._interpreter.get_prune_spec()
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
                    },
                )
            )

            # REDUCE: Merge survivors into single node
            merged_node = operators.reduce(tree, root.node_id, survivor_ids)

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
                    },
                )
            )

            # Build artifact for sign-off
            artifact = operators.finalize(state, merged_node)

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
