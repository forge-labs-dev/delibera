"""Deliberation engine orchestrator.

The engine is the central control loop that owns the deliberation tree,
applies operators, enforces protocols, and determines convergence.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from delibera.agents.stub import PlannerStub, ProposerStub, ResearcherStub
from delibera.engine import operators
from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree
from delibera.epistemics.models import Evidence
from delibera.gates import (
    GATE_ALLOWED_ACTIONS,
    AutoApproveGateHandler,
    GateAborted,
    GateHandler,
    GateSummary,
    GateType,
    apply_final_signoff_response,
    apply_scope_response,
    get_response_changes,
    needs_final_signoff_gate,
    needs_scope_gate,
)
from delibera.protocol import DEFAULT_PROTOCOL, ProtocolInterpreter, ProtocolSpec
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
    ) -> None:
        """Initialize the engine.

        Args:
            runs_dir: Directory for run outputs. Defaults to ./runs.
            tool_registry: Tool registry for available tools.
            policy_engine: Policy engine for tool access control.
            gate_handler: Handler for user gates. Defaults to AutoApproveGateHandler.
            gates_enabled: Whether gates are enabled. Defaults to True.
            protocol: Protocol specification to use. Defaults to DEFAULT_PROTOCOL.
        """
        self.runs_dir = runs_dir or Path("runs")
        self._tool_registry = tool_registry or create_default_registry()
        self._policy_engine = policy_engine or create_default_policy_engine()
        self._gate_handler = gate_handler or AutoApproveGateHandler()
        self._gates_enabled = gates_enabled
        self._protocol = protocol or DEFAULT_PROTOCOL

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
                        },
                    )
                )

            # PRUNE: Keep top-k by epistemic quality (from protocol)
            child_ids = [c.node_id for c in children]
            # Note: _prune_rule reserved for future prune strategies (e.g., score_only)
            keep_k, _prune_rule = self._interpreter.get_prune_spec()
            survivor_ids, pruned_ids = operators.prune(tree, child_ids, keep_count=keep_k)

            writer.emit(
                TraceEvent(
                    event_type="prune",
                    run_id=run_id,
                    payload={
                        "survivor_ids": survivor_ids,
                        "pruned_ids": pruned_ids,
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
                self._handle_final_signoff_gate(
                    artifact=artifact,
                    node_id=merged_node.node_id,
                )

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

    def _handle_final_signoff_gate(
        self,
        artifact: dict[str, Any],
        node_id: str,
    ) -> None:
        """Handle the final sign-off gate before writing artifact.

        Args:
            artifact: The final artifact to approve.
            node_id: The merged node ID.

        Raises:
            GateAborted: If user chooses to abort.
        """
        # Build summary
        summary = GateSummary(
            gate_type=GateType.FINAL_SIGNOFF,
            allowed_actions=GATE_ALLOWED_ACTIONS[GateType.FINAL_SIGNOFF],
            recommendation=artifact.get("recommendation", ""),
            claim_check_summary=artifact.get("claim_check_summary"),
            open_questions=artifact.get("open_questions", []),
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
        apply_final_signoff_response(response)

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
