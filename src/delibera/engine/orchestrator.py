"""Deliberation engine orchestrator.

The engine is the central control loop that owns the deliberation tree,
applies operators, enforces protocols, and determines convergence.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from delibera.agents.stub import PlannerStub, ProposerStub
from delibera.engine import operators
from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree
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

    Implements the v0.3 protocol with deterministic stub agents:
    1. PLAN - Generate branch labels
    2. EXPAND - Create child nodes
    3. PROPOSE - Generate proposals per branch (with optional tool use)
    4. VALIDATE - Extract and validate claims
    5. PRUNE - Keep top-2 by epistemic quality
    6. REDUCE - Merge survivors
    7. FINALIZE - Write artifact with claim_check_summary
    """

    def __init__(
        self,
        runs_dir: Path | None = None,
        tool_registry: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            runs_dir: Directory for run outputs. Defaults to ./runs.
            tool_registry: Tool registry for available tools.
            policy_engine: Policy engine for tool access control.
        """
        self.runs_dir = runs_dir or Path("runs")
        self._tool_registry = tool_registry or create_default_registry()
        self._policy_engine = policy_engine or create_default_policy_engine()

        # These are set during run execution
        self._current_run_id: str = ""
        self._current_writer: TraceWriter | None = None
        self._tool_router: ToolRouter | None = None

    def run(self, question: str) -> Path:
        """Execute a full deliberation run.

        Args:
            question: The question or problem to deliberate on.

        Returns:
            Path to the run directory containing trace.jsonl and artifact.json.

        Raises:
            ValueError: If question is empty or whitespace-only.
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

            # Emit run_start
            writer.emit(
                TraceEvent(
                    event_type="run_start",
                    run_id=run_id,
                    payload={"question": question, "created_at": created_at},
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

            # EXPAND: Create child nodes
            branch_labels = plan_output["branches"]
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

            # VALIDATE: Extract and validate claims for each branch
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

            # PRUNE: Keep top-2 by epistemic quality
            child_ids = [c.node_id for c in children]
            survivor_ids, pruned_ids = operators.prune(tree, child_ids, keep_count=2)

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

            # FINALIZE: Build and write artifact
            artifact = operators.finalize(state, merged_node)
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
