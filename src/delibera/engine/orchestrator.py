"""Deliberation engine orchestrator.

The engine is the central control loop that owns the deliberation tree,
applies operators, enforces protocols, and determines convergence.
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from delibera.agents.stub import PlannerStub, ProposerStub
from delibera.engine import operators
from delibera.engine.state import RunState
from delibera.engine.tree import DeliberationTree
from delibera.trace.events import TraceEvent
from delibera.trace.writer import TraceWriter


class Engine:
    """The deliberation engine orchestrator.

    Implements the v0 protocol with deterministic stub agents:
    1. PLAN - Generate branch labels
    2. EXPAND - Create child nodes
    3. PROPOSE - Generate proposals per branch
    4. PRUNE - Keep top-2 by score
    5. REDUCE - Merge survivors
    6. FINALIZE - Write artifact
    """

    def __init__(self, runs_dir: Path | None = None) -> None:
        """Initialize the engine.

        Args:
            runs_dir: Directory for run outputs. Defaults to ./runs.
        """
        self.runs_dir = runs_dir or Path("runs")

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

        # Initialize trace writer
        writer = TraceWriter(run_dir)

        try:
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
                propose_output = proposer.execute(
                    {"label": child.label, "question": question}
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

            # PRUNE: Keep top-2 by score
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
            # Emit run_end with failed status on any error
            writer.emit(
                TraceEvent(
                    event_type="run_end",
                    run_id=run_id,
                    payload={"status": "failed", "error": str(e)},
                )
            )
            raise

    def _generate_run_id(self) -> str:
        """Generate a unique, filesystem-safe run ID."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"{timestamp}_{short_uuid}"
