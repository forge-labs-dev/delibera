# Delibera — Tracing and Replay

This document defines how Delibera records execution, enables replay, and supports auditability.

Tracing is a **first-class requirement** in Delibera.
If a deliberation cannot be traced and replayed, it is considered incorrect.

---

## 1. Purpose

Delibera is designed for **decision-grade reasoning**, not conversational output.

As such, every run must be:
- inspectable
- auditable
- reproducible
- explainable after the fact

Tracing and replay provide these guarantees.

---

## 2. What is a trace?

A **trace** is an append-only record of all significant events that occur during a Delibera run.

A trace must be sufficient to:
- reconstruct the deliberation tree
- inspect all agent outputs
- verify policy enforcement
- understand convergence decisions
- regenerate the final artifact **without re-running agents or tools**

---

## 3. Trace structure

Traces are stored as **JSONL** (one event per line).

Directory layout:

```
runs/<run_id>/
  trace.jsonl      # Append-only event log
  artifact.json    # Final decision artifact
  metadata.json    # Run metadata (see below)
```

Each run has a unique, immutable `run_id`.

### metadata.json contents

The metadata file captures run-level information:

```json
{
  "run_id": "abc123",
  "protocol": "tree_protocol_v1",
  "protocol_version": "1.0",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:45:00Z",
  "status": "converged",
  "question": "What cloud provider should we use?",
  "constraints": {
    "max_depth": 3,
    "max_tool_calls": 100
  },
  "summary": {
    "total_nodes": 12,
    "branches_pruned": 3,
    "tool_calls": 47,
    "user_gates_triggered": 2
  }
}
```

Metadata is written at run completion and is derived from trace events.

---

## 4. Trace events

Each trace entry is a structured event.

```
TraceEvent :=
  event_type
  timestamp
  run_id
  payload
```


Events are ordered strictly by time of occurrence.

---

## 5. Required event types

### 5.1 Run lifecycle

- `run_start`
- `run_end`
- `aborted`

---

### 5.2 Tree structure

- `node_created`
- `expand`
- `prune`
- `reduce`

These events are sufficient to reconstruct the deliberation tree.

---

### 5.3 Agent execution

- `work_start`
- `work_output`

Payload includes:
- role
- step
- structured agent output
- references to node IDs

---

### 5.4 Epistemics

- `claims_extracted`
- `claim_validation_report`
- `objection_added`
- `objection_resolved`

These events capture epistemic state transitions.

---

### 5.5 Tooling and policy

- `tool_call_requested`
- `tool_call_executed`
- `tool_call_denied`

Payload includes:
- tool name
- role
- step
- input/output
- policy decision

---

### 5.6 User gates

- `gate_triggered`
- `gate_response_applied`

Payload includes:
- gate type
- summary shown to user
- structured user response

---

### 5.7 Convergence

- `score_computed`
- `convergence_checked`
- `budget_exhausted`

These events explain why the run stopped.

---

### 5.8 Finalization

- `final_artifact_written`

Payload references `artifact.json`.

---

## 6. Artifact storage

The final decision artifact is stored separately as `artifact.json`.

The artifact must include:
- final recommendation
- supporting claims
- evidence references
- resolved and accepted objections
- remaining uncertainties
- confidence estimates

The artifact must be derivable from the trace.

---

### 6.1 Artifact persistence

When available, Delibera may use Strata as the persistence backend for artifacts referenced in traces.

In this mode:
- trace events reference Strata artifact IDs
- artifacts are immutable and versioned
- lineage between artifacts is preserved independently of trace logs

Replay uses trace data for control flow reconstruction and Strata artifacts for content reconstruction, without re-executing agents or tools.


## 7. Replay semantics

### 7.1 Definition

**Replay** is the deterministic reconstruction of a run from its trace.

Replay must:
- rebuild the deliberation tree
- restore node states and ledgers
- reapply operator effects
- regenerate the final artifact

Replay must **not**:
- invoke LLMs
- invoke tools
- request user input

---

### 7.2 Replay process

1. Load `trace.jsonl`
2. Reconstruct tree structure from structural events
3. Apply state updates from epistemic and operator events
4. Rebuild final artifact from recorded outputs
5. Validate convergence predicates

Any mismatch indicates a trace or implementation bug.

---

## 8. Determinism guarantees

Delibera guarantees determinism under replay because:
- agents do not control operators
- all nondeterministic outputs are recorded
- policy decisions are logged
- convergence is engine-computed

The trace is the **single source of truth**.

---

## 9. Partial replay and inspection

Traces support partial replay:
- inspect a single node’s evolution
- analyze why a branch was pruned
- audit tool usage
- review unresolved objections

Partial replay is read-only.

---

## 10. Privacy and redaction

Traces may contain sensitive data.

Delibera supports:
- configurable redaction at trace write time
- metadata-only traces
- separation of artifact and trace storage

Redaction must preserve structural integrity.

---

## 11. Invariants

Tracing and replay must preserve:

1. Append-only event logs
2. Complete operator coverage
3. Deterministic replay
4. Artifact derivability from trace
5. Auditability of policy decisions

Violating any invariant is a correctness bug.

---

## 12. Summary

Tracing and replay make Delibera:
- auditable
- explainable
- reproducible
- trustworthy

They are not optional features.
They are foundational to decision-grade AI deliberation.
