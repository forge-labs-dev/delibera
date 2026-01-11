# Delibera — Architecture

This document describes the **software architecture** of Delibera and how the formalism maps to concrete system components.

Its purpose is to:
- define clear module boundaries
- enforce engine authority
- prevent accidental coupling between agents, protocols, and state

All implementation must respect the constraints defined here.

---

## 1. Architectural overview

Delibera follows a **layered architecture** with strict separation of concerns:

```
+----------------------------------------------------+
|                    CLI / API                       |
+---------------------------+------------------------+
|        Protocol Layer     |      User Gates        |
+---------------------------+------------------------+
|      Deliberation Engine (Orchestrator)            |
+----------------------------------------------------+
|   Epistemics   |   Tools & Policy   |   Tracing    |
+----------------------------------------------------+
|            Agents (Roles, LLM-backed)              |
+----------------------------------------------------+
|             External Systems (LLMs, Docs, etc.)    |
+----------------------------------------------------+
```

The **Deliberation Engine** is the authority.
All other components operate under its control.

---

## 2. Core components

### 2.1 Deliberation Engine (Orchestrator)

The engine is the **central control loop**.

Responsibilities:
- owns the deliberation tree
- applies all operators:
  - Expand
  - Work
  - Validate
  - Score
  - Prune
  - Reduce
  - Converge
- enforces protocol order and budgets
- triggers user gates
- records trace events

Non-responsibilities:
- generating content
- deciding factual correctness
- performing tool actions directly

The engine **never delegates authority** over structure or termination.

---

### 2.2 Protocol Layer

The protocol layer defines **what sequence of operators is allowed**.

A protocol specifies:
- valid steps and their order
- where Expand is allowed (possibly multiple times)
- per-branch pipelines
- prune and reduce rules
- convergence predicates
- user gate locations
- budget constraints

Protocols are:
- declarative
- workflow-agnostic
- independent of agent implementations

The engine interprets protocols.
Protocols never execute logic themselves.

---

### 2.3 Agents (Roles)

Agents are **heuristic generators**, not controllers.

Each agent:
- has a single role (Planner, Researcher, Skeptic, etc.)
- consumes structured input
- produces structured output:
  - artifacts
  - claims
  - objections
  - recommendations

Agents:
- may request tool usage via the engine
- may recommend actions (e.g., “consider expanding”)
- may score or critique

Agents cannot:
- mutate the tree
- invoke operators
- terminate execution
- bypass policies

Removing agents entirely should not break the engine.

---

### 2.4 Persistence and lineage (Strata)

Delibera does not manage persistence, caching, or lineage internally.

Instead, it delegates artifact materialization and provenance tracking to **Strata**, an external persistence layer for long-horizon computation.

Strata provides:
- immutable, versioned artifacts
- deterministic deduplication by provenance
- explicit lineage between artifacts
- crash-safe finalization and recovery

Delibera uses Strata to persist:
- agent work outputs
- validation reports
- reduction results
- final decision artifacts

Strata does not:
- control deliberation flow
- manage retries or loops
- execute agents or tools

This separation ensures that Delibera remains an orchestration and reasoning system, while Strata serves as the durable memory and history of deliberation runs.

---

## 3. Epistemics subsystem

The epistemics layer implements:
- claims
- evidence
- objections
- ledgers
- validation logic

Responsibilities:
- extract claims from artifacts
- track support relations
- classify claim status
- merge ledgers during Reduce

The epistemics layer:
- contains no control flow
- does not know about protocols
- does not manage tools directly

It is invoked **only by engine operators**.

---

## 4. Tooling and policy subsystem

### 4.1 ToolRouter

All tool calls flow through a ToolRouter:

```
Agent → Engine → ToolRouter → Tool
```

The ToolRouter:
- validates tool requests
- consults the PolicyEngine
- executes or denies the call
- logs full provenance

Agents never call tools directly.

---

### 4.2 PolicyEngine

The PolicyEngine enforces layered constraints:
1. Global policy (session/org)
2. Role-based policy
3. Step-level overrides (e.g. CLAIM_CHECK)
4. Budget constraints

During validation:
- tool access is restricted to evidence-local operations
- no new sources may be introduced

Policy violations are surfaced as trace events.

---

## 5. User gates

User gates are implemented as **engine-controlled interrupts**.

Flow:
1. Engine evaluates a gate predicate
2. Engine emits a GateEvent
3. Execution pauses
4. Structured user input is collected
5. Constraints are updated
6. Execution resumes

User gates:
- do not modify tree structure
- do not inject free-form text into agent context
- are fully logged and replayable

---

## 6. Tracing and replay

Tracing is a first-class subsystem.

Every run records:
- operator applications
- agent outputs
- tool calls
- policy decisions
- user gates
- final artifact

Traces are:
- append-only
- JSON-serializable
- sufficient to replay the run without LLMs or tools

Replay uses trace data to:
- reconstruct tree state
- regenerate the final artifact
- inspect decisions and convergence

---

## 7. Data flow (end-to-end)

1. User provides question and constraints
2. Engine initializes root node
3. Protocol dictates next operator
4. Engine invokes agent(s) for Work steps
5. Agents propose structured outputs
6. Engine applies operators and updates state
7. Tool requests are routed and logged
8. Validation and pruning occur
9. Reduction collapses branches
10. Convergence is evaluated
11. Final artifact is produced and logged

At no point do agents control the flow.

---

## 8. Module boundaries (hard rules)

The following dependencies are **forbidden**:

- Agents importing engine internals
- Agents mutating tree or node state
- Epistemics invoking tools
- Protocols executing logic
- User input bypassing gates

Violations indicate architectural bugs.

---

## 9. Implementation guidance

Recommended module split:

```
delibera/
├── engine/
│   ├── orchestrator.py
│   ├── operators.py
│   └── state.py
├── protocol/
│   ├── spec.py
│   └── interpreter.py
├── agents/
│   ├── base.py
│   ├── planner.py
│   ├── proposer.py
│   ├── skeptic.py
│   └── judge.py
├── epistemics/
│   ├── claims.py
│   ├── evidence.py
│   ├── objections.py
│   ├── ledger.py
│   └── validate.py
├── tools/
│   ├── base.py
│   ├── router.py
│   └── policy.py
├── gates/
│   ├── models.py
│   └── handlers.py
└── trace/
    ├── events.py
    ├── writer.py
    └── replay.py
```

This layout mirrors responsibility boundaries.

---

## 10. Architectural invariants

The architecture must preserve:

1. Engine-only authority over structure and termination
2. Explicit epistemic objects
3. Governed tool access
4. Deterministic convergence
5. Replayable execution

Breaking any invariant compromises Delibera’s correctness.

---

## 11. Summary

Delibera’s architecture enforces a strict separation between:
- **thinking** (agents),
- **knowing** (epistemics),
- **doing** (tools),
- and **deciding** (engine).

This separation is what makes Delibera:
- auditable,
- reproducible,
- and suitable for high-stakes decision support.
