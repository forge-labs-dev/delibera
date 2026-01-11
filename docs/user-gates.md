# Delibera — User Gates

This document defines **user gates**, Delibera’s mechanism for structured human input.

User gates allow humans to intervene **without breaking determinism, auditability, or engine authority**.

They are the only sanctioned interface for human-in-the-loop interaction.

---

## 1. Purpose

Some aspects of deliberation cannot be safely inferred by agents:
- intent clarification
- value tradeoffs
- hidden constraints
- accountability and sign-off

User gates exist to handle these irreducible human inputs **without turning Delibera into a chat system**.

---

## 2. What is a user gate?

A **user gate** is a protocol-defined interruption point where:
- execution pauses
- the engine presents a structured summary
- bounded user input is requested
- constraints are updated
- execution resumes deterministically

Formally:

```
Gate :=
  gate_type
  trigger_predicate
  summary_payload
  allowed_actions
```


User gates:
- do not mutate the deliberation tree
- do not inject free-form text
- do not bypass operators
- are fully logged and replayable

---

## 3. User gate execution model

1. Engine evaluates gate predicate
2. Gate predicate returns true
3. Engine emits a `gate_triggered` event
4. Execution pauses
5. User provides structured input
6. Engine validates input
7. Constraints are updated
8. Execution resumes

Agents are not executed during a gate.

---

## 4. What users can and cannot do

### Users can:
- approve or reject interpretations
- adjust priorities and weights
- add constraints or assumptions
- explicitly accept risks
- approve final output

### Users cannot:
- add or delete tree nodes
- bypass validation
- override claim status
- directly edit agent outputs
- force convergence arbitrarily

All user actions are bounded by gate type.

---

## 5. Canonical user gates

Delibera defines a small, fixed set of **canonical gates**.
Protocols may enable or disable them, but must not redefine their semantics.

---

### 5.1 Gate: Intake

**Purpose**
Collect initial question, constraints, and success criteria.

**When**
At run start, before any operator execution.

**User input**
- question confirmation or edit
- constraints (budgets, risk tolerance)
- output format preferences

**Effect**
- initializes or updates global constraints

---

### 5.2 Gate: Scope clarification

**Purpose**
Ensure the system is solving the right problem.

**When**
After PLAN, before the first EXPAND.

**Trigger predicate (examples)**
- low confidence in extracted assumptions
- ambiguous question interpretation
- large or diverse proposed branches

**User input**
- approve or modify assumptions
- veto specific branches
- add missing constraints

**Effect**
- updates planning assumptions
- constrains subsequent EXPAND operations

---

### 5.3 Gate: Evidence access

**Purpose**
Handle blocked validation due to missing or restricted sources.

**When**
During VALIDATE, when evidence is insufficient or policy blocks access.

**Trigger predicate (examples)**
- unsupported fact claims due to missing sources
- denied tool calls required for validation

**User input**
- approve additional source access
- provide a document or reference
- decline (forcing rewrite or pruning)

**Effect**
- updates tool policy constraints
- may unblock validation

---

### 5.4 Gate: Tradeoff resolution

**Purpose**
Resolve value-based ambiguity between strong alternatives.

**When**
After scoring and before PRUNE, when multiple branches are near-tied.

**Trigger predicate (examples)**
- score difference below threshold
- conflicting value claims (e.g. speed vs safety)

**User input**
- set priority weights
- explicitly choose among options
- request further analysis on a specific dimension

**Effect**
- updates scoring weights
- may alter pruning outcome

---

### 5.5 Gate: Final sign-off

**Purpose**
Provide accountability for the final decision.

**When**
After convergence criteria are satisfied.

**User input**
- approve and finalize
- request additional bounded refinement
- abort the run

**Effect**
- approval finalizes artifact
- refinement resumes execution
- abort terminates run explicitly

---

## 6. Gate summaries

At each gate, the engine presents a **GateSummary**.

A GateSummary:
- is minimal
- is structured
- exposes only relevant state
- avoids overwhelming the user

Example (tradeoff gate):

```
GateSummary :=
  gate_type: tradeoff
  top_branches:
    - name
      score
      key_pros
      key_cons
      unresolved_objections
  allowed_actions
```


Gate summaries are included in the trace.

---

## 7. Structured user responses

User responses must conform to a predefined schema.

Example:

```
GateResponse :=
  action
  parameters
```


Examples:
- `{ action: "approve" }`
- `{ action: "set_priority", parameters: { "safety": 0.7, "speed": 0.3 } }`
- `{ action: "abort" }`

Free-form text is not permitted.

---

## 8. User gates and replay

During replay:
- gates are not re-triggered
- user responses are read from the trace
- execution resumes deterministically

User gates do not introduce nondeterminism.

---

## 9. User gates and convergence

User gates:
- do not replace convergence predicates
- do not force termination
- may update constraints that affect convergence

Final convergence remains engine-determined.

---

## 10. Failure modes and safeguards

If a user:
- provides invalid input → engine rejects and requests correction
- refuses to respond → protocol-defined timeout applies
- aborts → run ends with explicit `aborted` state

All outcomes are logged.

---

## 11. Invariants

User gates must preserve:

1. Engine authority
2. Bounded user input
3. Structured summaries
4. Deterministic replay
5. Full traceability

Violating any invariant is a correctness bug.

---

## 12. Summary

User gates allow Delibera to:
- incorporate human intent and values
- preserve accountability
- avoid silent assumptions
- remain auditable

They are the **only** sanctioned path for human input
in decision-grade AI deliberation.
