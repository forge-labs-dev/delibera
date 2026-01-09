# Delibera — Protocols

This document defines how **deliberation protocols** are specified and executed in Delibera.

Protocols describe *what deliberation process is allowed*.
They do not execute logic themselves; they are interpreted and enforced by the engine.

This document is the authoritative reference for protocol design.

---

## 1. What is a protocol?

A **protocol** is a declarative specification that defines:

- the sequence of deliberation steps
- where and how the tree may expand
- how branches are evaluated, pruned, and reduced
- how convergence is determined
- where user gates may occur
- what budgets apply

Protocols are:
- configurable
- workflow-agnostic
- independent of agent implementations

---

## 2. Protocol execution model

At runtime:

1. The engine loads a ProtocolSpec.
2. The engine initializes the deliberation tree.
3. The engine executes protocol-defined operators in order.
4. Agents are invoked only for `Work` steps.
5. All structural operations are engine-applied.
6. Execution stops only when convergence predicates hold.

Protocols constrain *possibility space*;  
the engine enforces correctness.

---

## 3. ProtocolSpec (conceptual schema)

A protocol specification defines:

- metadata
- budgets
- expand points
- per-branch pipelines
- prune rules
- reduce rules
- refinement loops
- convergence predicates
- user gate hooks

Conceptually:

```
ProtocolSpec :=
  name
  budgets
  expand_rules[]
  branch_pipeline[]
  prune_rule
  reduce_rule
  refine_loop[]
  convergence
  gates
```


This schema may be serialized as YAML or JSON, but is defined here conceptually.

---

## 4. Steps and operators

Protocols reference **operators**, not agent logic.

### Supported operators
- `PLAN` (Work)
- `EXPAND` (Expand)
- `WORK` (Work)
- `VALIDATE` (Validate)
- `SCORE` (Score)
- `PRUNE` (Prune)
- `REDUCE` (Reduce)
- `CONVERGE` (Check)

Agents are associated only with `Work` steps.

---

## 5. Expand rules (branching)

### 5.1 Purpose of EXPAND

EXPAND introduces **alternatives** into the deliberation tree.

Examples:
- options
- hypotheses
- plans
- risks

EXPAND is never open-ended.

---

### 5.2 Multiple EXPANDs

Protocols may allow **multiple EXPAND operations** at different depths.

Each EXPAND:
- has a semantic purpose
- has a maximum branching factor
- produces a specific child kind
- must eventually be paired with a REDUCE

Example conceptual pattern:

```
Root
└─ EXPAND (options)
   └─ WORK / VALIDATE / SCORE
      └─ EXPAND (plans)
         └─ WORK / VALIDATE / SCORE
            └─ REDUCE
               └─ REDUCE
```


---

### 5.3 Expand rule definition

Each expand rule specifies:
- at which step EXPAND is allowed
- maximum number of children
- child node kind
- minimum quality threshold to allow expansion
- maximum depth

The engine enforces all constraints.

---

## 6. Per-branch pipeline

After EXPAND, each branch executes the **branch pipeline** independently.

A branch pipeline is an ordered list of steps, typically including:
- WORK (e.g. PROPOSE, REDTEAM)
- VALIDATE (claim checking)
- SCORE

Branch pipelines:
- operate on a single node
- do not modify sibling branches
- may be parallelized

---

## 7. Validation and scoring

### 7.1 Validation

VALIDATE:
- extracts claims
- classifies claims as supported / weak / unsupported
- updates the node ledger

During VALIDATE:
- tool policy is tightened
- only evidence-local access is permitted
- no new sources may be introduced

---

### 7.2 Scoring

SCORE computes a score vector based on:
- claim validation results
- unresolved objections
- evidence coverage
- artifact completeness
- optional judge recommendations

Scoring is deterministic given node state and constraints.

---

## 8. Pruning

PRUNE removes weak branches deterministically.

Common prune criteria:
- unsupported blocking claims
- unresolved blocking objections
- low aggregate score
- dominance by another branch

PRUNE rules must be:
- explicit
- deterministic
- auditable

Agents may recommend pruning;  
the engine decides.

---

## 9. Reduction

REDUCE merges surviving branches.

Rules:
- only supported or accepted claims survive
- conflicts must be resolved or explicitly recorded
- provenance is preserved
- merged artifacts must remain structured

REDUCE is how the tree collapses.

---

## 10. Refinement loops

After REDUCE, protocols may define a **refinement loop** to improve the merged result.

Typical refinement steps:
- REDTEAM
- REVISE
- VALIDATE
- SCORE

Refinement loops are:
- bounded (max rounds)
- budget-aware
- convergence-checked each iteration

---

## 11. Convergence

Protocols must define a **convergence specification**.

Convergence may depend on:
- structural conditions (single branch remaining)
- epistemic quality (no blocking objections, bounded unsupported claims, evidence coverage threshold)
- stability across rounds
- budget exhaustion

Convergence is evaluated **only by the engine**.

---

## 12. User gates in protocols

Protocols may specify **user gates** at defined points.

Common gates:
- scope clarification (after PLAN)
- evidence access (during VALIDATE)
- tradeoff resolution (before PRUNE)
- final sign-off (after convergence)

User gates:
- pause execution
- accept bounded input
- update constraints only
- are fully logged

---

## 13. Tree Protocol v1 (reference protocol)

Tree Protocol v1 defines a minimal, generic deliberation flow:

1. PLAN (root)
2. EXPAND (options)
3. Per-branch pipeline:
   - WORK (propose)
   - WORK (redteam)
   - VALIDATE
   - SCORE
4. PRUNE
5. REDUCE
6. Refinement loop (optional)
7. CONVERGE
8. FINALIZE

This protocol:
- supports multiple EXPANDs
- guarantees one root and one leaf
- is suitable for most decision workflows

---

## 14. Protocol invariants

All protocols must preserve:

1. Engine-only authority over structure
2. Bounded expansion
3. Deterministic pruning and reduction
4. Explicit convergence predicates
5. Replayable execution

Violating any invariant is a protocol error.

---

## 15. Summary

Protocols in Delibera define **how deliberation proceeds**, not *what agents think*.

By separating:
- protocol (rules)
- engine (control)
- agents (heuristics)

Delibera enables configurable, auditable, and convergent multi-agent deliberation.
