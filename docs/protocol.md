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

### Operator types

The engine implements these core operators:
- **Expand** — create child nodes (alternatives)
- **Work** — execute agent logic on a node
- **Validate** — check claims against evidence
- **Score** — compute quality metrics
- **Prune** — remove weak branches
- **Reduce** — merge surviving branches

### Protocol step names

Protocol steps are named actions that map to operators:

| Step | Operator | Description |
|------|----------|-------------|
| `PLAN` | Work | Initial planning using Planner role |
| `EXPAND` | Expand | Create alternative branches |
| `PROPOSE` | Work | Generate proposals using Proposer role |
| `RESEARCH` | Work | Gather evidence using Researcher role |
| `CLAIM_CHECK` | Validate | Validate claims against evidence |
| `REDTEAM` | Work | Challenge proposals using Skeptic role |
| `VALIDATE` | Validate | Check claims against evidence |
| `SCORE` | Score | Compute branch quality scores |
| `PRUNE` | Prune | Remove weak branches |
| `REDUCE` | Reduce | Merge surviving branches |
| `CONVERGE` | (check) | Evaluate convergence predicates |
| `FINALIZE` | Work | Produce final artifact |

Multiple step names may use the same operator (e.g., `PLAN`, `PROPOSE`, `REDTEAM` all invoke Work with different roles).

Agents are associated only with Work-based steps.

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
- WORK (e.g. PROPOSE, RESEARCH, REDTEAM)
- VALIDATE (claim checking)
- SCORE

Branch pipelines:
- operate on a single node
- do not modify sibling branches
- are parallelized when `max_parallel_branches > 1` (via `ThreadPoolExecutor`)

### 6.1 Parallel branch execution

Branch pipelines can be executed concurrently for significant speedup:

```bash
delibera run --question "Your question" --max-parallel-branches 3
```

Parallelism is implemented via `ThreadPoolExecutor`. Each branch runs its full pipeline independently. The `TraceWriter` is thread-safe (uses `threading.Lock`) to ensure correct event ordering. With real LLM agents, parallel execution achieves ~3x speedup.

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

## 11. Dynamic protocols

Protocols can adapt based on intermediate results.

### 11.1 Conditional expansion

Expand rules can include a `condition` that is evaluated against the node's ledger metrics before expansion occurs:

```yaml
expand_rules:
  - id: expand_sub_options
    at_step_id: propose
    child_kind: sub_option
    max_children: 2
    depth: 2
    condition: "evidence_coverage > 0.5"
```

The condition is evaluated by the interpreter against the node's computed metrics. If the condition is false, expansion is skipped.

### 11.2 Dominance threshold (early termination)

The convergence spec can include a `dominance_threshold` that triggers early termination when the top-scoring option clearly dominates:

```yaml
convergence:
  max_rounds: 0
  dominance_threshold: 2.0
```

If `top_score / second_score >= dominance_threshold`, the engine overrides `keep_k` to keep only the dominant option during pruning.

---

## 12. Convergence

Protocols must define a **convergence specification**.

Convergence may depend on:
- structural conditions (single branch remaining)
- epistemic quality (no blocking objections, bounded unsupported claims, evidence coverage threshold)
- stability across rounds
- budget exhaustion

Convergence is evaluated **only by the engine**.

---

## 13. User gates in protocols

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

## 14. Tree Protocol v1 (reference protocol)

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

### Example: Tree Protocol v1 specification

```yaml
name: tree_protocol_v1
version: "1.0"

budgets:
  max_depth: 3
  max_tool_calls: 100
  max_branches_per_expand: 5

expand_rules:
  - step: after_plan
    max_children: 5
    child_kind: option
    min_quality: 0.3

branch_pipeline:
  - step: PROPOSE
    role: proposer
  - step: REDTEAM
    role: skeptic
  - step: VALIDATE
  - step: SCORE

prune_rule:
  strategy: threshold_and_topk
  min_score: 0.4
  keep_top: 3
  block_on_unsupported_facts: true

reduce_rule:
  strategy: merge_supported
  conflict_handling: record_disagreement

refinement:
  enabled: true
  max_rounds: 2
  steps:
    - REDTEAM
    - VALIDATE
    - SCORE

convergence:
  structural: single_branch
  quality:
    max_unsupported_claims: 2
    min_evidence_coverage: 0.7
    no_blocking_objections: true

gates:
  - type: scope_clarification
    after: PLAN
    trigger: low_confidence
  - type: tradeoff_resolution
    after: SCORE
    trigger: score_tie
  - type: final_signoff
    after: CONVERGE
    trigger: always
```

---

## 15. Protocol invariants

All protocols must preserve:

1. Engine-only authority over structure
2. Bounded expansion
3. Deterministic pruning and reduction
4. Explicit convergence predicates
5. Replayable execution

Violating any invariant is a protocol error.

---

## 16. Summary

Protocols in Delibera define **how deliberation proceeds**, not *what agents think*.

By separating:
- protocol (rules)
- engine (control)
- agents (heuristics)

Delibera enables configurable, auditable, and convergent multi-agent deliberation.
