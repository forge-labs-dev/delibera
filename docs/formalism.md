# Delibera — Formalism

This document defines the **formal model and terminology** used throughout Delibera.
It is the canonical reference for concepts, invariants, and operator semantics.

All implementation and protocol design must conform to this document.

---

## 1. Overview

Delibera models deliberation as a **governed, tree-structured process** that transforms an initial question into a single final decision artifact.

Formally, a Delibera run consists of:

- a **deliberation tree**
- a set of **engine-applied operators**
- a collection of **epistemic objects** (claims, evidence, objections)
- explicit **convergence predicates**
- optional **user gates**

Agents generate proposals and evaluations;
the **engine controls structure, progression, and termination**.

---

## 2. Inputs

### 2.1 Question and constraints

A run is initialized with a tuple:

$$
\mathcal{Q} = (q, C)
$$

Where:
- $q$ is the natural-language question or problem statement
- $C$ is a set of constraints, including:
  - budgets (time, cost, depth)
  - tool access policies
  - risk tolerance
  - output format requirements

---

## 3. Deliberation tree

### 3.1 Tree structure

A deliberation run constructs a rooted tree:

$$
\mathcal{D} = (V, \rightarrow, v_0)
$$

Where:
- $V$ is a set of nodes
- $\rightarrow$ is a parent–child relation (a tree)
- $v_0$ is the root node

The tree satisfies:
- exactly one root
- no cycles
- bounded depth (protocol-defined)

---

### 3.2 Nodes

Each node $v \in V$ represents a **branch of deliberation** (e.g., an option, hypothesis, plan, or risk).

A node has state:

$$
S(v) = (context, artifact, L, meta)
$$

Where:
- `context`: inherited constraints and branch-specific intent
- `artifact`: structured output produced for this node
- $L$: an epistemic ledger (claims, evidence, objections)
- `meta`: scores, timestamps, tool traces, and bookkeeping data

Nodes are immutable except through **engine-applied operators**.

---

## 4. Epistemic objects

### 4.1 Claims

A **claim** is an atomic unit of assertion:

$$
c = (id, text, type, confidence, owner)
$$

Where:
- `type` ∈ {`fact`, `inference`, `value`, `plan`}
- `confidence` ∈ [0,1]
- `owner` identifies the role that introduced the claim

Claims are extracted from agent outputs and explicitly tracked.

---

### 4.2 Evidence

An **evidence item** supports one or more claims:

$$
e = (id, source, excerpt, provenance)
$$

Where:
- `source` is a document, URL, or dataset
- `excerpt` is the supporting content
- `provenance` links to tool calls or document identifiers

---

### 4.3 Objections

An **objection** challenges a claim or artifact:

$$
o = (id, target, severity, status, rationale)
$$

Where:
- `severity` ∈ {`blocking`, `nonblocking`}
- `status` ∈ {`open`, `resolved`, `accepted`}

Blocking objections prevent convergence unless resolved or explicitly accepted.

---

### 4.4 Ledger

Each node maintains a **ledger**:

$$
L(v) = (\mathcal{C}, \mathcal{E}, \mathcal{O}, supp)
$$

Where:
- $\mathcal{C}$ = set of claims
- $\mathcal{E}$ = set of evidence items
- $\mathcal{O}$ = set of objections
- $supp(c)$ ⊆ $\mathcal{E}$ is the support relation

The ledger is the authoritative epistemic state for a node.

---

## 5. Operators (engine-applied)

All operators are applied **exclusively by the deliberation engine**.

Agents may recommend actions, but cannot invoke operators directly.

---

### 5.1 Expand

$$
Expand(v) \rightarrow \{v_1, \dots, v_k\}
$$

Creates child nodes representing alternatives.

Constraints:
- $k$ ≤ max branching factor
- depth ≤ max depth
- expansion must be protocol-authorized

Each Expand must eventually be paired with a Reduce.

---

### 5.2 Work

$$
Work_{step}(v) \rightarrow S(v)'
$$

Executes one protocol step on a node using one or more roles.

Effects:
- updates artifact
- adds claims, evidence, objections
- may request tool usage (policy-gated)

---

### 5.3 Validate (claim checking)

$$
Validate(v) \rightarrow report
$$

Classifies claims as:
- `supported`
- `weak`
- `unsupported`

Validation operates under **tightened tool policy**:
- only evidence-local access is permitted
- no new sources may be introduced

Validation updates claim status in the ledger.

---

### 5.4 Score

$$
Score(v) \rightarrow s(v) \in \mathbb{R}^m
$$

Computes a score vector based on:
- ledger statistics
- artifact properties
- unresolved objections
- optional judge recommendations

---

### 5.5 Prune

$$
Prune(\{v_1,\dots,v_n\}) \rightarrow K
$$

Selects a subset $K$ of surviving nodes using a deterministic rule.

Examples:
- remove nodes with unsupported blocking claims
- keep top-k by score
- drop dominated alternatives

---

### 5.6 Reduce

$$
Reduce(K) \rightarrow v_{merged}
$$

Merges surviving nodes into a single node.

Rules:
- only supported or accepted claims survive
- conflicts must be resolved or explicitly recorded
- ledgers are merged with provenance preserved

Reduce collapses branches upward in the tree.

---

## 6. Convergence

Convergence is **engine-determined**, not agent-determined.

### 6.1 Structural convergence

Structural convergence holds when:
- exactly one branch remains at the root
- required Reduce operations have completed

---

### 6.2 Quality convergence

Quality convergence is satisfied when all hold:

- no unresolved blocking objections
- unsupported claims ≤ threshold
- evidence coverage ≥ minimum
- ledger changes stabilize across rounds

---

### 6.3 Budget stop

If budgets are exhausted:
- the engine may converge with a **best-effort artifact**
- uncertainty and open issues must be surfaced explicitly

---

### 6.4 Convergence predicate

$$
Converged(\mathcal{D}, S_\star) =
Structural \land (Quality \lor BudgetStop)
$$

---

## 7. User gates

A **user gate** is a protocol-defined interruption point:

$$
Gate(S_\star, v) \rightarrow (Decision, \Delta C)
$$

Properties:
- gates pause execution
- present a structured summary
- accept bounded input
- update constraints only

Users cannot directly mutate the tree.

---

## 8. Invariants

The following invariants must always hold:

1. Only the engine applies operators.
2. The deliberation graph is a tree.
3. Every run produces one final artifact or an explicit abort.
4. Claims and evidence are explicit.
5. Validation runs under stricter policy than research.
6. Convergence is measurable and auditable.

Violating any invariant is a correctness bug.

---

## 9. Summary

Delibera formalizes deliberation as:

> A governed tree search over epistemic artifacts,
> with explicit operators, constrained agent roles,
> and deterministic convergence.

This formalism underpins all protocols, implementations, and evaluations.
