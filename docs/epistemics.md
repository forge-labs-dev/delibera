# Delibera — Epistemics

This document defines the **epistemic model** used by Delibera.

Epistemics governs how claims, evidence, and objections are represented, validated, merged, and used to determine deliberation quality and convergence.

This document is the authoritative reference for epistemic correctness.

---

## 1. Purpose

Most AI systems treat reasoning as unstructured text.

Delibera instead treats reasoning as a set of **explicit epistemic objects**:
- claims
- evidence
- objections

These objects are:
- structured
- attributable
- auditable
- governable

The epistemics subsystem enables:
- claim validation
- principled pruning
- accountable convergence

---

## 2. Core epistemic objects

### 2.1 Claims

A **claim** is an atomic assertion extracted from agent outputs.

Each claim has the form:

```
Claim :=
  id
  text
  type
  confidence
  owner
  status
```

Where:
- `type` ∈ {`fact`, `inference`, `value`, `plan`}
- `confidence` ∈ [0,1]
- `owner` identifies the role that introduced the claim
- `status` ∈ {`supported`, `weak`, `unsupported`, `unvalidated`}

Claims must be:
- minimal (one assertion per claim)
- attributable to a role
- traceable through the deliberation tree

---

### 2.2 Claim types

Claim types have different epistemic requirements:

- **fact**
  Asserts something about the world.
  Requires evidence.

- **inference**
  Draws a conclusion from other claims or evidence.
  Requires justification and inherits uncertainty.

- **value**
  Expresses a preference or judgment.
  Does not require evidence, but must be explicit.

- **plan**
  Proposes an action or sequence of steps.
  Does not require evidence, but may introduce risks.

Claim type determines validation behavior.

---

### 2.3 Evidence

An **evidence item** supports one or more claims.

```
Evidence :=
  id
  source
  excerpt
  provenance
```

Where:
- `source` is a document, URL, dataset, or internal reference
- `excerpt` is the relevant supporting content
- `provenance` links to tool calls, document IDs, or trace events

Evidence must be:
- immutable once recorded
- explicitly linked to claims
- reproducible or inspectable

---

### 2.4 Objections

An **objection** challenges a claim, plan, or artifact.

```
Objection :=
  id
  target
  severity
  status
  rationale
```

Where:
- `target` references a claim or artifact
- `severity` ∈ {`blocking`, `nonblocking`}
- `status` ∈ {`open`, `resolved`, `accepted`}

Blocking objections prevent convergence unless resolved or explicitly accepted.

---

## 3. Ledger

Each node maintains a **ledger** that aggregates epistemic objects.

```
Ledger :=
  claims
  evidence
  objections
  support_relations
```

Where:
- `claims` is a set of Claim objects
- `evidence` is a set of Evidence objects
- `objections` is a set of Objection objects
- `support_relations` maps claims to supporting evidence

The ledger is the **single source of truth** for epistemic state.

---

## 4. Claim extraction

Claims are extracted from agent-produced artifacts.

Requirements:
- extraction is deterministic
- claims are minimal and non-overlapping
- extraction preserves provenance (source artifact, role, step)

In v1, claim extraction may use simple heuristics (e.g. sentence splitting with filters).

Claim extraction is invoked only by the engine.

---

## 5. Validation (claim checking)

### 5.1 Purpose

Validation determines whether claims are supported by evidence.

Validation produces a **ClaimCheckReport** that updates claim statuses.

---

### 5.2 Validation rules (v1)

For each claim:

- **fact**
  - supported if at least one evidence item exists
    whose excerpt materially supports the claim
  - weak if evidence is indirect or partial
  - unsupported if no valid evidence exists

- **inference**
  - supported if underlying facts are supported
  - weak if based on weak facts
  - unsupported if based on unsupported facts

- **value**
  - automatically supported (but explicitly marked as value)

- **plan**
  - automatically supported (but may introduce objections)

Validation rules are intentionally conservative.

---

### 5.3 Evidence-local restriction

During validation:
- tool access is restricted
- only already-cited evidence may be inspected
- no new sources may be introduced

This prevents post-hoc rationalization.

---

## 6. Objection lifecycle

Objections progress through states:

1. **open**
   Objection has been raised and not addressed.

2. **resolved**
   Objection has been addressed through revision,
   mitigation, or additional evidence.

3. **accepted**
   Objection remains valid but is explicitly accepted
   as a known risk or tradeoff.

Blocking objections must not remain open at convergence.

---

## 7. Ledger merging (Reduce)

During REDUCE, ledgers from multiple branches are merged.

Rules:
- claims with status `unsupported` are dropped
- supported and accepted claims survive
- conflicting claims must be:
  - resolved, or
  - carried forward as explicit disagreements
- evidence provenance is preserved
- objections are merged with updated targets

Ledger merging is deterministic.

---

## 8. Epistemic metrics

Epistemic state feeds scoring and convergence.

Common metrics:
- number of unsupported claims
- number of weak claims
- evidence coverage (fraction of fact claims with evidence)
- number of unresolved blocking objections

These metrics are computed by the engine.

---

## 9. Epistemics and pruning

Pruning decisions may depend on epistemic quality.

Examples:
- prune branches with any unsupported blocking claims
- prune branches with evidence coverage below threshold
- prune dominated branches with strictly worse epistemic metrics

Agents may recommend pruning;
the engine decides.

---

## 10. Epistemics and convergence

Convergence requires epistemic sufficiency.

At convergence:
- no unresolved blocking objections remain
- unsupported claims are below threshold
- remaining uncertainty is explicit

Epistemic weakness must never be hidden.

---

## 11. Invariants

The epistemics subsystem must preserve:

1. Claims are explicit and attributable
2. Evidence is required for factual claims
3. Validation is conservative
4. Unsupported claims do not silently survive
5. Epistemic state is auditable and replayable

Violating any invariant is a correctness bug.

---

## 12. Summary

Delibera’s epistemic model ensures that:

- reasoning is explicit,
- evidence is accountable,
- disagreement is structured,
- and convergence is justified.

This epistemic rigor distinguishes Delibera from chat-based and agent-centric systems.
