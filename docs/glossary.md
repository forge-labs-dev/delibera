# Delibera — Glossary

This glossary defines key terms used throughout Delibera's documentation.

---

## Core Concepts

### Artifact
A structured output produced by deliberation. Each node has an artifact, and the final artifact is the canonical decision output of a run.

### Branch
A path in the deliberation tree representing one alternative, hypothesis, or option being explored.

### Claim
An atomic assertion extracted from agent outputs. Claims have types (`fact`, `inference`, `value`, `plan`), confidence scores, and validation status.

### Convergence
The condition under which a deliberation run terminates. Convergence is engine-determined based on structural conditions (single branch remaining) and quality predicates (no blocking objections, sufficient evidence coverage).

### Deliberation Tree
The rooted tree structure that organizes a deliberation run. Has exactly one root (the initial question) and collapses to one leaf (the final decision).

---

## Epistemic Objects

### Evidence
Data that supports a claim. Evidence has a source, excerpt, and provenance linking it to tool calls or documents.

### Ledger
The epistemic state of a node, containing all claims, evidence, objections, and support relations.

### Objection
A challenge to a claim or artifact. Objections have severity (`blocking` or `nonblocking`) and status (`open`, `resolved`, `accepted`).

### Support Relation
The mapping from claims to evidence items that substantiate them.

---

## Operators

### Expand
Engine operator that creates child nodes representing alternatives. Must eventually be paired with a Reduce.

### Prune
Engine operator that removes weak branches based on deterministic criteria (e.g., unsupported claims, low scores).

### Reduce
Engine operator that merges surviving branches into a single node, preserving supported claims and resolving conflicts.

### Score
Engine operator that computes a score vector for a node based on ledger statistics and artifact properties.

### Validate
Engine operator that classifies claims as `supported`, `weak`, or `unsupported` based on available evidence.

### Work
Engine operator that executes a protocol step on a node using one or more agent roles.

---

## System Components

### Agent
A heuristic generator with a specific role (Planner, Researcher, Skeptic, Judge). Agents propose content but cannot control structure or termination.

### Engine (Orchestrator)
The central control loop that owns the deliberation tree, applies operators, enforces protocols, and determines convergence.

### PolicyEngine
The component that enforces layered constraints on tool access (global, role-based, step-level, budget).

### Protocol
A declarative specification defining the sequence of operators, branching rules, and convergence predicates for a deliberation run.

### ToolRouter
The component that mediates all tool calls, validating requests against policy and logging provenance.

### Trace
An append-only record of all events during a run, sufficient to replay the deliberation without re-running agents or tools.

---

## User Interaction

### User Gate
A protocol-defined interruption point where execution pauses, a structured summary is presented, and bounded user input is collected.

### Gate Types
- **Intake**: Initial question and constraint collection
- **Scope Clarification**: Verify problem interpretation after planning
- **Evidence Access**: Handle blocked validation due to missing sources
- **Tradeoff Resolution**: Resolve value-based ambiguity between alternatives
- **Final Sign-off**: Accountability approval for the final decision

---

## Claim Types

| Type | Description | Evidence Required |
|------|-------------|-------------------|
| `fact` | Assertion about the world | Yes |
| `inference` | Conclusion drawn from other claims | Inherits from premises |
| `value` | Preference or judgment | No (must be explicit) |
| `plan` | Proposed action or steps | No (may introduce objections) |

---

## Claim Status

| Status | Meaning |
|--------|---------|
| `supported` | Adequate evidence exists |
| `weak` | Evidence is indirect or partial |
| `unsupported` | No valid evidence |
| `unvalidated` | Not yet checked |

---

## Objection Status

| Status | Meaning |
|--------|---------|
| `open` | Raised and not addressed |
| `resolved` | Addressed through revision or mitigation |
| `accepted` | Acknowledged as a known risk or tradeoff |
