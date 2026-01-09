# Delibera — Tooling and Policy

This document defines how **external tools** are integrated into Delibera and how **policy** governs their use.

Tooling in Delibera is designed for:
- safety
- reproducibility
- auditability
- epistemic integrity

Tools are treated as **capabilities**, not conveniences.

---

## 1. Purpose

Unrestricted tool use leads to:
- irreproducible reasoning
- silent data leakage
- post-hoc evidence injection
- untraceable decisions

Delibera enforces **governed tool access** so that:
- every tool call is intentional
- every result is attributable
- every decision is auditable

---

## 2. Core principles

Tooling in Delibera follows these principles:

1. **Engine-mediated access**  
   Agents never call tools directly.

2. **Policy-first design**  
   Tool access is allowed only when permitted by policy.

3. **Least privilege**  
   Roles and steps receive only the tools they require.

4. **Traceability**  
   Every tool call is logged with inputs and outputs.

5. **Epistemic safety**  
   Validation cannot introduce new evidence.

---

## 3. Tool abstraction

### 3.1 ToolSpec

Each tool is defined by a ToolSpec:

```
ToolSpec :=
  name
  input_schema
  output_schema
  risk_level
  execute()
```

Where:
- `name` uniquely identifies the tool
- `input_schema` defines allowed arguments
- `output_schema` defines returned data
- `risk_level` ∈ {low, medium, high}
- `execute()` performs the tool action

Tools must be deterministic where possible or clearly document nondeterminism.

---

### 3.2 Examples of tools

Common tools include:
- `web.search`
- `web.open`
- `docs.retrieve`
- `db.query`
- `calculator`
- `code.exec` (explicitly restricted; not enabled in v1)

Each tool is registered with the engine at startup.

---

## 4. ToolRouter

All tool calls pass through the **ToolRouter**.

Flow:

```
Agent → Engine → ToolRouter → Tool
```


The ToolRouter:
- validates the request against the ToolSpec
- consults the PolicyEngine
- executes or denies the request
- emits trace events

Agents never bypass the ToolRouter.

---

## 5. PolicyEngine

The PolicyEngine enforces **layered constraints**.

A tool call is permitted only if **all** policy layers allow it.

---

### 5.1 Policy layers

#### 1. Global policy
Applies to the entire run.

Controls:
- enabled tools
- allowed domains or sources
- maximum tool calls
- total cost or time budgets
- data handling constraints

---

#### 2. Role-based policy
Applies to a specific agent role.

Examples:
- Researcher may use `web.search`
- ClaimChecker may only inspect existing evidence
- Judge may use no external tools

---

#### 3. Step-level policy overrides
Applies during specific protocol steps.

Most important override:
- `CLAIM_CHECK` step enforces *evidence-local access only*

Overrides may:
- further restrict tools
- tighten domain allowlists
- reduce call limits

Overrides may not expand permissions beyond global policy.

---

#### 4. Budget constraints
Applies across all layers.

Budgets include:
- max tool calls
- max wall-clock time
- max cost units

Budget exhaustion may trigger graceful convergence.

---

## 6. Evidence-local restriction

### 6.1 Motivation

Claim validation must not:
- introduce new sources,
- search for supporting evidence after the fact,
- or silently strengthen weak claims.

---

### 6.2 Rule

During validation:
- only tools that **inspect already-cited evidence** are allowed
- tools that discover new sources are denied
- attempts to violate this restriction are logged

Examples:
- Allowed: `web.open` on a previously cited URL
- Allowed: `docs.retrieve` for an already referenced document
- Denied: `web.search`

This restriction is enforced by a step-level policy override.

---

## 7. Tool call traceability

Every tool call emits a trace event containing:

```
ToolCall :=
  tool_name
  role
  step
  input
  output
  timestamp
  policy_decision
```

Trace data must be sufficient to:
- inspect tool behavior
- reproduce reasoning context
- audit policy enforcement

Tool calls are immutable once recorded.

---

## 8. Policy violations

If a tool call is denied:
- execution continues unless the protocol specifies otherwise
- a trace event is emitted
- agents may adapt their output or raise objections

Repeated or critical violations may:
- trigger a user gate
- affect scoring
- block convergence

---

## 9. Tooling and convergence

Tool usage affects convergence indirectly:
- lack of evidence may block claim validation
- denied tool access may require user intervention
- excessive tool use may exhaust budgets

The engine incorporates these signals when evaluating convergence.

---

## 10. Security and non-goals (v1)

Explicit non-goals in v1:
- arbitrary code execution
- unrestricted filesystem access
- network access without policy
- silent data exfiltration

High-risk tools must be explicitly enabled and audited.

---

## 11. Invariants

The tooling subsystem must preserve:

1. No direct tool access by agents
2. Policy-enforced tool usage
3. Evidence-local validation
4. Full traceability of tool calls
5. Deterministic policy decisions

Violating any invariant is a correctness and safety bug.

---

## 12. Summary

Delibera’s tooling and policy system ensures that:
- tool use is intentional
- evidence is not fabricated
- decisions are auditable
- safety constraints are enforceable

This governance layer is essential for decision-grade AI deliberation.
