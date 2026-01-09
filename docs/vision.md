# Delibera — Vision

## Purpose

Delibera is an open-source engine for **decision-grade AI deliberation**.

Its goal is to make multi-agent reasoning **structured, governed, and auditable**, so that AI systems can support real decisions—not just generate persuasive text.

Delibera treats deliberation as a **process**, not a conversation.

---

## The problem

Current multi-agent LLM systems suffer from the same core issues:

- reasoning is implicit and unstructured  
- disagreement is uncontrolled  
- convergence is undefined (“stop when it feels done”)  
- claims are mixed with opinions and plans  
- evidence is untracked or post-hoc  
- outputs are chat logs, not decisions  

As a result, these systems:
- hallucinate confidently,
- fail silently,
- cannot be audited or replayed,
- and are unsafe for high-stakes use.

Delibera exists to address these failures **at the system level**, not via better prompts.

---

## Core idea

Delibera models deliberation as a **governed tree search**:

- one **root**: the user’s question and constraints  
- multiple **branches**: alternatives, hypotheses, plans, or risks  
- one **leaf**: a single, canonical decision artifact  

The system explicitly:
- expands alternatives,
- evaluates and validates them,
- prunes weak branches,
- reduces survivors,
- and converges using measurable criteria.

This process is **engine-controlled**, not agent-driven.

---

## What Delibera is

Delibera is:

- a **deliberation engine**, not a chatbot
- a **protocol runner**, not a fixed workflow
- **multi-agent**, but not autonomous
- **LLM-powered**, but not LLM-dependent
- **open and inspectable**, by design

It is intended to be infrastructure for:
- research synthesis
- design and architecture review
- due diligence
- strategy analysis
- policy or risk assessment

---

## What Delibera is not

Delibera is **not**:

- a conversational interface
- an autonomous decision-maker
- a black-box AI system
- a single application or domain-specific tool
- an agent “playground”

Delibera does not aim to:
- remove humans from decisions
- replace institutional judgment
- optimize for speed over correctness

---

## Design principles

### 1. Governance over autonomy
Agents propose content.  
The engine governs structure, operators, and stopping.

No agent can:
- expand the tree,
- prune branches,
- merge results,
- or decide when to stop.

---

### 2. Explicit structure beats implicit reasoning
All reasoning artifacts are explicit:
- claims
- evidence
- objections
- scores
- decisions

Nothing important lives only in free-form text.

---

### 3. Convergence is a first-class concept
Delibera never stops because “agents agree”.

It stops only when:
- explicit convergence criteria are satisfied, or
- budgets are exhausted (with uncertainty surfaced).

---

### 4. Claims require accountability
Factual claims must be:
- identifiable,
- attributable,
- and checkable against evidence.

Unsupported claims are not silently accepted.

---

### 5. Tool use is governed
Tools are capabilities, not shortcuts.

All tool access is:
- policy-controlled,
- role-scoped,
- step-restricted,
- logged for replay.

---

### 6. Humans intervene only where necessary
Human input is incorporated through **user gates**:
- scope clarification
- evidence access
- value tradeoffs
- final sign-off

User input is structured, bounded, and auditable.

---

## Success criteria (v1)

Delibera v1 is successful if:

1. A full run produces **exactly one final decision artifact** (or an explicit abort).
2. The deliberation process is **replayable without re-running agents or tools**.
3. Tree expansion, pruning, and reduction are **engine-controlled**.
4. Claims and evidence are explicitly tracked and validated.
5. Convergence is determined by **measurable predicates**, not agent preference.
6. User input only enters via defined gates.
7. Removing LLMs does not break the core control logic.

If these are true, Delibera is correct—even if incomplete.

---

## Long-term vision

Delibera aims to become:

- a **reference implementation** for deliberative AI systems
- a foundation for **auditable decision workflows**
- a bridge between:
  - argumentation theory,
  - Tree-of-Thought reasoning,
  - and real-world decision processes

Its success is measured not by fluency, but by **trustworthiness**.

---

## Guiding statement

> If a decision cannot be explained, traced, and replayed,  
> it should not be automated.

Delibera exists to make that principle practical.
