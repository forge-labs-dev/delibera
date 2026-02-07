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
|                    CLI / API                        |
+---------------------------+------------------------+
|        Protocol Layer     |      User Gates        |
+---------------------------+------------------------+
|      Deliberation Engine (Orchestrator)             |
+----------------------------------------------------+
|  Epistemics  |  Tools & Policy  |  Tracing          |
+----------------------------------------------------+
|         Agents (Stubs or LLM-backed)               |
+----------------------------------------------------+
|  LLM Providers  |  Retrieval + Verification        |
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
- supports **multi-level tree expansion** via recursive `_process_level()`
- supports **parallel branch execution** via `ThreadPoolExecutor`
- evaluates **dynamic protocol conditions** (conditional expansion, dominance threshold)
- enforces protocol order and budgets
- triggers user gates
- records trace events (thread-safe)

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

### 2.4 LLM Integration

The LLM layer provides a pluggable interface for language model access.

Components:
- **LLMClient** protocol — `generate(LLMRequest) -> LLMResponse`
- **GeminiClient** — Google Gemini implementation (SDK + HTTP fallback)
- **Prompt templates** — Structured prompts for all agent roles
- **Redaction** — Sensitive data filtering before sending to LLMs

LLM-backed agents:
- **LLM Planner** — Generates contextual branch labels from the question
- **LLM Proposer** — Generates structured proposals with claims, evidence, and sub-branches for depth-2 expansion
- **LLM Researcher** — Formulates search queries and executes via retriever (web search auto-enabled)
- **LLM Red-Teamer** — Generates meaningful objections with severity classification
- **LLM Refiner** — Addresses objections and improves proposals iteratively
- **LLM Validator** — Semantic claim-evidence matching (replaces keyword heuristic)

LLM calls are:
- always traced (request metadata, response, latency, token usage)
- never made during replay or validation
- opt-in via `--use-all-llm` or per-role flags (e.g., `--use-llm-proposer`)
- gracefully degraded (falls back to stubs on failure)

---

### 2.5 Evidence Retrieval

The retrieval layer provides multi-source evidence gathering.

Retrievers:
- **KeywordRetriever** — local keyword matching (default, no API needed)
- **EmbeddingRetriever** — semantic search using Gemini embeddings
- **WebRetriever** — web search via Gemini Google Search grounding
- **HybridRetriever** — combines local + web with Reciprocal Rank Fusion

All retrievers implement the `EvidenceRetriever` protocol and return `RetrievalResult` objects with source, excerpt, score, source_type, and method metadata.

---

### 2.6 Evidence Verification

The verification layer validates web search results before they enter the epistemic system.

Verifiers:
- **FetchVerifier** — follows redirect URLs, fetches actual content, checks excerpt containment
- **LLMVerifier** — uses Gemini to fact-check claims against sources
- **CrossReferenceVerifier** — searches for corroborating sources

All verifiers implement the `EvidenceVerifier` protocol. Verification is opt-in via `--verify`.

---

### 2.7 Persistence and lineage (Strata — planned)

Delibera does not yet manage persistence, caching, or lineage internally.

Future integration with **Strata** (an external persistence layer) will provide:
- immutable, versioned artifacts
- deterministic deduplication by provenance
- explicit lineage between artifacts
- crash-safe finalization and recovery

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

Current module layout:

```
delibera/
├── engine/
│   ├── orchestrator.py      # Central control loop
│   ├── operators.py         # Expand, work, validate, score, prune, reduce, finalize
│   ├── tree.py              # Deliberation tree data structures
│   └── state.py             # Run state tracking
├── protocol/
│   ├── spec.py              # Protocol dataclasses
│   ├── loader.py            # YAML protocol loading
│   ├── interpreter.py       # Protocol execution logic
│   └── defaults.py          # Default protocol values
├── agents/
│   ├── base.py              # Agent protocol
│   ├── stub.py              # Deterministic stubs (planner, proposer, researcher, redteam, refiner)
│   ├── llm_planner.py       # LLM-backed planner
│   ├── llm_proposer.py      # LLM-backed proposer (with sub_branches for multi-level trees)
│   ├── llm_researcher.py    # LLM-backed researcher (query generation + retriever)
│   ├── llm_redteam.py       # LLM-backed red-teamer
│   ├── llm_refiner.py       # LLM-backed refiner
│   └── llm_validator.py     # LLM-backed claim validator (semantic matching)
├── epistemics/
│   ├── models.py            # Claim, Evidence, Objection dataclasses
│   ├── extract.py           # Extract claims from artifacts
│   ├── ledger.py            # Evidence ledger and support tracking
│   └── validate.py          # Claim validation and evidence linking
├── retrieval/
│   ├── base.py              # Retriever/verifier protocols, result types
│   ├── keyword.py           # Keyword-based retrieval
│   ├── embedding.py         # Embedding-based semantic search
│   ├── web.py               # Web search via Gemini grounding
│   ├── hybrid.py            # RRF fusion of local + web
│   └── verify.py            # FetchVerifier, LLMVerifier, CrossReferenceVerifier
├── llm/
│   ├── base.py              # LLMClient protocol, request/response types
│   ├── gemini.py            # Google Gemini integration
│   ├── prompts.py           # Prompt templates
│   └── redaction.py         # Sensitive data redaction
├── scoring/
│   ├── score.py             # Core scoring logic
│   ├── metrics.py           # Epistemic quality metrics
│   └── weights.py           # Weighted score combining
├── tools/
│   ├── spec.py              # ToolSpec protocol
│   ├── registry.py          # Tool registry
│   ├── router.py            # Tool request routing
│   ├── policy.py            # Policy engine
│   └── builtin/             # Calculator, docs search/read
├── gates/
│   ├── models.py            # Gate types, summaries, responses
│   ├── predicates.py        # Gate trigger conditions
│   ├── handler.py           # CLI, auto-approve, scripted handlers
│   └── apply.py             # Gate response application
├── trace/
│   ├── events.py            # TraceEvent dataclass
│   ├── writer.py            # Write trace to JSONL
│   ├── reader.py            # Read and parse traces
│   ├── replay.py            # Deterministic replay
│   └── validate.py          # Trace consistency validation
├── inspect/
│   ├── summarize.py         # Run summarization
│   ├── render_md.py         # Markdown report generation
│   └── render_text.py       # Text summary rendering
├── eval/
│   ├── runner.py            # Evaluation suite runner
│   ├── loader.py            # Load evaluation YAML
│   ├── metrics.py           # Evaluation metrics
│   ├── compare.py           # Compare runs
│   └── models.py            # Eval data models
└── cli.py                   # Click-based CLI
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
