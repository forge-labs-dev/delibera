# Delibera Roadmap

This document outlines the current state and planned next steps for Delibera.

## Current State (v0.1.0)

The core engine is functional with 513 tests passing. All phases of the deliberation loop are implemented: plan, expand, propose, research, claim-check, red-team, score, prune, reduce, refine, and finalize. The system runs end-to-end with stub agents or with LLM-backed agents (Gemini) for all roles.

### Completed

- Core deliberation engine with 15-phase orchestration
- Declarative YAML protocol system
- Epistemic layer (claims, evidence, objections, validation, support linking)
- Multi-source evidence retrieval (keyword, embedding, web, hybrid)
- Evidence verification for web results (fetch, LLM, cross-reference)
- LLM integration with Gemini for all agent roles (planner, proposer, researcher, red-teamer, refiner, validator)
- LLM-backed claim validator with semantic evidence matching
- Human-in-the-loop gates (scope, tradeoff, final sign-off)
- Tool system with policy engine and built-in tools
- Complete tracing and deterministic replay
- Inspection and Markdown report generation
- Evaluation harness with suite runner
- CLI with extensive configuration

### Stub-Only (deterministic, for testing)

- All agent roles have stub fallbacks for deterministic testing without API keys

---

## Next Steps

### Priority 1: LLM-Backed Agents (COMPLETED)

All stub agents replaced with LLM-backed implementations using Gemini, with stub fallback on failure.

- **LLM Planner** - Generates contextual branch labels from the question
- **LLM Proposer** - Generates structured proposals with claims and evidence
- **LLM Researcher** - Formulates search queries and executes via retriever
- **LLM Red-Teamer** - Generates meaningful objections with severity classification
- **LLM Refiner** - Addresses objections and improves proposals iteratively

### Priority 2: Additional LLM Providers

Currently only Gemini is supported. Adding more providers increases accessibility and allows comparison.

**OpenAI Provider**
- Implement LLMClient for GPT-4o / GPT-4o-mini
- Support structured JSON output via response_format
- Handle rate limiting and token counting

**Anthropic Provider**
- Implement LLMClient for Claude
- Support tool use for structured output
- Handle streaming and token counting

**Ollama / Local Provider**
- Support local models via Ollama API
- Useful for testing and environments without API access
- Lower quality but zero cost and full privacy

### Priority 3: Improved Claim Validation (COMPLETED)

LLM-backed claim validation replaces keyword-matching heuristic with semantic understanding.

- **LLM Claim Validator** - Uses LLM to semantically assess claim-evidence support
  - Scores confidence of support relationship (not just binary)
  - Detects contradictions between claims and evidence
  - Identifies claims that need additional evidence
  - Provides reasoning for each assessment
  - Falls back to heuristic validation on LLM error
- **Semantic Evidence Matching** - LLM evaluates semantic support, not just keyword overlap
  - Handles paraphrasing and indirect support
  - Ranks evidence by relevance via confidence scoring

### Priority 4: Protocol Enhancements

**Multi-Level Tree Expansion**
- Support deeper trees (current protocols use max_depth=1)
- Sub-plans within options (e.g., "How to implement Option A?")
- Risk analysis branches per option

**Dynamic Protocols**
- Allow protocols to adapt based on intermediate results
- Early termination when one option clearly dominates
- Conditional branches based on claim validation results

**Parallel Branch Execution**
- Run branch pipelines concurrently (currently sequential)
- Would significantly speed up deliberation with multiple options

### Priority 5: Persistence and Integration

**Strata Integration**
- Integrate with Strata for artifact persistence and lineage tracking
- Immutable, versioned artifacts with provenance
- Cross-run artifact deduplication
- Crash recovery from partially completed runs

**API Server**
- REST/gRPC API for programmatic access
- WebSocket support for streaming gate interactions
- Run management (start, pause, resume, inspect)

**Web UI**
- Visual deliberation tree explorer
- Interactive gate responses
- Real-time run monitoring
- Report viewer

### Priority 6: Production Hardening

**Async Execution**
- Convert synchronous engine to async
- Non-blocking LLM calls and retrieval
- Concurrent branch evaluation

**Cost Tracking**
- Track LLM token usage and cost per run
- Budget limits per run/step
- Cost breakdown in reports

**Error Recovery**
- Resume interrupted runs from last checkpoint
- Retry failed LLM calls with backoff
- Partial results when some branches fail

**Observability**
- Structured logging with correlation IDs
- Metrics export (Prometheus/OpenTelemetry)
- Run duration and performance tracking

---

## Non-Goals (for now)

These are explicitly out of scope:

- **Autonomous decision-making** — Delibera produces artifacts for human review, not autonomous actions
- **Real-time streaming** — Runs are batch processes, not interactive streams
- **Multi-user collaboration** — Single-user or single-system runs only
- **Fine-tuning or training** — Delibera uses LLMs as-is, no model training

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. When working on roadmap items:

1. Open an issue to discuss approach before large PRs
2. Keep PRs focused on a single feature or fix
3. Add tests for all new functionality
4. Ensure `uv run pytest`, `uv run mypy src/`, and `uv run ruff check src/ tests/` pass
5. Respect architectural invariants (engine-only control, complete tracing, governed access)
