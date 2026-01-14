# Delibera

An engine for **decision-grade AI deliberation**.

Delibera is an open-source framework that makes multi-agent reasoning **structured, governed, and auditable**. Unlike chat-based AI systems, Delibera treats deliberation as a process—not a conversation. The engine controls tree expansion, pruning, and convergence while agents contribute content. Every run can be replayed without re-invoking LLMs.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Engine** | Central orchestrator that controls tree structure and termination. Agents propose; the engine decides. |
| **Protocol** | Declarative YAML spec defining expansion rules, branch pipelines, pruning, and convergence criteria. |
| **Epistemics** | Explicit tracking of claims, evidence, and objections. Fact claims require evidence; inferences must follow from supported facts. |
| **Gates** | Structured human-in-the-loop checkpoints for scope clarification, tradeoffs, and final approval. |
| **Replay** | Full runs can be reconstructed from trace logs without calling agents or tools. |

## Quick Start

### Installation

```bash
# Install from PyPI
pip install delibera

# Or clone and install with uv
git clone https://github.com/forge-labs-dev/delibera.git
cd delibera
uv sync
```

### Run a Deliberation

```bash
# Basic run with interactive gates (uses stub agents)
delibera run --question "Should we adopt uv for dependency management?"

# Run with auto-approved gates (for CI/scripts)
delibera run --question "Should we adopt uv?" --auto-approve-gates

# Run with a custom protocol
delibera run --question "Your question" --protocol protocols/tree_v1.yaml
```

### Run with LLM-Backed Proposer

To use an LLM for generating proposals (instead of deterministic stubs):

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key"

# Run with LLM proposer
delibera run --question "Should we adopt uv?" --use-llm-proposer --auto-approve-gates

# Specify model and parameters
delibera run --question "Your question" \
  --use-llm-proposer \
  --llm-model gemini-1.5-pro \
  --llm-temperature 0.3 \
  --auto-approve-gates
```

**Note:** LLM mode requires the `google-generativeai` package. Install with:
```bash
pip install delibera[llm]
```

Replay and inspection work identically whether the run used LLM or stubs - replay never re-invokes the LLM.

### Inspect a Run

```bash
# Print a human-readable summary
delibera inspect --run-id <run_id>

# Generate a Markdown report
delibera report --run-id <run_id> --out report.md
```

### Replay a Run

```bash
# Validate trace and artifact consistency
delibera replay --run-id <run_id>
```

### Run Evaluation Suites

```bash
# Run an evaluation suite
delibera eval --suite suites/basic.yaml

# Save results to JSON
delibera eval --suite suites/basic.yaml --save-results results.json
```

## Example Protocol

```yaml
name: simple_protocol
protocol_version: v1
max_depth: 1
gates_enabled: true

expand_rules:
  - id: expand_options
    at_step_id: plan
    child_kind: option
    max_children: 3
    depth: 1

branch_pipeline:
  - id: propose
    kind: work
    step_name: PROPOSE
    role: proposer

  - id: research
    kind: work
    step_name: RESEARCH
    role: researcher

  - id: validate
    kind: validate
    step_name: CLAIM_CHECK

prune:
  rule: epistemic_then_score
  keep_k: 2

reduce:
  rule: merge_artifacts

convergence:
  max_rounds: 0
```

## What Delibera Is Not

- **Not an agent framework** — Delibera is not LangChain, CrewAI, or AutoGPT. It's a deliberation engine with strict governance.
- **Not a workflow orchestrator** — Delibera is not Airflow or Prefect. It's specifically for reasoning processes that require epistemic tracking.
- **Not autonomous** — Delibera does not "decide for you". It produces structured decision artifacts for humans to review.
- **Not a chatbot** — Outputs are artifacts, not conversations.

## Documentation

See the [docs/](docs/README.md) directory for detailed documentation:

- [Vision](docs/vision.md) — Why Delibera exists
- [Architecture](docs/architecture.md) — System structure and invariants
- [Formalism](docs/formalism.md) — Formal model and terminology
- [Protocols](docs/protocol.md) — Protocol specification
- [Epistemics](docs/epistemics.md) — Claims, evidence, and validation
- [Tooling and Policy](docs/tooling-and-policy.md) — Tool access governance
- [User Gates](docs/user-gates.md) — Human-in-the-loop checkpoints
- [Tracing and Replay](docs/tracing-and-replay.md) — Audit and replay

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidelines.

## Roadmap

- **v0.1** (current) — Core engine, protocols, epistemics, replay, evaluation
- **v0.2** — Strata integration for artifact persistence and lineage tracking
- **v0.3** — Remote tool execution with policy sandboxing
- **v1.0** — Production hardening, performance optimization

## License

See [LICENSE](LICENSE) file for details.
