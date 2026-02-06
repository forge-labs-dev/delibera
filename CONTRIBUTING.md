# Contributing to Delibera

Thank you for your interest in contributing to Delibera. This document provides guidelines for contributing safely and effectively.

## Repository Structure

```
delibera/
├── src/delibera/           # Main source code
│   ├── engine/             # Orchestrator, operators, tree, state
│   ├── protocol/           # Protocol spec, loader, interpreter, defaults
│   ├── agents/             # Stubs + LLM proposer
│   ├── epistemics/         # Claims, evidence, objections, ledger, validation
│   ├── retrieval/          # Keyword, embedding, web, hybrid retrievers + verification
│   ├── llm/                # LLM client protocol, Gemini provider, prompts, redaction
│   ├── tools/              # Tool registry, router, policy, built-ins
│   ├── gates/              # Gate models, predicates, handlers, response application
│   ├── trace/              # Events, writer, reader, replay, validation
│   ├── scoring/            # Scoring metrics and weights
│   ├── inspect/            # Run summarization and report rendering
│   ├── eval/               # Evaluation harness, suite runner, metrics
│   └── cli.py              # CLI entry point
├── tests/                  # Test suite (411 tests)
├── docs/                   # Design documentation
├── protocols/              # Example protocol YAML files
├── suites/                 # Evaluation suites
└── evidence/               # Default evidence pack
```

## Invariants You Must Respect

These invariants are **non-negotiable**. PRs violating them will be rejected.

### 1. Engine-Only Control

The engine alone controls:
- Tree structure (expand, prune, reduce)
- Run termination (convergence)
- Operator sequencing

Agents **must not**:
- Mutate tree state
- Call operators directly
- Bypass policy checks

### 2. Replay Must Be Read-Only

Replay reconstructs runs from traces without:
- Calling agents
- Calling tools
- Modifying any state

If your change breaks replay determinism, it's a bug.

### 3. Tracing Must Be Complete

Every significant action must emit a trace event:
- All operator applications
- All agent outputs
- All tool calls (requested, executed, denied)
- All gate interactions

If something happens but isn't traced, it's invisible to audit and replay.

### 4. Network Access Is Governed

Network-capable features (web retrieval, LLM calls, URL verification) must:
- Go through the retrieval or LLM abstraction layers
- Be opt-in via CLI flags (e.g., `--retrieval-method web`, `--verify`)
- Never be called during replay or validation steps
- Be fully traced for auditability

Do not add uncontrolled network access outside these abstractions.

## How to Add Things

### Adding a New Agent Stub

1. Create a new class in `src/delibera/agents/stub.py` or a new file
2. Implement the agent interface (see existing stubs)
3. Register it in the appropriate factory
4. Add tests proving it produces valid output
5. Ensure it doesn't call operators or mutate state

### Adding a New Tool

1. Create a tool class implementing `ToolSpec` protocol
2. Define `name`, `risk_level`, `capability`, and `execute()`
3. Register in `create_default_registry()` if built-in
4. Add policy rules if needed
5. Add tests for validation and execution

### Adding a New Protocol

1. Create a YAML file in `protocols/`
2. Follow the schema (see `tree_v1.yaml` as reference)
3. Test with `delibera run --protocol your_protocol.yaml`
4. Ensure all steps have corresponding agent implementations

### Adding a New Retriever

1. Implement the `EvidenceRetriever` protocol from `retrieval/base.py`
2. Implement `retrieve(query, max_results) -> list[RetrievalResult]`
3. Register in `create_retriever()` factory in `retrieval/__init__.py`
4. Add CLI option in `cli.py` if user-selectable
5. Add tests with mocked external calls

### Adding a New LLM Provider

1. Implement the `LLMClient` protocol from `llm/base.py`
2. Implement `generate(LLMRequest) -> LLMResponse`
3. Register in the CLI factory logic
4. Add tests with a fake client (no real API calls in CI)
5. Ensure replay works (LLM calls are traced, not re-invoked)

### Adding a New Gate Type

1. Define the gate type in `gates/models.py`
2. Implement predicate in `gates/predicates.py`
3. Add handler support in `gates/handler.py`
4. Add tests for trigger and response handling

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_v0_run.py

# Run with coverage
uv run pytest --cov=delibera
```

## Code Quality

```bash
# Type checking (required to pass)
uv run mypy src/

# Linting (required to pass)
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/
```

## PR Requirements

Before submitting a PR:

1. **Tests required**: Add tests for new functionality
2. **Tests pass**: `uv run pytest` must pass
3. **Types pass**: `uv run mypy src/` must pass
4. **Lint passes**: `uv run ruff check src/ tests/` must pass
5. **Determinism**: New code must not introduce non-determinism
6. **No behavior changes**: Unless explicitly intended, PRs should not change deliberation semantics

## Determinism Guidelines

Delibera requires deterministic behavior for replay. Avoid:

- `dict` iteration order assumptions (use `sorted()`)
- Timestamp-based sorting without stable tiebreakers
- Random number generation without seeding
- File system order assumptions

Good pattern:
```python
# Sort by stable key
items = sorted(items, key=lambda x: (x.score, x.id))
```

Bad pattern:
```python
# Non-deterministic order
for key in some_dict:
    process(key)
```

## Commit Messages

Use clear, descriptive commit messages:

```
Feat: add tradeoff gate for close-score decisions

- Add predicate checking score difference
- Add gate handler for weight selection
- Add tests for trigger and response
```

Prefixes:
- `Feat:` — New feature
- `Fix:` — Bug fix
- `Refactor:` — Code restructuring
- `Docs:` — Documentation only
- `Test:` — Test-only changes
- `Chore:` — Maintenance tasks

## Getting Help

- Read the [docs/](docs/README.md) for architecture and design
- Check existing tests for usage examples
- Open an issue for design questions before large PRs

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
