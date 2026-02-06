# Delibera Documentation

This directory contains the design documentation for Delibera, an engine for decision-grade AI deliberation.

## How to Read These Docs

**If you want to understand what Delibera is:**
Start with [Vision](vision.md). It explains the problem, the approach, and the design principles.

**If you want to understand how it works:**
Read [Architecture](architecture.md) for system structure, then [Formalism](formalism.md) for the formal model.

**If you want to use Delibera:**
- [Protocols](protocol.md) — How to define deliberation flows
- [User Gates](user-gates.md) — How to add human checkpoints
- [Tracing and Replay](tracing-and-replay.md) — How to inspect and audit runs

**If you want to extend Delibera:**
- [Tooling and Policy](tooling-and-policy.md) — How to add tools
- [Epistemics](epistemics.md) — How claims and evidence work

## Document Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [Vision](vision.md) | Why Delibera exists | First |
| [Formalism](formalism.md) | Formal model, definitions, invariants | Reference |
| [Architecture](architecture.md) | System structure, module boundaries | Understanding internals |
| [Protocols](protocol.md) | Protocol YAML specification | Defining workflows |
| [Epistemics](epistemics.md) | Claims, evidence, objections | Understanding validation |
| [Tooling and Policy](tooling-and-policy.md) | Tool integration, policy layers | Adding tools |
| [User Gates](user-gates.md) | Human-in-the-loop checkpoints | Adding interaction |
| [Tracing and Replay](tracing-and-replay.md) | Trace format, replay semantics | Debugging, auditing |
| [Glossary](glossary.md) | Term definitions | Quick lookup |

## Recommended Reading Order

For a complete understanding, read in this order:

1. **[Vision](vision.md)** — The "why"
2. **[Formalism](formalism.md)** — The canonical model
3. **[Architecture](architecture.md)** — The implementation structure
4. **[Protocols](protocol.md)** — Workflow specification
5. **[Epistemics](epistemics.md)** — Knowledge tracking
6. **[Tooling and Policy](tooling-and-policy.md)** — External capabilities
7. **[Tracing and Replay](tracing-and-replay.md)** — Auditability
8. **[User Gates](user-gates.md)** — Human interaction

## Key Invariants

These invariants are enforced throughout the system:

1. **Engine-only control** — Only the engine modifies tree structure and determines termination
2. **Explicit epistemics** — Claims, evidence, and objections are first-class objects
3. **Deterministic convergence** — Runs stop based on measurable predicates
4. **Complete tracing** — Every action is recorded for replay
5. **Policy-governed access** — All external access (tools, LLMs, web) is controlled and opt-in

Violating these invariants is a correctness bug.

## See Also

- [README.md](../README.md) — Project overview and quick start
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Contributor guidelines
- [protocols/](../protocols/) — Example protocol files
