# Contributing Guidelines

This document defines how changes should be made safely.

---

## Guiding Principle

> Do not surprise the next maintainer.

---

## Onboarding for Contributors

Start here after cloning the repo:

1. `uv sync --all-extras`
2. `uv run pytest`
3. Run the end-to-end HITL tutorial (mirrors
   `tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow`):

```bash
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

4. Review `docs/agentframework/overview.md` and run agent framework tests:

```bash
uv run pytest tests/test_agentframework_models.py tests/test_agentframework_mixins.py -v
```

---

## Before Making Changes

Ask:
- Does this respect MCP boundaries?
- Does this move logic into the wrong plane?
- Can this be tested?
- Can this be explained simply?

If unsure, choose the simpler option.

---

## Required for All Changes

- Tests covering new behavior
- Docs updated if behavior changes
- No breaking changes to MCP routes without versioning

---

## What Requires Extra Scrutiny

- New orchestration logic
- New persistence models
- New long-lived state
- Anything labeled “agent”

These often indicate architectural drift.

---

## What Is Encouraged

- Smaller tools
- Clearer schemas
- Better examples
- Better test coverage

---

## Development Setup

```bash
# Clone and install
git clone <repo>
cd qmcp
uv sync --all-extras

# Run tests (with auto cleanup)
uv run qmcp test -v

# Run tests with coverage
uv run qmcp test --coverage

# Run specific test file
uv run qmcp test tests/test_hitl.py

# Smoke-check example flows (skips Metaflow flows on Windows)
uv run pytest tests/test_examples_smoke.py

# Cookbook CLI tests
uv run pytest tests/test_cli_cookbook.py

# Run linter
uv run ruff check .

# Start dev server
uv run qmcp serve --reload
```

---

## Project Structure

```
qmcp/
├── __init__.py       # Package metadata
├── cli.py            # Click CLI interface
├── config.py         # Pydantic settings
├── server.py         # FastAPI MCP server
├── schemas/          # Pydantic models for MCP
│   └── mcp.py
├── tools/            # Tool system
│   ├── registry.py   # Tool registration
│   └── builtin.py    # Built-in tools
├── db/               # Database persistence
│   ├── engine.py     # Async SQLModel engine
│   └── models.py     # ToolInvocation, HumanRequest, HumanResponse
├── agentframework/   # Agent types, mixins, topologies
│   ├── models/       # Data models and registry
│   ├── mixins.py     # Capability mixins
│   ├── topologies.py # Collaboration patterns
│   └── runners.py    # Execution environments
└── integrations/     # External library integrations
    └── pydantic_ai/  # PydanticAI adapter

tests/
├── conftest.py       # Shared fixtures (DB isolation)
├── test_db.py        # Database model tests
├── test_hitl.py      # Human-in-the-loop tests
├── test_server.py    # API endpoint tests
├── test_tools.py     # Tool registry tests
├── test_agentframework_*.py  # Agent framework tests
└── test_pydantic_ai_integration.py  # PydanticAI integration tests
```

---

## Adding a New Tool

1. Add to `qmcp/tools/builtin.py` (or create a new module)
2. Use the `@tool_registry.register()` decorator
3. Follow the tool design rules:
   - Stateless
   - Single responsibility
   - No side effects unless required
   - No calling other tools

Example:
```python
@tool_registry.register(
    name="my_tool",
    description="What this tool does",
    input_schema={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."}
        },
        "required": ["param"],
    },
)
def my_tool(params: dict) -> Any:
    return params.get("param")
```

---

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Test changes
- `refactor:` Code refactoring

Example: `feat: add human request endpoint`
