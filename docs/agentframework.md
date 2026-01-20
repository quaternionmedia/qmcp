# Agent Framework

The agent framework is a schema and mixin layer for defining agent types,
topologies, and execution records. It is designed to live in the client
plane and persist data in the same QMCP database, without adding orchestration
logic to the server.

## What Is Implemented

- SQLModel tables and schemas in `qmcp.agentframework.models`
- Capability mixins in `qmcp.agentframework.mixins`
- Topology and runner registries (skeletons) in `qmcp.agentframework.topologies` and `qmcp.agentframework.runners`
- Tests for models and mixins in `tests/`

## What Is Planned (Design Docs)

- Topology execution semantics (`qmcp/agentframework/03-TOPOLOGIES.md`)
- Runner orchestration patterns (`qmcp/agentframework/04-RUNNERS.md`)
- Extended test plan (`qmcp/agentframework/05-TESTS.md`)

These documents are design references. The repository includes skeleton
registries for topologies and runners, but execution is not implemented yet.

## Copy-Paste Quickstart

Validate the agent framework models and mixins:

```bash
uv run pytest tests/test_agentframework_models.py tests/test_agentframework_mixins.py -v
```

For an end-to-end run that exercises the server, database, and HITL flow,
use the tutorial in `README.md`:

```bash
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

## Where to Look Next

- `qmcp/agentframework/00-MASTER-PLAN.md` for the overall design
- `docs/architecture.md` for system boundaries and integration points
