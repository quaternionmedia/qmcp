# MCP Service – Overview

This project implements a **Model Context Protocol (MCP) server** with:

- A **spec-aligned MCP server** (FastAPI)
- A **Python client library** (`qmcp.client.MCPClient`) for programmatic access
- **Metaflow integration** with ready-to-use example flows
- **Human-in-the-loop (HITL)** as a first-class capability
- **Persistence** with SQLite for audit and durability
- **Agent framework** schemas and mixins for modeling agent types and topologies
- Strong emphasis on **clarity, correctness, and testability**

This is not an autonomous agent system.
This is not an experimentation playground.

It is a **boring, reliable MCP implementation** intended for real-world use.

---

Setting it up: [`setup/README.md`](setup/README.md), for the role at hand.
The roles and their boundaries: [`roles.md`](roles.md).

---

## What This System Does

### Server Capabilities

- Exposes MCP-compatible **tools** over HTTP
- Allows MCP clients to:
  - Discover tools (`GET /v1/tools`)
  - Invoke tools (`POST /v1/tools/{name}`)
  - Query invocation history (`GET /v1/invocations`)
  - Request human input or approval (`POST /v1/human/requests`)
  - Read a request with its answer (`GET /v1/human/requests/{id}`). The read
    stores the expiry of a pending request past its `expires_at`, so a loop
    watching for answers lists them (`GET /v1/human/requests`) rather than
    reading each one
- Records instructions people give against a rostered project, asks consent
  through the human queue, acts in the project's clone through a runtime, and
  hands each act the project's earlier instructions, on loopback
  (`/v1/instructions`; [`voice-loop-demo.md`](voice-loop-demo.md))
- Serves the topology vocabulary to a window that must not carry its own copy:
  - Draw a shape (`GET /v1/topology/shape/{kind}`) and build a form from its
    configuration class (`GET /v1/topology/schema/{kind}`)
  - Ask what every shape would do (`GET /v1/orchestration/plane`) and what a
    hand could run now (`GET /v1/orchestration/runnable`)
  - Keep designs (`/v1/topologies`), each answered with the plane's verdict --
    a refused shape saves, because designing is not an act and running is
- Archives conversations with assistants as units of work, on loopback
  (`/v1/threads`; `docs/threads.md`)
- Reports on itself: a dashboard, deterministic audits, a self-check of its
  own gates and a feedback loop (`docs/self-checks.md`)
- Provides durability via:
  - SQLite persistence for all interactions
  - Audit trails for every tool invocation
  - Persistent human request/response tracking

### Client Library

- Full Python client (`qmcp.client.MCPClient`) with:
  - Tool discovery and invocation
  - HITL request/response handling
  - Blocking wait for human responses
  - Invocation history queries
  - Custom exception handling

### Example Flows

- **simple_plan.py** - Basic tool invocation from Metaflow
- **approved_deploy.py** - HITL approval workflow with human gating

### Agent Framework (Schema + Mixins)

The agent framework provides:
- SQLModel tables for agent types, topologies, and executions
- Mixins for optional capabilities such as memory, reasoning, and HITL
- Multi-agent topologies (Debate, Pipeline, Ensemble, etc.)

Runtime orchestration is intentionally not implemented on the server.
See [Agent Framework](agentframework/overview.md) for implementation status.

### PydanticAI Integration

QMCP integrates with [PydanticAI](https://ai.pydantic.dev/) for agent execution:
- `create_agent()` - Create PydanticAI agents from QMCP model configs
- `QMCPToolset` - Connect agents to QMCP server with full audit trail
- Model metadata (pricing, limits, capabilities) preserved

See [PydanticAI Integration](integrations/pydantic-ai.md) for usage.

### Local Dev Workflows

- **local_agent_chain.py** - Local LLM plan -> review -> refine
- **local_qc_gauntlet.py** - QC checklist and gate generation
- **local_release_notes.py** - Release notes + doc update suggestions

These flows chain local LLM agents with PydanticAI and persist artifacts with
SQLModel. On Windows, run them in a Linux container (Docker or k3s) and set
`MCP_URL` / `LLM_BASE_URL` to point at local services.

---

## What This System Does Not Do

- ❌ Tools do not call each other
- ❌ Nothing answers a human request on a person's behalf
- ❌ Nothing acts on an instruction before a person says "approve" to its
  consent
- ❌ The server does not run flows: the cookbook's flows run in Docker as its
  clients
- ❌ It does not display its record: dossier shows it, codecartographer draws
  its shapes, and joe's page shows the spoken conversation

The server does start processes of its own: `serve --converse` runs the
standing conversation beside it, and acting on an instruction runs the act in
a process of its own. [`roles.md`](roles.md) draws the stack around it and
says what each role owns.

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Foundation | ✅ Complete | Core MCP server with tools |
| 2. Persistence | ✅ Complete | SQLite with audit logging |
| 3. HITL | ✅ Complete | Human-in-the-loop endpoints |
| 4. Client | ✅ Complete | Python client + Metaflow examples |
| 5. Agent Framework | ✅ Complete | Models, mixins, topologies |
| 6. PydanticAI | ✅ Complete | Integration with PydanticAI agents |
| 7. Hardening | 🔄 Next | Production observability |

See [ROADMAP.md](ROADMAP.md) for details.

---

## Design Philosophy

- Prefer explicit schemas over implicit context
- Prefer examples over abstractions
- Prefer boring patterns over clever ones
- Optimize for maintainability over novelty

---

## Adoption Checklist

- Decide hosting model (local, container, VM) and who can reach it.
- Set `QMCP_HOST`, `QMCP_PORT`, and `QMCP_DATABASE_URL` for the environment.
- Standardize `X-Correlation-ID` formatting for audit trails.
- Decide how humans submit HITL responses (UI or API).
- Connect `/metrics` to monitoring.

Copy-pasteable end-to-end tutorial:

```bash
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

---

## Auditability and Accountability

QMCP is designed to provide a durable audit trail:
- Every tool invocation is recorded with input, output, status, and timestamps.
- Human requests and responses are persisted, including who responded.
- Clients can supply a `X-Correlation-ID` to tie logs, requests, and database records together.
