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

Quickstart: see `../quickstart.md` for a copy-paste walkthrough.

---

## What This System Does

### Server Capabilities

- Exposes MCP-compatible **tools** over HTTP
- Allows MCP clients to:
  - Discover tools (`GET /v1/tools`)
  - Invoke tools (`POST /v1/tools/{name}`)
  - Query invocation history (`GET /v1/invocations`)
  - Request human input or approval (`POST /v1/human/requests`)
  - Poll for human responses (`GET /v1/human/requests/{id}`)
- Serves the topology vocabulary to a window that must not carry its own copy:
  - Draw a shape (`GET /v1/topology/shape/{kind}`) and build a form from its
    configuration class (`GET /v1/topology/schema/{kind}`)
  - Ask what every shape would do (`GET /v1/orchestration/plane`) and what a
    hand could run now (`GET /v1/orchestration/runnable`)
  - Keep designs (`/v1/topologies`), each answered with the plane's verdict --
    a refused shape saves, because designing is not an act and running is
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

## What This System Explicitly Does NOT Do

- ❌ The MCP server does not orchestrate workflows
- ❌ The server does not manage agents or chains
- ❌ Tools do not call each other
- ❌ There is no autonomous loop
- ❌ There is no hidden shared state

All orchestration lives in the **client**, not the server.

---

## Mental Model

```
User / UI / IDE
       ↓
Metaflow (MCP Client)
       ↓
MCP Server (FastAPI)
       ↓
Tools / Human Requests / Database
```

If logic appears above the MCP server boundary, it is correct.
If logic appears below it, it is likely a bug.

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
