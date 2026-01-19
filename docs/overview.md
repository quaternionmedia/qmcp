# MCP Service – Overview

This project implements a **Model Context Protocol (MCP) server** with:

- A **spec-aligned MCP server** (FastAPI)
- A **Python client library** (`qmcp.client.MCPClient`) for programmatic access
- **Metaflow integration** with ready-to-use example flows
- **Human-in-the-loop (HITL)** as a first-class capability
- **Persistence** with SQLite for audit and durability
- Strong emphasis on **clarity, correctness, and testability**

This is not an autonomous agent system.
This is not an experimentation playground.

It is a **boring, reliable MCP implementation** intended for real-world use.

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
| 5. Hardening | 🔄 Next | Production observability |

See [ROADMAP.md](ROADMAP.md) for details.

---

## Design Philosophy

- Prefer explicit schemas over implicit context
- Prefer examples over abstractions
- Prefer boring patterns over clever ones
- Optimize for maintainability over novelty

---

## Auditability and Accountability

QMCP is designed to provide a durable audit trail:
- Every tool invocation is recorded with input, output, status, and timestamps.
- Human requests and responses are persisted, including who responded.
- Clients can supply a `X-Correlation-ID` to tie logs, requests, and database records together.
