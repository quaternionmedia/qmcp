# Architecture

This document is the authoritative description of the system architecture.
If code, tests, or future changes conflict with this document, the code is wrong.

---

## Architectural Goal

Provide a spec-aligned MCP server that can safely support:
- Multi-step workflows
- Human-in-the-loop decision points
- Deterministic, auditable execution

While remaining:
- Simple
- Testable
- Maintainable over time

---

## High-Level Components

The system is intentionally split into three planes:

+------------------------------+
| Client / Orchestration Plane | <- Metaflow
+--------------+---------------+
               |
          MCP (HTTP / stdio)
+--------------v---------------+
| MCP Server Plane             | <- FastAPI
+--------------+---------------+
               |
+--------------v---------------+
| Execution / Storage          | <- Tools, DB
+------------------------------+

Each plane has strict responsibilities.

---

## Plane 1: Client / Orchestration (Metaflow)

### Responsibilities

- Define workflows (DAGs)
- Call MCP tools
- Request and await human input
- Branch based on results
- Own all business logic
- Define agent types and topologies using the agent framework models

### Constraints

- Must not import server-side code
- Must treat MCP server as a black box
- Must be replayable and deterministic

### Why Metaflow

Metaflow provides:
- Explicit step boundaries
- Durable state
- Retry and resume semantics
- Clear audit trails

Metaflow is the MCP client, not part of the server.

### Local Dev Runners

Metaflow flows can run:
- Directly on Linux/macOS with `uv run`
- In a Linux container on Windows (recommended) using Docker or k3s

Local dev flows may chain local LLM agents with PydanticAI and persist artifacts
with SQLModel. This orchestration stays in the client plane.

### PydanticAI Integration

QMCP integrates with [PydanticAI](https://ai.pydantic.dev/) for agent execution:
- PydanticAI provides the agent runtime (tools, deps injection, retries)
- QMCP provides model metadata, audit trails, and HITL
- The `QMCPToolset` connects PydanticAI agents to the MCP server

See [PydanticAI Integration](integrations/pydantic-ai.md) for usage.

### Agent Framework

The agent framework is a client-side schema and mixin layer:
- SQLModel tables define agent types, topologies, and executions
- Mixins provide optional capabilities (memory, reasoning, HITL)
- Multi-agent topologies orchestrate agent collaboration
- No server-side orchestration is implemented; the server only persists data

See [Agent Framework](agentframework/overview.md) for current implementation status.

---

## Plane 2: MCP Server (FastAPI)

### Responsibilities

- Expose MCP-compatible endpoints
- Register and serve tools
- Persist human requests and responses
- Persist interaction summaries
- Enforce schemas and contracts

### Constraints

- No orchestration logic
- No multi-step flows
- No agent chaining (in the server)
- No hidden state

The server should remain boring.

---

## Plane 3: Execution and Storage

### Tools

- Stateless
- Synchronous by default
- Side effects only when explicitly intended
- One responsibility per tool

Tools do not:
- Call other tools
- Wait for human input
- Store global state

---

## Human-in-the-Loop Architecture

Humans are modeled as explicit MCP resources.

### Human interaction lifecycle

1. Client creates a human request
2. Server persists the request
3. Human responds via UI or API
4. Server persists the response
5. Client polls and continues

### Design Rationale

- Explicit over implicit
- Auditable over convenient
- Durable over ephemeral

No human decision should disappear into logs.

---

## Data Flow Summary

User Input ->
Metaflow Step ->
MCP Tool Call ->
Tool Execution ->
(Optional) Human Approval ->
Next Step

There are no hidden loops, callbacks, or side channels.

---

## Failure Modes and Handling

### Tool Failure
- Propagated to client
- Client decides retry or abort

### Human Timeout
- Client enforces timeout
- Flow aborts or compensates

### Server Restart
- Tools remain stateless
- Human requests remain persisted
- Client can resume safely

---

## Non-Goals (Architectural)

The following are explicitly excluded:
- Autonomous agent loops
- Self-modifying flows
- Hidden memory
- Implicit context sharing
- Model-specific behavior leaks

These may be explored only if the architecture remains intact.

---

## Architectural Invariants (Must Hold)

- Server does not orchestrate
- Client owns control flow
- Tools are stateless
- Humans are explicit
- Everything is testable

If any invariant is violated, stop and refactor.
