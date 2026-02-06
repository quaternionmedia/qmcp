# QMCP Development Roadmap

This document outlines the phased implementation plan for building a production-grade MCP server.

---

## Current State

- ✅ Project skeleton with `pyproject.toml`
- ✅ Documentation vision (architecture, overview, tools)
- ✅ Dependencies: `fastapi`, `click`, `sqlmodel`, `aiosqlite`, `httpx`, `metaflow`, `structlog`
- ✅ MCP server with tool discovery and invocation
- ✅ Persistence layer with audit logging
- ✅ Human-in-the-loop (HITL) endpoints
- ✅ Python client library (`qmcp.client`)
- ✅ Example Metaflow flows
- ✅ Structured logging with structlog
- ✅ Request tracing middleware (correlation IDs)
- ✅ Prometheus-compatible metrics endpoint
- ✅ 4 built-in tools
- ✅ Agent framework schemas + mixins
- ✅ PydanticAI integration for agent execution
- ✅ 148+ passing tests

---

## Phase 1: Foundation (Core MCP Server)

**Goal**: Minimal working MCP server with tool discovery and invocation.

### Deliverables

1. **Project Structure**
   ```
   qmcp/
   ├── __init__.py
   ├── server.py          # FastAPI app
   ├── cli.py             # Click CLI entrypoint
   ├── config.py          # Settings via pydantic-settings
   ├── tools/
   │   ├── __init__.py
   │   ├── registry.py    # Tool registration system
   │   └── builtin.py     # Example tools (echo, planner)
   └── schemas/
       ├── __init__.py
       └── mcp.py         # MCP protocol schemas
   ```

2. **MCP Endpoints**
   - `GET /health` – Health check
   - `GET /v1/tools` – List available tools
   - `POST /v1/tools/{tool_name}` – Invoke a tool

3. **CLI Commands**
   - `qmcp serve` – Start the MCP server
   - `qmcp tools list` – List registered tools

4. **Dependencies to Add**
   - `pydantic-settings` – Configuration management
   - `uvicorn` – ASGI server (included in fastapi[standard])

### Acceptance Criteria
- [x] Server starts via `uv run qmcp serve`
- [x] `GET /v1/tools` returns tool list
- [x] `POST /v1/tools/echo` returns echoed input
- [x] Basic tests pass (15/15)

---

## Phase 2: Persistence Layer

**Goal**: Add SQLModel-based persistence for audit and HITL support.

### Deliverables

1. **Database Models**
   ```
   qmcp/
   └── db/
       ├── __init__.py
       ├── engine.py       # SQLModel engine setup
       └── models.py       # ToolInvocation, HumanRequest, etc.
   ```

2. **Models**
   - `ToolInvocation` – Log of every tool call
   - `HumanRequest` – Pending human approval requests
   - `HumanResponse` – Completed human responses

3. **Dependencies to Add**
   - `sqlmodel` – ORM with Pydantic integration
   - `aiosqlite` – Async SQLite driver

### Acceptance Criteria
- [x] Tool invocations are logged to database
- [x] Database initializes on startup
- [x] Query endpoints for invocation history (`GET /v1/invocations`)

---

## Phase 3: Human-in-the-Loop ✅ COMPLETE

**Goal**: First-class HITL as described in architecture.

### Deliverables

1. **HITL Endpoints**
   - `POST /v1/human/requests` – Create approval request
   - `GET /v1/human/requests` – List requests with filtering
   - `GET /v1/human/requests/{id}` – Get request with response
   - `POST /v1/human/responses` – Submit human decision

2. **HITL Lifecycle**
   - Request creation with configurable timeout (default 1 hour)
   - Durable persistence in SQLite
   - Status tracking (pending, responded, expired)
   - Polling endpoint returns request + response together
   - Options validation for constrained choices

### Acceptance Criteria
- [x] Can create human request with timeout, options, context
- [x] Can submit response (validates against options if provided)
- [x] Can poll and receive request status + response
- [x] Expired requests are detected and marked
- [x] 15 HITL-specific tests passing

---

## Phase 4: Metaflow Client Integration ✅ COMPLETE

**Goal**: Example Metaflow flows demonstrating MCP client usage.

### Deliverables

1. **Client Library**
   ```
   qmcp/
   └── client/
       ├── __init__.py
       └── mcp_client.py   # HTTP client for MCP server
   ```

2. **Example Flows**
   ```
   examples/
   └── flows/
       ├── simple_plan.py       # Basic tool invocation
       └── approved_deploy.py   # HITL approval flow
   ```

3. **Dependencies Added**
   - `metaflow` – Workflow orchestration
   - `httpx` – HTTP client

### Acceptance Criteria
- [x] Python client library with full API coverage
- [x] Example flow calls MCP tools
- [x] Example flow demonstrates HITL
- [x] 16 client tests passing

---

## Phase 5: Production Hardening ✅ COMPLETE

**Goal**: Make the system production-ready.

### Deliverables

1. **Observability**
   ```
   qmcp/
   ├── logging.py      # Structured logging with structlog
   ├── middleware.py   # Request tracing middleware
   └── metrics.py      # Prometheus-compatible metrics
   ```

2. **Features Added**
   - JSON structured logging (production) / console logging (dev)
   - Request tracing with `X-Request-ID` and `X-Correlation-ID` headers
   - `/metrics` endpoint (Prometheus text format)
   - `/metrics/json` endpoint (JSON format)
   - HTTP request counters and latency histograms
   - Tool invocation metrics
   - HITL request metrics

3. **Testing**
   - Unit tests for tools
   - Contract tests for MCP routes
   - Client library tests
   - Metrics and observability tests

### Acceptance Criteria
- [x] Tests passing
- [x] Structured logs in JSON (production mode)
- [x] Request tracing with correlation IDs
- [x] Prometheus-compatible metrics

---

## Phase 6: Agent Framework (Schema + Mixins) ✅ COMPLETE

**Goal**: Provide agent schemas and capability mixins without server-side orchestration.

### Deliverables

1. **Agent Framework Models**
   ```
   qmcp/
   └── agentframework/
       ├── models.py     # AgentType, Topology, Execution, etc.
       └── mixins.py     # Capability mixins + registry
   ```

2. **Topology and Runner Registries (Skeletons)**
   ```
   qmcp/
   └── agentframework/
       ├── topologies.py
       └── runners.py
   ```

3. **Tests**
   - `tests/test_agentframework_models.py`
   - `tests/test_agentframework_mixins.py`

### Acceptance Criteria
- [x] Agent framework imports cleanly from `qmcp.agentframework`
- [x] Agent framework tests pass

---

## Phase 7: PydanticAI Integration ✅ COMPLETE

**Goal**: Integrate with PydanticAI for agent execution while preserving QMCP's unique capabilities.

### Deliverables

1. **Integration Module**
   ```
   qmcp/
   └── integrations/
       └── pydantic_ai/
           ├── __init__.py    # Public exports
           ├── models.py      # Model conversion utilities
           ├── agents.py      # Agent creation adapters
           └── toolsets.py    # QMCPToolset for server connection
   ```

2. **Features**
   - `model_to_pydantic_ai()` - Convert QMCP ModelConfig to PydanticAI string
   - `create_agent()` - Create PydanticAI agents from QMCP model configs
   - `QMCPToolset` - Connect PydanticAI agents to QMCP server with audit trail
   - `estimate_cost()` - Cost estimation using QMCP pricing metadata
   - `AgentBuilder` - Fluent API for agent configuration

3. **Documentation**
   - `docs/integrations/index.md` - Integration overview
   - `docs/integrations/pydantic-ai.md` - Full usage documentation

### Acceptance Criteria
- [x] PydanticAI agents can be created from QMCP model configs
- [x] QMCPToolset connects agents to QMCP server
- [x] HITL support through toolset
- [x] 15 integration tests passing

---

## Prioritization Rationale

| Phase | Value | Complexity | Dependencies |
|-------|-------|------------|--------------|
| 1     | High  | Low        | None         |
| 2     | Medium| Medium     | Phase 1      |
| 3     | High  | Medium     | Phase 2      |
| 4     | Medium| Medium     | Phase 1, 3   |
| 5     | High  | High       | All          |

**Start with Phase 1** – it delivers immediate value and validates the architecture.

---

## Current Sprint: ALL PHASES COMPLETE ✅

All 7 phases are complete. QMCP is a production-ready MCP server with PydanticAI integration:

**Phase Summary:**
| Phase | Description | Tests |
|-------|-------------|-------|
| 1. Foundation | Core MCP server | 20 |
| 2. Persistence | SQLite audit logging | 6 |
| 3. HITL | Human-in-the-loop | 15 |
| 4. Client | Python client + Metaflow | 16 |
| 5. Hardening | Observability + metrics | 18 |
| 6. Agent Framework | Schemas + mixins | 47 |
| 7. PydanticAI | Agent runtime integration | 15 |
| **Total** | | **137+** |

**Production Features:**
- Structured JSON logging (structlog)
- Request tracing with correlation IDs
- Prometheus-compatible `/metrics` endpoint
- PydanticAI agent execution with QMCP audit trail
- Comprehensive test coverage

**Next Steps (Future Work):**
- Add authentication/authorization
- Webhook notifications for HITL
- Redis/PostgreSQL backend options
- Kubernetes deployment manifests
- Implement agent topology execution and runner orchestration
