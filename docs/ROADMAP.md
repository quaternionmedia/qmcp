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

## Phase 8: Composable Cookbook & MetaflowRunner ✅ COMPLETE

**Goal**: Extract duplicated flow patterns into `qmcp.cookbook` as composable, first-class modules and wire the MetaflowRunner to execute flows.

### Deliverables

1. **Cookbook Package**
   ```
   qmcp/
   └── cookbook/
       ├── __init__.py          # Re-exports for convenience
       ├── agent_builders.py    # LocalLLMConfig, build_local_agent, build_qmcp_agent
       ├── persistence.py       # FlowPersistence context manager + SQLModel entities
       └── mcp_tools.py         # MCPToolInvoker + tool input models
   ```

2. **MetaflowRunner** (`qmcp/agentframework/runners.py`)
   - Resolves topology type → flow script via `_TOPOLOGY_FLOW_MAP`
   - Executes flows via subprocess with timeout handling
   - Injects `--mcp-url` and `METAFLOW_USER` automatically
   - Returns `RunResult` with execution status, output, and timing

3. **Flow Refactoring**
   - All 4 flows (`local_agent_chain`, `local_qc_gauntlet`, `local_release_notes`, `council_deliberation`) import from `qmcp.cookbook`
   - Deleted `local_dev_db.py` and `local_mcp.py` (no shims)
   - Council flow now uses `FlowPersistence` for deliberation record tracking

4. **Tests**
   - `tests/test_cookbook.py` — agent builders, persistence lifecycle, MCP invoker
   - `tests/test_metaflow_runner.py` — registry, flow resolution, command building, async run

### Acceptance Criteria
- [x] `from qmcp.cookbook import build_local_agent, FlowPersistence, MCPToolInvoker` works
- [x] All flows import from `qmcp.cookbook` (no local helper references)
- [x] MetaflowRunner resolves topologies and launches flows
- [x] Cookbook modules auto-available in Docker runner (copied with `qmcp/`)
- [x] New tests pass

---

## Phase 9: The Designer's Seam

The routes a window draws from, so that no window carries its own copy of what
a shape is or what running one would do. `docs/agentframework/topologies.md`,
*Over HTTP, for a designer*, is the reference; `walkthrough/07-saving-a-shape-is-not-running-it.md`
exercises every route. The window that reads them is codecartographer, and the
plan they serve is the corpus's `plans/the-web-window.md`; its
`handbook/handoffs/the-web-window.md` names what each remaining phase asks of
this server.

### Deliverables

- [x] `GET /v1/orchestration/plane` and `GET /v1/orchestration/runnable` -- the plane as a document, and what a declared hand could run
- [x] `GET /v1/topology/schema/{kind}` -- the form a designer builds from
- [x] `/v1/topologies` (POST, GET, GET one, PUT) over the table that existed with no route; a refused shape saves and says so
- [x] The `topology` address kind, ahead of the corpus's grammar; the shared-vectors test reads the corpus's vector once it exists
- [ ] **An execution route.** Run a shape whose plane status is `runs` against declared workers and a declared budget, through the governed seam; the spend declared and consented here (`records/DRAFT-no-unattended-spending.md` in the corpus), the refusal asked of `orchestration.refuses` at run. Nothing a window sends turns a draft into a decision
- [ ] **An event stream** (`/v1/events`), after the window's polling overlay exists -- one more surface to govern, so it comes second by decision
- [ ] Answering the human queue from a web window is **deferred by decision** in the corpus's plan; build nothing for it until it has been reviewed

### Acceptance Criteria

- [x] Every designer route is documented and walked through in `walkthrough/07`
- [ ] The execution route's walkthrough shows a run refused before it spends, and a run consented before it starts
- [ ] The corpus's `project-seed/address-vectors.json` carries a `topology` vector and `tests/test_addresses.py` passes against it

---

## Phase 10: The Spoken Instruction

The voice loop answers questions an agent asks: a closed choice, spoken and
recorded on the human queue (`docs/integrations/voice.md`). This phase runs the
other direction. A person speaks an instruction, it is recorded against a
project, an agent acts on it in that project's clone, and the result is spoken
back -- so work continues across sessions and repositories from the record this
server keeps, not from whichever conversation happens to be open.

What it composes already exists: joe records until the speaker stops and
transcribes (`/api/voice/listen`, with the cap and the pause both adjustable);
vox carries the engine contract and the synthesizer; the human queue holds
consent; `qmcp.spend` and `qmcp.governed` refuse unconsented spending; and the
thread archive (`/v1/threads`, `qmcp threads consolidate`) reads which session
worked in which checkout, on which branch, toward which pull requests, and which
projects a thread is about.

### Deliverables

One pull request each, in this order; each is useful before the next exists.

- [ ] **Free-text answers by voice.** `run_once` asks every request as a closed choice and gives a request without `options` the options approve and reject, so an `input` request answered "yes" records `approve`. A request without options takes the transcript as its answer, read back once for confirmation
- [ ] **An agent can ask.** `qmcp_mcp.py` lists and answers the human queue but cannot create a request or wait on one. A tool that creates one and then polls the pending listing -- which expires nothing -- until it is answered or expires lets a session in any repository put a question to the voice loop
- [ ] **"Where was I?"** A spoken summary of a project's latest work, read from the thread archive: the session, its branch, its pull requests, its last turn. Read-only; no agent runs and nothing is spent
- [ ] **An instruction inbox.** `/v1/instructions` (POST, GET), loopback-only as the voice routes are, records an utterance, the project it resolved to, and its source. Recording executes nothing. Resolution reuses `consolidate`'s roster matching, and an ambiguous or unmatched name is asked back as a closed choice rather than guessed. joe's page offers it beside "Answer by voice"
- [ ] **Acting on an instruction.** A worker takes an instruction, declares its spend, asks consent on the human queue -- the existing voice approval -- and runs an agent in the project's clone, continuing the session the archive names where there is one. The agent runtime sits behind an adapter, as speech engines do behind `vox.adapters`. This is Phase 9's execution route or its sibling, through the same governed seam
- [ ] **The result, spoken.** The outcome is recorded against the instruction and spoken as a short summary; the full result stays in the record. Progress uses the announcement states joe's conversation panel already reads

### Acceptance Criteria

- [ ] An instruction spoken at joe's page appears in `/v1/instructions` with its project, and nothing runs until consent is given
- [ ] A refused or unanswered consent spends nothing, and a walkthrough shows it
- [ ] "Where was I?" about a project gives the same answer after every process has restarted
- [ ] A cookbook check in the shape of `qmcp cookbook voice` runs the path offline on vox's deterministic engine, and has been seen to go red

### Constraints going in

- **Endpointing is tuned for short answers.** The listen route ends a take `silence_ms` after the speaker stops (default 800), and an instruction has pauses mid-thought. The route already accepts a longer pause and cap, so this is a parameter rather than new code. joe's `/api/capture/start` and `/stop` would serve as push-to-talk, but their docstring says they record from the default device rather than the microphone `joe voice setup` saved; establish which before relying on them
- **Whether project names transcribe reliably is unmeasured.** Resolution confirms rather than trusts, so it does not have to be
- **Latency on a sentence-length utterance is unmeasured.** The faster-whisper comparison in `governance/qm/perspectives/2026-09-28-the-voice-loop-meets-the-field.md` bears on it, and is evidence rather than a decision
- **`run_forever` has no bound on an idle queue**, and a standing worker needs one
- **One conversation at a time, on loopback, with synthesis in a process of its own** (the synthesizer needs a main thread on Windows). An instruction and its consent question share that slot

---

## Current Sprint

Phases 1 through 8 are complete and Phase 9's routes are shipped; its runtime items above are the open work, and Phase 10 is planned. QMCP is a production-ready MCP server with PydanticAI integration and composable workflow building blocks:

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
| 8. Cookbook & Runner | Composable modules + MetaflowRunner | 30+ |
| **Total** | | **167+** |

**Production Features:**
- Structured JSON logging (structlog)
- Request tracing with correlation IDs
- Prometheus-compatible `/metrics` endpoint
- PydanticAI agent execution with QMCP audit trail
- Composable `qmcp.cookbook` modules for flow building
- MetaflowRunner for topology-to-flow execution
- Comprehensive test coverage

**Next Steps (Future Work):**
- Topology runtime execution (Pipeline, Council, etc.) with PydanticAI built-in -- Phase 9's execution route is the governed way in
- CI/CD pipeline (GitHub Actions)
- HumanInLoopMixin HITL API integration
- AsyncRunner implementation
- Add authentication/authorization
- Webhook notifications for HITL
- Redis/PostgreSQL backend options
- Kubernetes deployment manifests
