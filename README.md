# QMCP - Model Context Protocol Server

A spec-aligned **Model Context Protocol (MCP) server** built with FastAPI.

## Features

- ✅ **Tool Discovery** - List available tools via `/v1/tools`
- ✅ **Tool Invocation** - Execute tools via `/v1/tools/{name}`
- ✅ **Invocation History** - Audit trail via `/v1/invocations`
- ✅ **Human-in-the-Loop** - Request human input via `/v1/human/*`
- ✅ **Persistence** - SQLite with SQLModel/aiosqlite
- ✅ **Python Client** - `qmcp.client.MCPClient` for workflows
- ✅ **Metaflow Examples** - Ready-to-use flow templates
- ✅ **Agent Framework** - SQLModel schemas + mixins for agent types/topologies
- ✅ **Structured Logging** - JSON logs with structlog
- ✅ **Request Tracing** - Correlation IDs across requests
- ✅ **Metrics** - Prometheus-compatible `/metrics` endpoint
- ✅ **CLI Interface** - Manage via `qmcp` command

## Quick Start

```bash
# Install dependencies
uv sync

# Start the server
uv run qmcp serve

# Or with development reload
uv run qmcp serve --reload
```

See `quickstart.md` for a copy-paste walkthrough.

## Adoption and Onboarding

Adoption checklist:
- Decide how the server is hosted (local, container, or VM) and who can reach it.
- Set `QMCP_HOST`, `QMCP_PORT`, and `QMCP_DATABASE_URL` for your environment.
- Standardize `X-Correlation-ID` values for audit trails across clients.
- Decide how humans submit HITL responses (UI or API).
- Wire `/metrics` into your monitoring stack.

Onboarding path:
1. `uv sync --all-extras`
2. Run the end-to-end tutorial below.
3. `uv run qmcp serve` for local exploration.

### End-to-End Tutorial (HITL approval workflow)

This tutorial mirrors the end-to-end test
`tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow`.

Copy and paste:
```bash
uv sync --all-extras
uv run pytest tests/test_hitl.py::TestHITLWorkflow::test_complete_approval_workflow -v
```

## Client Library

```python
from qmcp.client import MCPClient

with MCPClient(base_url="http://localhost:3333") as client:
    # List tools
    tools = client.list_tools()

    # Invoke a tool
    result = client.invoke_tool("echo", {"message": "Hello!"})
    print(result.result)

    # Human-in-the-loop
    request = client.create_human_request(
        request_id="approval-001",
        request_type="approval",
        prompt="Approve deployment?",
        options=["approve", "reject"]
    )
    response = client.wait_for_response("approval-001", timeout=3600)
```

See [docs/client.md](docs/client.md) for full API documentation.

## CLI Commands

```bash
# Start the server
qmcp serve [--host HOST] [--port PORT] [--reload]

# Start the server for Docker-based flows
qmcp cookbook serve [--host 0.0.0.0] [--port PORT] [--reload]

# List registered tools
qmcp tools list

# Show configuration
qmcp info

# Run a cookbook flow in Docker
qmcp cookbook simple-plan --goal "Deploy a web service"

# Start the server + run a cookbook flow (unified dev)
qmcp cookbook dev simple-plan --goal "Deploy a web service"

# Run a cookbook flow via the generic runner
qmcp cookbook run simple-plan --goal "Deploy a web service"

# Run other cookbook recipes (flow args are passed through)
qmcp cookbook run approved-deploy --service "api-gateway" --environment "staging"
qmcp cookbook dev local-qc-gauntlet --change-summary "Add audit fields" --target-area "metrics, logging"

# Run a cookbook flow in Docker explicitly
qmcp cookbook docker simple-plan --goal "Deploy a web service"

# If the qmcp shim cannot be installed (Windows)
uv run --no-sync python -m qmcp cookbook run simple-plan --goal "Deploy a web service"

# Run tests with auto setup/teardown
qmcp test [-v] [--coverage] [TEST_PATH]
```

Cookbook flows run in Docker and require Docker Desktop (Linux engine).
Add `--no-sync` to skip syncing flow dependencies if the image is already built.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/tools` | GET | List available tools |
| `/v1/tools/{name}` | POST | Invoke a tool |
| `/v1/invocations` | GET | List invocation history |
| `/v1/invocations/{id}` | GET | Get single invocation |
| `/v1/human/requests` | POST | Create human request |
| `/v1/human/requests` | GET | List human requests |
| `/v1/human/requests/{id}` | GET | Get request with response |
| `/v1/human/responses` | POST | Submit human response |
| `/metrics` | GET | Prometheus metrics |
| `/metrics/json` | GET | Metrics as JSON |

## Built-in Tools

- **echo** - Echo input back (for testing)
- **planner** - Create execution plans
- **executor** - Execute approved plans
- **reviewer** - Review and assess results

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests (with auto cleanup)
uv run qmcp test -v

# Run tests with coverage
uv run qmcp test --coverage

# Run linter
uv run ruff check .
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architectural overview.

The system follows a three-plane architecture:

1. **Client/Orchestration** - Metaflow workflows (MCP client)
2. **MCP Server** - FastAPI service (this project)
3. **Execution/Storage** - Tools and database

## Documentation

- [Quickstart](quickstart.md) - Copy-paste setup and validation
- [Overview](docs/overview.md) - What and why
- [Architecture](docs/architecture.md) - How and constraints
- [Tools](docs/tools.md) - Tool capabilities
- [Client Library](docs/client.md) - Python client API
- [Human-in-the-Loop](docs/human_in_loop.md) - HITL guide
- [Agent Framework](docs/agentframework.md) - Agent schemas and mixins
- [Deployment](docs/deployment.md) - Production deployment guide
- [Contributing](docs/contributing.md) - Development guidelines
- [Roadmap](docs/ROADMAP.md) - Development phases

## Example Flows

See [examples/flows/](examples/flows/) for Metaflow integration examples:

- **simple_plan.py** - Basic tool invocation
- **approved_deploy.py** - HITL approval workflow
- **local_agent_chain.py** - Local LLM plan -> review -> refine with SQLModel artifacts
- **local_qc_gauntlet.py** - Local LLM QC checklist/task/gate builder
- **local_release_notes.py** - Local LLM release notes and doc update suggestions

For local LLM flows, install extras with `uv sync --extra flows`.
Start `uv run qmcp serve --host 0.0.0.0` when `--use-mcp True` to enable MCP calls
from Docker-based flows.
On Windows, prefer running flows in a Linux container to avoid platform-specific
Metaflow dependencies.

Docker runner (recommended on Windows):
```bash
docker compose -f docker-compose.flows.yml build
docker compose -f docker-compose.flows.yml run --rm flow-runner \
  examples/flows/local_agent_chain.py run --use-mcp True --goal "..."
```

Set `MCP_URL` and `LLM_BASE_URL` (or pass `--mcp-url` / `--llm-base-url`) when
running in Docker, e.g. `http://host.docker.internal:3333`.

## License

MIT
