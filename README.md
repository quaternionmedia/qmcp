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

## Client Library

```python
from qmcp.client import MCPClient

with MCPClient(base_url="http://localhost:8000") as client:
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

# List registered tools
qmcp tools list

# Show configuration
qmcp info

# Run tests with auto setup/teardown
qmcp test [-v] [--coverage] [TEST_PATH]
```

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

- [Overview](docs/overview.md) - What and why
- [Architecture](docs/architecture.md) - How and constraints
- [Tools](docs/tools.md) - Tool capabilities
- [Client Library](docs/client.md) - Python client API
- [Human-in-the-Loop](docs/human_in_loop.md) - HITL guide
- [Deployment](docs/deployment.md) - Production deployment guide
- [Contributing](docs/contributing.md) - Development guidelines
- [Roadmap](docs/ROADMAP.md) - Development phases

## Example Flows

See [examples/flows/](examples/flows/) for Metaflow integration examples:

- **simple_plan.py** - Basic tool invocation
- **approved_deploy.py** - HITL approval workflow

## License

MIT
