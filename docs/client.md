# MCP Client Library

The `qmcp.client` module provides a Python client for interacting with the MCP server.

## Installation

The client is included with the main package:

```bash
pip install qmcp
# or with uv
uv pip install qmcp
```

## Quick Start

```python
from qmcp.client import MCPClient

# Connect to MCP server
with MCPClient(base_url="http://localhost:8000") as client:
    # Check server health
    health = client.health()
    print(f"Server status: {health['status']}")

    # List available tools
    tools = client.list_tools()
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")

    # Invoke a tool
    result = client.invoke_tool("echo", {"message": "Hello, MCP!"})
    print(f"Result: {result.result}")
```

## API Reference

### MCPClient

The main client class for MCP server interaction.

```python
MCPClient(
    base_url: str = "http://localhost:8000",
    timeout: float = 30.0,
    http_client: Optional[httpx.Client] = None
)
```

**Parameters:**
- `base_url` – MCP server URL (default: `http://localhost:8000`)
- `timeout` – Request timeout in seconds (default: `30.0`)
- `http_client` – Optional custom httpx client for testing

### Health Check

```python
def health(self) -> dict
```

Returns server health status.

**Example:**
```python
status = client.health()
# {"status": "healthy", "version": "0.1.0"}
```

### Tool Discovery

```python
def list_tools(self) -> list[ToolDefinition]
```

Returns list of available tools with their schemas.

**Example:**
```python
tools = client.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
    print(f"  Schema: {tool.input_schema}")
```

### Tool Invocation

```python
def invoke_tool(
    self,
    tool_name: str,
    input_params: dict,
    correlation_id: Optional[str] = None
) -> ToolResult
```

Invokes a tool and returns the result.

**Parameters:**
- `tool_name` – Name of the tool to invoke
- `input_params` – Tool input parameters
- `correlation_id` – Optional ID for tracking across systems

**Returns:** `ToolResult` with `result`, `error`, and `invocation_id`

**Raises:**
- `ToolNotFoundError` – Tool doesn't exist
- `ToolInvocationError` – Tool execution failed

**Example:**
```python
# Simple invocation
result = client.invoke_tool("echo", {"message": "Hello"})
print(result.result)

# With correlation ID for tracking
result = client.invoke_tool(
    "planner",
    {"prompt": "Build a REST API"},
    correlation_id="workflow-123"
)
print(result.invocation_id)
```

### Human-in-the-Loop

#### Create Request

```python
def create_human_request(
    self,
    request_id: str,
    request_type: str,
    prompt: str,
    options: Optional[list[str]] = None,
    context: Optional[dict] = None,
    timeout_seconds: int = 3600,
    correlation_id: Optional[str] = None
) -> HumanRequest
```

Creates a human approval/input request.

**Parameters:**
- `request_id` – Unique request identifier
- `request_type` – Type: "approval", "input", or "review"
- `prompt` – Human-readable prompt
- `options` – Optional list of valid responses
- `context` – Optional context data for decision support
- `timeout_seconds` – Expiration timeout (default 1 hour)
- `correlation_id` – Optional tracking ID

**Raises:**
- `HumanRequestConflictError` – Request ID already exists

**Example:**
```python
request = client.create_human_request(
    request_id="deploy-prod-001",
    request_type="approval",
    prompt="Approve deployment to production?",
    options=["approve", "reject"],
    context={
        "environment": "production",
        "changes": ["Update API version", "Add new endpoint"]
    },
    timeout_seconds=1800  # 30 minutes
)
print(f"Created request: {request.id}, expires: {request.expires_at}")
```

#### Get Request

```python
def get_human_request(
    self,
    request_id: str
) -> tuple[HumanRequest, Optional[HumanResponse]]
```

Gets a request and its response (if any).

**Returns:** Tuple of (request, response or None)

**Example:**
```python
request, response = client.get_human_request("deploy-prod-001")
print(f"Status: {request.status}")
if response:
    print(f"Response: {response.response} by {response.responded_by}")
```

#### Submit Response

```python
def submit_human_response(
    self,
    request_id: str,
    response: str,
    responded_by: Optional[str] = None,
    metadata: Optional[dict] = None
) -> HumanResponse
```

Submits a human response to a request.

**Raises:**
- `HumanRequestNotFoundError` – Request doesn't exist
- `HumanRequestExpiredError` – Request has expired

**Example:**
```python
response = client.submit_human_response(
    request_id="deploy-prod-001",
    response="approve",
    responded_by="alice@example.com",
    metadata={"approved_via": "slack"}
)
```

#### Wait for Response (Blocking)

```python
def wait_for_response(
    self,
    request_id: str,
    timeout: float = 3600,
    poll_interval: float = 2.0
) -> HumanResponse
```

Blocks until a response is received or timeout.

**Parameters:**
- `request_id` – Request to wait for
- `timeout` – Max wait time in seconds
- `poll_interval` – Time between polls

**Raises:**
- `TimeoutError` – No response within timeout
- `HumanRequestExpiredError` – Request expired

**Example:**
```python
# Create request and wait for human response
request = client.create_human_request(
    request_id="review-pr-42",
    request_type="review",
    prompt="Review PR #42: Add user authentication"
)

try:
    response = client.wait_for_response(
        "review-pr-42",
        timeout=600,  # 10 minutes
        poll_interval=5.0
    )
    print(f"Received: {response.response}")
except TimeoutError:
    print("No response received in time")
```

### Invocation History

#### List Invocations

```python
def list_invocations(
    self,
    tool_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> list[dict]
```

Lists tool invocation history with optional filtering.

**Example:**
```python
# All invocations
invocations = client.list_invocations()

# Filter by tool
echo_calls = client.list_invocations(tool_name="echo")

# Paginate
page2 = client.list_invocations(limit=10, offset=10)
```

#### Get Invocation

```python
def get_invocation(self, invocation_id: str) -> dict
```

Gets a single invocation by ID.

**Example:**
```python
result = client.invoke_tool("planner", {"prompt": "Build API"})
invocation = client.get_invocation(result.invocation_id)
print(f"Tool: {invocation['tool_name']}")
print(f"Duration: {invocation['duration_ms']}ms")
```

## Data Classes

### ToolDefinition

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Optional[dict] = None
```

### ToolResult

```python
@dataclass
class ToolResult:
    result: Any
    error: Optional[str] = None
    invocation_id: Optional[str] = None
```

### HumanRequest

```python
@dataclass
class HumanRequest:
    id: str
    request_type: str
    prompt: str
    status: str
    created_at: str
    expires_at: Optional[str] = None
    options: Optional[list[str]] = None
    context: Optional[dict] = None
```

### HumanResponse

```python
@dataclass
class HumanResponse:
    id: str
    request_id: str
    response: str
    responded_by: Optional[str] = None
    created_at: str
```

## Exceptions

```python
from qmcp.client import (
    MCPClientError,           # Base exception
    ToolNotFoundError,        # Tool doesn't exist
    ToolInvocationError,      # Tool execution failed
    HumanRequestNotFoundError, # Request doesn't exist
    HumanRequestExpiredError, # Request has expired
    HumanRequestConflictError # Request ID already exists
)
```

## Using with Metaflow

See [examples/flows/](../examples/flows/) for complete Metaflow integration examples.

### Basic Flow

```python
from metaflow import FlowSpec, step
from qmcp.client import MCPClient

class MCPToolFlow(FlowSpec):
    @step
    def start(self):
        with MCPClient() as client:
            result = client.invoke_tool("echo", {"message": "Hello from Metaflow"})
            self.echo_result = result.result
        self.next(self.end)

    @step
    def end(self):
        print(f"Result: {self.echo_result}")

if __name__ == "__main__":
    MCPToolFlow()
```

### HITL Approval Flow

```python
from metaflow import FlowSpec, step
from qmcp.client import MCPClient

class ApprovalFlow(FlowSpec):
    @step
    def start(self):
        self.run_id = current.run_id
        self.next(self.request_approval)

    @step
    def request_approval(self):
        with MCPClient() as client:
            client.create_human_request(
                request_id=f"approval-{self.run_id}",
                request_type="approval",
                prompt="Approve this workflow?",
                options=["approve", "reject"]
            )
            response = client.wait_for_response(
                f"approval-{self.run_id}",
                timeout=3600
            )
            self.approved = response.response == "approve"
        self.next(self.end)

    @step
    def end(self):
        if self.approved:
            print("Workflow approved!")
        else:
            print("Workflow rejected.")

if __name__ == "__main__":
    ApprovalFlow()
```

## Testing

The client can be used with a test server:

```python
import pytest
from starlette.testclient import TestClient
from qmcp.server import app
from qmcp.client import MCPClient

@pytest.fixture
def client():
    return TestClient(app)

def test_tool_invocation(client):
    # Use the test client directly or wrap in MCPClient
    response = client.post("/v1/tools/echo", json={"input": {"message": "test"}})
    assert response.status_code == 200
```
