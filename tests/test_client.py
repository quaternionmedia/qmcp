"""Tests for the MCP client library."""

import pytest

from qmcp.client import MCPClient
from qmcp.client.mcp_client import (
    HumanRequestConflictError,
    MCPClientError,
    ToolNotFoundError,
)


class TestMCPClientHealth:
    """Tests for health check."""

    def test_health_check(self, client, mcp_client):
        """Test that health check returns expected fields."""
        health = mcp_client.health()
        assert health["status"] == "healthy"
        assert "version" in health


class TestMCPClientTools:
    """Tests for tool operations."""

    def test_list_tools(self, client, mcp_client):
        """Test listing tools."""
        tools = mcp_client.list_tools()
        assert len(tools) >= 4  # echo, planner, executor, reviewer
        tool_names = [t.name for t in tools]
        assert "echo" in tool_names
        assert "planner" in tool_names

    def test_invoke_echo_tool(self, client, mcp_client):
        """Test invoking the echo tool."""
        result = mcp_client.invoke_tool("echo", {"message": "hello"})
        assert result.result == "hello"
        assert result.error is None
        assert result.invocation_id is not None

    def test_invoke_planner_tool(self, client, mcp_client):
        """Test invoking the planner tool."""
        result = mcp_client.invoke_tool("planner", {"goal": "Test goal"})
        assert result.error is None
        assert "steps" in result.result
        assert "estimated_steps" in result.result

    def test_invoke_nonexistent_tool(self, client, mcp_client):
        """Test that invoking a nonexistent tool raises an error."""
        with pytest.raises(ToolNotFoundError):
            mcp_client.invoke_tool("nonexistent", {})

    def test_invoke_with_correlation_id(self, client, mcp_client):
        """Test invoking a tool with correlation ID."""
        result = mcp_client.invoke_tool(
            "echo",
            {"message": "test"},
            correlation_id="test-correlation-123",
        )
        assert result.invocation_id is not None


class TestMCPClientHITL:
    """Tests for human-in-the-loop operations."""

    def test_create_human_request(self, client, mcp_client):
        """Test creating a human request."""
        request = mcp_client.create_human_request(
            request_id="test-request-001",
            request_type="approval",
            prompt="Test approval?",
            options=["yes", "no"],
        )
        assert request.id == "test-request-001"
        assert request.status == "pending"
        assert request.request_type == "approval"

    def test_create_duplicate_request_fails(self, client, mcp_client):
        """Test that creating a duplicate request raises an error."""
        mcp_client.create_human_request(
            request_id="duplicate-001",
            request_type="input",
            prompt="Test input",
        )

        with pytest.raises(HumanRequestConflictError):
            mcp_client.create_human_request(
                request_id="duplicate-001",
                request_type="input",
                prompt="Test input again",
            )

    def test_get_human_request(self, client, mcp_client):
        """Test getting a human request."""
        # Create first
        mcp_client.create_human_request(
            request_id="get-test-001",
            request_type="review",
            prompt="Review this?",
        )

        # Then get
        request, response = mcp_client.get_human_request("get-test-001")
        assert request.id == "get-test-001"
        assert request.prompt == "Review this?"
        assert response is None  # No response yet

    def test_get_nonexistent_request(self, client, mcp_client):
        """Test that getting a nonexistent request raises an error."""
        with pytest.raises(MCPClientError):
            mcp_client.get_human_request("nonexistent-request")

    def test_submit_human_response(self, client, mcp_client):
        """Test submitting a human response."""
        # Create request
        mcp_client.create_human_request(
            request_id="respond-test-001",
            request_type="approval",
            prompt="Approve?",
            options=["approve", "reject"],
        )

        # Submit response
        response = mcp_client.submit_human_response(
            request_id="respond-test-001",
            response="approve",
            responded_by="test@example.com",
        )

        assert response.request_id == "respond-test-001"
        assert response.response == "approve"
        assert response.responded_by == "test@example.com"

    def test_get_request_with_response(self, client, mcp_client):
        """Test getting a request that has a response."""
        # Create and respond
        mcp_client.create_human_request(
            request_id="full-test-001",
            request_type="approval",
            prompt="Full test?",
            options=["yes", "no"],
        )
        mcp_client.submit_human_response(
            request_id="full-test-001",
            response="yes",
        )

        # Get and verify
        request, response = mcp_client.get_human_request("full-test-001")
        assert request.status == "responded"
        assert response is not None
        assert response.response == "yes"


class TestMCPClientInvocations:
    """Tests for invocation history."""

    def test_list_invocations(self, client, mcp_client):
        """Test listing invocations."""
        # Create some invocations
        mcp_client.invoke_tool("echo", {"message": "test1"})
        mcp_client.invoke_tool("echo", {"message": "test2"})

        # List
        invocations = mcp_client.list_invocations()
        assert len(invocations) >= 2

    def test_list_invocations_filtered(self, client, mcp_client):
        """Test listing invocations with filter."""
        # Create invocation
        mcp_client.invoke_tool("echo", {"message": "filtered"})

        # List filtered
        invocations = mcp_client.list_invocations(tool_name="echo")
        assert all(inv["tool_name"] == "echo" for inv in invocations)

    def test_get_invocation(self, client, mcp_client):
        """Test getting a single invocation."""
        result = mcp_client.invoke_tool("echo", {"message": "get-test"})

        invocation = mcp_client.get_invocation(result.invocation_id)
        assert invocation["id"] == result.invocation_id
        assert invocation["tool_name"] == "echo"
        assert invocation["status"] == "success"

    def test_get_nonexistent_invocation(self, client, mcp_client):
        """Test that getting a nonexistent invocation raises an error."""
        with pytest.raises(MCPClientError):
            mcp_client.get_invocation("nonexistent-id")


# Fixture for MCP client that uses the test server
@pytest.fixture
def mcp_client(client):
    """Create an MCP client that wraps the test client.

    This creates a wrapper that delegates HTTP calls to the Starlette TestClient
    while presenting the MCPClient interface.
    """

    class TestMCPClient(MCPClient):
        """MCPClient that uses Starlette TestClient for requests."""

        def __init__(self, test_client):
            self._test_client = test_client
            self.base_url = ""
            self.timeout = 30.0

        def close(self):
            pass

        def _request(self, method: str, path: str, **kwargs):
            """Make a request using the test client."""
            method_func = getattr(self._test_client, method.lower())
            return method_func(path, **kwargs)

        # Override methods to use test client
        def health(self):
            response = self._test_client.get("/health")
            response.raise_for_status()
            return response.json()

        def list_tools(self):
            from qmcp.client.mcp_client import ToolDefinition
            response = self._test_client.get("/v1/tools")
            response.raise_for_status()
            data = response.json()
            return [
                ToolDefinition(
                    name=t["name"],
                    description=t["description"],
                    input_schema=t.get("input_schema"),
                )
                for t in data["tools"]
            ]

        def invoke_tool(self, tool_name, input_params, correlation_id=None):
            from qmcp.client.mcp_client import ToolNotFoundError, ToolResult
            payload = {"input": input_params}
            if correlation_id:
                payload["correlation_id"] = correlation_id
            response = self._test_client.post(f"/v1/tools/{tool_name}", json=payload)
            if response.status_code == 404:
                raise ToolNotFoundError(f"Tool '{tool_name}' not found")
            response.raise_for_status()
            data = response.json()
            return ToolResult(
                result=data.get("result"),
                error=data.get("error"),
                invocation_id=data.get("invocation_id"),
            )

        def create_human_request(self, request_id, request_type, prompt, options=None,
                                  context=None, timeout_seconds=3600, correlation_id=None):
            from qmcp.client.mcp_client import HumanRequest, HumanRequestConflictError
            payload = {
                "id": request_id,
                "request_type": request_type,
                "prompt": prompt,
                "timeout_seconds": timeout_seconds,
            }
            if options:
                payload["options"] = options
            if context:
                payload["context"] = context
            if correlation_id:
                payload["correlation_id"] = correlation_id
            response = self._test_client.post("/v1/human/requests", json=payload)
            if response.status_code == 409:
                raise HumanRequestConflictError(f"Request ID '{request_id}' already exists")
            response.raise_for_status()
            data = response.json()
            return HumanRequest(
                id=data["id"],
                request_type=data["request_type"],
                prompt=data["prompt"],
                status=data["status"],
                created_at=data["created_at"],
                expires_at=data.get("expires_at"),
            )

        def get_human_request(self, request_id):
            from qmcp.client.mcp_client import HumanRequest, HumanResponse, MCPClientError
            response = self._test_client.get(f"/v1/human/requests/{request_id}")
            if response.status_code == 404:
                raise MCPClientError(f"Request '{request_id}' not found")
            response.raise_for_status()
            data = response.json()
            request_data = data["request"]
            request = HumanRequest(
                id=request_data["id"],
                request_type=request_data["request_type"],
                prompt=request_data["prompt"],
                status=request_data["status"],
                created_at=request_data["created_at"],
                expires_at=request_data.get("expires_at"),
                options=request_data.get("options"),
                context=request_data.get("context"),
            )
            human_response = None
            if data.get("response"):
                resp_data = data["response"]
                human_response = HumanResponse(
                    id=resp_data["id"],
                    request_id=resp_data["request_id"],
                    response=resp_data["response"],
                    responded_by=resp_data.get("responded_by"),
                    created_at=resp_data["created_at"],
                )
            return request, human_response

        def submit_human_response(self, request_id, response, responded_by=None, metadata=None):
            from qmcp.client.mcp_client import (
                HumanRequestExpiredError,
                HumanResponse,
                MCPClientError,
            )
            payload = {"request_id": request_id, "response": response}
            if responded_by:
                payload["responded_by"] = responded_by
            if metadata:
                payload["response_metadata"] = metadata
            resp = self._test_client.post("/v1/human/responses", json=payload)
            if resp.status_code == 404:
                raise MCPClientError(f"Request '{request_id}' not found")
            if resp.status_code == 410:
                raise HumanRequestExpiredError(f"Request '{request_id}' has expired")
            resp.raise_for_status()
            data = resp.json()
            return HumanResponse(
                id=data["id"],
                request_id=data["request_id"],
                response=data["response"],
                responded_by=data.get("responded_by"),
                created_at=data["created_at"],
            )

        def list_invocations(self, tool_name=None, status=None, limit=50, offset=0):
            params = {"limit": limit, "offset": offset}
            if tool_name:
                params["tool_name"] = tool_name
            if status:
                params["status"] = status
            response = self._test_client.get("/v1/invocations", params=params)
            response.raise_for_status()
            return response.json()["invocations"]

        def get_invocation(self, invocation_id):
            from qmcp.client.mcp_client import MCPClientError
            response = self._test_client.get(f"/v1/invocations/{invocation_id}")
            if response.status_code == 404:
                raise MCPClientError(f"Invocation '{invocation_id}' not found")
            response.raise_for_status()
            return response.json()

    return TestMCPClient(client)
