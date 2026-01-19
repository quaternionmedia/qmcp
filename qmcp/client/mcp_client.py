"""MCP Client for interacting with the QMCP server.

This client provides a clean interface for:
- Tool discovery and invocation
- Human-in-the-loop request management
- Invocation history queries

Usage:
    from qmcp.client import MCPClient

    client = MCPClient("http://localhost:3333")

    # List tools
    tools = client.list_tools()

    # Invoke a tool
    result = client.invoke_tool("echo", {"message": "hello"})

    # Create and wait for human approval
    response = client.request_human_approval(
        request_id="deploy-001",
        prompt="Deploy to production?",
        options=["approve", "reject"],
        timeout_seconds=3600,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


class MCPClientError(Exception):
    """Base exception for MCP client errors."""

    pass


class ToolNotFoundError(MCPClientError):
    """Raised when a tool is not found."""

    pass


class HumanRequestExpiredError(MCPClientError):
    """Raised when a human request has expired."""

    pass


class HumanRequestConflictError(MCPClientError):
    """Raised when a human request ID already exists."""

    pass


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] | None = None


@dataclass
class ToolResult:
    """Result of a tool invocation."""

    result: Any
    error: str | None
    invocation_id: str


@dataclass
class HumanRequest:
    """A human-in-the-loop request."""

    id: str
    request_type: str
    prompt: str
    status: str
    created_at: str
    expires_at: str | None = None
    options: list[str] | None = None
    context: dict[str, Any] | None = None


@dataclass
class HumanResponse:
    """A human response to a request."""

    id: str
    request_id: str
    response: str
    responded_by: str | None
    created_at: str


class MCPClient:
    """HTTP client for the QMCP server.

    This client is designed to be used from Metaflow flows or other
    orchestration systems. It is synchronous by default for simplicity.

    Args:
        base_url: The base URL of the MCP server (e.g., "http://localhost:3333")
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, base_url: str = "http://localhost:3333", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> MCPClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    # -------------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------------

    def health(self) -> dict[str, str]:
        """Check server health.

        Returns:
            Health status dict with "status" and "version" keys.
        """
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------

    def list_tools(self) -> list[ToolDefinition]:
        """List all available tools.

        Returns:
            List of ToolDefinition objects.
        """
        response = self._client.get("/v1/tools")
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

    def invoke_tool(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        correlation_id: str | None = None,
    ) -> ToolResult:
        """Invoke a tool by name.

        Args:
            tool_name: Name of the tool to invoke.
            input_params: Input parameters for the tool.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ToolResult with result, error, and invocation_id.

        Raises:
            ToolNotFoundError: If the tool doesn't exist.
        """
        payload = {"input": input_params}
        if correlation_id:
            payload["correlation_id"] = correlation_id

        response = self._client.post(f"/v1/tools/{tool_name}", json=payload)

        if response.status_code == 404:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")

        response.raise_for_status()
        data = response.json()

        return ToolResult(
            result=data.get("result"),
            error=data.get("error"),
            invocation_id=data.get("invocation_id"),
        )

    # -------------------------------------------------------------------------
    # Human-in-the-Loop
    # -------------------------------------------------------------------------

    def create_human_request(
        self,
        request_id: str,
        request_type: str,
        prompt: str,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
        timeout_seconds: int = 3600,
        correlation_id: str | None = None,
    ) -> HumanRequest:
        """Create a human-in-the-loop request.

        Args:
            request_id: Unique ID for this request.
            request_type: Type of request ("approval", "input", "review").
            prompt: The question or prompt for the human.
            options: Optional list of valid response options.
            context: Optional context dict to help the human decide.
            timeout_seconds: How long to wait for a response (default: 1 hour).
            correlation_id: Optional correlation ID for tracing.

        Returns:
            HumanRequest object.

        Raises:
            HumanRequestConflictError: If request ID already exists.
        """
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

        response = self._client.post("/v1/human/requests", json=payload)

        if response.status_code == 409:
            raise HumanRequestConflictError(
                f"Request ID '{request_id}' already exists"
            )

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

    def get_human_request(self, request_id: str) -> tuple[HumanRequest, HumanResponse | None]:
        """Get a human request and its response (if any).

        Args:
            request_id: The request ID to look up.

        Returns:
            Tuple of (HumanRequest, HumanResponse or None).

        Raises:
            MCPClientError: If request not found.
        """
        response = self._client.get(f"/v1/human/requests/{request_id}")

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

    def submit_human_response(
        self,
        request_id: str,
        response: str,
        responded_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HumanResponse:
        """Submit a response to a human request.

        Args:
            request_id: The request ID to respond to.
            response: The response value.
            responded_by: Optional identifier of who responded.
            metadata: Optional metadata about the response.

        Returns:
            HumanResponse object.

        Raises:
            MCPClientError: If request not found.
            HumanRequestExpiredError: If request has expired.
        """
        payload = {
            "request_id": request_id,
            "response": response,
        }
        if responded_by:
            payload["responded_by"] = responded_by
        if metadata:
            payload["response_metadata"] = metadata

        resp = self._client.post("/v1/human/responses", json=payload)

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

    def wait_for_human_response(
        self,
        request_id: str,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> HumanResponse:
        """Wait for a human response by polling.

        Args:
            request_id: The request ID to wait for.
            poll_interval: Seconds between polls (default: 5).
            timeout: Max seconds to wait (default: None = wait forever).

        Returns:
            HumanResponse when received.

        Raises:
            HumanRequestExpiredError: If request expires before response.
            TimeoutError: If timeout exceeded before response.
        """
        start_time = time.time()

        while True:
            request, response = self.get_human_request(request_id)

            if response is not None:
                return response

            if request.status == "expired":
                raise HumanRequestExpiredError(
                    f"Request '{request_id}' expired without response"
                )

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Timeout waiting for response to '{request_id}'"
                    )

            time.sleep(poll_interval)

    def request_human_approval(
        self,
        request_id: str,
        prompt: str,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
        timeout_seconds: int = 3600,
        poll_interval: float = 5.0,
        wait_timeout: float | None = None,
    ) -> HumanResponse:
        """Create an approval request and wait for the response.

        This is a convenience method that combines create_human_request
        and wait_for_human_response.

        Args:
            request_id: Unique ID for this request.
            prompt: The question for the human.
            options: Valid response options (e.g., ["approve", "reject"]).
            context: Context to help the human decide.
            timeout_seconds: Server-side timeout for the request.
            poll_interval: Client-side poll interval.
            wait_timeout: Client-side max wait time (None = use server timeout).

        Returns:
            HumanResponse with the human's decision.
        """
        self.create_human_request(
            request_id=request_id,
            request_type="approval",
            prompt=prompt,
            options=options,
            context=context,
            timeout_seconds=timeout_seconds,
        )

        return self.wait_for_human_response(
            request_id=request_id,
            poll_interval=poll_interval,
            timeout=wait_timeout,
        )

    # -------------------------------------------------------------------------
    # Invocation History
    # -------------------------------------------------------------------------

    def list_invocations(
        self,
        tool_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List tool invocations.

        Args:
            tool_name: Filter by tool name.
            status: Filter by status ("pending", "success", "error").
            limit: Max results to return.
            offset: Pagination offset.

        Returns:
            List of invocation dicts.
        """
        params = {"limit": limit, "offset": offset}
        if tool_name:
            params["tool_name"] = tool_name
        if status:
            params["status"] = status

        response = self._client.get("/v1/invocations", params=params)
        response.raise_for_status()
        return response.json()["invocations"]

    def get_invocation(self, invocation_id: str) -> dict[str, Any]:
        """Get a single invocation by ID.

        Args:
            invocation_id: The invocation ID.

        Returns:
            Invocation dict.

        Raises:
            MCPClientError: If invocation not found.
        """
        response = self._client.get(f"/v1/invocations/{invocation_id}")

        if response.status_code == 404:
            raise MCPClientError(f"Invocation '{invocation_id}' not found")

        response.raise_for_status()
        return response.json()
