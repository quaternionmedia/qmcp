"""QMCP Toolset for PydanticAI integration.

Provides a PydanticAI-compatible toolset that connects to a QMCP MCP server,
allowing agents to use QMCP tools with full audit trail and HITL support.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from pydantic_ai.tools import ToolDefinition as PAIToolDefinition

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    PAIToolDefinition = None  # type: ignore


def _check_pydantic_ai() -> None:
    """Raise ImportError if pydantic-ai is not installed."""
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is required for this functionality. "
            "Install it with: pip install qmcp[pydantic-ai]"
        )


@dataclass
class QMCPToolset:
    """A PydanticAI toolset that connects to a QMCP MCP server.

    This toolset allows PydanticAI agents to use tools exposed by a QMCP server,
    with full audit trail, human-in-the-loop support, and metrics.

    Example:
        >>> from pydantic_ai import Agent
        >>> from qmcp.integrations.pydantic_ai import QMCPToolset, create_agent
        >>>
        >>> async with QMCPToolset("http://localhost:3333") as toolset:
        ...     agent = create_agent(
        ...         "claude-sonnet-4-20250514",
        ...         toolsets=[toolset],
        ...     )
        ...     result = await agent.run("Use the echo tool to say hello")

    The toolset supports:
    - Async context manager for connection lifecycle
    - Tool prefix to avoid naming conflicts
    - Correlation ID for distributed tracing
    - Human-in-the-loop via the QMCP HITL API

    Attributes:
        base_url: The QMCP server URL
        tool_prefix: Optional prefix for tool names (e.g., "qmcp_")
        timeout: HTTP request timeout in seconds
        correlation_id: Optional correlation ID for tracing
    """

    base_url: str = "http://localhost:3333"
    tool_prefix: str = ""
    timeout: float = 30.0
    correlation_id: str | None = None
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _tools: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        _check_pydantic_ai()
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def __aenter__(self) -> QMCPToolset:
        """Enter async context and fetch tool definitions."""
        await self._fetch_tools()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context and close HTTP client."""
        await self._client.aclose()

    async def _fetch_tools(self) -> None:
        """Fetch tool definitions from the QMCP server."""
        response = await self._client.get("/v1/tools")
        response.raise_for_status()
        data = response.json()
        self._tools = data.get("tools", [])

    # -------------------------------------------------------------------------
    # PydanticAI AbstractToolset interface
    # -------------------------------------------------------------------------

    async def get_tools(self) -> Sequence[PAIToolDefinition]:
        """Return tool definitions for PydanticAI.

        This method is called by PydanticAI to discover available tools.
        """
        if not self._tools:
            await self._fetch_tools()

        definitions = []
        for tool in self._tools:
            name = f"{self.tool_prefix}{tool['name']}" if self.tool_prefix else tool["name"]

            # Convert QMCP schema to PydanticAI format
            definition = PAIToolDefinition(
                name=name,
                description=tool.get("description", ""),
                parameters_json_schema=tool.get("input_schema", {}),
            )
            definitions.append(definition)

        return definitions

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a tool on the QMCP server.

        Args:
            name: The tool name (with prefix stripped if applicable)
            tool_args: Arguments to pass to the tool
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            The tool result

        Raises:
            httpx.HTTPStatusError: If the tool call fails
        """
        # Strip prefix if present
        tool_name = name
        if self.tool_prefix and tool_name.startswith(self.tool_prefix):
            tool_name = tool_name[len(self.tool_prefix) :]

        # Build request payload
        payload: dict[str, Any] = {"input": tool_args}
        if self.correlation_id:
            payload["correlation_id"] = self.correlation_id

        # Call the tool
        response = await self._client.post(f"/v1/tools/{tool_name}", json=payload)
        response.raise_for_status()

        data = response.json()

        # Check for errors
        if data.get("error"):
            raise RuntimeError(f"Tool {tool_name} failed: {data['error']}")

        return data.get("result")

    # -------------------------------------------------------------------------
    # HITL convenience methods (not part of AbstractToolset)
    # -------------------------------------------------------------------------

    async def request_human_approval(
        self,
        request_id: str,
        prompt: str,
        options: list[str] | None = None,
        context: dict[str, Any] | None = None,
        timeout_seconds: int = 3600,
        poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        """Request human approval through QMCP's HITL API.

        This can be used within a tool to pause execution until a human
        approves an action.

        Args:
            request_id: Unique ID for this request
            prompt: Question to ask the human
            options: Valid response options
            context: Additional context for the human
            timeout_seconds: How long to wait for a response
            poll_interval: How often to poll for a response

        Returns:
            The human response dict

        Example:
            >>> @agent.tool
            ... async def deploy(ctx, env: str) -> str:
            ...     approval = await toolset.request_human_approval(
            ...         f"deploy-{env}",
            ...         f"Deploy to {env}?",
            ...         options=["approve", "reject"],
            ...     )
            ...     if approval["response"] == "approve":
            ...         return f"Deployed to {env}"
            ...     return "Deployment cancelled"
        """
        # Create the request
        payload: dict[str, Any] = {
            "id": request_id,
            "request_type": "approval",
            "prompt": prompt,
            "timeout_seconds": timeout_seconds,
        }
        if options:
            payload["options"] = options
        if context:
            payload["context"] = context
        if self.correlation_id:
            payload["correlation_id"] = self.correlation_id

        response = await self._client.post("/v1/human/requests", json=payload)

        # Handle conflict (request already exists)
        if response.status_code == 409:
            pass  # Request exists, we'll poll for response
        else:
            response.raise_for_status()

        # Poll for response
        while True:
            response = await self._client.get(f"/v1/human/requests/{request_id}")
            response.raise_for_status()
            data = response.json()

            request_status = data["request"]["status"]

            if data.get("response"):
                return data["response"]

            if request_status == "expired":
                raise TimeoutError(f"Human request {request_id} expired")

            await asyncio.sleep(poll_interval)

    async def health(self) -> dict[str, str]:
        """Check QMCP server health.

        Returns:
            Health status dict with "status" and "version" keys
        """
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return cached tool definitions."""
        return self._tools


__all__ = ["QMCPToolset"]
