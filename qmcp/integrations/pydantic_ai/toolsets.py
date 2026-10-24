"""QMCP Toolset for PydanticAI integration.

Provides a PydanticAI-compatible toolset that connects to a QMCP MCP server,
allowing agents to use QMCP tools with full audit trail and HITL support.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx

from qmcp.client.mcp_client import default_base_url

try:
    from pydantic_ai.tools import ToolDefinition as PAIToolDefinition
    from pydantic_ai.toolsets import AbstractToolset
    from pydantic_ai.toolsets.abstract import ToolsetTool
    from pydantic_core import SchemaValidator, core_schema

    # The arguments go to the server as they came, and the server validates
    # them against the tool's own schema -- as PydanticAI's ExternalToolset
    # passes its tools' arguments through.
    _ARGS = SchemaValidator(schema=core_schema.any_schema())
    PYDANTIC_AI_AVAILABLE = True
    _Toolset = AbstractToolset
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    PAIToolDefinition = None  # type: ignore
    _Toolset = object  # type: ignore


def _check_pydantic_ai() -> None:
    """Raise ImportError if pydantic-ai is not installed."""
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is required for this functionality. "
            "Install it with: pip install qmcp[pydantic-ai]"
        )


@dataclass
class QMCPToolset(_Toolset):
    """A PydanticAI toolset that connects to a QMCP MCP server.

    This toolset allows PydanticAI agents to use tools exposed by a QMCP server,
    with full audit trail, human-in-the-loop support, and metrics.

    Example:
        >>> from pydantic_ai import Agent
        >>> from qmcp.integrations.pydantic_ai import QMCPToolset, create_agent
        >>>
        >>> async with QMCPToolset("http://localhost:3141") as toolset:
        ...     agent = create_agent(
        ...         "claude-sonnet-4-20250514",
        ...         toolsets=[toolset],
        ...     )
        ...     result = await agent.run("Use the echo tool to say hello")

    The toolset supports:
    - Async context manager for connection lifecycle
    - Tool prefix to avoid naming conflicts
    - Correlation ID for distributed tracing
    - Human-in-the-loop via the qmcp human request API

    Attributes:
        base_url: The QMCP server URL
        tool_prefix: Optional prefix for tool names (e.g., "qmcp_")
        timeout: HTTP request timeout in seconds
        correlation_id: Optional correlation ID for tracing
    """

    base_url: str = field(default_factory=default_base_url)
    """Where this machine serves, unless told otherwise. Derived from
    `qmcp.config` rather than typed in: a hard-coded port here could not
    reach the server this package starts."""

    tool_prefix: str = ""
    timeout: float = 30.0
    correlation_id: str | None = None
    toolset_id: str | None = None
    """The toolset's id among an agent's toolsets (`AbstractToolset.id`)."""
    max_retries: int = 1
    """How often the agent may retry a call the server refused."""
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _tools: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _entered: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        _check_pydantic_ai()
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @property
    def id(self) -> str | None:
        return self.toolset_id

    async def __aenter__(self) -> QMCPToolset:
        """Enter async context and fetch tool definitions.

        Entries are counted: an agent enters its toolsets for every run, inside
        the caller's own `async with`, and the client is closed only when the
        outermost one exits.
        """
        if self._entered == 0:
            if self._client.is_closed:
                self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
            await self._fetch_tools()
        self._entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context, closing the HTTP client on the outermost exit."""
        self._entered -= 1
        if self._entered == 0:
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

    async def get_tools(self, ctx: Any = None) -> dict[str, ToolsetTool]:
        """The server's tools, by the name the agent calls them.

        This method is called by PydanticAI to discover available tools.
        """
        if not self._tools:
            await self._fetch_tools()

        found = {}
        for tool in self._tools:
            name = f"{self.tool_prefix}{tool['name']}" if self.tool_prefix else tool["name"]
            definition = PAIToolDefinition(
                name=name,
                description=tool.get("description", ""),
                parameters_json_schema=tool.get("input_schema") or {"type": "object"},
            )
            found[name] = ToolsetTool(toolset=self, tool_def=definition,
                                      max_retries=self.max_retries, args_validator=_ARGS)
        return found

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
