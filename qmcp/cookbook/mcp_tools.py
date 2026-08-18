"""MCP server tool invocation helpers for QMCP flows.

Provides typed tool input models and an invoker class that optionally
logs every call through a FlowPersistence instance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qmcp.client import MCPClient
from qmcp.schemas.mcp import ToolInvokeRequest, ToolInvokeResponse


# ---------------------------------------------------------------------------
# Tool input models
# ---------------------------------------------------------------------------


class EchoInput(BaseModel):
    """Echo tool input."""

    message: str


class PlannerInput(BaseModel):
    """Planner tool input."""

    goal: str
    context: str | None = None


class ExecutorInput(BaseModel):
    """Executor tool input."""

    plan: dict[str, Any]
    dry_run: bool = True


class ReviewerInput(BaseModel):
    """Reviewer tool input."""

    result: dict[str, Any]
    criteria: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level helpers (preserved from original local_mcp.py)
# ---------------------------------------------------------------------------


def check_health(mcp_url: str) -> dict[str, str]:
    """Check MCP server health."""
    with MCPClient(mcp_url) as client:
        return client.health()


def invoke_tool(
    mcp_url: str,
    tool_name: str,
    payload: BaseModel,
    correlation_id: str | None = None,
) -> ToolInvokeResponse:
    """Invoke a tool with Pydantic validation and MCP client."""
    request = ToolInvokeRequest(
        input=payload.model_dump(),
        correlation_id=correlation_id,
    )
    with MCPClient(mcp_url) as client:
        result = client.invoke_tool(
            tool_name,
            request.input,
            correlation_id=request.correlation_id,
        )
    return ToolInvokeResponse(
        result=result.result,
        error=result.error,
        invocation_id=result.invocation_id,
    )


def require_invocation_id(response: ToolInvokeResponse) -> str:
    """Ensure MCP responses include an invocation ID."""
    if not response.invocation_id:
        raise RuntimeError("MCP response missing invocation_id")
    return response.invocation_id


# ---------------------------------------------------------------------------
# MCPToolInvoker
# ---------------------------------------------------------------------------


class MCPToolInvoker:
    """MCP tool invoker with optional automatic persistence logging.

    When a ``FlowPersistence`` instance is provided, every invocation is
    automatically saved to the flow's local database.

    Usage::

        invoker = MCPToolInvoker(mcp_url, persistence=fp)
        result = invoker.invoke("executor", ExecutorInput(plan=plan, dry_run=True),
                                correlation_id="flow-123")
    """

    def __init__(
        self,
        mcp_url: str,
        persistence: Any | None = None,
    ) -> None:
        self.mcp_url = mcp_url
        self.persistence = persistence

    def health(self) -> dict[str, str]:
        """Check MCP server health, persisting as artifact if available."""
        result = check_health(self.mcp_url)
        if self.persistence is not None:
            self.persistence.artifact("mcp_health", result)
        return result

    def invoke(
        self,
        tool_name: str,
        payload: BaseModel,
        correlation_id: str | None = None,
        artifact_kind: str | None = None,
    ) -> ToolInvokeResponse:
        """Invoke a tool and optionally persist the invocation.

        Args:
            tool_name: Name of the MCP tool.
            payload: Pydantic model with tool input.
            correlation_id: Optional correlation ID for tracing.
            artifact_kind: If set, also saves the response as an artifact.

        Returns:
            The tool invocation response.

        Raises:
            RuntimeError: If the tool returns an error.
        """
        result = invoke_tool(
            self.mcp_url,
            tool_name,
            payload,
            correlation_id=correlation_id,
        )

        if result.error:
            raise RuntimeError(f"MCP {tool_name} failed: {result.error}")

        invocation_id = require_invocation_id(result)

        if self.persistence is not None:
            self.persistence.mcp_invocation(
                tool_name=tool_name,
                invocation_id=invocation_id,
                payload=payload.model_dump(),
                correlation_id=correlation_id,
            )
            if artifact_kind:
                self.persistence.artifact(artifact_kind, result.model_dump())

        return result


__all__ = [
    # Input models
    "EchoInput",
    "PlannerInput",
    "ExecutorInput",
    "ReviewerInput",
    # Helpers
    "check_health",
    "invoke_tool",
    "require_invocation_id",
    # Invoker
    "MCPToolInvoker",
]
