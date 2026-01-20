"""MCP helpers for local dev Metaflow flows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from qmcp.client import MCPClient
from qmcp.schemas.mcp import ToolInvokeRequest, ToolInvokeResponse


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
