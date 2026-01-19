"""MCP protocol schema definitions.

These schemas align with the Model Context Protocol specification.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Definition of an MCP tool for discovery."""

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for tool input",
    )


class ToolListResponse(BaseModel):
    """Response for tool listing endpoint."""

    tools: list[ToolDefinition]


class ToolInvokeRequest(BaseModel):
    """Request payload for tool invocation."""

    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool input parameters",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Optional correlation ID for tracing",
    )


class ToolInvokeResponse(BaseModel):
    """Response from tool invocation."""

    result: Any = Field(default=None, description="Tool execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    invocation_id: str | None = Field(
        default=None,
        description="ID of the invocation record for audit",
    )


class InvocationSummary(BaseModel):
    """Summary of a tool invocation for listing."""

    id: str
    tool_name: str
    status: str
    duration_ms: int | None
    created_at: datetime
    error: str | None = None


class InvocationListResponse(BaseModel):
    """Response for invocation listing endpoint."""

    invocations: list[Any]  # ToolInvocation objects
    count: int


# Human-in-the-Loop Schemas


class HumanRequestCreate(BaseModel):
    """Request to create a human approval/input request."""

    id: str = Field(..., description="Client-provided unique ID for this request")
    request_type: str = Field(
        ...,
        description="Type of request: 'approval', 'input', 'review'",
    )
    prompt: str = Field(..., description="The question or prompt for the human")
    options: list[str] | None = Field(
        default=None,
        description="Allowed response options (for approval type)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context to display to the human",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Timeout in seconds (1 minute to 24 hours)",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Optional correlation ID for tracing",
    )


class HumanRequestResponse(BaseModel):
    """Response after creating a human request."""

    id: str
    request_type: str
    prompt: str
    status: str
    created_at: datetime
    expires_at: datetime | None


class HumanResponseCreate(BaseModel):
    """Request to submit a human response."""

    request_id: str = Field(..., description="ID of the request being responded to")
    response: str = Field(..., description="The human's response")
    responded_by: str | None = Field(
        default=None,
        description="Optional identifier of who responded",
    )
    response_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response",
    )


class HumanResponseResult(BaseModel):
    """Response after submitting a human response."""

    id: str
    request_id: str
    response: str
    responded_by: str | None
    created_at: datetime


class HumanRequestListResponse(BaseModel):
    """Response for listing human requests."""

    requests: list[Any]  # HumanRequest objects
    count: int
