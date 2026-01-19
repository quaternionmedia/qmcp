"""Database models for QMCP persistence.

All models use SQLModel for Pydantic + SQLAlchemy integration.
These models support:
- Tool invocation audit logging
- Human-in-the-loop request/response tracking
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlmodel import JSON, Column, Field, SQLModel


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


class InvocationStatus(str, Enum):
    """Status of a tool invocation."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class HumanRequestStatus(str, Enum):
    """Status of a human request."""

    PENDING = "pending"
    RESPONDED = "responded"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ToolInvocation(SQLModel, table=True):
    """Record of a tool invocation.

    Every tool call is logged here for audit and debugging.
    """

    __tablename__ = "tool_invocations"
    __table_args__ = {"extend_existing": True}

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    tool_name: str = Field(index=True)
    input_params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: Any | None = Field(default=None, sa_column=Column(JSON))
    error: str | None = Field(default=None)
    status: InvocationStatus = Field(default=InvocationStatus.PENDING)
    duration_ms: int | None = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    completed_at: datetime | None = Field(default=None)

    # Optional correlation ID for tracing across systems
    correlation_id: str | None = Field(default=None, index=True)


class HumanRequest(SQLModel, table=True):
    """A request for human input or approval.

    Human requests are durable and survive server restarts.
    """

    __tablename__ = "human_requests"
    __table_args__ = {"extend_existing": True}

    id: str = Field(primary_key=True)  # Client-provided ID
    request_type: str = Field(index=True)  # e.g., "approval", "input", "review"
    prompt: str
    options: list[str] | None = Field(default=None, sa_column=Column(JSON))
    context: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    timeout_seconds: int = Field(default=3600)
    status: HumanRequestStatus = Field(default=HumanRequestStatus.PENDING, index=True)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    expires_at: datetime | None = Field(default=None)

    # Optional correlation ID for tracing
    correlation_id: str | None = Field(default=None, index=True)


class HumanResponse(SQLModel, table=True):
    """A human's response to a request.

    Linked to a HumanRequest by request_id.
    """

    __tablename__ = "human_responses"
    __table_args__ = {"extend_existing": True}

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    request_id: str = Field(index=True)  # Links to HumanRequest.id
    response: str  # The human's response (e.g., "approve", "reject", free text)
    response_metadata: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    responded_by: str | None = Field(default=None)  # Optional: who responded
    created_at: datetime = Field(default_factory=utc_now, index=True)
