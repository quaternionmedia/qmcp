"""Database models for QMCP persistence.

All models use SQLModel for Pydantic + SQLAlchemy integration.
These models support:
- Tool invocation audit logging
- Human-in-the-loop request/response tracking
- The instruction inbox: what a person asked for, recorded and not run
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlmodel import JSON, Column, Field, SQLModel

# Ensure agent framework tables are registered in SQLModel metadata.
from qmcp.agentframework import models as _agent_models  # noqa: F401


def utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


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


class InstructionSource(str, Enum):
    """How an instruction arrived: spoken, typed at a terminal, or sent by a page."""

    VOICE = "voice"
    TYPED = "typed"
    PAGE = "page"


class InstructionStatus(str, Enum):
    """Whether an instruction has a project.

    Both values describe a record and neither describes a run: nothing in this
    vocabulary says an instruction is being acted on or has been. Acting is a
    later change with a migration of its own, and the statuses it needs are
    its to add.
    """

    RECORDED = "recorded"
    UNRESOLVED = "unresolved"


class Instruction(SQLModel, table=True):
    """What a person asked for, in their words, against a project.

    Recording one executes nothing. `detail` carries the evidence for the
    project: which roster names the text matched and the rule that read them,
    and for a spoken instruction every transcript the dialog took, in order.
    """

    __tablename__ = "instructions"
    __table_args__ = {"extend_existing": True}

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    text: str
    project: str | None = Field(default=None, index=True)
    source: InstructionSource = Field(index=True)
    status: InstructionStatus = Field(default=InstructionStatus.RECORDED, index=True)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    updated_at: datetime = Field(default_factory=utc_now)
    detail: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
