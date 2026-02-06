"""Audit and metrics entity models."""

from __future__ import annotations

from .base import (
    Any,
    UUID,
    Column,
    Field,
    JSON,
    SQLModel,
    datetime,
    utc_now,
    uuid4,
)
from ..enums import EventType, LogLevel


class AuditLog(SQLModel, table=True):
    """Audit log for tracking agent actions."""

    __tablename__ = "audit_logs"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID | None = Field(default=None, foreign_key="executions.id")
    agent_instance_id: UUID | None = Field(default=None, foreign_key="agent_instances.id")
    event_type: EventType
    level: LogLevel = Field(default=LogLevel.INFO)
    message: str = Field(max_length=1024)
    details: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=utc_now)


class MetricRecord(SQLModel, table=True):
    """Metrics collected during execution."""

    __tablename__ = "metric_records"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    agent_instance_id: UUID | None = Field(default=None, foreign_key="agent_instances.id")
    metric_name: str = Field(max_length=64)
    value: float
    unit: str | None = Field(default=None, max_length=32)
    tags: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    recorded_at: datetime = Field(default_factory=utc_now)


class ToolInvocation(SQLModel, table=True):
    """Records of tool invocations by agents."""

    __tablename__ = "tool_invocations"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    agent_instance_id: UUID = Field(foreign_key="agent_instances.id")
    tool_name: str = Field(max_length=64)
    input_data: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None, max_length=512)
    duration_ms: int | None = Field(default=None, ge=0)
    invoked_at: datetime = Field(default_factory=utc_now)


__all__ = [
    "AuditLog",
    "MetricRecord",
    "ToolInvocation",
]
