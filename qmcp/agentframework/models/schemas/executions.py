"""Execution-related API schemas."""

from __future__ import annotations

from .base import Any, UUID, Field, SQLModel, datetime
from ..enums import ExecutionStatus, Priority


# =============================================================================
# Execution Schemas
# =============================================================================


class ExecutionCreate(SQLModel):
    """Schema for creating a new execution."""

    topology_name: str
    input_data: dict[str, Any]
    priority: Priority = Field(default=Priority.NORMAL)
    correlation_id: str | None = None
    metadata_: dict[str, Any] | None = None


class ExecutionRead(SQLModel):
    """Schema for reading execution data."""

    id: UUID
    topology_id: int
    status: ExecutionStatus
    priority: Priority
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    correlation_id: str | None
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    error: str | None


class ExecutionSummary(SQLModel):
    """Compact schema for listing executions."""

    id: UUID
    topology_id: int
    status: ExecutionStatus
    priority: Priority
    started_at: datetime
    duration_ms: int | None


class ExecutionStatusUpdate(SQLModel):
    """Schema for updating execution status."""

    status: ExecutionStatus
    error: str | None = None
    error_details: dict[str, Any] | None = None


# =============================================================================
# Result Schemas
# =============================================================================


class ResultCreate(SQLModel):
    """Schema for creating a new result."""

    output: dict[str, Any]
    confidence: float | None = None
    reasoning: str | None = None
    token_usage: dict[str, int] | None = None


class ResultRead(SQLModel):
    """Schema for reading result data."""

    id: UUID
    execution_id: UUID
    agent_instance_id: UUID
    output: dict[str, Any]
    confidence: float | None
    reasoning: str | None
    token_usage: dict[str, int] | None
    started_at: datetime
    completed_at: datetime | None


# =============================================================================
# Checkpoint Schemas
# =============================================================================


class CheckpointRead(SQLModel):
    """Schema for reading checkpoint data."""

    id: UUID
    execution_id: UUID
    step_name: str
    state_snapshot: dict[str, Any]
    created_at: datetime


__all__ = [
    # Execution
    "ExecutionCreate",
    "ExecutionRead",
    "ExecutionSummary",
    "ExecutionStatusUpdate",
    # Result
    "ResultCreate",
    "ResultRead",
    # Checkpoint
    "CheckpointRead",
]
