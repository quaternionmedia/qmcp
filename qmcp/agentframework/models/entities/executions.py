"""Execution-related entity models."""

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
from ..enums import ExecutionStatus, Priority


class Execution(SQLModel, table=True):
    """Records a single topology execution."""

    __tablename__ = "executions"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    topology_id: int = Field(foreign_key="topologies.id")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    priority: Priority = Field(default=Priority.NORMAL)
    input_data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    output_data: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    correlation_id: str | None = Field(default=None, index=True)
    parent_execution_id: UUID | None = Field(default=None, foreign_key="executions.id")
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False),
    )
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = Field(default=None)
    error: str | None = Field(default=None)
    error_details: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )

    @property
    def duration_ms(self) -> int | None:
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def mark_complete(self, output: dict[str, Any]) -> None:
        self.status = ExecutionStatus.COMPLETED
        self.output_data = output
        self.completed_at = utc_now()

    def mark_failed(self, error: str, details: dict[str, Any] | None = None) -> None:
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.error_details = details
        self.completed_at = utc_now()


class Result(SQLModel, table=True):
    """Individual agent result within an execution."""

    __tablename__ = "results"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    agent_instance_id: UUID = Field(foreign_key="agent_instances.id")
    output: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = Field(default=None)
    token_usage: dict[str, int] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = Field(default=None)


class Checkpoint(SQLModel, table=True):
    """Execution checkpoints for recovery."""

    __tablename__ = "checkpoints"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    step_name: str = Field(max_length=64)
    state_snapshot: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)


__all__ = [
    "Execution",
    "Result",
    "Checkpoint",
]
