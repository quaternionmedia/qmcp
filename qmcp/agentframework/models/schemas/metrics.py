"""Metrics and tool invocation API schemas."""

from __future__ import annotations

from .base import Any, UUID, Field, SQLModel, datetime


# =============================================================================
# Metric Schemas
# =============================================================================


class MetricCreate(SQLModel):
    """Schema for recording a metric."""

    metric_name: str = Field(max_length=64)
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class MetricRead(SQLModel):
    """Schema for reading metric data."""

    id: UUID
    execution_id: UUID
    agent_instance_id: UUID | None
    metric_name: str
    value: float
    unit: str | None
    tags: dict[str, str]
    recorded_at: datetime


class MetricAggregation(SQLModel):
    """Schema for aggregated metrics."""

    metric_name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    sum_value: float


# =============================================================================
# Tool Invocation Schemas
# =============================================================================


class ToolInvocationCreate(SQLModel):
    """Schema for recording a tool invocation."""

    tool_name: str = Field(max_length=64)
    input_data: dict[str, Any]


class ToolInvocationRead(SQLModel):
    """Schema for reading tool invocation data."""

    id: UUID
    execution_id: UUID
    agent_instance_id: UUID
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    success: bool
    error_message: str | None
    duration_ms: int | None
    invoked_at: datetime


__all__ = [
    # Metric
    "MetricCreate",
    "MetricRead",
    "MetricAggregation",
    # Tool Invocation
    "ToolInvocationCreate",
    "ToolInvocationRead",
]
