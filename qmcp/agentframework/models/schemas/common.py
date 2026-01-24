"""Common API schemas for pagination, filtering, and system status."""

from __future__ import annotations

from .base import Any, Field, SQLModel, datetime
from ..enums import ExecutionStatus, HealthStatus, Priority


# =============================================================================
# Pagination and Query Schemas
# =============================================================================


class PaginationParams(SQLModel):
    """Schema for pagination parameters."""

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(SQLModel):
    """Schema for paginated response wrapper."""

    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class FilterParams(SQLModel):
    """Schema for common filter parameters."""

    created_after: datetime | None = None
    created_before: datetime | None = None
    status: ExecutionStatus | None = None
    priority: Priority | None = None


class SortParams(SQLModel):
    """Schema for sorting parameters."""

    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc")


# =============================================================================
# Health and Status Schemas
# =============================================================================


class HealthCheck(SQLModel):
    """Schema for health check response."""

    status: HealthStatus
    version: str
    uptime_seconds: int
    active_executions: int
    active_agents: int


class SystemStats(SQLModel):
    """Schema for system statistics."""

    total_executions: int
    completed_executions: int
    failed_executions: int
    avg_execution_duration_ms: float | None
    total_messages: int
    total_tool_invocations: int


__all__ = [
    # Pagination and Query
    "PaginationParams",
    "PaginatedResponse",
    "FilterParams",
    "SortParams",
    # Health and Status
    "HealthCheck",
    "SystemStats",
]
