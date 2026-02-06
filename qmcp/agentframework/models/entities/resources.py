"""Resource allocation entity models."""

from __future__ import annotations

from .base import (
    UUID,
    Field,
    SQLModel,
    datetime,
    utc_now,
)
from ..enums import ResourceType


class ResourceAllocation(SQLModel, table=True):
    """Resource allocations for executions."""

    __tablename__ = "resource_allocations"

    id: int | None = Field(default=None, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    resource_type: ResourceType
    resource_id: str = Field(max_length=128)
    quantity: int = Field(default=1, ge=1)
    acquired_at: datetime = Field(default_factory=utc_now)
    released_at: datetime | None = Field(default=None)


__all__ = [
    "ResourceAllocation",
]
