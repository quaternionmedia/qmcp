"""Audit log API schemas."""

from __future__ import annotations

from .base import Any, UUID, SQLModel, datetime


class AuditLogRead(SQLModel):
    """Schema for reading audit log entries."""

    id: UUID
    execution_id: UUID | None
    agent_instance_id: UUID | None
    event_type: str
    level: str
    message: str
    details: dict[str, Any] | None
    created_at: datetime


__all__ = [
    "AuditLogRead",
]
