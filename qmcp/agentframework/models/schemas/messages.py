"""Message-related API schemas."""

from __future__ import annotations

from .base import Any, UUID, Field, SQLModel, datetime
from ..enums import Priority


class MessageCreate(SQLModel):
    """Schema for creating a new message."""

    recipient_id: UUID | None = None
    content: dict[str, Any]
    priority: Priority = Field(default=Priority.NORMAL)


class MessageRead(SQLModel):
    """Schema for reading message data."""

    id: UUID
    execution_id: UUID
    sender_id: UUID
    recipient_id: UUID | None
    priority: Priority
    content: dict[str, Any]
    round_number: int
    sequence_number: int
    created_at: datetime


__all__ = [
    "MessageCreate",
    "MessageRead",
]
