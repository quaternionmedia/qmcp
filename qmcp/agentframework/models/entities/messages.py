"""Message entity models."""

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
from ..enums import MessageType, Priority


class Message(SQLModel, table=True):
    """Inter-agent message within an execution."""

    __tablename__ = "messages"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    sender_id: UUID = Field(foreign_key="agent_instances.id")
    recipient_id: UUID | None = Field(default=None, foreign_key="agent_instances.id")
    message_type: MessageType
    priority: Priority = Field(default=Priority.NORMAL)
    content: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    round_number: int = Field(default=0)
    sequence_number: int = Field(default=0)
    parent_message_id: UUID | None = Field(default=None, foreign_key="messages.id")
    created_at: datetime = Field(default_factory=utc_now)


__all__ = [
    "Message",
]
