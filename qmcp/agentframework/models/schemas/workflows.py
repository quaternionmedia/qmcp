"""Workflow template API schemas."""

from __future__ import annotations

from .base import Any, Field, SQLModel, datetime


class WorkflowTemplateCreate(SQLModel):
    """Schema for creating a workflow template."""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    topology_name: str
    default_input: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    is_public: bool = Field(default=False)


class WorkflowTemplateUpdate(SQLModel):
    """Schema for updating a workflow template."""

    description: str | None = None
    default_input: dict[str, Any] | None = None
    tags: list[str] | None = None
    is_public: bool | None = None


class WorkflowTemplateRead(SQLModel):
    """Schema for reading workflow template data."""

    id: int
    name: str
    description: str
    topology_id: int
    default_input: dict[str, Any]
    tags: list[str]
    is_public: bool
    created_at: datetime
    updated_at: datetime


__all__ = [
    "WorkflowTemplateCreate",
    "WorkflowTemplateUpdate",
    "WorkflowTemplateRead",
]
