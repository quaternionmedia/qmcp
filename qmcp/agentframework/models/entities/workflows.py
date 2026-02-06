"""Workflow template entity models."""

from __future__ import annotations

from .base import (
    Any,
    Column,
    Field,
    JSON,
    SQLModel,
    datetime,
    field_validator,
    utc_now,
    validate_identifier,
)


class WorkflowTemplate(SQLModel, table=True):
    """Reusable workflow templates."""

    __tablename__ = "workflow_templates"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    topology_id: int = Field(foreign_key="topologies.id")
    default_input: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    is_public: bool = Field(default=False)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_identifier(value)


__all__ = [
    "WorkflowTemplate",
]
