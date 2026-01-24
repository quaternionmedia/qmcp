"""Agent-related API schemas."""

from __future__ import annotations

from .base import Any, UUID, Field, SQLModel, datetime
from ..enums import AgentRole, HealthStatus, SkillCategory


# =============================================================================
# Agent Type Schemas
# =============================================================================


class AgentTypeCreate(SQLModel):
    """Schema for creating a new agent type."""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    role: AgentRole
    config: dict[str, Any] = Field(default_factory=dict)


class AgentTypeUpdate(SQLModel):
    """Schema for updating an agent type."""

    description: str | None = None
    role: AgentRole | None = None
    version: str | None = None
    config: dict[str, Any] | None = None


class AgentTypeRead(SQLModel):
    """Schema for reading agent type data."""

    id: int
    name: str
    description: str
    role: AgentRole
    version: str
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AgentTypeSummary(SQLModel):
    """Compact schema for listing agent types."""

    id: int
    name: str
    role: AgentRole
    version: str


# =============================================================================
# Agent Instance Schemas
# =============================================================================


class AgentInstanceRead(SQLModel):
    """Schema for reading agent instance data."""

    id: UUID
    agent_type_id: int
    execution_id: UUID | None
    state: dict[str, Any]
    health: HealthStatus
    created_at: datetime
    last_active: datetime | None


class AgentInstanceSummary(SQLModel):
    """Compact schema for listing agent instances."""

    id: UUID
    agent_type_id: int
    health: HealthStatus
    last_active: datetime | None


# =============================================================================
# Skill Schemas
# =============================================================================


class SkillCreate(SQLModel):
    """Schema for creating an agent skill."""

    name: str = Field(min_length=1, max_length=64)
    category: SkillCategory
    proficiency: float = Field(default=0.8, ge=0.0, le=1.0)
    description: str | None = None


class SkillRead(SQLModel):
    """Schema for reading skill data."""

    id: int
    agent_type_id: int
    name: str
    category: SkillCategory
    proficiency: float
    description: str | None


__all__ = [
    # Agent Type
    "AgentTypeCreate",
    "AgentTypeUpdate",
    "AgentTypeRead",
    "AgentTypeSummary",
    # Agent Instance
    "AgentInstanceRead",
    "AgentInstanceSummary",
    # Skill
    "SkillCreate",
    "SkillRead",
]
