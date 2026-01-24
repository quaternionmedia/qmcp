"""Agent-related entity models."""

from __future__ import annotations

from .base import (
    Any,
    UUID,
    Column,
    Field,
    JSON,
    SQLModel,
    datetime,
    field_validator,
    utc_now,
    uuid4,
    validate_identifier,
)
from ..enums import AgentRole, HealthStatus, SkillCategory
from ..configs import AgentCapability


class AgentType(SQLModel, table=True):
    """Persistent agent type definition."""

    __tablename__ = "agent_types"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    role: AgentRole
    version: str = Field(default="1.0.0")
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_identifier(value)

    def get_capabilities(self) -> list[AgentCapability]:
        raw_caps = self.config.get("capabilities", [])
        return [AgentCapability(**c) if isinstance(c, dict) else c for c in raw_caps]

    def has_capability(self, name: str) -> bool:
        return any(cap.name == name for cap in self.get_capabilities())


class AgentInstance(SQLModel, table=True):
    """A running instance of an agent type."""

    __tablename__ = "agent_instances"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    agent_type_id: int = Field(foreign_key="agent_types.id")
    execution_id: UUID | None = Field(default=None, foreign_key="executions.id")
    state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False),
    )
    health: HealthStatus = Field(default=HealthStatus.UNKNOWN)
    created_at: datetime = Field(default_factory=utc_now)
    last_active: datetime | None = Field(default=None)

    def update_state(self, updates: dict[str, Any]) -> None:
        self.state = {**self.state, **updates}
        self.last_active = utc_now()


class AgentSkill(SQLModel, table=True):
    """Mapping of skills to agent types."""

    __tablename__ = "agent_skills"

    id: int | None = Field(default=None, primary_key=True)
    agent_type_id: int = Field(foreign_key="agent_types.id")
    name: str = Field(min_length=1, max_length=64)
    category: SkillCategory
    proficiency: float = Field(default=0.8, ge=0.0, le=1.0)
    description: str | None = Field(default=None, max_length=256)


__all__ = [
    "AgentType",
    "AgentInstance",
    "AgentSkill",
]
