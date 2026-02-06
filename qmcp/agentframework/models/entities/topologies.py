"""Topology-related entity models."""

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
from ..enums import TopologyType


class Topology(SQLModel, table=True):
    """Persistent topology definition."""

    __tablename__ = "topologies"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    topology_type: TopologyType
    version: str = Field(default="1.0.0")
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_identifier(value)


class TopologyMembership(SQLModel, table=True):
    """Links agents to topologies."""

    __tablename__ = "topology_memberships"

    id: int | None = Field(default=None, primary_key=True)
    topology_id: int = Field(foreign_key="topologies.id")
    agent_type_id: int = Field(foreign_key="agent_types.id")
    slot_name: str = Field(description="Named slot in topology")
    position: int = Field(default=0)
    config_overrides: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


__all__ = [
    "Topology",
    "TopologyMembership",
]
