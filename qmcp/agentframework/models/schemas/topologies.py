"""Topology-related API schemas."""

from __future__ import annotations

from .base import Any, Field, SQLModel, datetime
from ..enums import TopologyType


class TopologyCreate(SQLModel):
    """Schema for creating a new topology."""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)
    topology_type: TopologyType
    config: dict[str, Any] = Field(default_factory=dict)
    agent_names: list[str] = Field(default_factory=list)


class TopologyUpdate(SQLModel):
    """Schema for updating a topology."""

    description: str | None = None
    config: dict[str, Any] | None = None
    version: str | None = None


class TopologyRead(SQLModel):
    """Schema for reading topology data."""

    id: int
    name: str
    description: str
    topology_type: TopologyType
    version: str
    config: dict[str, Any]
    created_at: datetime


class TopologySummary(SQLModel):
    """Compact schema for listing topologies."""

    id: int
    name: str
    topology_type: TopologyType
    version: str


__all__ = [
    "TopologyCreate",
    "TopologyUpdate",
    "TopologyRead",
    "TopologySummary",
]
