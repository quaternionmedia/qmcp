"""Topology skeletons for the QMCP agent framework."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeVar

from sqlmodel import Field, SQLModel

from qmcp.agentframework.models import (
    AgentType,
    ChainOfCommandConfig,
    CompoundConfig,
    CrossCheckConfig,
    DebateConfig,
    DelegationConfig,
    EnsembleConfig,
    PipelineConfig,
    Topology,
    TopologyType,
)


class ExecutionContext(SQLModel):
    """Context passed through topology execution."""

    execution_id: str | None = None
    topology_id: int | None = None
    input_data: dict[str, Any] = Field(default_factory=dict)
    metadata_: dict[str, Any] = Field(default_factory=dict)
    round_number: int = 0
    parent_context: ExecutionContext | None = None


class AgentInvocationResult(SQLModel):
    """Result from a single agent invocation."""

    agent_instance_id: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    reasoning: str | None = None
    token_usage: dict[str, int] | None = None
    duration_ms: int | None = None
    error: str | None = None


T = TypeVar("T", bound="BaseTopology")


class BaseTopology(ABC):
    """Abstract base class for topology implementations."""

    topology_type: ClassVar[TopologyType]
    config_class: ClassVar[type[SQLModel]]

    def __init__(
        self,
        topology: Topology,
        agents: dict[str, AgentType],
        db_session: Any,
    ) -> None:
        self.topology = topology
        self.agents = agents
        self.db_session = db_session

    def get_config(self) -> SQLModel:
        return self.config_class(**self.topology.config)

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("Topology runtime is not implemented yet.")


class TopologyRegistry:
    """Registry for topology implementations."""

    _topologies: dict[TopologyType, type[BaseTopology]] = {}

    @classmethod
    def register(cls, topology_class: type[BaseTopology]) -> type[BaseTopology]:
        cls._topologies[topology_class.topology_type] = topology_class
        return topology_class

    @classmethod
    def get(cls, topology_type: TopologyType) -> type[BaseTopology] | None:
        return cls._topologies.get(topology_type)

    @classmethod
    def create(
        cls,
        topology: Topology,
        agents: dict[str, AgentType],
        db_session: Any,
    ) -> BaseTopology:
        topology_class = cls.get(topology.topology_type)
        if topology_class is None:
            raise ValueError(f"Unknown topology type: {topology.topology_type}")
        return topology_class(topology, agents, db_session)


def topology(cls: type[T]) -> type[T]:
    """Decorator to register a topology implementation."""
    return TopologyRegistry.register(cls)


@topology
class DebateTopology(BaseTopology):
    topology_type = TopologyType.DEBATE
    config_class = DebateConfig


@topology
class ChainOfCommandTopology(BaseTopology):
    topology_type = TopologyType.CHAIN_OF_COMMAND
    config_class = ChainOfCommandConfig


@topology
class DelegationTopology(BaseTopology):
    topology_type = TopologyType.DELEGATION
    config_class = DelegationConfig


@topology
class CrossCheckTopology(BaseTopology):
    topology_type = TopologyType.CROSS_CHECK
    config_class = CrossCheckConfig


@topology
class EnsembleTopology(BaseTopology):
    topology_type = TopologyType.ENSEMBLE
    config_class = EnsembleConfig


@topology
class PipelineTopology(BaseTopology):
    topology_type = TopologyType.PIPELINE
    config_class = PipelineConfig


@topology
class CompoundTopology(BaseTopology):
    topology_type = TopologyType.COMPOUND
    config_class = CompoundConfig


__all__ = [
    "ExecutionContext",
    "AgentInvocationResult",
    "BaseTopology",
    "TopologyRegistry",
    "topology",
    "DebateTopology",
    "ChainOfCommandTopology",
    "DelegationTopology",
    "CrossCheckTopology",
    "EnsembleTopology",
    "PipelineTopology",
    "CompoundTopology",
]
