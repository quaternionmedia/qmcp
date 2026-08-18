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

    def get_typed_config(self) -> Any:
        """Return the topology config parsed into its typed config model.

        Resolves the topology type to the corresponding config class and
        validates ``self.config`` through it.

        Returns:
            An instance of the appropriate config class (e.g. ``DebateConfig``).

        Raises:
            ValueError: If no config class is registered for this topology type.
        """
        from ..configs.topology import (
            ChainOfCommandConfig,
            CompoundConfig,
            CouncilConfig,
            CrossCheckConfig,
            DebateConfig,
            DelegationConfig,
            EnsembleConfig,
            MeshConfig,
            PipelineConfig,
            RingConfig,
            StarConfig,
        )
        from ..enums import TopologyType

        _config_map = {
            TopologyType.DEBATE: DebateConfig,
            TopologyType.CHAIN_OF_COMMAND: ChainOfCommandConfig,
            TopologyType.DELEGATION: DelegationConfig,
            TopologyType.CROSS_CHECK: CrossCheckConfig,
            TopologyType.ENSEMBLE: EnsembleConfig,
            TopologyType.PIPELINE: PipelineConfig,
            TopologyType.COMPOUND: CompoundConfig,
            TopologyType.MESH: MeshConfig,
            TopologyType.STAR: StarConfig,
            TopologyType.RING: RingConfig,
            TopologyType.COUNCIL: CouncilConfig,
        }

        config_cls = _config_map.get(self.topology_type)
        if config_cls is None:
            raise ValueError(
                f"No config class registered for topology type {self.topology_type!r}"
            )
        return config_cls.model_validate(self.config)


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
