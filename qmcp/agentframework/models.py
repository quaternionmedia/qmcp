"""Data models for the QMCP agent framework."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import field_validator
from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class AgentRole(str, Enum):
    """Primary function of an agent within a topology."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"


class TopologyType(str, Enum):
    """Collaboration pattern for a group of agents."""

    DEBATE = "debate"
    CHAIN_OF_COMMAND = "chain"
    DELEGATION = "delegation"
    CROSS_CHECK = "crosscheck"
    ENSEMBLE = "ensemble"
    PIPELINE = "pipeline"
    COMPOUND = "compound"


class ExecutionStatus(str, Enum):
    """Status of a topology execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(str, Enum):
    """Type of inter-agent message."""

    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"
    BROADCAST = "broadcast"
    SYSTEM = "system"


class ConsensusMethod(str, Enum):
    """Methods for reaching consensus."""

    MAJORITY_VOTE = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED_VOTE = "weighted"
    MEDIATOR_DECISION = "mediator"
    FIRST_AGREEMENT = "first_agree"


class AggregationMethod(str, Enum):
    """Methods for aggregating ensemble outputs."""

    VOTE = "vote"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CONCAT = "concat"
    BEST_OF = "best_of"
    SYNTHESIS = "synthesis"


class AgentCapability(SQLModel):
    """A capability that can be attached to an agent."""

    name: str = Field(description="Unique capability identifier")
    version: str = Field(default="1.0.0")
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)


class AgentConfig(SQLModel):
    """Configuration for an agent type."""

    model: str = Field(default="claude-sonnet-4-20250514")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    system_prompt: str | None = Field(default=None)
    output_format: str | None = Field(default=None)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=300, gt=0)
    max_tool_calls: int = Field(default=10, ge=0)
    capabilities: list[AgentCapability] = Field(default_factory=list)


class DebateConfig(SQLModel):
    """Configuration for Debate topology."""

    max_rounds: int = Field(default=3, ge=1, le=10)
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.MEDIATOR_DECISION)
    convergence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    allow_early_termination: bool = Field(default=True)
    mediator_agent_name: str | None = Field(default=None)


class ChainOfCommandConfig(SQLModel):
    """Configuration for Chain of Command topology."""

    authority_levels: list[str] = Field(
        default_factory=lambda: ["commander", "lieutenant", "worker"]
    )
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_delegation_depth: int = Field(default=3, ge=1)


class DelegationConfig(SQLModel):
    """Configuration for Delegation topology."""

    routing_strategy: str = Field(default="capability_match")
    load_balance: bool = Field(default=True)
    fallback_agent_name: str | None = Field(default=None)


class CrossCheckConfig(SQLModel):
    """Configuration for Cross-Check topology."""

    num_checkers: int = Field(default=3, ge=2, le=10)
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.MAJORITY_VOTE)
    require_unanimous_for_approval: bool = Field(default=False)
    checker_specializations: list[str] = Field(default_factory=list)


class EnsembleConfig(SQLModel):
    """Configuration for Ensemble topology."""

    aggregation_method: AggregationMethod = Field(default=AggregationMethod.SYNTHESIS)
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    failure_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PipelineConfig(SQLModel):
    """Configuration for Pipeline topology."""

    stages: list[str] = Field(default_factory=list)
    checkpoint_after: list[str] = Field(default_factory=list)
    retry_failed_stages: bool = Field(default=True)
    max_stage_retries: int = Field(default=2, ge=0)


class CompoundConfig(SQLModel):
    """Configuration for Compound topology."""

    sub_topologies: list[str] = Field(default_factory=list)
    composition_type: str = Field(default="sequential")


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
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must be alphanumeric with underscores/hyphens")
        return value.lower()

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
    created_at: datetime = Field(default_factory=utc_now)
    last_active: datetime | None = Field(default=None)

    def update_state(self, updates: dict[str, Any]) -> None:
        self.state = {**self.state, **updates}
        self.last_active = utc_now()


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
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must be alphanumeric with underscores/hyphens")
        return value.lower()

    def get_typed_config(self) -> SQLModel:
        config_map = {
            TopologyType.DEBATE: DebateConfig,
            TopologyType.CHAIN_OF_COMMAND: ChainOfCommandConfig,
            TopologyType.DELEGATION: DelegationConfig,
            TopologyType.CROSS_CHECK: CrossCheckConfig,
            TopologyType.ENSEMBLE: EnsembleConfig,
            TopologyType.PIPELINE: PipelineConfig,
            TopologyType.COMPOUND: CompoundConfig,
        }
        config_class = config_map.get(self.topology_type)
        if config_class:
            return config_class(**self.config)
        return SQLModel()


class TopologyMembership(SQLModel, table=True):
    """Links agents to topologies."""

    __tablename__ = "topology_memberships"

    id: int | None = Field(default=None, primary_key=True)
    topology_id: int = Field(foreign_key="topologies.id")
    agent_type_id: int = Field(foreign_key="agent_types.id")
    slot_name: str = Field(description="Named slot in topology")
    position: int = Field(default=0)
    config_overrides: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))


class Execution(SQLModel, table=True):
    """Records a single topology execution."""

    __tablename__ = "executions"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    topology_id: int = Field(foreign_key="topologies.id")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    input_data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    output_data: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    correlation_id: str | None = Field(default=None, index=True)
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False),
    )
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = Field(default=None)
    error: str | None = Field(default=None)
    error_details: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )

    @property
    def duration_ms(self) -> int | None:
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    def mark_complete(self, output: dict[str, Any]) -> None:
        self.status = ExecutionStatus.COMPLETED
        self.output_data = output
        self.completed_at = utc_now()

    def mark_failed(self, error: str, details: dict[str, Any] | None = None) -> None:
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.error_details = details
        self.completed_at = utc_now()


class Message(SQLModel, table=True):
    """Inter-agent message within an execution."""

    __tablename__ = "messages"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    sender_id: UUID = Field(foreign_key="agent_instances.id")
    recipient_id: UUID | None = Field(default=None, foreign_key="agent_instances.id")
    message_type: MessageType
    content: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    round_number: int = Field(default=0)
    sequence_number: int = Field(default=0)
    created_at: datetime = Field(default_factory=utc_now)


class Result(SQLModel, table=True):
    """Individual agent result within an execution."""

    __tablename__ = "results"

    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    execution_id: UUID = Field(foreign_key="executions.id")
    agent_instance_id: UUID = Field(foreign_key="agent_instances.id")
    output: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str | None = Field(default=None)
    token_usage: dict[str, int] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = Field(default=None)


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


class ExecutionCreate(SQLModel):
    """Schema for creating a new execution."""

    topology_name: str
    input_data: dict[str, Any]
    metadata_: dict[str, Any] | None = None


class ExecutionRead(SQLModel):
    """Schema for reading execution data."""

    id: UUID
    topology_id: int
    status: ExecutionStatus
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    error: str | None


__all__ = [
    "AgentRole",
    "TopologyType",
    "ExecutionStatus",
    "MessageType",
    "ConsensusMethod",
    "AggregationMethod",
    "AgentCapability",
    "AgentConfig",
    "DebateConfig",
    "ChainOfCommandConfig",
    "DelegationConfig",
    "CrossCheckConfig",
    "EnsembleConfig",
    "PipelineConfig",
    "CompoundConfig",
    "AgentType",
    "AgentInstance",
    "Topology",
    "TopologyMembership",
    "Execution",
    "Message",
    "Result",
    "AgentTypeCreate",
    "AgentTypeUpdate",
    "AgentTypeRead",
    "TopologyCreate",
    "TopologyUpdate",
    "TopologyRead",
    "ExecutionCreate",
    "ExecutionRead",
]
