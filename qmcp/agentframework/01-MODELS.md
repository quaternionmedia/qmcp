# QMCP Agent Framework: Models Specification

Note: Design reference. Implementation status is documented in docs/agentframework.md.


## Overview

This document specifies the SQLModel/Pydantic data models that form the foundation of the agent framework. All models follow QMCP conventions for consistency with existing tool invocations and human-in-the-loop patterns.

## Dependencies

```python
# Required imports for all model files
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import field_validator, model_validator
from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel
```

## Enumerations

### File: `qmcp/agentframework/models/enums.py`

```python
"""Agent framework enumerations."""

from enum import Enum


class AgentRole(str, Enum):
    """
    Defines the primary function of an agent within a topology.
    
    Each role implies certain capabilities and behavioral expectations.
    """
    PLANNER = "planner"           # Strategic planning, task decomposition
    EXECUTOR = "executor"         # Task execution, action taking
    REVIEWER = "reviewer"         # Quality assurance, validation
    CRITIC = "critic"             # Adversarial analysis, finding flaws
    SYNTHESIZER = "synthesizer"   # Information aggregation, summarization
    SPECIALIST = "specialist"     # Domain-specific expertise
    COORDINATOR = "coordinator"   # Orchestration, delegation
    OBSERVER = "observer"         # Monitoring, logging, metrics


class TopologyType(str, Enum):
    """
    Defines the collaboration pattern for a group of agents.
    
    Each type has distinct execution semantics and configuration options.
    """
    DEBATE = "debate"                 # Structured argumentation
    CHAIN_OF_COMMAND = "chain"        # Hierarchical delegation
    DELEGATION = "delegation"         # Capability-based routing
    CROSS_CHECK = "crosscheck"        # Independent validation
    ENSEMBLE = "ensemble"             # Parallel aggregation
    PIPELINE = "pipeline"             # Sequential processing
    COMPOUND = "compound"             # Nested topologies


class ExecutionStatus(str, Enum):
    """Status of a topology execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"           # Awaiting human input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(str, Enum):
    """Type of inter-agent message."""
    REQUEST = "request"         # Task assignment
    RESPONSE = "response"       # Task completion
    FEEDBACK = "feedback"       # Review/critique
    ESCALATION = "escalation"   # Authority escalation
    BROADCAST = "broadcast"     # Multi-recipient
    SYSTEM = "system"           # Framework messages


class ConsensusMethod(str, Enum):
    """Methods for reaching consensus in multi-agent scenarios."""
    MAJORITY_VOTE = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED_VOTE = "weighted"
    MEDIATOR_DECISION = "mediator"
    FIRST_AGREEMENT = "first_agree"


class AggregationMethod(str, Enum):
    """Methods for aggregating ensemble outputs."""
    VOTE = "vote"               # Most common answer
    AVERAGE = "average"         # Numeric averaging
    WEIGHTED = "weighted"       # Weighted combination
    CONCAT = "concat"           # Concatenate all
    BEST_OF = "best_of"         # Highest confidence
    SYNTHESIS = "synthesis"     # LLM synthesis
```

## Core Agent Models

### File: `qmcp/agentframework/models/agent.py`

```python
"""Agent type and instance models."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from pydantic import field_validator, model_validator
from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel

from .enums import AgentRole

if TYPE_CHECKING:
    from .topology import TopologyMembership
    from .execution import Message


# ============================================================================
# Base Models (Non-Table)
# ============================================================================

class AgentCapability(SQLModel):
    """
    Represents a single capability that can be attached to an agent.
    
    Capabilities are composable mixins that provide specific functionality.
    """
    name: str = Field(description="Unique capability identifier")
    version: str = Field(default="1.0.0", description="Capability version")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Capability-specific configuration"
    )
    enabled: bool = Field(default=True, description="Whether capability is active")


class AgentConfig(SQLModel):
    """
    Configuration for an agent type.
    
    This is embedded in AgentType.config as JSON.
    """
    # LLM Settings
    model: str = Field(default="claude-sonnet-4-20250514", description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    
    # Behavioral Settings
    system_prompt: Optional[str] = Field(default=None)
    output_format: Optional[str] = Field(
        default=None,
        description="Expected output format (json, markdown, etc.)"
    )
    
    # Resource Limits
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=300, gt=0)
    max_tool_calls: int = Field(default=10, ge=0)
    
    # Capabilities (loaded from mixin registry)
    capabilities: list[AgentCapability] = Field(default_factory=list)


class AgentTypeBase(SQLModel):
    """Base fields for AgentType."""
    name: str = Field(
        index=True,
        unique=True,
        min_length=1,
        max_length=64,
        description="Unique agent type identifier"
    )
    description: str = Field(
        min_length=1,
        max_length=1024,
        description="Human-readable description"
    )
    role: AgentRole = Field(description="Primary agent role")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")


class AgentTypeCreate(AgentTypeBase):
    """Schema for creating a new agent type."""
    config: AgentConfig = Field(default_factory=AgentConfig)


class AgentTypeUpdate(SQLModel):
    """Schema for updating an agent type (all fields optional)."""
    description: Optional[str] = None
    role: Optional[AgentRole] = None
    version: Optional[str] = None
    config: Optional[AgentConfig] = None


class AgentTypeRead(AgentTypeBase):
    """Schema for reading agent type data."""
    id: int
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Database Tables
# ============================================================================

class AgentType(AgentTypeBase, table=True):
    """
    Persistent agent type definition.
    
    An AgentType defines a template for creating agent instances.
    It includes role, capabilities, and configuration.
    """
    __tablename__ = "agent_types"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    config: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    instances: list["AgentInstance"] = Relationship(back_populates="agent_type")
    topology_memberships: list["TopologyMembership"] = Relationship(
        back_populates="agent_type"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must contain only alphanumeric, underscore, or hyphen")
        return v.lower()
    
    def get_capabilities(self) -> list[AgentCapability]:
        """Extract capabilities from config."""
        raw_caps = self.config.get("capabilities", [])
        return [AgentCapability(**c) for c in raw_caps]
    
    def has_capability(self, name: str) -> bool:
        """Check if agent has a specific capability."""
        return any(c.name == name for c in self.get_capabilities())


class AgentInstance(SQLModel, table=True):
    """
    A running instance of an agent type.
    
    Instances track state for a specific execution context.
    """
    __tablename__ = "agent_instances"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"type_": "UUID"}
    )
    agent_type_id: int = Field(foreign_key="agent_types.id")
    execution_id: Optional[UUID] = Field(
        default=None,
        foreign_key="executions.id"
    )
    
    # Instance State
    state: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
        description="Mutable instance state (memory, context, etc.)"
    )
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False),
        description="Instance metadata"
    )
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = Field(default=None)
    
    # Relationships
    agent_type: AgentType = Relationship(back_populates="instances")
    sent_messages: list["Message"] = Relationship(
        back_populates="sender",
        sa_relationship_kwargs={"foreign_keys": "[Message.sender_id]"}
    )
    received_messages: list["Message"] = Relationship(
        back_populates="recipient",
        sa_relationship_kwargs={"foreign_keys": "[Message.recipient_id]"}
    )
    
    def update_state(self, updates: dict[str, Any]) -> None:
        """Merge updates into instance state."""
        self.state = {**self.state, **updates}
        self.last_active = datetime.utcnow()
```

## Topology Models

### File: `qmcp/agentframework/models/topology.py`

```python
"""Topology and membership models."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from pydantic import field_validator, model_validator
from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel

from .enums import (
    AggregationMethod,
    ConsensusMethod,
    TopologyType,
)

if TYPE_CHECKING:
    from .agent import AgentType
    from .execution import Execution


# ============================================================================
# Topology Configuration Models (Non-Table)
# ============================================================================

class DebateConfig(SQLModel):
    """Configuration for Debate topology."""
    max_rounds: int = Field(default=3, ge=1, le=10)
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.MEDIATOR_DECISION)
    convergence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    allow_early_termination: bool = Field(default=True)
    mediator_agent_name: Optional[str] = Field(default=None)


class ChainOfCommandConfig(SQLModel):
    """Configuration for Chain of Command topology."""
    authority_levels: list[str] = Field(
        default_factory=lambda: ["commander", "lieutenant", "worker"]
    )
    escalation_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence below which to escalate"
    )
    max_delegation_depth: int = Field(default=3, ge=1)


class DelegationConfig(SQLModel):
    """Configuration for Delegation topology."""
    routing_strategy: str = Field(
        default="capability_match",
        description="How to match tasks to specialists"
    )
    load_balance: bool = Field(default=True)
    fallback_agent_name: Optional[str] = Field(default=None)


class CrossCheckConfig(SQLModel):
    """Configuration for Cross-Check topology."""
    num_checkers: int = Field(default=3, ge=2, le=10)
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.MAJORITY_VOTE)
    require_unanimous_for_approval: bool = Field(default=False)
    checker_specializations: list[str] = Field(default_factory=list)


class EnsembleConfig(SQLModel):
    """Configuration for Ensemble topology."""
    aggregation_method: AggregationMethod = Field(default=AggregationMethod.SYNTHESIS)
    diversity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight given to diverse outputs"
    )
    failure_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Max fraction of agents that can fail"
    )


class PipelineConfig(SQLModel):
    """Configuration for Pipeline topology."""
    stages: list[str] = Field(
        default_factory=list,
        description="Ordered list of stage names"
    )
    checkpoint_after: list[str] = Field(
        default_factory=list,
        description="Stages after which to checkpoint"
    )
    retry_failed_stages: bool = Field(default=True)
    max_stage_retries: int = Field(default=2, ge=0)


class CompoundConfig(SQLModel):
    """Configuration for Compound (nested) topology."""
    sub_topologies: list[str] = Field(
        default_factory=list,
        description="Names of nested topologies"
    )
    composition_type: str = Field(
        default="sequential",
        description="How to compose: sequential, parallel, conditional"
    )


# ============================================================================
# Base Models (Non-Table)
# ============================================================================

class TopologyBase(SQLModel):
    """Base fields for Topology."""
    name: str = Field(
        index=True,
        unique=True,
        min_length=1,
        max_length=64
    )
    description: str = Field(min_length=1, max_length=1024)
    topology_type: TopologyType = Field(description="Type of collaboration pattern")
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")


class TopologyCreate(TopologyBase):
    """Schema for creating a new topology."""
    config: dict[str, Any] = Field(default_factory=dict)
    agent_names: list[str] = Field(
        default_factory=list,
        description="Names of agents to include"
    )


class TopologyUpdate(SQLModel):
    """Schema for updating a topology."""
    description: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    version: Optional[str] = None


class TopologyRead(TopologyBase):
    """Schema for reading topology data."""
    id: int
    config: dict[str, Any]
    created_at: datetime
    agent_count: int


# ============================================================================
# Database Tables
# ============================================================================

class Topology(TopologyBase, table=True):
    """
    Persistent topology definition.
    
    A Topology defines how multiple agents collaborate.
    """
    __tablename__ = "topologies"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    config: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    memberships: list["TopologyMembership"] = Relationship(back_populates="topology")
    executions: list["Execution"] = Relationship(back_populates="topology")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Name must contain only alphanumeric, underscore, or hyphen")
        return v.lower()
    
    def get_typed_config(self) -> SQLModel:
        """Get configuration as typed model based on topology_type."""
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
    """
    Links agents to topologies with role-specific configuration.
    """
    __tablename__ = "topology_memberships"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    topology_id: int = Field(foreign_key="topologies.id")
    agent_type_id: int = Field(foreign_key="agent_types.id")
    
    # Role within topology
    slot_name: str = Field(
        description="Named slot in topology (e.g., 'mediator', 'checker_1')"
    )
    position: int = Field(
        default=0,
        description="Order in sequential topologies"
    )
    
    # Role-specific overrides
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
        description="Overrides for agent config in this topology"
    )
    
    # Relationships
    topology: Topology = Relationship(back_populates="memberships")
    agent_type: "AgentType" = Relationship(back_populates="topology_memberships")
```

## Execution Models

### File: `qmcp/agentframework/models/execution.py`

```python
"""Execution, message, and result models."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel

from .enums import ExecutionStatus, MessageType

if TYPE_CHECKING:
    from .agent import AgentInstance
    from .topology import Topology


# ============================================================================
# Base Models (Non-Table)
# ============================================================================

class ExecutionCreate(SQLModel):
    """Schema for creating a new execution."""
    topology_name: str
    input_data: dict[str, Any]
    metadata_: Optional[dict[str, Any]] = None


class ExecutionRead(SQLModel):
    """Schema for reading execution data."""
    id: UUID
    topology_id: int
    status: ExecutionStatus
    input_data: dict[str, Any]
    output_data: Optional[dict[str, Any]]
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    error: Optional[str]


class MessageCreate(SQLModel):
    """Schema for creating an inter-agent message."""
    sender_id: UUID
    recipient_id: Optional[UUID] = None  # None for broadcasts
    message_type: MessageType
    content: dict[str, Any]


class MessageRead(SQLModel):
    """Schema for reading message data."""
    id: UUID
    execution_id: UUID
    sender_id: UUID
    recipient_id: Optional[UUID]
    message_type: MessageType
    content: dict[str, Any]
    created_at: datetime


# ============================================================================
# Database Tables
# ============================================================================

class Execution(SQLModel, table=True):
    """
    Records a single topology execution.
    
    Tracks input, output, status, timing, and errors.
    """
    __tablename__ = "executions"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"type_": "UUID"}
    )
    topology_id: int = Field(foreign_key="topologies.id")
    
    # Execution State
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    input_data: dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False)
    )
    output_data: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True)
    )
    
    # Metadata
    correlation_id: Optional[str] = Field(
        default=None,
        index=True,
        description="External correlation ID"
    )
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSON, nullable=False)
    )
    
    # Lifecycle
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Error tracking
    error: Optional[str] = Field(default=None)
    error_details: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True)
    )
    
    # Relationships
    topology: "Topology" = Relationship(back_populates="executions")
    messages: list["Message"] = Relationship(back_populates="execution")
    results: list["Result"] = Relationship(back_populates="execution")
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate execution duration in milliseconds."""
        if self.completed_at and self.started_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None
    
    def mark_complete(self, output: dict[str, Any]) -> None:
        """Mark execution as completed with output."""
        self.status = ExecutionStatus.COMPLETED
        self.output_data = output
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self, error: str, details: Optional[dict] = None) -> None:
        """Mark execution as failed with error."""
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.error_details = details
        self.completed_at = datetime.utcnow()


class Message(SQLModel, table=True):
    """
    Inter-agent message within an execution.
    
    Tracks all communication between agents for debugging and analysis.
    """
    __tablename__ = "messages"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"type_": "UUID"}
    )
    execution_id: UUID = Field(foreign_key="executions.id")
    
    # Participants
    sender_id: UUID = Field(foreign_key="agent_instances.id")
    recipient_id: Optional[UUID] = Field(
        default=None,
        foreign_key="agent_instances.id"
    )
    
    # Message Content
    message_type: MessageType
    content: dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False)
    )
    
    # Metadata
    round_number: int = Field(
        default=0,
        description="Round number in iterative topologies"
    )
    sequence_number: int = Field(
        default=0,
        description="Sequential order within execution"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    execution: Execution = Relationship(back_populates="messages")
    sender: "AgentInstance" = Relationship(
        back_populates="sent_messages",
        sa_relationship_kwargs={"foreign_keys": "[Message.sender_id]"}
    )
    recipient: Optional["AgentInstance"] = Relationship(
        back_populates="received_messages",
        sa_relationship_kwargs={"foreign_keys": "[Message.recipient_id]"}
    )


class Result(SQLModel, table=True):
    """
    Individual agent result within an execution.
    
    Tracks each agent's contribution for aggregation and analysis.
    """
    __tablename__ = "results"
    
    id: Optional[UUID] = Field(
        default_factory=uuid4,
        primary_key=True,
        sa_column_kwargs={"type_": "UUID"}
    )
    execution_id: UUID = Field(foreign_key="executions.id")
    agent_instance_id: UUID = Field(foreign_key="agent_instances.id")
    
    # Result Content
    output: dict[str, Any] = Field(
        sa_column=Column(JSON, nullable=False)
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in result"
    )
    
    # Metadata
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for this result"
    )
    token_usage: Optional[dict[str, int]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True)
    )
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Relationships
    execution: Execution = Relationship(back_populates="results")


# ============================================================================
# Model Registry
# ============================================================================

# All tables for migration
__all_tables__ = [
    "AgentType",
    "AgentInstance", 
    "Topology",
    "TopologyMembership",
    "Execution",
    "Message",
    "Result",
]
```

## Package Initialization

### File: `qmcp/agentframework/models/__init__.py`

```python
"""Agent framework data models."""

from .enums import (
    AgentRole,
    AggregationMethod,
    ConsensusMethod,
    ExecutionStatus,
    MessageType,
    TopologyType,
)

from .agent import (
    AgentCapability,
    AgentConfig,
    AgentInstance,
    AgentType,
    AgentTypeBase,
    AgentTypeCreate,
    AgentTypeRead,
    AgentTypeUpdate,
)

from .topology import (
    ChainOfCommandConfig,
    CompoundConfig,
    CrossCheckConfig,
    DebateConfig,
    DelegationConfig,
    EnsembleConfig,
    PipelineConfig,
    Topology,
    TopologyBase,
    TopologyCreate,
    TopologyMembership,
    TopologyRead,
    TopologyUpdate,
)

from .execution import (
    Execution,
    ExecutionCreate,
    ExecutionRead,
    Message,
    MessageCreate,
    MessageRead,
    Result,
)

__all__ = [
    # Enums
    "AgentRole",
    "AggregationMethod",
    "ConsensusMethod",
    "ExecutionStatus",
    "MessageType",
    "TopologyType",
    # Agent models
    "AgentCapability",
    "AgentConfig",
    "AgentInstance",
    "AgentType",
    "AgentTypeBase",
    "AgentTypeCreate",
    "AgentTypeRead",
    "AgentTypeUpdate",
    # Topology models
    "ChainOfCommandConfig",
    "CompoundConfig",
    "CrossCheckConfig",
    "DebateConfig",
    "DelegationConfig",
    "EnsembleConfig",
    "PipelineConfig",
    "Topology",
    "TopologyBase",
    "TopologyCreate",
    "TopologyMembership",
    "TopologyRead",
    "TopologyUpdate",
    # Execution models
    "Execution",
    "ExecutionCreate",
    "ExecutionRead",
    "Message",
    "MessageCreate",
    "MessageRead",
    "Result",
]
```

## Database Migration

### File: `qmcp/agentframework/models/migrations.py`

```python
"""Database migration utilities for agent models."""

from sqlmodel import SQLModel

from . import (
    AgentInstance,
    AgentType,
    Execution,
    Message,
    Result,
    Topology,
    TopologyMembership,
)


def get_all_tables() -> list[type[SQLModel]]:
    """Return all SQLModel table classes for migration."""
    return [
        AgentType,
        AgentInstance,
        Topology,
        TopologyMembership,
        Execution,
        Message,
        Result,
    ]


async def create_agent_tables(engine) -> None:
    """Create all agent-related tables."""
    from sqlmodel import SQLModel as SM
    
    # Import all models to ensure they're registered
    _ = get_all_tables()
    
    async with engine.begin() as conn:
        await conn.run_sync(SM.metadata.create_all)


async def drop_agent_tables(engine) -> None:
    """Drop all agent-related tables (use with caution!)."""
    from sqlmodel import SQLModel as SM
    
    _ = get_all_tables()
    
    async with engine.begin() as conn:
        await conn.run_sync(SM.metadata.drop_all)
```

## Implementation Notes

### Type Coercion

SQLModel handles JSON columns specially. When storing complex types:

```python
# Store AgentConfig as dict
agent = AgentType(
    name="planner",
    description="Strategic planner",
    role=AgentRole.PLANNER,
    config=AgentConfig(model="claude-sonnet-4-20250514").model_dump()
)
```

### Relationship Loading

For performance, relationships are lazy-loaded by default. Use selectin loading for eager loading:

```python
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

async def get_topology_with_agents(session: AsyncSession, name: str):
    stmt = (
        select(Topology)
        .where(Topology.name == name)
        .options(selectinload(Topology.memberships))
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
```

### Validation

Pydantic validators run at instantiation. Add custom validators for complex constraints:

```python
@model_validator(mode="after")
def validate_topology_config(self) -> "Topology":
    """Ensure config matches topology_type."""
    required_fields = {
        TopologyType.DEBATE: ["max_rounds"],
        TopologyType.PIPELINE: ["stages"],
    }
    if self.topology_type in required_fields:
        for field in required_fields[self.topology_type]:
            if field not in self.config:
                raise ValueError(f"{field} required for {self.topology_type}")
    return self
```

## Next Steps

1. Implement mixin system (see `02-MIXINS.md`)
2. Implement topology execution engine (see `03-TOPOLOGIES.md`)
3. Add comprehensive tests (see `05-TESTS.md`)
