"""Topology configuration models."""

from __future__ import annotations

from .base import Field, SQLModel
from ..enums import (
    AggregationMethod,
    ConsensusMethod,
    ErrorStrategy,
)


class DebateConfig(SQLModel):
    """Configuration for Debate topology."""

    max_rounds: int = Field(default=3, ge=1, le=10)
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.MEDIATOR_DECISION)
    convergence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    allow_early_termination: bool = Field(default=True)
    mediator_agent_name: str | None = Field(default=None)
    require_justification: bool = Field(default=True)
    min_argument_length: int = Field(default=50, ge=0)


class ChainOfCommandConfig(SQLModel):
    """Configuration for Chain of Command topology."""

    authority_levels: list[str] = Field(
        default_factory=lambda: ["commander", "lieutenant", "worker"]
    )
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_delegation_depth: int = Field(default=3, ge=1)
    require_acknowledgment: bool = Field(default=True)
    timeout_per_level_ms: int = Field(default=60000, ge=5000)


class DelegationConfig(SQLModel):
    """Configuration for Delegation topology."""

    routing_strategy: str = Field(default="capability_match")
    load_balance: bool = Field(default=True)
    fallback_agent_name: str | None = Field(default=None)
    max_queue_size: int = Field(default=100, ge=10)
    priority_queue: bool = Field(default=False)


class CrossCheckConfig(SQLModel):
    """Configuration for Cross-Check topology."""

    num_checkers: int = Field(default=3, ge=2, le=10)
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.MAJORITY_VOTE)
    require_unanimous_for_approval: bool = Field(default=False)
    checker_specializations: list[str] = Field(default_factory=list)
    independent_execution: bool = Field(default=True)
    share_reasoning: bool = Field(default=False)


class EnsembleConfig(SQLModel):
    """Configuration for Ensemble topology."""

    aggregation_method: AggregationMethod = Field(default=AggregationMethod.SYNTHESIS)
    diversity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    failure_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_responses: int = Field(default=2, ge=1)
    weight_by_confidence: bool = Field(default=True)


class PipelineConfig(SQLModel):
    """Configuration for Pipeline topology."""

    stages: list[str] = Field(default_factory=list)
    checkpoint_after: list[str] = Field(default_factory=list)
    retry_failed_stages: bool = Field(default=True)
    max_stage_retries: int = Field(default=2, ge=0)
    error_strategy: ErrorStrategy = Field(default=ErrorStrategy.RETRY)
    parallel_stages: list[list[str]] = Field(default_factory=list)


class CompoundConfig(SQLModel):
    """Configuration for Compound topology."""

    sub_topologies: list[str] = Field(default_factory=list)
    composition_type: str = Field(default="sequential")
    share_context: bool = Field(default=True)
    merge_outputs: bool = Field(default=True)


class MeshConfig(SQLModel):
    """Configuration for Mesh topology."""

    connection_density: float = Field(default=0.5, ge=0.0, le=1.0)
    bidirectional: bool = Field(default=True)
    broadcast_enabled: bool = Field(default=True)
    max_hops: int = Field(default=3, ge=1)


class StarConfig(SQLModel):
    """Configuration for Star topology."""

    hub_agent_name: str = Field(description="Central hub agent")
    spoke_timeout_ms: int = Field(default=30000, ge=5000)
    parallel_spokes: bool = Field(default=True)
    hub_aggregation: AggregationMethod = Field(default=AggregationMethod.SYNTHESIS)


class RingConfig(SQLModel):
    """Configuration for Ring topology."""

    direction: str = Field(default="clockwise")
    max_iterations: int = Field(default=3, ge=1)
    termination_condition: str = Field(default="consensus")
    pass_full_context: bool = Field(default=False)


class CheckpointConfig(SQLModel):
    """Configuration for execution checkpoints."""

    enabled: bool = Field(default=True)
    interval_ms: int = Field(default=60000, ge=10000)
    max_checkpoints: int = Field(default=10, ge=1)
    persist_to_disk: bool = Field(default=False)


class MetricsConfig(SQLModel):
    """Configuration for metrics collection."""

    enabled: bool = Field(default=True)
    collect_latency: bool = Field(default=True)
    collect_tokens: bool = Field(default=True)
    collect_errors: bool = Field(default=True)
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    export_interval_ms: int = Field(default=30000, ge=5000)


__all__ = [
    "DebateConfig",
    "ChainOfCommandConfig",
    "DelegationConfig",
    "CrossCheckConfig",
    "EnsembleConfig",
    "PipelineConfig",
    "CompoundConfig",
    "MeshConfig",
    "StarConfig",
    "RingConfig",
    "CheckpointConfig",
    "MetricsConfig",
]
