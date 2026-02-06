"""Agent configuration models."""

from __future__ import annotations

from .base import Any, Field, SQLModel
from ..enums import (
    AuthScope,
    CommunicationProtocol,
    LogLevel,
    Priority,
    SkillCategory,
)


class AgentCapability(SQLModel):
    """A capability that can be attached to an agent."""

    name: str = Field(description="Unique capability identifier")
    version: str = Field(default="1.0.0")
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)


class SkillConfig(SQLModel):
    """Configuration for an agent skill."""

    name: str = Field(description="Skill name")
    category: SkillCategory = Field(default=SkillCategory.REASONING)
    proficiency: float = Field(default=0.8, ge=0.0, le=1.0)
    priority: Priority = Field(default=Priority.NORMAL)


class RetryConfig(SQLModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_ms: int = Field(default=1000, ge=100)
    max_delay_ms: int = Field(default=30000, ge=1000)
    exponential_base: float = Field(default=2.0, ge=1.0, le=4.0)
    jitter: bool = Field(default=True)


class TimeoutConfig(SQLModel):
    """Configuration for timeout behavior."""

    request_timeout_ms: int = Field(default=30000, ge=1000)
    connection_timeout_ms: int = Field(default=5000, ge=500)
    idle_timeout_ms: int = Field(default=60000, ge=5000)
    total_timeout_ms: int = Field(default=300000, ge=10000)


class LoggingConfig(SQLModel):
    """Configuration for agent logging."""

    level: LogLevel = Field(default=LogLevel.INFO)
    include_timestamps: bool = Field(default=True)
    include_agent_id: bool = Field(default=True)
    log_messages: bool = Field(default=True)
    log_tool_calls: bool = Field(default=True)
    redact_sensitive: bool = Field(default=True)


class SecurityConfig(SQLModel):
    """Configuration for agent security."""

    scopes: list[AuthScope] = Field(default_factory=lambda: [AuthScope.READ, AuthScope.EXECUTE])
    require_approval: bool = Field(default=False)
    sandbox_enabled: bool = Field(default=True)
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    max_cost_per_execution: float | None = Field(default=None, ge=0.0)


class ResourceLimits(SQLModel):
    """Resource limits for agent execution."""

    max_tokens_per_call: int = Field(default=4096, gt=0)
    max_tokens_per_execution: int = Field(default=100000, gt=0)
    max_tool_calls: int = Field(default=50, ge=0)
    max_concurrent_requests: int = Field(default=5, ge=1)
    max_memory_mb: int = Field(default=512, ge=64)


class CommunicationConfig(SQLModel):
    """Configuration for agent communication."""

    protocol: CommunicationProtocol = Field(default=CommunicationProtocol.ASYNC)
    buffer_size: int = Field(default=100, ge=10)
    ack_required: bool = Field(default=False)
    compress_messages: bool = Field(default=False)
    encrypt_messages: bool = Field(default=False)


# Import ModelConfig for AgentConfig (avoiding circular import)
# This will be resolved at runtime through the __init__.py
class AgentConfig(SQLModel):
    """Configuration for an agent type.

    The `model` field accepts either a string model ID (e.g., "claude-sonnet-4-20250514")
    for simple configuration, or use `model_config_obj` for full model configuration.
    """

    # Model configuration - supports simple string ID or full ModelConfig
    model: str = Field(default="claude-sonnet-4-20250514", description="Model ID for simple config")
    # Note: model_config_obj is typed as Any to avoid circular imports
    # At runtime, it should be a ModelConfig instance
    model_config_obj: Any | None = Field(
        default=None,
        description="Full model configuration (overrides 'model' string if provided)",
    )

    # Sampling parameters (used when model is a string; ignored if model_config_obj is set)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, gt=0)
    system_prompt: str | None = Field(default=None)
    output_format: str | None = Field(default=None)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: int = Field(default=300, gt=0)
    max_tool_calls: int = Field(default=10, ge=0)
    capabilities: list[AgentCapability] = Field(default_factory=list)
    skills: list[SkillConfig] = Field(default_factory=list)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)


__all__ = [
    "AgentCapability",
    "SkillConfig",
    "RetryConfig",
    "TimeoutConfig",
    "LoggingConfig",
    "SecurityConfig",
    "ResourceLimits",
    "CommunicationConfig",
    "AgentConfig",
]
