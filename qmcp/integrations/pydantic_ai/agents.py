"""Agent creation utilities for PydanticAI integration.

Provides adapters to create PydanticAI agents from QMCP configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, Generic

if TYPE_CHECKING:
    from qmcp.agentframework.models import AgentConfig, ModelConfig

try:
    from pydantic_ai import Agent
    from pydantic_ai.agent import AgentRunResult

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None  # type: ignore
    AgentRunResult = None  # type: ignore

from .models import model_to_pydantic_ai, get_model_settings, get_usage_limits

T = TypeVar("T")
DepsT = TypeVar("DepsT")


def _check_pydantic_ai() -> None:
    """Raise ImportError if pydantic-ai is not installed."""
    if not PYDANTIC_AI_AVAILABLE:
        raise ImportError(
            "pydantic-ai is required for this functionality. "
            "Install it with: pip install qmcp[pydantic-ai]"
        )


def create_agent(
    model: ModelConfig | str,
    *,
    system_prompt: str | None = None,
    deps_type: type[DepsT] | None = None,
    output_type: type[T] | None = None,
    tools: list[Any] | None = None,
    toolsets: list[Any] | None = None,
    retries: int = 1,
    **kwargs: Any,
) -> Agent[DepsT, T]:
    """Create a PydanticAI Agent from a QMCP model configuration.

    This is the primary way to create agents that leverage QMCP's model
    registry while using PydanticAI's agent runtime.

    Args:
        model: A QMCP ModelConfig or model ID string
        system_prompt: The system prompt for the agent
        deps_type: Type for dependency injection
        output_type: Expected output type (Pydantic model or primitive)
        tools: List of tool functions to register
        toolsets: List of toolsets (including QMCPToolset)
        retries: Number of retries on validation failure
        **kwargs: Additional arguments passed to Agent constructor

    Returns:
        A configured PydanticAI Agent

    Example:
        >>> from qmcp.agentframework.models import Models
        >>> from qmcp.integrations.pydantic_ai import create_agent
        >>>
        >>> agent = create_agent(
        ...     Models.CLAUDE_SONNET_4,
        ...     system_prompt="You are a helpful assistant.",
        ... )
        >>> result = await agent.run("Hello!")
    """
    _check_pydantic_ai()

    # Convert model to PydanticAI string
    model_str = model_to_pydantic_ai(model)

    # Build model settings if we have a ModelConfig
    model_settings = None
    if hasattr(model, "sampling"):
        model_settings = get_model_settings(model)
        if model_settings:
            kwargs.setdefault("model_settings", model_settings)

    # Build agent kwargs
    agent_kwargs: dict[str, Any] = {
        "retries": retries,
    }

    if system_prompt:
        agent_kwargs["system_prompt"] = system_prompt
    if deps_type:
        agent_kwargs["deps_type"] = deps_type
    if output_type:
        agent_kwargs["output_type"] = output_type
    if tools:
        agent_kwargs["tools"] = tools
    if toolsets:
        agent_kwargs["toolsets"] = toolsets

    agent_kwargs.update(kwargs)

    return Agent(model_str, **agent_kwargs)


def create_agent_from_config(
    config: AgentConfig,
    *,
    deps_type: type[DepsT] | None = None,
    output_type: type[T] | None = None,
    tools: list[Any] | None = None,
    toolsets: list[Any] | None = None,
    **kwargs: Any,
) -> Agent[DepsT, T]:
    """Create a PydanticAI Agent from a full QMCP AgentConfig.

    This provides deeper integration, using all configuration from AgentConfig
    including system prompt, retries, and security settings.

    Args:
        config: A QMCP AgentConfig instance
        deps_type: Type for dependency injection
        output_type: Expected output type
        tools: List of tool functions
        toolsets: List of toolsets
        **kwargs: Additional arguments passed to Agent constructor

    Returns:
        A configured PydanticAI Agent
    """
    _check_pydantic_ai()

    # Determine model
    model = config.model_config_obj if config.model_config_obj else config.model

    # Extract settings from AgentConfig
    return create_agent(
        model,
        system_prompt=config.system_prompt,
        deps_type=deps_type,
        output_type=output_type,
        tools=tools,
        toolsets=toolsets,
        retries=config.max_retries,
        **kwargs,
    )


@dataclass
class AgentBuilder(Generic[DepsT, T]):
    """Fluent builder for creating PydanticAI agents with QMCP integration.

    Provides a more discoverable API for agent configuration.

    Example:
        >>> agent = (
        ...     AgentBuilder(Models.CLAUDE_SONNET_4)
        ...     .with_system_prompt("You are helpful.")
        ...     .with_output_type(MyResponse)
        ...     .with_toolset(QMCPToolset("http://localhost:3333"))
        ...     .build()
        ... )
    """

    model: ModelConfig | str
    system_prompt: str | None = None
    deps_type: type[DepsT] | None = None
    output_type: type[T] | None = None
    tools: list[Any] = field(default_factory=list)
    toolsets: list[Any] = field(default_factory=list)
    retries: int = 1
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def with_system_prompt(self, prompt: str) -> AgentBuilder[DepsT, T]:
        """Set the system prompt."""
        self.system_prompt = prompt
        return self

    def with_deps_type(self, deps_type: type[DepsT]) -> AgentBuilder[DepsT, T]:
        """Set the dependency type for injection."""
        self.deps_type = deps_type
        return self

    def with_output_type(self, output_type: type[T]) -> AgentBuilder[DepsT, T]:
        """Set the expected output type."""
        self.output_type = output_type
        return self

    def with_tool(self, tool: Any) -> AgentBuilder[DepsT, T]:
        """Add a tool function."""
        self.tools.append(tool)
        return self

    def with_tools(self, tools: list[Any]) -> AgentBuilder[DepsT, T]:
        """Add multiple tool functions."""
        self.tools.extend(tools)
        return self

    def with_toolset(self, toolset: Any) -> AgentBuilder[DepsT, T]:
        """Add a toolset (e.g., QMCPToolset)."""
        self.toolsets.append(toolset)
        return self

    def with_retries(self, retries: int) -> AgentBuilder[DepsT, T]:
        """Set the number of retries on validation failure."""
        self.retries = retries
        return self

    def with_option(self, key: str, value: Any) -> AgentBuilder[DepsT, T]:
        """Set an additional option passed to the Agent constructor."""
        self.extra_kwargs[key] = value
        return self

    def build(self) -> Agent[DepsT, T]:
        """Build and return the configured Agent."""
        return create_agent(
            self.model,
            system_prompt=self.system_prompt,
            deps_type=self.deps_type,
            output_type=self.output_type,
            tools=self.tools if self.tools else None,
            toolsets=self.toolsets if self.toolsets else None,
            retries=self.retries,
            **self.extra_kwargs,
        )


__all__ = [
    "create_agent",
    "create_agent_from_config",
    "AgentBuilder",
    "PYDANTIC_AI_AVAILABLE",
]
