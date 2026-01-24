"""PydanticAI integration for QMCP.

This module provides adapters to use QMCP's model registry and MCP server
with PydanticAI's agent framework.

Usage:
    from qmcp.integrations.pydantic_ai import (
        model_to_pydantic_ai,
        create_agent,
        QMCPToolset,
    )
    from qmcp.agentframework.models import Models

    # Convert QMCP model to PydanticAI string
    model_str = model_to_pydantic_ai(Models.CLAUDE_SONNET_4)

    # Create agent with QMCP configuration
    agent = create_agent(
        model=Models.CLAUDE_SONNET_4,
        system_prompt="You are helpful.",
    )

    # Use QMCP server as a toolset
    async with QMCPToolset("http://localhost:3333") as toolset:
        agent = create_agent(Models.CLAUDE_SONNET_4, toolsets=[toolset])
        result = await agent.run("Hello!")
"""

from .models import (
    model_to_pydantic_ai,
    get_model_settings,
    get_usage_limits,
    estimate_cost,
)
from .agents import (
    create_agent,
    create_agent_from_config,
    AgentBuilder,
    PYDANTIC_AI_AVAILABLE,
)
from .toolsets import QMCPToolset

__all__ = [
    # Model conversion
    "model_to_pydantic_ai",
    "get_model_settings",
    "get_usage_limits",
    "estimate_cost",
    # Agent creation
    "create_agent",
    "create_agent_from_config",
    "AgentBuilder",
    # Toolsets
    "QMCPToolset",
    # Utilities
    "PYDANTIC_AI_AVAILABLE",
]
