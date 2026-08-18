"""Composable agent creation patterns for QMCP flows.

Provides unified builders for local LLM agents and QMCP-integrated agents
that work inside Metaflow flows and the Docker runner.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LocalLLMConfig(BaseModel):
    """Configuration for a local OpenAI-compatible LLM endpoint."""

    model: str
    base_url: str
    api_key: str = "local"
    temperature: float = 0.3
    max_tokens: int = 512


def build_local_agent(
    config: LocalLLMConfig,
    system_prompt: str,
    output_type: type[T],
    retries: int = 3,
) -> Any:
    """Build a PydanticAI agent targeting a local OpenAI-compatible LLM.

    Works with Ollama, LM Studio, vLLM, or any endpoint that implements
    the OpenAI chat completions API.

    Args:
        config: Local LLM connection configuration.
        system_prompt: System prompt for the agent.
        output_type: Pydantic model for structured output.
        retries: Retries on validation failure (local LLMs need more).

    Returns:
        A configured PydanticAI Agent.
    """
    from openai import AsyncOpenAI
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    provider = OpenAIProvider(openai_client=client)
    model = OpenAIChatModel(config.model, provider=provider)

    json_instruction = (
        "\n\nIMPORTANT: Respond ONLY with valid JSON. "
        "No explanation or text before/after."
    )

    return Agent(
        model=model,
        system_prompt=system_prompt + json_instruction,
        output_type=output_type,
        retries=retries,
        model_settings=ModelSettings(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ),
    )


def build_qmcp_agent(
    model: Any,
    system_prompt: str,
    output_type: type[T],
    mcp_url: str | None = None,
    retries: int = 1,
) -> Any:
    """Build a PydanticAI agent using QMCP's model registry.

    Uses the QMCP integration layer so agents benefit from model metadata,
    pricing, and optionally connect to the MCP server for audited tool calls.

    Args:
        model: A QMCP ModelConfig or model ID string.
        system_prompt: System prompt for the agent.
        output_type: Pydantic model for structured output.
        mcp_url: If provided, attaches a QMCPToolset for audited tool calls.
        retries: Retries on validation failure.

    Returns:
        A configured PydanticAI Agent.
    """
    from qmcp.integrations.pydantic_ai import AgentBuilder, QMCPToolset

    builder = (
        AgentBuilder(model)
        .with_system_prompt(system_prompt)
        .with_output_type(output_type)
        .with_retries(retries)
    )

    if mcp_url:
        builder = builder.with_toolset(QMCPToolset(mcp_url))

    return builder.build()


__all__ = [
    "LocalLLMConfig",
    "build_local_agent",
    "build_qmcp_agent",
]
