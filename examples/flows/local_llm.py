"""Local LLM helpers for PydanticAI-driven Metaflow flows."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for an OpenAI-compatible local LLM."""

    model: str
    base_url: str
    api_key: str = "local"


def build_agent(
    config: LocalLLMConfig,
    system_prompt: str,
    result_type: type[BaseModel],
) -> Agent:
    """Create a PydanticAI agent wired to a local OpenAI-compatible endpoint."""
    model = OpenAIModel(
        config.model,
        base_url=config.base_url,
        api_key=config.api_key,
    )
    return Agent(
        model=model,
        system_prompt=system_prompt,
        result_type=result_type,
    )
