"""Pre-defined model configurations for common LLM models.

This module provides ready-to-use ModelConfig instances for popular models,
eliminating the need for string literals and ensuring consistent configuration.

Usage:
    from qmcp.agentframework.models import Models

    # Use a predefined model
    agent_config = AgentConfig(model_config_obj=Models.CLAUDE_SONNET_4)

    # Or get by string ID for dynamic selection
    model = Models.get("claude-sonnet-4-20250514")
"""

from __future__ import annotations

from .configs import (
    ModelCapabilities,
    ModelConfig,
    ModelFallbackConfig,
    ModelLimits,
    ModelPricing,
)
from .enums import (
    ModelAvailability,
    ModelCapabilityType,
    ModelFamily,
    ModelProvider,
    ModelTier,
    PricingUnit,
)


def _claude_capabilities(vision: bool = True, tools: bool = True) -> ModelCapabilities:
    """Standard Claude model capabilities."""
    supported = [
        ModelCapabilityType.TEXT_GENERATION,
        ModelCapabilityType.MULTI_TURN,
        ModelCapabilityType.SYSTEM_PROMPT,
        ModelCapabilityType.STREAMING,
    ]
    if vision:
        supported.append(ModelCapabilityType.VISION)
    if tools:
        supported.append(ModelCapabilityType.TOOL_USE)
        supported.append(ModelCapabilityType.STRUCTURED_OUTPUT)

    return ModelCapabilities(
        supported=supported,
        vision_formats=["png", "jpg", "gif", "webp"] if vision else [],
        supports_parallel_tool_calls=tools,
        supports_tool_choice=tools,
        structured_output_formats=["json_schema"] if tools else [],
    )


class Models:
    """Registry of pre-defined model configurations.

    Provides constants for common models with accurate configurations.

    Example:
        >>> from qmcp.agentframework.models import Models
        >>> config = AgentConfig(model_config_obj=Models.CLAUDE_SONNET_4)
    """

    # =========================================================================
    # Anthropic Claude Models
    # =========================================================================

    CLAUDE_OPUS_4 = ModelConfig(
        model_id="claude-opus-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        family=ModelFamily.CLAUDE,
        tier=ModelTier.FLAGSHIP,
        display_name="Claude Opus 4",
        description="Most capable Claude model for complex tasks",
        limits=ModelLimits(
            context_window=200000,
            max_output_tokens=32000,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.VISION,
                ModelCapabilityType.TOOL_USE,
                ModelCapabilityType.STREAMING,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
                ModelCapabilityType.STRUCTURED_OUTPUT,
                ModelCapabilityType.THINKING,
            ],
            vision_formats=["png", "jpg", "gif", "webp"],
            supports_parallel_tool_calls=True,
            supports_tool_choice=True,
            structured_output_formats=["json_schema"],
        ),
        pricing=ModelPricing(
            input_cost=15.0,
            output_cost=75.0,
            unit=PricingUnit.PER_1M_TOKENS,
            cache_read_cost=1.50,
            cache_write_cost=18.75,
        ),
        fallback=ModelFallbackConfig(
            enabled=True,
            fallback_models=["claude-sonnet-4-20250514"],
        ),
    )

    CLAUDE_SONNET_4 = ModelConfig(
        model_id="claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        family=ModelFamily.CLAUDE,
        tier=ModelTier.STANDARD,
        display_name="Claude Sonnet 4",
        description="Balanced performance for most tasks",
        limits=ModelLimits(
            context_window=200000,
            max_output_tokens=16000,
        ),
        capabilities=_claude_capabilities(vision=True, tools=True),
        pricing=ModelPricing(
            input_cost=3.0,
            output_cost=15.0,
            unit=PricingUnit.PER_1M_TOKENS,
            cache_read_cost=0.30,
            cache_write_cost=3.75,
        ),
        fallback=ModelFallbackConfig(
            enabled=True,
            fallback_models=["claude-3-5-haiku-20241022"],
        ),
    )

    CLAUDE_HAIKU_35 = ModelConfig(
        model_id="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        family=ModelFamily.CLAUDE,
        tier=ModelTier.FAST,
        display_name="Claude 3.5 Haiku",
        description="Fast and efficient for simple tasks",
        limits=ModelLimits(
            context_window=200000,
            max_output_tokens=8192,
        ),
        capabilities=_claude_capabilities(vision=True, tools=True),
        pricing=ModelPricing(
            input_cost=0.80,
            output_cost=4.0,
            unit=PricingUnit.PER_1M_TOKENS,
            cache_read_cost=0.08,
            cache_write_cost=1.0,
        ),
    )

    # Legacy aliases
    CLAUDE_SONNET = CLAUDE_SONNET_4
    CLAUDE_OPUS = CLAUDE_OPUS_4
    CLAUDE_HAIKU = CLAUDE_HAIKU_35

    # =========================================================================
    # OpenAI GPT Models
    # =========================================================================

    GPT_4O = ModelConfig(
        model_id="gpt-4o",
        provider=ModelProvider.OPENAI,
        family=ModelFamily.GPT,
        tier=ModelTier.FLAGSHIP,
        display_name="GPT-4o",
        description="OpenAI's most capable model",
        limits=ModelLimits(
            context_window=128000,
            max_output_tokens=16384,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.VISION,
                ModelCapabilityType.TOOL_USE,
                ModelCapabilityType.STREAMING,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
                ModelCapabilityType.JSON_MODE,
            ],
            vision_formats=["png", "jpg", "gif", "webp"],
            supports_parallel_tool_calls=True,
            supports_tool_choice=True,
        ),
        pricing=ModelPricing(
            input_cost=2.50,
            output_cost=10.0,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
        fallback=ModelFallbackConfig(
            enabled=True,
            fallback_models=["gpt-4o-mini"],
        ),
    )

    GPT_4O_MINI = ModelConfig(
        model_id="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        family=ModelFamily.GPT,
        tier=ModelTier.FAST,
        display_name="GPT-4o Mini",
        description="Fast and affordable GPT-4 class model",
        limits=ModelLimits(
            context_window=128000,
            max_output_tokens=16384,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.VISION,
                ModelCapabilityType.TOOL_USE,
                ModelCapabilityType.STREAMING,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
                ModelCapabilityType.JSON_MODE,
            ],
            vision_formats=["png", "jpg", "gif", "webp"],
            supports_parallel_tool_calls=True,
            supports_tool_choice=True,
        ),
        pricing=ModelPricing(
            input_cost=0.15,
            output_cost=0.60,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
    )

    # =========================================================================
    # Google Gemini Models
    # =========================================================================

    GEMINI_PRO = ModelConfig(
        model_id="gemini-1.5-pro",
        provider=ModelProvider.GOOGLE,
        family=ModelFamily.GEMINI,
        tier=ModelTier.FLAGSHIP,
        display_name="Gemini 1.5 Pro",
        description="Google's most capable model",
        limits=ModelLimits(
            context_window=2000000,
            max_output_tokens=8192,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.VISION,
                ModelCapabilityType.TOOL_USE,
                ModelCapabilityType.STREAMING,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
                ModelCapabilityType.AUDIO,
                ModelCapabilityType.VIDEO,
            ],
            vision_formats=["png", "jpg", "gif", "webp"],
            audio_formats=["mp3", "wav", "flac"],
            supports_parallel_tool_calls=True,
        ),
        pricing=ModelPricing(
            input_cost=1.25,
            output_cost=5.0,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
        fallback=ModelFallbackConfig(
            enabled=True,
            fallback_models=["gemini-1.5-flash"],
        ),
    )

    GEMINI_FLASH = ModelConfig(
        model_id="gemini-1.5-flash",
        provider=ModelProvider.GOOGLE,
        family=ModelFamily.GEMINI,
        tier=ModelTier.FAST,
        display_name="Gemini 1.5 Flash",
        description="Fast Gemini model for high-volume tasks",
        limits=ModelLimits(
            context_window=1000000,
            max_output_tokens=8192,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.VISION,
                ModelCapabilityType.TOOL_USE,
                ModelCapabilityType.STREAMING,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
            ],
            vision_formats=["png", "jpg", "gif", "webp"],
            supports_parallel_tool_calls=True,
        ),
        pricing=ModelPricing(
            input_cost=0.075,
            output_cost=0.30,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
    )

    # =========================================================================
    # Local / Open Source Models (Ollama)
    # =========================================================================

    LLAMA_3_70B = ModelConfig(
        model_id="llama3:70b",
        provider=ModelProvider.OLLAMA,
        family=ModelFamily.LLAMA,
        tier=ModelTier.STANDARD,
        display_name="Llama 3 70B",
        description="Meta's Llama 3 70B via Ollama",
        limits=ModelLimits(
            context_window=8192,
            max_output_tokens=4096,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
            ],
        ),
        pricing=ModelPricing(
            input_cost=0.0,
            output_cost=0.0,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
    )

    LLAMA_3_8B = ModelConfig(
        model_id="llama3:8b",
        provider=ModelProvider.OLLAMA,
        family=ModelFamily.LLAMA,
        tier=ModelTier.FAST,
        display_name="Llama 3 8B",
        description="Meta's Llama 3 8B via Ollama",
        limits=ModelLimits(
            context_window=8192,
            max_output_tokens=4096,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
            ],
        ),
        pricing=ModelPricing(
            input_cost=0.0,
            output_cost=0.0,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
    )

    MISTRAL_7B = ModelConfig(
        model_id="mistral:7b",
        provider=ModelProvider.OLLAMA,
        family=ModelFamily.MISTRAL,
        tier=ModelTier.FAST,
        display_name="Mistral 7B",
        description="Mistral 7B via Ollama",
        limits=ModelLimits(
            context_window=32000,
            max_output_tokens=4096,
        ),
        capabilities=ModelCapabilities(
            supported=[
                ModelCapabilityType.TEXT_GENERATION,
                ModelCapabilityType.SYSTEM_PROMPT,
                ModelCapabilityType.MULTI_TURN,
            ],
        ),
        pricing=ModelPricing(
            input_cost=0.0,
            output_cost=0.0,
            unit=PricingUnit.PER_1M_TOKENS,
        ),
    )

    # =========================================================================
    # Registry methods
    # =========================================================================

    _registry: dict[str, ModelConfig] = {}

    @classmethod
    def _build_registry(cls) -> None:
        """Build the model registry from class attributes."""
        if cls._registry:
            return
        for name in dir(cls):
            if name.startswith("_") or name[0].islower():
                continue
            attr = getattr(cls, name)
            if isinstance(attr, ModelConfig):
                cls._registry[attr.model_id] = attr

    @classmethod
    def get(cls, model_id: str) -> ModelConfig | None:
        """Get a model configuration by its ID.

        Args:
            model_id: The model identifier (e.g., "claude-sonnet-4-20250514")

        Returns:
            The ModelConfig if found, None otherwise
        """
        cls._build_registry()
        return cls._registry.get(model_id)

    @classmethod
    def list_all(cls) -> list[ModelConfig]:
        """List all registered models.

        Returns:
            List of all ModelConfig instances
        """
        cls._build_registry()
        return list(cls._registry.values())

    @classmethod
    def by_provider(cls, provider: ModelProvider) -> list[ModelConfig]:
        """Get all models from a specific provider.

        Args:
            provider: The model provider

        Returns:
            List of ModelConfig instances from that provider
        """
        cls._build_registry()
        return [m for m in cls._registry.values() if m.provider == provider]

    @classmethod
    def by_tier(cls, tier: ModelTier) -> list[ModelConfig]:
        """Get all models of a specific tier.

        Args:
            tier: The model tier (flagship, standard, fast, etc.)

        Returns:
            List of ModelConfig instances of that tier
        """
        cls._build_registry()
        return [m for m in cls._registry.values() if m.tier == tier]


__all__ = ["Models"]
