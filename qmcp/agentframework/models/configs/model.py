"""LLM model configuration models."""

from __future__ import annotations

from .base import Any, Field, SQLModel
from ..enums import (
    ModelAvailability,
    ModelCapabilityType,
    ModelFamily,
    ModelProvider,
    ModelTier,
    PricingUnit,
)


class ModelPricing(SQLModel):
    """Pricing information for an LLM model."""

    input_cost: float = Field(default=0.0, ge=0.0, description="Cost per unit for input tokens")
    output_cost: float = Field(default=0.0, ge=0.0, description="Cost per unit for output tokens")
    unit: PricingUnit = Field(default=PricingUnit.PER_1M_TOKENS)
    cache_read_cost: float | None = Field(default=None, ge=0.0, description="Cost for cached input reads")
    cache_write_cost: float | None = Field(default=None, ge=0.0, description="Cost for cache writes")
    image_cost: float | None = Field(default=None, ge=0.0, description="Cost per image processed")
    currency: str = Field(default="USD")


class ModelLimits(SQLModel):
    """Token and rate limits for an LLM model."""

    context_window: int = Field(default=200000, gt=0, description="Maximum context length in tokens")
    max_output_tokens: int = Field(default=4096, gt=0, description="Maximum output tokens per request")
    requests_per_minute: int | None = Field(default=None, gt=0, description="Rate limit: requests per minute")
    tokens_per_minute: int | None = Field(default=None, gt=0, description="Rate limit: tokens per minute")
    tokens_per_day: int | None = Field(default=None, gt=0, description="Rate limit: tokens per day")
    concurrent_requests: int | None = Field(default=None, gt=0, description="Max concurrent requests")
    max_images_per_request: int | None = Field(default=None, gt=0, description="Max images per request")


class ModelCapabilities(SQLModel):
    """Capabilities supported by an LLM model."""

    supported: list[ModelCapabilityType] = Field(
        default_factory=lambda: [
            ModelCapabilityType.TEXT_GENERATION,
            ModelCapabilityType.MULTI_TURN,
            ModelCapabilityType.SYSTEM_PROMPT,
        ]
    )
    vision_formats: list[str] = Field(default_factory=list, description="Supported image formats (png, jpg, etc.)")
    audio_formats: list[str] = Field(default_factory=list, description="Supported audio formats")
    max_tool_definitions: int | None = Field(default=None, gt=0, description="Max tools in a single request")
    supports_parallel_tool_calls: bool = Field(default=False)
    supports_tool_choice: bool = Field(default=False)
    structured_output_formats: list[str] = Field(default_factory=list, description="json_schema, regex, etc.")


class ModelEndpoint(SQLModel):
    """Endpoint configuration for accessing a model."""

    base_url: str | None = Field(default=None, description="Custom API base URL")
    api_version: str | None = Field(default=None, description="API version string")
    deployment_name: str | None = Field(default=None, description="Azure/custom deployment name")
    region: str | None = Field(default=None, description="Cloud region for the endpoint")
    headers: dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers")


class ModelFallbackConfig(SQLModel):
    """Configuration for model fallback behavior."""

    enabled: bool = Field(default=True)
    fallback_models: list[str] = Field(default_factory=list, description="Ordered list of fallback model IDs")
    trigger_on_rate_limit: bool = Field(default=True)
    trigger_on_timeout: bool = Field(default=True)
    trigger_on_error: bool = Field(default=False)
    max_fallback_attempts: int = Field(default=2, ge=1, le=5)


class ModelSamplingParams(SQLModel):
    """Sampling parameters for model inference."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int | None = Field(default=None, gt=0, description="Top-k sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: list[str] = Field(default_factory=list, description="Stop generation sequences")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class ModelConfig(SQLModel):
    """Complete configuration for an LLM model."""

    model_id: str = Field(description="Model identifier (e.g., claude-sonnet-4-20250514)")
    provider: ModelProvider = Field(default=ModelProvider.ANTHROPIC)
    family: ModelFamily = Field(default=ModelFamily.CLAUDE)
    tier: ModelTier = Field(default=ModelTier.STANDARD)
    display_name: str | None = Field(default=None, description="Human-readable model name")
    description: str | None = Field(default=None)
    version: str | None = Field(default=None, description="Model version string")
    availability: ModelAvailability = Field(default=ModelAvailability.AVAILABLE)
    release_date: str | None = Field(default=None, description="ISO date of model release")
    deprecation_date: str | None = Field(default=None, description="ISO date of planned deprecation")

    # Nested configurations
    sampling: ModelSamplingParams = Field(default_factory=ModelSamplingParams)
    limits: ModelLimits = Field(default_factory=ModelLimits)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    endpoint: ModelEndpoint = Field(default_factory=ModelEndpoint)
    fallback: ModelFallbackConfig = Field(default_factory=ModelFallbackConfig)

    # Inference settings
    max_tokens: int = Field(default=4096, gt=0, description="Default max output tokens")
    timeout_seconds: int = Field(default=300, gt=0)
    stream: bool = Field(default=False, description="Enable streaming responses")

    # Custom data
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class ModelRegistryEntry(SQLModel):
    """Entry in the model registry for tracking available models."""

    model_id: str = Field(description="Unique model identifier")
    provider: ModelProvider
    family: ModelFamily
    tier: ModelTier
    display_name: str
    availability: ModelAvailability = Field(default=ModelAvailability.AVAILABLE)
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    supports_vision: bool = Field(default=False)
    supports_tools: bool = Field(default=False)
    supports_streaming: bool = Field(default=True)
    input_cost_per_1m: float = Field(default=0.0, ge=0.0)
    output_cost_per_1m: float = Field(default=0.0, ge=0.0)


__all__ = [
    "ModelPricing",
    "ModelLimits",
    "ModelCapabilities",
    "ModelEndpoint",
    "ModelFallbackConfig",
    "ModelSamplingParams",
    "ModelConfig",
    "ModelRegistryEntry",
]
