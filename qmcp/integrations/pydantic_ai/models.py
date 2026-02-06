"""Model conversion utilities for PydanticAI integration.

Converts QMCP's ModelConfig to PydanticAI model specifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qmcp.agentframework.models import ModelConfig

# Provider mapping: QMCP ModelProvider -> PydanticAI prefix
_PROVIDER_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "groq": "groq",
    "mistral": "mistral",
    "ollama": "ollama",
    "bedrock": "bedrock",
    "azure": "azure",
}


def model_to_pydantic_ai(model: ModelConfig | str) -> str:
    """Convert a QMCP ModelConfig to a PydanticAI model string.

    PydanticAI uses strings like "anthropic:claude-sonnet-4-20250514" or
    "openai:gpt-4o" to specify models.

    Args:
        model: Either a ModelConfig instance or a model ID string

    Returns:
        A PydanticAI-compatible model string

    Examples:
        >>> from qmcp.agentframework.models import Models
        >>> model_to_pydantic_ai(Models.CLAUDE_SONNET_4)
        'anthropic:claude-sonnet-4-20250514'

        >>> model_to_pydantic_ai(Models.GPT_4O)
        'openai:gpt-4o'

        >>> model_to_pydantic_ai("claude-sonnet-4-20250514")
        'anthropic:claude-sonnet-4-20250514'
    """
    if isinstance(model, str):
        # Try to look up in registry
        from qmcp.agentframework.models import Models

        config = Models.get(model)
        if config is None:
            # Assume it's already a PydanticAI string or guess provider
            if ":" in model:
                return model
            return _infer_provider(model)
        model = config

    # Get provider prefix
    provider_value = model.provider.value if hasattr(model.provider, "value") else str(model.provider)
    provider_prefix = _PROVIDER_MAP.get(provider_value.lower(), provider_value.lower())

    return f"{provider_prefix}:{model.model_id}"


def _infer_provider(model_id: str) -> str:
    """Infer the provider from a model ID string.

    Args:
        model_id: A model identifier like "claude-sonnet-4-20250514"

    Returns:
        A PydanticAI model string with inferred provider
    """
    model_lower = model_id.lower()

    if "claude" in model_lower:
        return f"anthropic:{model_id}"
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return f"openai:{model_id}"
    if "gemini" in model_lower:
        return f"google:{model_id}"
    if "llama" in model_lower or "mistral" in model_lower:
        # Could be Ollama or cloud - default to Ollama for local models
        if ":" in model_id:  # Ollama format like "llama3:70b"
            return f"ollama:{model_id}"
        return f"groq:{model_id}"

    # Unknown - return as-is, let PydanticAI handle it
    return model_id


def get_model_settings(model: ModelConfig) -> dict[str, Any]:
    """Extract PydanticAI ModelSettings from a QMCP ModelConfig.

    Args:
        model: A QMCP ModelConfig instance

    Returns:
        A dict suitable for PydanticAI's model_settings parameter

    Example:
        >>> settings = get_model_settings(Models.CLAUDE_SONNET_4)
        >>> agent = Agent("anthropic:claude-sonnet-4", model_settings=settings)
    """
    settings: dict[str, Any] = {}

    # Sampling parameters
    if model.sampling:
        if model.sampling.temperature is not None:
            settings["temperature"] = model.sampling.temperature
        if model.sampling.top_p is not None:
            settings["top_p"] = model.sampling.top_p
        if model.sampling.top_k is not None:
            settings["top_k"] = model.sampling.top_k

    # Token limits
    if model.max_tokens:
        settings["max_tokens"] = model.max_tokens

    # Timeout
    if model.timeout_seconds:
        settings["timeout"] = model.timeout_seconds

    return settings


def get_usage_limits(model: ModelConfig) -> dict[str, Any]:
    """Extract PydanticAI UsageLimits from a QMCP ModelConfig.

    Args:
        model: A QMCP ModelConfig instance

    Returns:
        A dict suitable for PydanticAI's usage_limits parameter
    """
    limits: dict[str, Any] = {}

    if model.limits:
        if model.limits.max_output_tokens:
            limits["response_tokens_limit"] = model.limits.max_output_tokens

    return limits


def estimate_cost(model: ModelConfig, input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost of a request using QMCP's pricing metadata.

    Args:
        model: A QMCP ModelConfig instance
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in the model's currency (usually USD)

    Example:
        >>> cost = estimate_cost(Models.CLAUDE_SONNET_4, 1000, 500)
        >>> print(f"Estimated cost: ${cost:.4f}")
    """
    if not model.pricing:
        return 0.0

    # Pricing is typically per 1M tokens
    input_cost = (input_tokens / 1_000_000) * model.pricing.input_cost
    output_cost = (output_tokens / 1_000_000) * model.pricing.output_cost

    return input_cost + output_cost


__all__ = [
    "model_to_pydantic_ai",
    "get_model_settings",
    "get_usage_limits",
    "estimate_cost",
]
