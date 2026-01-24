"""Tests for PydanticAI integration module."""

import pytest

from qmcp.agentframework.models import (
    ModelConfig,
    ModelProvider,
    ModelFamily,
    ModelTier,
    Models,
)


class TestModelConversion:
    """Tests for model_to_pydantic_ai and related functions."""

    def test_model_to_pydantic_ai_from_config(self):
        """Test converting ModelConfig to PydanticAI string."""
        from qmcp.integrations.pydantic_ai.models import model_to_pydantic_ai

        result = model_to_pydantic_ai(Models.CLAUDE_SONNET_4)
        assert result == "anthropic:claude-sonnet-4-20250514"

    def test_model_to_pydantic_ai_gpt(self):
        """Test converting GPT model."""
        from qmcp.integrations.pydantic_ai.models import model_to_pydantic_ai

        result = model_to_pydantic_ai(Models.GPT_4O)
        assert result == "openai:gpt-4o"

    def test_model_to_pydantic_ai_gemini(self):
        """Test converting Gemini model."""
        from qmcp.integrations.pydantic_ai.models import model_to_pydantic_ai

        result = model_to_pydantic_ai(Models.GEMINI_PRO)
        assert result == "google:gemini-1.5-pro"

    def test_model_to_pydantic_ai_from_string(self):
        """Test converting string model ID."""
        from qmcp.integrations.pydantic_ai.models import model_to_pydantic_ai

        # Known model in registry
        result = model_to_pydantic_ai("claude-sonnet-4-20250514")
        assert result == "anthropic:claude-sonnet-4-20250514"

    def test_model_to_pydantic_ai_unknown_infers_provider(self):
        """Test that unknown models get provider inferred."""
        from qmcp.integrations.pydantic_ai.models import model_to_pydantic_ai

        # Claude-like name
        result = model_to_pydantic_ai("claude-new-model")
        assert result == "anthropic:claude-new-model"

        # GPT-like name
        result = model_to_pydantic_ai("gpt-5-turbo")
        assert result == "openai:gpt-5-turbo"

    def test_model_to_pydantic_ai_passthrough(self):
        """Test that already-formatted strings pass through."""
        from qmcp.integrations.pydantic_ai.models import model_to_pydantic_ai

        result = model_to_pydantic_ai("anthropic:claude-3-opus")
        assert result == "anthropic:claude-3-opus"

    def test_get_model_settings(self):
        """Test extracting model settings."""
        from qmcp.integrations.pydantic_ai.models import get_model_settings

        settings = get_model_settings(Models.CLAUDE_SONNET_4)

        assert "temperature" in settings
        assert settings["temperature"] == 0.7
        assert "max_tokens" in settings

    def test_estimate_cost(self):
        """Test cost estimation."""
        from qmcp.integrations.pydantic_ai.models import estimate_cost

        # Claude Sonnet 4: $3/1M input, $15/1M output
        cost = estimate_cost(Models.CLAUDE_SONNET_4, input_tokens=1000, output_tokens=500)

        # Expected: (1000/1M * 3) + (500/1M * 15) = 0.003 + 0.0075 = 0.0105
        assert cost == pytest.approx(0.0105, rel=0.01)

    def test_estimate_cost_zero_for_local(self):
        """Test that local models have zero cost."""
        from qmcp.integrations.pydantic_ai.models import estimate_cost

        cost = estimate_cost(Models.LLAMA_3_70B, input_tokens=10000, output_tokens=5000)
        assert cost == 0.0


class TestAgentCreation:
    """Tests for agent creation (mocked - PydanticAI may not be installed)."""

    def test_pydantic_ai_available_flag(self):
        """Test that PYDANTIC_AI_AVAILABLE flag is set correctly."""
        from qmcp.integrations.pydantic_ai.agents import PYDANTIC_AI_AVAILABLE

        # This should be either True or False based on installation
        assert isinstance(PYDANTIC_AI_AVAILABLE, bool)

    @pytest.mark.skipif(
        not __import__("qmcp.integrations.pydantic_ai.agents", fromlist=["PYDANTIC_AI_AVAILABLE"]).PYDANTIC_AI_AVAILABLE,
        reason="pydantic-ai not installed",
    )
    def test_create_agent_basic(self):
        """Test creating an agent with basic configuration."""
        from qmcp.integrations.pydantic_ai import create_agent

        agent = create_agent(
            Models.CLAUDE_SONNET_4,
            system_prompt="You are helpful.",
        )

        assert agent is not None

    @pytest.mark.skipif(
        not __import__("qmcp.integrations.pydantic_ai.agents", fromlist=["PYDANTIC_AI_AVAILABLE"]).PYDANTIC_AI_AVAILABLE,
        reason="pydantic-ai not installed",
    )
    def test_agent_builder(self):
        """Test the fluent builder API."""
        from qmcp.integrations.pydantic_ai import AgentBuilder

        agent = (
            AgentBuilder(Models.CLAUDE_SONNET_4)
            .with_system_prompt("You are helpful.")
            .with_retries(3)
            .build()
        )

        assert agent is not None


class TestQMCPToolset:
    """Tests for QMCPToolset (basic initialization tests)."""

    @pytest.mark.skipif(
        not __import__("qmcp.integrations.pydantic_ai.agents", fromlist=["PYDANTIC_AI_AVAILABLE"]).PYDANTIC_AI_AVAILABLE,
        reason="pydantic-ai not installed",
    )
    def test_toolset_initialization(self):
        """Test QMCPToolset initializes correctly."""
        from qmcp.integrations.pydantic_ai import QMCPToolset

        toolset = QMCPToolset(
            base_url="http://localhost:3333",
            tool_prefix="test_",
        )

        assert toolset.base_url == "http://localhost:3333"
        assert toolset.tool_prefix == "test_"

    @pytest.mark.skipif(
        not __import__("qmcp.integrations.pydantic_ai.agents", fromlist=["PYDANTIC_AI_AVAILABLE"]).PYDANTIC_AI_AVAILABLE,
        reason="pydantic-ai not installed",
    )
    def test_toolset_strips_trailing_slash(self):
        """Test that base_url trailing slash is stripped."""
        from qmcp.integrations.pydantic_ai import QMCPToolset

        toolset = QMCPToolset(base_url="http://localhost:3333/")
        assert toolset.base_url == "http://localhost:3333"


class TestIntegrationExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from qmcp.integrations.pydantic_ai import (
            model_to_pydantic_ai,
            get_model_settings,
            get_usage_limits,
            estimate_cost,
            create_agent,
            create_agent_from_config,
            AgentBuilder,
            QMCPToolset,
            PYDANTIC_AI_AVAILABLE,
        )

        # All should be importable
        assert callable(model_to_pydantic_ai)
        assert callable(get_model_settings)
        assert callable(get_usage_limits)
        assert callable(estimate_cost)
