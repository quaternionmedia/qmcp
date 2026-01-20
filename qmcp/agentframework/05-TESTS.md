# QMCP Agent Framework: Tests Specification

Note: Design reference. Implementation status is documented in docs/agentframework.md.


## Overview

This document specifies the test suite for the agent framework. Tests are organized by component and follow pytest conventions with async support via pytest-asyncio.

## Test Configuration

### File: `tests/conftest.py`

```python
"""Pytest configuration and shared fixtures."""

import asyncio
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import SQLModel

from qmcp.agentframework.models import (
    AgentRole,
    AgentType,
    Execution,
    ExecutionStatus,
    Topology,
    TopologyMembership,
    TopologyType,
)


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create in-memory database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session."""
    session_factory = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        yield session
        await session.rollback()


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def sample_agent_config() -> dict:
    """Sample agent configuration."""
    return {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 4096,
        "system_prompt": "You are a helpful assistant.",
        "capabilities": [
            {"name": "tool_use", "config": {"max_tool_calls_per_turn": 5}},
            {"name": "reasoning", "config": {"reasoning_style": "step_by_step"}},
        ],
    }


@pytest_asyncio.fixture
async def planner_agent(db_session: AsyncSession, sample_agent_config) -> AgentType:
    """Create planner agent."""
    agent = AgentType(
        name=f"planner_{uuid4().hex[:8]}",
        description="Strategic planning agent",
        role=AgentRole.PLANNER,
        config=sample_agent_config,
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def executor_agent(db_session: AsyncSession) -> AgentType:
    """Create executor agent."""
    agent = AgentType(
        name=f"executor_{uuid4().hex[:8]}",
        description="Task executor agent",
        role=AgentRole.EXECUTOR,
        config={"model": "claude-sonnet-4-20250514"},
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def reviewer_agent(db_session: AsyncSession) -> AgentType:
    """Create reviewer agent."""
    agent = AgentType(
        name=f"reviewer_{uuid4().hex[:8]}",
        description="Quality reviewer agent",
        role=AgentRole.REVIEWER,
        config={"model": "claude-sonnet-4-20250514"},
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def critic_agent(db_session: AsyncSession) -> AgentType:
    """Create critic agent."""
    agent = AgentType(
        name=f"critic_{uuid4().hex[:8]}",
        description="Critical analysis agent",
        role=AgentRole.CRITIC,
        config={"model": "claude-sonnet-4-20250514"},
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def synthesizer_agent(db_session: AsyncSession) -> AgentType:
    """Create synthesizer agent."""
    agent = AgentType(
        name=f"synthesizer_{uuid4().hex[:8]}",
        description="Information synthesis agent",
        role=AgentRole.SYNTHESIZER,
        config={"model": "claude-sonnet-4-20250514"},
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


# ============================================================================
# Topology Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def debate_topology(db_session: AsyncSession) -> Topology:
    """Create debate topology."""
    topology = Topology(
        name=f"debate_{uuid4().hex[:8]}",
        description="Test debate topology",
        topology_type=TopologyType.DEBATE,
        config={
            "max_rounds": 2,
            "consensus_method": "mediator",
            "allow_early_termination": True,
        },
    )
    db_session.add(topology)
    await db_session.commit()
    await db_session.refresh(topology)
    return topology


@pytest_asyncio.fixture
async def ensemble_topology(db_session: AsyncSession) -> Topology:
    """Create ensemble topology."""
    topology = Topology(
        name=f"ensemble_{uuid4().hex[:8]}",
        description="Test ensemble topology",
        topology_type=TopologyType.ENSEMBLE,
        config={
            "aggregation_method": "synthesis",
            "failure_threshold": 0.5,
        },
    )
    db_session.add(topology)
    await db_session.commit()
    await db_session.refresh(topology)
    return topology


@pytest_asyncio.fixture
async def pipeline_topology(db_session: AsyncSession) -> Topology:
    """Create pipeline topology."""
    topology = Topology(
        name=f"pipeline_{uuid4().hex[:8]}",
        description="Test pipeline topology",
        topology_type=TopologyType.PIPELINE,
        config={
            "stages": ["parse", "analyze", "generate"],
            "retry_failed_stages": True,
        },
    )
    db_session.add(topology)
    await db_session.commit()
    await db_session.refresh(topology)
    return topology


# ============================================================================
# Mock LLM Fixture
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Factory for mock LLM responses."""
    def _create(content: str = "Mock response", **kwargs):
        return {
            "content": content,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            **kwargs,
        }
    return _create
```

## Model Tests

### File: `tests/agents/test_models.py`

```python
"""Tests for agent data models."""

import pytest
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from qmcp.agentframework.models import (
    AgentCapability,
    AgentConfig,
    AgentInstance,
    AgentRole,
    AgentType,
    AgentTypeCreate,
    AggregationMethod,
    ConsensusMethod,
    DebateConfig,
    EnsembleConfig,
    Execution,
    ExecutionStatus,
    Message,
    MessageType,
    PipelineConfig,
    Result,
    Topology,
    TopologyMembership,
    TopologyType,
)


class TestAgentRole:
    """Tests for AgentRole enum."""
    
    def test_all_roles_defined(self):
        """Verify all expected roles exist."""
        expected_roles = [
            "planner", "executor", "reviewer", "critic",
            "synthesizer", "specialist", "coordinator", "observer"
        ]
        actual_roles = [r.value for r in AgentRole]
        assert set(expected_roles) == set(actual_roles)
    
    def test_role_string_values(self):
        """Verify roles are string enums."""
        assert AgentRole.PLANNER.value == "planner"
        assert isinstance(AgentRole.PLANNER.value, str)


class TestAgentCapability:
    """Tests for AgentCapability model."""
    
    def test_create_minimal(self):
        """Create capability with minimal fields."""
        cap = AgentCapability(name="tool_use")
        assert cap.name == "tool_use"
        assert cap.version == "1.0.0"
        assert cap.enabled is True
        assert cap.config == {}
    
    def test_create_with_config(self):
        """Create capability with configuration."""
        cap = AgentCapability(
            name="memory",
            version="2.0.0",
            config={"max_memories": 100},
            enabled=True,
        )
        assert cap.config["max_memories"] == 100


class TestAgentConfig:
    """Tests for AgentConfig model."""
    
    def test_default_values(self):
        """Verify default configuration values."""
        config = AgentConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.max_retries == 3
    
    def test_temperature_validation(self):
        """Temperature must be 0-2."""
        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            AgentConfig(temperature=2.5)
        
        # Valid range
        config = AgentConfig(temperature=0.0)
        assert config.temperature == 0.0
        config = AgentConfig(temperature=2.0)
        assert config.temperature == 2.0
    
    def test_with_capabilities(self):
        """Config with capabilities list."""
        config = AgentConfig(
            capabilities=[
                AgentCapability(name="tool_use"),
                AgentCapability(name="memory"),
            ]
        )
        assert len(config.capabilities) == 2


class TestAgentType:
    """Tests for AgentType model."""
    
    def test_create_agent_type(self, db_session, sample_agent_config):
        """Create agent type with valid data."""
        agent = AgentType(
            name="test_planner",
            description="A test planning agent",
            role=AgentRole.PLANNER,
            config=sample_agent_config,
        )
        assert agent.name == "test_planner"
        assert agent.role == AgentRole.PLANNER
    
    def test_name_validation(self):
        """Name must be alphanumeric with underscores/hyphens."""
        # Valid names
        AgentType(name="valid_name", description="Test", role=AgentRole.PLANNER)
        AgentType(name="valid-name-2", description="Test", role=AgentRole.PLANNER)
        
        # Invalid names (special characters)
        with pytest.raises(ValidationError):
            AgentType(name="invalid name", description="Test", role=AgentRole.PLANNER)
        with pytest.raises(ValidationError):
            AgentType(name="invalid@name", description="Test", role=AgentRole.PLANNER)
    
    def test_name_lowercased(self):
        """Names are automatically lowercased."""
        agent = AgentType(
            name="MyAgent",
            description="Test",
            role=AgentRole.PLANNER,
        )
        assert agent.name == "myagent"
    
    def test_get_capabilities(self, sample_agent_config):
        """Extract capabilities from config."""
        agent = AgentType(
            name="test",
            description="Test",
            role=AgentRole.PLANNER,
            config=sample_agent_config,
        )
        caps = agent.get_capabilities()
        assert len(caps) == 2
        assert caps[0].name == "tool_use"
    
    def test_has_capability(self, sample_agent_config):
        """Check capability existence."""
        agent = AgentType(
            name="test",
            description="Test",
            role=AgentRole.PLANNER,
            config=sample_agent_config,
        )
        assert agent.has_capability("tool_use") is True
        assert agent.has_capability("nonexistent") is False


class TestAgentInstance:
    """Tests for AgentInstance model."""
    
    @pytest.mark.asyncio
    async def test_create_instance(self, db_session, planner_agent):
        """Create agent instance."""
        instance = AgentInstance(
            agent_type_id=planner_agent.id,
            state={"context": "test"},
        )
        db_session.add(instance)
        await db_session.commit()
        
        assert instance.id is not None
        assert instance.state["context"] == "test"
    
    @pytest.mark.asyncio
    async def test_update_state(self, db_session, planner_agent):
        """Update instance state."""
        instance = AgentInstance(
            agent_type_id=planner_agent.id,
            state={"key1": "value1"},
        )
        db_session.add(instance)
        await db_session.commit()
        
        instance.update_state({"key2": "value2"})
        
        assert instance.state["key1"] == "value1"
        assert instance.state["key2"] == "value2"
        assert instance.last_active is not None


class TestTopology:
    """Tests for Topology model."""
    
    def test_create_topology(self):
        """Create topology with valid data."""
        topology = Topology(
            name="test_debate",
            description="Test debate topology",
            topology_type=TopologyType.DEBATE,
            config={"max_rounds": 3},
        )
        assert topology.name == "test_debate"
        assert topology.topology_type == TopologyType.DEBATE
    
    def test_get_typed_config(self):
        """Get strongly-typed configuration."""
        topology = Topology(
            name="test",
            description="Test",
            topology_type=TopologyType.DEBATE,
            config={"max_rounds": 5, "allow_early_termination": False},
        )
        config = topology.get_typed_config()
        assert isinstance(config, DebateConfig)
        assert config.max_rounds == 5
        assert config.allow_early_termination is False


class TestTopologyConfigs:
    """Tests for topology configuration models."""
    
    def test_debate_config_defaults(self):
        """DebateConfig default values."""
        config = DebateConfig()
        assert config.max_rounds == 3
        assert config.consensus_method == ConsensusMethod.MEDIATOR_DECISION
        assert config.allow_early_termination is True
    
    def test_ensemble_config_defaults(self):
        """EnsembleConfig default values."""
        config = EnsembleConfig()
        assert config.aggregation_method == AggregationMethod.SYNTHESIS
        assert config.failure_threshold == 0.5
    
    def test_pipeline_config(self):
        """PipelineConfig with stages."""
        config = PipelineConfig(
            stages=["parse", "analyze", "generate"],
            checkpoint_after=["analyze"],
        )
        assert len(config.stages) == 3
        assert "analyze" in config.checkpoint_after


class TestExecution:
    """Tests for Execution model."""
    
    @pytest.mark.asyncio
    async def test_create_execution(self, db_session, debate_topology):
        """Create execution record."""
        execution = Execution(
            topology_id=debate_topology.id,
            input_data={"topic": "Test topic"},
            status=ExecutionStatus.PENDING,
        )
        db_session.add(execution)
        await db_session.commit()
        
        assert execution.id is not None
        assert execution.status == ExecutionStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_mark_complete(self, db_session, debate_topology):
        """Mark execution as complete."""
        execution = Execution(
            topology_id=debate_topology.id,
            input_data={},
            status=ExecutionStatus.RUNNING,
        )
        db_session.add(execution)
        await db_session.commit()
        
        execution.mark_complete({"result": "success"})
        
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.output_data["result"] == "success"
        assert execution.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_mark_failed(self, db_session, debate_topology):
        """Mark execution as failed."""
        execution = Execution(
            topology_id=debate_topology.id,
            input_data={},
        )
        db_session.add(execution)
        await db_session.commit()
        
        execution.mark_failed("Test error", {"code": 500})
        
        assert execution.status == ExecutionStatus.FAILED
        assert execution.error == "Test error"
        assert execution.error_details["code"] == 500
    
    @pytest.mark.asyncio
    async def test_duration_calculation(self, db_session, debate_topology):
        """Calculate execution duration."""
        from datetime import timedelta
        
        execution = Execution(
            topology_id=debate_topology.id,
            input_data={},
        )
        execution.started_at = datetime.utcnow()
        execution.completed_at = execution.started_at + timedelta(seconds=5)
        
        assert execution.duration_ms == 5000


class TestMessage:
    """Tests for Message model."""
    
    @pytest.mark.asyncio
    async def test_create_message(self, db_session, debate_topology, planner_agent):
        """Create inter-agent message."""
        execution = Execution(
            topology_id=debate_topology.id,
            input_data={},
        )
        db_session.add(execution)
        
        instance = AgentInstance(agent_type_id=planner_agent.id)
        db_session.add(instance)
        await db_session.commit()
        
        message = Message(
            execution_id=execution.id,
            sender_id=instance.id,
            message_type=MessageType.REQUEST,
            content={"task": "Analyze data"},
        )
        db_session.add(message)
        await db_session.commit()
        
        assert message.id is not None
        assert message.message_type == MessageType.REQUEST
```

## Mixin Tests

### File: `tests/agents/test_mixins.py`

```python
"""Tests for agent capability mixins."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from qmcp.agentframework.mixins import (
    BaseMixin,
    MixinConfig,
    MixinRegistry,
    MemoryConfig,
    MemoryEntry,
    MemoryMixin,
    ReasoningConfig,
    ReasoningMixin,
    StructuredOutputConfig,
    StructuredOutputMixin,
    ToolUseConfig,
    ToolUseMixin,
    mixin,
)


class TestMixinRegistry:
    """Tests for MixinRegistry."""
    
    def test_register_and_get(self):
        """Register and retrieve mixin."""
        @mixin
        class TestMixin(BaseMixin):
            name = "test_mixin"
            description = "Test mixin"
        
        retrieved = MixinRegistry.get("test_mixin")
        assert retrieved is TestMixin
    
    def test_create_mixin(self):
        """Create mixin instance with config."""
        instance = MixinRegistry.create(
            "tool_use",
            {"max_tool_calls_per_turn": 3}
        )
        assert isinstance(instance, ToolUseMixin)
        assert instance.tool_config.max_tool_calls_per_turn == 3
    
    def test_create_unknown_raises(self):
        """Creating unknown mixin raises error."""
        with pytest.raises(ValueError, match="Unknown mixin"):
            MixinRegistry.create("nonexistent_mixin")
    
    def test_list_all(self):
        """List all registered mixins."""
        mixins = MixinRegistry.list_all()
        assert len(mixins) > 0
        
        names = [m["name"] for m in mixins]
        assert "tool_use" in names
        assert "memory" in names
    
    def test_resolve_dependencies(self):
        """Resolve mixin dependencies."""
        # human_in_loop depends on tool_use
        ordered = MixinRegistry.resolve_dependencies(["human_in_loop", "memory"])
        
        # tool_use should come before human_in_loop
        tool_idx = ordered.index("tool_use")
        hitl_idx = ordered.index("human_in_loop")
        assert tool_idx < hitl_idx
    
    def test_circular_dependency_raises(self):
        """Circular dependencies raise error."""
        # Would need to set up circular dependency to test
        # This is a placeholder for when such a case exists
        pass


class TestToolUseMixin:
    """Tests for ToolUseMixin."""
    
    def test_default_config(self):
        """Default configuration values."""
        mixin = ToolUseMixin()
        assert mixin.tool_config.max_tool_calls_per_turn == 5
        assert mixin.tool_config.require_confirmation is False
    
    def test_custom_config(self):
        """Custom configuration."""
        config = ToolUseConfig(
            allowed_tools=["web_search", "calculator"],
            max_tool_calls_per_turn=3,
        )
        mixin = ToolUseMixin(config)
        assert mixin.tool_config.allowed_tools == ["web_search", "calculator"]
    
    @pytest.mark.asyncio
    async def test_on_start_resets_counter(self):
        """on_start resets tool call counter."""
        mixin = ToolUseMixin()
        mixin._tool_call_count = 10
        
        await mixin.on_start({})
        
        assert mixin._tool_call_count == 0
    
    @pytest.mark.asyncio
    async def test_on_tool_call_allowed(self):
        """Tool call allowed when in allowed list."""
        config = ToolUseConfig(allowed_tools=["calculator"])
        mixin = ToolUseMixin(config)
        
        result = await mixin.on_tool_call("calculator", {"x": 1})
        assert result is None  # Pass through
    
    @pytest.mark.asyncio
    async def test_on_tool_call_denied(self):
        """Tool call denied when not in allowed list."""
        config = ToolUseConfig(allowed_tools=["calculator"])
        mixin = ToolUseMixin(config)
        
        with pytest.raises(PermissionError, match="not in allowed list"):
            await mixin.on_tool_call("web_search", {})
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Rate limit enforcement."""
        config = ToolUseConfig(max_tool_calls_per_turn=2)
        mixin = ToolUseMixin(config)
        
        await mixin.on_tool_call("tool1", {})
        await mixin.on_tool_call("tool2", {})
        
        with pytest.raises(RuntimeError, match="Exceeded max tool calls"):
            await mixin.on_tool_call("tool3", {})


class TestMemoryMixin:
    """Tests for MemoryMixin."""
    
    def test_default_config(self):
        """Default configuration values."""
        mixin = MemoryMixin()
        assert mixin.memory_config.max_memories == 100
        assert mixin.memory_config.auto_summarize is True
    
    @pytest.mark.asyncio
    async def test_remember(self):
        """Store memory."""
        mixin = MemoryMixin()
        
        entry = await mixin.remember("Test memory", importance=0.8, tags=["test"])
        
        assert entry.content == "Test memory"
        assert entry.importance == 0.8
        assert "test" in entry.tags
    
    @pytest.mark.asyncio
    async def test_recall(self):
        """Retrieve memories."""
        mixin = MemoryMixin()
        
        await mixin.remember("Memory 1", importance=0.5)
        await mixin.remember("Memory 2", importance=0.9)
        await mixin.remember("Memory 3", importance=0.3)
        
        recalled = await mixin.recall(limit=2)
        
        assert len(recalled) == 2
        # Higher importance first
        assert recalled[0].importance >= recalled[1].importance
    
    @pytest.mark.asyncio
    async def test_recall_with_tags(self):
        """Recall memories filtered by tags."""
        mixin = MemoryMixin()
        
        await mixin.remember("Tagged", tags=["important"])
        await mixin.remember("Untagged")
        
        recalled = await mixin.recall(tags=["important"])
        
        assert len(recalled) == 1
        assert recalled[0].content == "Tagged"
    
    @pytest.mark.asyncio
    async def test_forget(self):
        """Remove memory."""
        mixin = MemoryMixin()
        
        await mixin.remember("To forget")
        await mixin.remember("To keep")
        
        result = await mixin.forget("To forget")
        
        assert result is True
        assert len(mixin._memories) == 1
    
    @pytest.mark.asyncio
    async def test_memory_decay(self):
        """Memory importance decays over time."""
        config = MemoryConfig(memory_decay=True, decay_half_life_hours=0.001)
        mixin = MemoryMixin(config)
        
        await mixin.remember("Old memory", importance=1.0)
        
        import time
        time.sleep(0.01)  # Small delay
        
        mixin._apply_decay()
        
        # Importance should have decreased
        assert mixin._memories[0].importance < 1.0
    
    @pytest.mark.asyncio
    async def test_memory_pruning(self):
        """Memories pruned when over limit."""
        config = MemoryConfig(max_memories=2)
        mixin = MemoryMixin(config)
        
        await mixin.remember("Memory 1", importance=0.3)
        await mixin.remember("Memory 2", importance=0.9)
        await mixin.remember("Memory 3", importance=0.5)
        
        mixin._prune_memories()
        
        assert len(mixin._memories) == 2
        # Lowest importance removed
        contents = [m.content for m in mixin._memories]
        assert "Memory 1" not in contents


class TestReasoningMixin:
    """Tests for ReasoningMixin."""
    
    def test_available_styles(self):
        """Verify available reasoning styles."""
        assert "step_by_step" in ReasoningMixin.REASONING_PROMPTS
        assert "tree_of_thought" in ReasoningMixin.REASONING_PROMPTS
        assert "socratic" in ReasoningMixin.REASONING_PROMPTS
    
    @pytest.mark.asyncio
    async def test_on_invoke_injects_prompt(self):
        """Reasoning prompt injected into request."""
        mixin = ReasoningMixin()
        
        request = {"messages": [{"role": "user", "content": "Test question"}]}
        result = await mixin.on_invoke(request)
        
        # System message added with reasoning prompt
        assert result["messages"][0]["role"] == "system"
        assert "step by step" in result["messages"][0]["content"].lower()
    
    @pytest.mark.asyncio
    async def test_on_response_extracts_reasoning(self):
        """Reasoning trace extracted from response."""
        mixin = ReasoningMixin()
        
        response = {
            "content": "1. First step\n2. Second step\n3. Conclusion"
        }
        result = await mixin.on_response(response)
        
        assert "_reasoning" in result
        assert len(result["_reasoning"]["steps"]) >= 2


class TestStructuredOutputMixin:
    """Tests for StructuredOutputMixin."""
    
    @pytest.mark.asyncio
    async def test_on_invoke_adds_schema(self):
        """Schema instruction added to request."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        config = StructuredOutputConfig(output_schema=schema)
        mixin = StructuredOutputMixin(config)
        
        request = {"messages": [{"role": "user", "content": "Question"}]}
        result = await mixin.on_invoke(request)
        
        assert "JSON" in result["messages"][0]["content"]
    
    def test_extract_json_direct(self):
        """Extract JSON from direct response."""
        mixin = StructuredOutputMixin()
        
        result = mixin._extract_json('{"key": "value"}')
        
        assert result == {"key": "value"}
    
    def test_extract_json_from_markdown(self):
        """Extract JSON from markdown code block."""
        mixin = StructuredOutputMixin()
        
        content = '''Here is the answer:
```json
{"key": "value"}
```
'''
        result = mixin._extract_json(content)
        
        assert result == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_validation_passes(self):
        """Valid response passes validation."""
        schema = {
            "type": "object",
            "required": ["answer"],
        }
        config = StructuredOutputConfig(
            output_schema=schema,
            strict_validation=True,
        )
        mixin = StructuredOutputMixin(config)
        
        response = {"content": '{"answer": "test"}'}
        result = await mixin.on_response(response)
        
        assert result["_structured_output"]["answer"] == "test"
    
    @pytest.mark.asyncio
    async def test_validation_fails(self):
        """Invalid response fails validation."""
        schema = {
            "type": "object",
            "required": ["missing_field"],
        }
        config = StructuredOutputConfig(
            output_schema=schema,
            strict_validation=True,
            auto_repair=False,
        )
        mixin = StructuredOutputMixin(config)
        
        response = {"content": '{"other": "value"}'}
        
        with pytest.raises(ValueError, match="Missing required field"):
            await mixin.on_response(response)
```

## Topology Tests

### File: `tests/agents/test_topologies.py`

```python
"""Tests for agent collaboration topologies."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from qmcp.agentframework.models import (
    AgentRole,
    AgentType,
    DebateConfig,
    EnsembleConfig,
    ExecutionStatus,
    PipelineConfig,
    Topology,
    TopologyType,
)
from qmcp.agentframework.topologies import (
    BaseTopology,
    ChainOfCommandTopology,
    CrossCheckTopology,
    DebateTopology,
    EnsembleTopology,
    ExecutionContext,
    PipelineTopology,
    TopologyRegistry,
)


class TestTopologyRegistry:
    """Tests for TopologyRegistry."""
    
    def test_all_topologies_registered(self):
        """All topology types are registered."""
        for topo_type in TopologyType:
            if topo_type != TopologyType.COMPOUND:  # Compound is special
                assert TopologyRegistry.get(topo_type) is not None
    
    def test_create_topology(self, db_session, debate_topology, planner_agent, critic_agent, synthesizer_agent):
        """Create topology instance from model."""
        agents = {
            "proponent": planner_agent,
            "opponent": critic_agent,
            "mediator": synthesizer_agent,
        }
        
        topo = TopologyRegistry.create(debate_topology, agents, db_session)
        
        assert isinstance(topo, DebateTopology)


class TestDebateTopology:
    """Tests for DebateTopology."""
    
    @pytest.fixture
    def debate_agents(self, planner_agent, critic_agent, synthesizer_agent):
        """Agents for debate topology."""
        return {
            "proponent": planner_agent,
            "opponent": critic_agent,
            "mediator": synthesizer_agent,
        }
    
    @pytest.mark.asyncio
    async def test_requires_all_slots(self, db_session, debate_topology, planner_agent):
        """Debate requires proponent, opponent, mediator."""
        incomplete_agents = {"proponent": planner_agent}
        
        topo = DebateTopology(debate_topology, incomplete_agents, db_session)
        context = ExecutionContext(
            execution_id=uuid4(),
            topology_id=debate_topology.id,
            input_data={"topic": "Test"},
        )
        
        with pytest.raises(ValueError, match="requires 'opponent' slot"):
            await topo._run(context)
    
    @pytest.mark.asyncio
    async def test_debate_structure(self, db_session, debate_topology, debate_agents):
        """Verify debate execution structure."""
        topo = DebateTopology(debate_topology, debate_agents, db_session)
        
        # Mock LLM calls
        with patch.object(topo, 'invoke_agent') as mock_invoke:
            mock_invoke.return_value = MagicMock(
                output={"content": "Test argument"},
                confidence=0.8,
            )
            with patch.object(topo, 'record_result'):
                with patch.object(topo, 'broadcast_message'):
                    with patch.object(topo, 'setup'):
                        with patch.object(topo, 'teardown'):
                            context = ExecutionContext(
                                execution_id=uuid4(),
                                topology_id=debate_topology.id,
                                input_data={"topic": "AI Safety"},
                            )
                            
                            result = await topo._run(context)
        
        assert "debate_history" in result
        assert "synthesis" in result


class TestEnsembleTopology:
    """Tests for EnsembleTopology."""
    
    @pytest.fixture
    def ensemble_agents(self, planner_agent, executor_agent, synthesizer_agent):
        """Agents for ensemble topology."""
        return {
            "ensemble_0": planner_agent,
            "ensemble_1": executor_agent,
            "aggregator": synthesizer_agent,
        }
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, db_session, ensemble_topology, ensemble_agents):
        """Ensemble executes agents in parallel."""
        topo = EnsembleTopology(ensemble_topology, ensemble_agents, db_session)
        
        call_order = []
        
        async def mock_invoke(slot, messages, context):
            call_order.append(slot)
            return MagicMock(
                output={"content": f"Response from {slot}"},
                confidence=0.8,
                error=None,
            )
        
        with patch.object(topo, 'invoke_agent', side_effect=mock_invoke):
            with patch.object(topo, 'record_result'):
                with patch.object(topo, 'setup'):
                    with patch.object(topo, 'teardown'):
                        context = ExecutionContext(
                            execution_id=uuid4(),
                            topology_id=ensemble_topology.id,
                            input_data={"prompt": "Test prompt"},
                        )
                        
                        result = await topo._run(context)
        
        assert result["ensemble_size"] == 2
        assert result["successful"] == 2
    
    @pytest.mark.asyncio
    async def test_failure_threshold(self, db_session, ensemble_topology, ensemble_agents):
        """Ensemble fails when too many agents fail."""
        ensemble_topology.config["failure_threshold"] = 0.3
        topo = EnsembleTopology(ensemble_topology, ensemble_agents, db_session)
        
        async def mock_invoke_failing(slot, messages, context):
            return MagicMock(
                output={},
                error="Simulated failure",
            )
        
        with patch.object(topo, 'invoke_agent', side_effect=mock_invoke_failing):
            with patch.object(topo, 'record_result'):
                with patch.object(topo, 'setup'):
                    context = ExecutionContext(
                        execution_id=uuid4(),
                        topology_id=ensemble_topology.id,
                        input_data={"prompt": "Test"},
                    )
                    
                    with pytest.raises(RuntimeError, match="Too many ensemble failures"):
                        await topo._run(context)


class TestPipelineTopology:
    """Tests for PipelineTopology."""
    
    @pytest.fixture
    def pipeline_agents(self, planner_agent, executor_agent, reviewer_agent):
        """Agents for pipeline topology."""
        return {
            "parse": planner_agent,
            "analyze": executor_agent,
            "generate": reviewer_agent,
        }
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self, db_session, pipeline_topology, pipeline_agents):
        """Pipeline executes stages sequentially."""
        topo = PipelineTopology(pipeline_topology, pipeline_agents, db_session)
        
        execution_order = []
        
        async def mock_invoke(slot, messages, context):
            execution_order.append(slot)
            return MagicMock(
                output={"content": f"Output from {slot}"},
                duration_ms=100,
            )
        
        with patch.object(topo, 'invoke_agent', side_effect=mock_invoke):
            with patch.object(topo, 'record_result'):
                with patch.object(topo, 'send_message'):
                    with patch.object(topo, 'setup'):
                        with patch.object(topo, 'teardown'):
                            context = ExecutionContext(
                                execution_id=uuid4(),
                                topology_id=pipeline_topology.id,
                                input_data={"data": "test"},
                            )
                            
                            result = await topo._run(context)
        
        # Verify order
        assert execution_order == ["parse", "analyze", "generate"]
        assert len(result["stage_results"]) == 3
```

## Runner Tests

### File: `tests/agents/test_runners.py`

```python
"""Tests for topology runners."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from qmcp.agentframework.models import (
    AgentType,
    ExecutionStatus,
    Topology,
    TopologyType,
)
from qmcp.agentframework.runners import (
    AsyncRunner,
    LocalRunner,
    MetaflowRunner,
    RunConfig,
    RunnerRegistry,
)


class TestRunnerRegistry:
    """Tests for RunnerRegistry."""
    
    def test_all_runners_registered(self):
        """All runner types are registered."""
        assert RunnerRegistry.get("local") is not None
        assert RunnerRegistry.get("async") is not None
        assert RunnerRegistry.get("metaflow") is not None
    
    def test_create_runner(self):
        """Create runner instance."""
        runner = RunnerRegistry.create("local", RunConfig())
        assert isinstance(runner, LocalRunner)
    
    def test_create_unknown_raises(self):
        """Creating unknown runner raises error."""
        with pytest.raises(ValueError, match="Unknown runner type"):
            RunnerRegistry.create("nonexistent")


class TestLocalRunner:
    """Tests for LocalRunner."""
    
    @pytest.mark.asyncio
    async def test_run_simple_topology(self, planner_agent, ensemble_topology):
        """Run simple topology locally."""
        runner = LocalRunner(db_url="sqlite+aiosqlite:///:memory:")
        
        agents = {"ensemble_0": planner_agent}
        
        with patch.object(runner, '_execute') as mock_execute:
            mock_execution = MagicMock()
            mock_execution.id = uuid4()
            mock_execution.status = ExecutionStatus.COMPLETED
            mock_execution.output_data = {"result": "success"}
            mock_execution.error = None
            mock_execute.return_value = mock_execution
            
            result = await runner.run(
                ensemble_topology,
                agents,
                {"prompt": "Test"},
            )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.runner_type == "local"
    
    @pytest.mark.asyncio
    async def test_validation_error(self, planner_agent):
        """Invalid topology raises validation error."""
        runner = LocalRunner(db_url="sqlite+aiosqlite:///:memory:")
        
        # Missing agents
        topology = Topology(
            name="test",
            description="Test",
            topology_type=TopologyType.DEBATE,
            config={},
        )
        
        result = await runner.run(topology, {}, {})
        
        assert result.status == ExecutionStatus.FAILED
        assert "at least one agent" in result.error.lower()


class TestAsyncRunner:
    """Tests for AsyncRunner."""
    
    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """Concurrent execution respects limit."""
        runner = AsyncRunner(
            db_url="sqlite+aiosqlite:///:memory:",
            max_concurrent=2,
        )
        
        assert runner._semaphore._value == 2


class TestMetaflowRunner:
    """Tests for MetaflowRunner."""
    
    def test_generates_flow_file(self, tmp_path, debate_topology, planner_agent, critic_agent, synthesizer_agent):
        """Generates valid Metaflow flow file."""
        runner = MetaflowRunner(output_dir=str(tmp_path))
        
        agents = {
            "proponent": planner_agent,
            "opponent": critic_agent,
            "mediator": synthesizer_agent,
        }
        
        flow_path = runner._generate_flow(debate_topology, agents)
        
        assert flow_path.exists()
        content = flow_path.read_text()
        assert "from metaflow import" in content
        assert "FlowSpec" in content
        assert "@step" in content
    
    def test_pipeline_flow_generation(self, tmp_path, pipeline_topology, planner_agent, executor_agent, reviewer_agent):
        """Generate pipeline flow."""
        runner = MetaflowRunner(output_dir=str(tmp_path))
        
        agents = {
            "parse": planner_agent,
            "analyze": executor_agent,
            "generate": reviewer_agent,
        }
        
        code = runner._generate_pipeline_flow(pipeline_topology, agents)
        
        assert "def parse(self)" in code
        assert "def analyze(self)" in code
        assert "def generate(self)" in code
    
    def test_ensemble_flow_generation(self, tmp_path, ensemble_topology, planner_agent, executor_agent, synthesizer_agent):
        """Generate ensemble flow with foreach."""
        runner = MetaflowRunner(output_dir=str(tmp_path))
        
        agents = {
            "ensemble_0": planner_agent,
            "ensemble_1": executor_agent,
            "aggregator": synthesizer_agent,
        }
        
        code = runner._generate_ensemble_flow(ensemble_topology, agents)
        
        assert "foreach" in code
        assert "invoke_ensemble" in code
        assert "aggregate" in code
```

## Integration Tests

### File: `tests/agents/test_integration.py`

```python
"""Integration tests for the agent framework."""

import pytest
from uuid import uuid4

from qmcp.agentframework.models import (
    AgentConfig,
    AgentRole,
    AgentType,
    DebateConfig,
    EnsembleConfig,
    ExecutionStatus,
    Topology,
    TopologyMembership,
    TopologyType,
)
from qmcp.agentframework.mixins import MixinRegistry
from qmcp.agentframework.topologies import TopologyRegistry
from qmcp.agentframework.runners import LocalRunner, RunConfig


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.mark.asyncio
    async def test_create_and_run_debate(self, db_session):
        """Create agents, topology, and run debate."""
        # Create agents
        proponent = AgentType(
            name=f"proponent_{uuid4().hex[:8]}",
            description="Argues in favor",
            role=AgentRole.CRITIC,
            config=AgentConfig(
                system_prompt="You argue in favor of the topic.",
            ).model_dump(),
        )
        opponent = AgentType(
            name=f"opponent_{uuid4().hex[:8]}",
            description="Argues against",
            role=AgentRole.CRITIC,
            config=AgentConfig(
                system_prompt="You argue against the topic.",
            ).model_dump(),
        )
        mediator = AgentType(
            name=f"mediator_{uuid4().hex[:8]}",
            description="Synthesizes debate",
            role=AgentRole.SYNTHESIZER,
            config=AgentConfig(
                system_prompt="You synthesize the debate.",
            ).model_dump(),
        )
        
        for agent in [proponent, opponent, mediator]:
            db_session.add(agent)
        await db_session.flush()
        
        # Create topology
        topology = Topology(
            name=f"debate_{uuid4().hex[:8]}",
            description="Test debate",
            topology_type=TopologyType.DEBATE,
            config=DebateConfig(max_rounds=1).model_dump(),
        )
        db_session.add(topology)
        await db_session.flush()
        
        # Create memberships
        for slot, agent in [("proponent", proponent), ("opponent", opponent), ("mediator", mediator)]:
            membership = TopologyMembership(
                topology_id=topology.id,
                agent_type_id=agent.id,
                slot_name=slot,
            )
            db_session.add(membership)
        await db_session.commit()
        
        # Verify setup
        assert topology.id is not None
        assert proponent.id is not None
    
    @pytest.mark.asyncio
    async def test_mixin_composition(self, db_session, planner_agent):
        """Test that multiple mixins compose correctly."""
        # Add capabilities to agent
        planner_agent.config["capabilities"] = [
            {"name": "tool_use", "config": {"max_tool_calls_per_turn": 3}},
            {"name": "memory", "config": {"max_memories": 50}},
            {"name": "reasoning", "config": {"reasoning_style": "step_by_step"}},
        ]
        
        # Resolve dependencies
        cap_names = [c["name"] for c in planner_agent.config["capabilities"]]
        ordered = MixinRegistry.resolve_dependencies(cap_names)
        
        # Create mixin instances
        mixins = []
        for name in ordered:
            cap = next(c for c in planner_agent.config["capabilities"] if c["name"] == name)
            mixin = MixinRegistry.create(name, cap.get("config", {}))
            mixins.append(mixin)
        
        assert len(mixins) == 3
        
        # Apply mixins to request
        request = {"messages": [{"role": "user", "content": "Test"}]}
        for mixin in mixins:
            request = await mixin.on_invoke(request)
        
        # Verify all mixins modified request
        messages = request["messages"]
        # Reasoning adds system message
        assert any(m["role"] == "system" for m in messages)


class TestDatabasePersistence:
    """Test database persistence across operations."""
    
    @pytest.mark.asyncio
    async def test_agent_persistence(self, db_session):
        """Agents persist with all fields."""
        agent = AgentType(
            name=f"persist_test_{uuid4().hex[:8]}",
            description="Persistence test agent",
            role=AgentRole.EXECUTOR,
            config={
                "model": "claude-sonnet-4-20250514",
                "capabilities": [{"name": "tool_use"}],
            },
        )
        db_session.add(agent)
        await db_session.commit()
        
        # Refresh from database
        await db_session.refresh(agent)
        
        assert agent.id is not None
        assert agent.config["model"] == "claude-sonnet-4-20250514"
        assert len(agent.config["capabilities"]) == 1
    
    @pytest.mark.asyncio
    async def test_execution_history(self, db_session, debate_topology):
        """Execution history is recorded."""
        from qmcp.agentframework.models import Execution
        
        # Create multiple executions
        for i in range(3):
            execution = Execution(
                topology_id=debate_topology.id,
                input_data={"iteration": i},
                status=ExecutionStatus.COMPLETED,
                output_data={"result": f"Result {i}"},
            )
            db_session.add(execution)
        
        await db_session.commit()
        
        # Query history
        from sqlmodel import select
        
        stmt = select(Execution).where(Execution.topology_id == debate_topology.id)
        result = await db_session.execute(stmt)
        executions = result.scalars().all()
        
        assert len(executions) == 3
```

## Running Tests

```bash
# Run all tests
pytest tests/agents/ -v

# Run with coverage
pytest tests/agents/ --cov=qmcp.agentframework --cov-report=html

# Run specific test file
pytest tests/agents/test_models.py -v

# Run specific test class
pytest tests/agents/test_models.py::TestAgentType -v

# Run with async debug
pytest tests/agents/ -v --asyncio-mode=auto
```

## Test Utilities

### File: `tests/agents/utils.py`

```python
"""Test utilities and helpers."""

from typing import Any
from uuid import uuid4

from qmcp.agentframework.models import AgentRole, AgentType, Topology, TopologyType


def create_test_agent(
    name: str = None,
    role: AgentRole = AgentRole.EXECUTOR,
    config: dict = None,
) -> AgentType:
    """Create agent for testing."""
    return AgentType(
        name=name or f"test_agent_{uuid4().hex[:8]}",
        description="Test agent",
        role=role,
        config=config or {},
    )


def create_test_topology(
    name: str = None,
    topology_type: TopologyType = TopologyType.PIPELINE,
    config: dict = None,
) -> Topology:
    """Create topology for testing."""
    return Topology(
        name=name or f"test_topology_{uuid4().hex[:8]}",
        description="Test topology",
        topology_type=topology_type,
        config=config or {},
    )


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: list[dict[str, Any]] = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls = []
    
    async def complete(self, messages: list[dict], **kwargs) -> dict:
        """Return mock response."""
        self.calls.append({"messages": messages, **kwargs})
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = {"content": "Mock response"}
        
        self.call_count += 1
        return response
```
