"""Tests for agent framework data models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from qmcp.agentframework.models import (
    AgentCapability,
    AgentConfig,
    AgentRole,
    AgentType,
    AggregationMethod,
    ConsensusMethod,
    DebateConfig,
    EnsembleConfig,
    Execution,
    ExecutionStatus,
    PipelineConfig,
    Topology,
    TopologyType,
)


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_all_roles_defined(self):
        expected_roles = {
            "planner",
            "executor",
            "reviewer",
            "critic",
            "synthesizer",
            "specialist",
            "coordinator",
            "observer",
        }
        actual_roles = {role.value for role in AgentRole}
        assert expected_roles == actual_roles

    def test_role_string_values(self):
        assert AgentRole.PLANNER.value == "planner"
        assert isinstance(AgentRole.PLANNER.value, str)


class TestAgentCapability:
    """Tests for AgentCapability model."""

    def test_create_minimal(self):
        cap = AgentCapability(name="tool_use")
        assert cap.name == "tool_use"
        assert cap.version == "1.0.0"
        assert cap.enabled is True
        assert cap.config == {}

    def test_create_with_config(self):
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
        config = AgentConfig()
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.max_retries == 3

    def test_temperature_validation(self):
        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            AgentConfig(temperature=2.5)

        assert AgentConfig(temperature=0.0).temperature == 0.0
        assert AgentConfig(temperature=2.0).temperature == 2.0

    def test_with_capabilities(self):
        config = AgentConfig(
            capabilities=[
                AgentCapability(name="tool_use"),
                AgentCapability(name="memory"),
            ]
        )
        assert len(config.capabilities) == 2


class TestAgentType:
    """Tests for AgentType model."""

    def test_create_agent_type(self):
        agent = AgentType(
            name="test_planner",
            description="A test planning agent",
            role=AgentRole.PLANNER,
            config={"model": "claude-sonnet-4-20250514"},
        )
        assert agent.name == "test_planner"
        assert agent.role == AgentRole.PLANNER

    def test_name_validation_valid(self):
        AgentType(name="valid_name", description="Test", role=AgentRole.PLANNER)
        AgentType(name="valid-name-2", description="Test", role=AgentRole.PLANNER)

    def test_name_validation_invalid(self):
        with pytest.raises(ValueError):
            AgentType.validate_name("invalid name")
        with pytest.raises(ValueError):
            AgentType.validate_name("invalid@name")

    def test_name_lowercased(self):
        result = AgentType.validate_name("MyAgent")
        assert result == "myagent"

    def test_get_capabilities(self):
        agent = AgentType(
            name="test",
            description="Test",
            role=AgentRole.PLANNER,
            config={
                "capabilities": [
                    {"name": "tool_use", "config": {}},
                    {"name": "memory", "config": {}},
                ]
            },
        )
        caps = agent.get_capabilities()
        assert len(caps) == 2
        assert caps[0].name == "tool_use"

    def test_has_capability(self):
        agent = AgentType(
            name="test",
            description="Test",
            role=AgentRole.PLANNER,
            config={"capabilities": [{"name": "tool_use"}]},
        )
        assert agent.has_capability("tool_use") is True
        assert agent.has_capability("nonexistent") is False


class TestTopology:
    """Tests for Topology model."""

    def test_create_topology(self):
        topology = Topology(
            name="test_debate",
            description="Test debate topology",
            topology_type=TopologyType.DEBATE,
            config={"max_rounds": 3},
        )
        assert topology.name == "test_debate"
        assert topology.topology_type == TopologyType.DEBATE

    def test_get_typed_config(self):
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
        config = DebateConfig()
        assert config.max_rounds == 3
        assert config.consensus_method == ConsensusMethod.MEDIATOR_DECISION
        assert config.allow_early_termination is True

    def test_ensemble_config_defaults(self):
        config = EnsembleConfig()
        assert config.aggregation_method == AggregationMethod.SYNTHESIS
        assert config.failure_threshold == 0.5

    def test_pipeline_config(self):
        config = PipelineConfig(
            stages=["parse", "analyze", "generate"],
            checkpoint_after=["analyze"],
        )
        assert len(config.stages) == 3
        assert "analyze" in config.checkpoint_after


class TestExecution:
    """Tests for Execution model."""

    def test_create_execution(self):
        execution = Execution(
            topology_id=1,
            input_data={"topic": "Test topic"},
            status=ExecutionStatus.PENDING,
        )
        assert execution.status == ExecutionStatus.PENDING

    def test_mark_complete(self):
        execution = Execution(
            topology_id=1,
            input_data={},
            status=ExecutionStatus.RUNNING,
        )
        execution.mark_complete({"result": "success"})

        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.output_data["result"] == "success"
        assert execution.completed_at is not None

    def test_mark_failed(self):
        execution = Execution(
            topology_id=1,
            input_data={},
        )
        execution.mark_failed("Test error", {"code": 500})

        assert execution.status == ExecutionStatus.FAILED
        assert execution.error == "Test error"
        assert execution.error_details["code"] == 500

    def test_duration_calculation(self):
        execution = Execution(topology_id=1, input_data={})
        execution.started_at = datetime.now(UTC)
        execution.completed_at = execution.started_at + timedelta(seconds=5)

        assert execution.duration_ms == 5000
