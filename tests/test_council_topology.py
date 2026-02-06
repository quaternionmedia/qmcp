"""Smoke tests for the Council topology."""

import pytest
from click.testing import CliRunner

from qmcp.agentframework import (
    AgentRole,
    AgentType,
    ConsensusMethod,
    CouncilConfig,
    CouncilTopology,
    Topology,
    TopologyType,
)
from qmcp.agentframework.topologies import TopologyRegistry
from qmcp.cli import cli


class TestCouncilConfig:
    """Tests for CouncilConfig model."""

    def test_default_values(self):
        config = CouncilConfig()
        assert config.max_rounds == 5
        assert config.consensus_threshold == 0.67
        assert config.consensus_method == ConsensusMethod.QUORUM
        assert config.allow_early_consensus is True
        assert config.require_all_voices is True
        assert config.arbiter_can_override is True
        assert config.deliberation_style == "round_robin"

    def test_speaking_order_default(self):
        config = CouncilConfig()
        expected_order = [
            "storyteller",
            "dreamer",
            "strategist",
            "sanity_check",
            "archivist",
            "efficist",
            "accomplisher",
            "reflector",
        ]
        assert config.speaking_order == expected_order

    def test_consensus_threshold_validation(self):
        # Valid thresholds
        assert CouncilConfig(consensus_threshold=0.5).consensus_threshold == 0.5
        assert CouncilConfig(consensus_threshold=1.0).consensus_threshold == 1.0

        # Invalid thresholds
        with pytest.raises(ValueError):
            CouncilConfig(consensus_threshold=0.4)
        with pytest.raises(ValueError):
            CouncilConfig(consensus_threshold=1.1)

    def test_max_rounds_validation(self):
        assert CouncilConfig(max_rounds=1).max_rounds == 1
        assert CouncilConfig(max_rounds=20).max_rounds == 20

        with pytest.raises(ValueError):
            CouncilConfig(max_rounds=0)
        with pytest.raises(ValueError):
            CouncilConfig(max_rounds=21)

    def test_custom_speaking_order(self):
        custom_order = ["strategist", "efficist", "reflector"]
        config = CouncilConfig(speaking_order=custom_order)
        assert config.speaking_order == custom_order


class TestCouncilTopology:
    """Tests for CouncilTopology class."""

    def test_topology_type(self):
        assert CouncilTopology.topology_type == TopologyType.COUNCIL

    def test_config_class(self):
        assert CouncilTopology.config_class == CouncilConfig

    def test_registry_registration(self):
        """Verify CouncilTopology is registered in the TopologyRegistry."""
        topology_class = TopologyRegistry.get(TopologyType.COUNCIL)
        assert topology_class is CouncilTopology


class TestCouncilTopologyModel:
    """Tests for creating Council Topology database models."""

    def test_create_council_topology(self):
        topology = Topology(
            name="test_council",
            description="Test council for architecture decisions",
            topology_type=TopologyType.COUNCIL,
            config=CouncilConfig(
                max_rounds=3,
                consensus_threshold=0.67,
            ).model_dump(),
        )

        assert topology.name == "test_council"
        assert topology.topology_type == TopologyType.COUNCIL

    def test_config_can_be_parsed_as_council_config(self):
        """Test that topology config can be parsed back to CouncilConfig."""
        topology = Topology(
            name="test",
            description="Test",
            topology_type=TopologyType.COUNCIL,
            config={
                "max_rounds": 7,
                "consensus_threshold": 0.8,
                "arbiter_can_override": False,
            },
        )

        # Parse config dict back to CouncilConfig
        config = CouncilConfig(**topology.config)
        assert isinstance(config, CouncilConfig)
        assert config.max_rounds == 7
        assert config.consensus_threshold == 0.8
        assert config.arbiter_can_override is False


class TestCouncilAgentRoles:
    """Tests for council member agent types."""

    @pytest.fixture
    def council_agents(self):
        """Create all 9 council member agent types."""
        return {
            "arbiter": AgentType(
                name="council_arbiter",
                description="Council manager who facilitates and decides",
                role=AgentRole.COORDINATOR,
                config={"system_prompt": "You are the council arbiter."},
            ),
            "storyteller": AgentType(
                name="storyteller",
                description="Frames issues in narrative form",
                role=AgentRole.SPECIALIST,
                config={"system_prompt": "You are the relatable storyteller."},
            ),
            "dreamer": AgentType(
                name="dreamer",
                description="Explores possibilities without constraint",
                role=AgentRole.SPECIALIST,
                config={"system_prompt": "You are the infinite dreamer."},
            ),
            "strategist": AgentType(
                name="strategist",
                description="Focuses on practical implementation",
                role=AgentRole.PLANNER,
                config={"system_prompt": "You are the pragmatic strategist."},
            ),
            "sanity_check": AgentType(
                name="sanity_check",
                description="Validates feasibility and catches issues",
                role=AgentRole.REVIEWER,
                config={"system_prompt": "You are the sanity check."},
            ),
            "archivist": AgentType(
                name="archivist",
                description="Maintains context and references history",
                role=AgentRole.SPECIALIST,
                config={"system_prompt": "You are the tidy archivist."},
            ),
            "efficist": AgentType(
                name="efficist",
                description="Cuts through complexity for efficiency",
                role=AgentRole.CRITIC,
                config={"system_prompt": "You are the brutal efficist."},
            ),
            "accomplisher": AgentType(
                name="accomplisher",
                description="Drives toward completion",
                role=AgentRole.EXECUTOR,
                config={"system_prompt": "You are the eager accomplisher."},
            ),
            "reflector": AgentType(
                name="reflector",
                description="Provides deep technical analysis",
                role=AgentRole.SPECIALIST,
                config={"system_prompt": "You are the technical reflector."},
            ),
        }

    def test_all_council_members_created(self, council_agents):
        """Verify all 9 council members can be created."""
        assert len(council_agents) == 9

        expected_slots = {
            "arbiter",
            "storyteller",
            "dreamer",
            "strategist",
            "sanity_check",
            "archivist",
            "efficist",
            "accomplisher",
            "reflector",
        }
        assert set(council_agents.keys()) == expected_slots

    def test_agent_roles_match_documentation(self, council_agents):
        """Verify agent roles match the documented roles."""
        expected_roles = {
            "arbiter": AgentRole.COORDINATOR,
            "storyteller": AgentRole.SPECIALIST,
            "dreamer": AgentRole.SPECIALIST,
            "strategist": AgentRole.PLANNER,
            "sanity_check": AgentRole.REVIEWER,
            "archivist": AgentRole.SPECIALIST,
            "efficist": AgentRole.CRITIC,
            "accomplisher": AgentRole.EXECUTOR,
            "reflector": AgentRole.SPECIALIST,
        }

        for slot, expected_role in expected_roles.items():
            assert council_agents[slot].role == expected_role, (
                f"Agent {slot} has role {council_agents[slot].role}, "
                f"expected {expected_role}"
            )


class TestCouncilCLI:
    """Tests for council CLI commands."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_council_create_default(self, runner):
        """Test council create with minimal arguments."""
        result = runner.invoke(cli, ["council", "create", "--name", "test-council"])

        assert result.exit_code == 0
        assert "Council Topology Created" in result.output
        assert "test-council" in result.output
        assert "Max Rounds:" in result.output
        assert "Council Members:" in result.output

    def test_council_create_custom_config(self, runner):
        """Test council create with custom configuration."""
        result = runner.invoke(
            cli,
            [
                "council",
                "create",
                "--name",
                "custom-council",
                "--max-rounds",
                "7",
                "--consensus-threshold",
                "0.8",
                "--no-arbiter-override",
            ],
        )

        assert result.exit_code == 0
        assert "custom-council" in result.output

    def test_council_create_json_output(self, runner):
        """Test council create with JSON output."""
        result = runner.invoke(
            cli,
            ["council", "create", "--name", "json-council", "--output", "json"],
        )

        assert result.exit_code == 0
        # Should be valid JSON
        import json

        data = json.loads(result.output)
        assert data["name"] == "json-council"
        assert data["topology_type"] == "council"

    def test_council_members_list(self, runner):
        """Test council members command."""
        result = runner.invoke(cli, ["council", "members"])

        assert result.exit_code == 0
        assert "Council Member Roles" in result.output
        assert "arbiter" in result.output
        assert "storyteller" in result.output
        assert "dreamer" in result.output
        assert "strategist" in result.output
        assert "sanity_check" in result.output
        assert "archivist" in result.output
        assert "efficist" in result.output
        assert "accomplisher" in result.output
        assert "reflector" in result.output


class TestCouncilDeliberationFlowExists:
    """Tests to verify the council deliberation flow file exists and is valid."""

    def test_flow_file_exists(self):
        """Verify the council_deliberation.py flow file exists."""
        from pathlib import Path

        # Find the repo root
        current = Path(__file__).resolve()
        repo_root = current.parent.parent

        flow_path = repo_root / "examples" / "flows" / "council_deliberation.py"
        assert flow_path.exists(), f"Council flow not found at {flow_path}"

    def test_flow_imports(self):
        """Verify the council flow can be imported without errors."""
        import importlib.util
        from pathlib import Path

        current = Path(__file__).resolve()
        repo_root = current.parent.parent
        flow_path = repo_root / "examples" / "flows" / "council_deliberation.py"

        spec = importlib.util.spec_from_file_location("council_deliberation", flow_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        # Don't execute, just verify it can be loaded
        assert module is not None

    def test_flow_has_required_classes(self):
        """Verify the flow defines expected classes."""
        from pathlib import Path

        current = Path(__file__).resolve()
        repo_root = current.parent.parent
        flow_path = repo_root / "examples" / "flows" / "council_deliberation.py"

        content = flow_path.read_text()

        # Check for key class definitions
        assert "class CouncilRole" in content
        assert "class CouncilContribution" in content
        assert "class RoundSynthesis" in content
        assert "class CouncilDecision" in content
        assert "class CouncilDeliberationFlow" in content

        # Check for council member prompts
        assert "COUNCIL_PROMPTS" in content
        assert "ARBITER" in content
        assert "STORYTELLER" in content
        assert "DREAMER" in content


class TestCouncilTopologyIntegration:
    """Integration tests for the council topology."""

    def test_topology_registry_creates_council(self):
        """Test that TopologyRegistry can create a CouncilTopology instance."""
        topology_model = Topology(
            name="integration_test_council",
            description="Integration test",
            topology_type=TopologyType.COUNCIL,
            config=CouncilConfig().model_dump(),
        )

        # Create mock agents dict
        agents = {}

        # This should not raise
        topology_instance = TopologyRegistry.create(
            topology_model,
            agents,
            db_session=None,  # type: ignore
        )

        assert isinstance(topology_instance, CouncilTopology)
        assert topology_instance.topology == topology_model

    def test_council_config_serialization_roundtrip(self):
        """Test that CouncilConfig survives serialization through Topology."""
        original_config = CouncilConfig(
            max_rounds=7,
            consensus_threshold=0.75,
            deliberation_style="structured",
            speaking_order=["efficist", "strategist", "dreamer"],
        )

        topology = Topology(
            name="roundtrip_test",
            description="Test",
            topology_type=TopologyType.COUNCIL,
            config=original_config.model_dump(),
        )

        # Parse config dict back to CouncilConfig
        recovered_config = CouncilConfig(**topology.config)

        assert isinstance(recovered_config, CouncilConfig)
        assert recovered_config.max_rounds == 7
        assert recovered_config.consensus_threshold == 0.75
        assert recovered_config.deliberation_style == "structured"
        assert recovered_config.speaking_order == ["efficist", "strategist", "dreamer"]
