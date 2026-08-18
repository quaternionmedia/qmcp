"""Tests for compound recipe flows and CLI registration."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from qmcp.cli import cli


REPO_ROOT = Path(__file__).resolve().parent.parent
FLOW_DIR = REPO_ROOT / "examples" / "flows"


# ---------------------------------------------------------------------------
# Flow file existence
# ---------------------------------------------------------------------------


class TestCompoundFlowFilesExist:
    """Verify compound recipe flow files exist."""

    def test_qc_release_exists(self):
        assert (FLOW_DIR / "qc_release.py").exists()

    def test_plan_council_exists(self):
        assert (FLOW_DIR / "plan_council.py").exists()

    def test_change_impact_exists(self):
        assert (FLOW_DIR / "change_impact.py").exists()


# ---------------------------------------------------------------------------
# Flow file content checks
# ---------------------------------------------------------------------------


class TestQCReleaseFlowContent:
    """Verify qc_release.py has expected structure."""

    @pytest.fixture
    def content(self):
        return (FLOW_DIR / "qc_release.py").read_text()

    def test_has_pipeline_definition(self, content):
        assert "QC_RELEASE_PIPELINE" in content

    def test_has_flow_class(self, content):
        assert "class QCReleaseFlow" in content

    def test_uses_agent_pipeline(self, content):
        assert "AgentPipeline" in content

    def test_has_all_steps(self, content):
        for step_name in ["qc_checklist", "qc_gate", "release_notes", "doc_updates"]:
            assert step_name in content, f"Missing step: {step_name}"

    def test_imports_from_cookbook(self, content):
        assert "from qmcp.cookbook" in content


class TestPlanCouncilFlowContent:
    """Verify plan_council.py has expected structure."""

    @pytest.fixture
    def content(self):
        return (FLOW_DIR / "plan_council.py").read_text()

    def test_has_flow_class(self, content):
        assert "class PlanCouncilFlow" in content

    def test_has_council_members(self, content):
        assert "COUNCIL_MEMBERS" in content
        for member in ["strategist", "sanity_check", "efficist"]:
            assert member in content

    def test_has_plan_and_refine_steps(self, content):
        assert "def plan(" in content
        assert "def council(" in content
        assert "def refine(" in content

    def test_uses_agent_step(self, content):
        assert "AgentStep" in content


class TestChangeImpactFlowContent:
    """Verify change_impact.py has expected structure."""

    @pytest.fixture
    def content(self):
        return (FLOW_DIR / "change_impact.py").read_text()

    def test_has_pipeline_definition(self, content):
        assert "CHANGE_IMPACT_PIPELINE" in content

    def test_has_flow_class(self, content):
        assert "class ChangeImpactFlow" in content

    def test_has_all_steps(self, content):
        for step_name in ["summarizer", "risk_assessor", "test_planner", "migration_guide"]:
            assert step_name in content, f"Missing step: {step_name}"

    def test_no_mcp_tools(self, content):
        """Change impact is a pure pipeline — no MCP integration."""
        assert "MCPToolInvoker" not in content
        assert "mcp_tool=" not in content


# ---------------------------------------------------------------------------
# CLI recipe registration
# ---------------------------------------------------------------------------


class TestCompoundRecipesInCLI:
    """Verify compound recipes are registered in the CLI."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_cookbook_list_includes_compounds(self, runner):
        result = runner.invoke(cli, ["cookbook", "list"])
        assert result.exit_code == 0
        for recipe in ["qc-release", "plan-council", "change-impact"]:
            assert recipe in result.output, f"Recipe {recipe!r} not in cookbook list"

    def test_qc_release_requires_change_summary(self, runner):
        result = runner.invoke(cli, ["cookbook", "run", "qc-release"])
        assert result.exit_code != 0
        assert "--change-summary" in result.output

    def test_plan_council_requires_goal(self, runner):
        result = runner.invoke(cli, ["cookbook", "run", "plan-council"])
        assert result.exit_code != 0
        assert "--goal" in result.output

    def test_change_impact_requires_change_summary(self, runner):
        result = runner.invoke(cli, ["cookbook", "run", "change-impact"])
        assert result.exit_code != 0
        assert "--change-summary" in result.output
