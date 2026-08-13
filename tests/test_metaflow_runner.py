"""Tests for the MetaflowRunner implementation."""

from __future__ import annotations

import sys

import pytest

from qmcp.agentframework.models import Topology, TopologyType
from qmcp.agentframework.runners import (
    MetaflowRunner,
    RunConfig,
    RunResult,
    RunnerRegistry,
    _TOPOLOGY_FLOW_MAP,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestMetaflowRunnerRegistered:
    """Verify MetaflowRunner is in the RunnerRegistry."""

    def test_registered(self):
        runner_cls = RunnerRegistry.get("metaflow")
        assert runner_cls is MetaflowRunner

    def test_create_via_registry(self):
        runner = RunnerRegistry.create("metaflow")
        assert isinstance(runner, MetaflowRunner)

    def test_all_runners_registered(self):
        for name in ("local", "async", "metaflow"):
            assert RunnerRegistry.get(name) is not None


# ---------------------------------------------------------------------------
# Flow path resolution
# ---------------------------------------------------------------------------


class TestMetaflowRunnerResolvesFlow:
    """Tests for _resolve_flow_path."""

    def test_pipeline_resolves(self):
        runner = MetaflowRunner()
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.PIPELINE,
            config={},
        )
        path = runner._resolve_flow_path(topology)
        assert path == "examples/flows/local_agent_chain.py"

    def test_council_resolves(self):
        runner = MetaflowRunner()
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.COUNCIL,
            config={},
        )
        path = runner._resolve_flow_path(topology)
        assert path == "examples/flows/council_deliberation.py"

    def test_explicit_flow_path_overrides(self):
        runner = MetaflowRunner(flow_path="custom/flow.py")
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.PIPELINE,
            config={},
        )
        assert runner._resolve_flow_path(topology) == "custom/flow.py"

    def test_unmapped_topology_raises(self):
        runner = MetaflowRunner()
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.DEBATE,
            config={},
        )
        with pytest.raises(ValueError, match="No default flow registered"):
            runner._resolve_flow_path(topology)


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


class TestMetaflowRunnerBuildsCommand:
    """Tests for _build_command."""

    def test_basic_command(self):
        runner = MetaflowRunner()
        cmd = runner._build_command("my_flow.py", [])
        assert cmd[0] == sys.executable
        assert cmd[1] == "my_flow.py"
        assert cmd[2] == "run"
        assert "--mcp-url" in cmd

    def test_extra_args_forwarded(self):
        runner = MetaflowRunner()
        cmd = runner._build_command("flow.py", ["--goal", "test", "--context", "x"])
        assert "--goal" in cmd
        assert "test" in cmd
        assert "--context" in cmd

    def test_mcp_url_not_duplicated(self):
        runner = MetaflowRunner()
        cmd = runner._build_command("flow.py", ["--mcp-url", "http://custom:9999"])
        # Should only appear once (from flow_args, not injected again)
        assert cmd.count("--mcp-url") == 1
        assert "http://custom:9999" in cmd

    def test_mcp_url_from_config(self):
        config = RunConfig(mcp_url="http://config:5555")
        runner = MetaflowRunner(config=config)
        cmd = runner._build_command("flow.py", [])
        assert "http://config:5555" in cmd


# ---------------------------------------------------------------------------
# RunConfig fields
# ---------------------------------------------------------------------------


class TestRunConfigMetaflowFields:
    """Tests for Metaflow-specific RunConfig fields."""

    def test_defaults(self):
        config = RunConfig()
        assert config.docker is False
        assert config.mcp_url is None
        assert config.metaflow_user is None
        assert config.build_image is True
        assert config.sync_deps is True
        assert config.timeout_seconds == 300

    def test_custom_values(self):
        config = RunConfig(
            docker=True,
            mcp_url="http://mcp:3333",
            metaflow_user="testuser",
            build_image=False,
            sync_deps=False,
            timeout_seconds=600,
        )
        assert config.docker is True
        assert config.mcp_url == "http://mcp:3333"
        assert config.metaflow_user == "testuser"
        assert config.build_image is False
        assert config.sync_deps is False
        assert config.timeout_seconds == 600


# ---------------------------------------------------------------------------
# RunResult model
# ---------------------------------------------------------------------------


class TestRunResult:
    """Tests for RunResult model."""

    def test_defaults(self):
        result = RunResult()
        assert result.status == "unknown"
        assert result.execution_id is None
        assert result.output is None
        assert result.error is None
        assert result.duration_ms is None

    def test_completed(self):
        result = RunResult(
            execution_id="exec-1",
            status="completed",
            output={"stdout": "ok", "returncode": 0},
            duration_ms=1234,
        )
        assert result.status == "completed"
        assert result.output["returncode"] == 0

    def test_failed(self):
        result = RunResult(
            execution_id="exec-2",
            status="failed",
            error="timeout",
            duration_ms=5000,
        )
        assert result.status == "failed"
        assert result.error == "timeout"


# ---------------------------------------------------------------------------
# Async run (mocked subprocess)
# ---------------------------------------------------------------------------


class TestMetaflowRunnerRun:
    """Tests for MetaflowRunner.run() with mocked subprocess."""

    def test_successful_run(self):
        import asyncio
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Flow completed"
        mock_result.stderr = ""

        runner = MetaflowRunner(flow_path="test_flow.py")
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.PIPELINE,
            config={},
        )

        with patch("qmcp.agentframework.runners.subprocess.run", return_value=mock_result):
            result = asyncio.run(runner.run(topology, {}))

        assert result.status == "completed"
        assert result.execution_id is not None
        assert result.duration_ms is not None

    def test_failed_run(self):
        import asyncio
        from unittest.mock import patch, MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"

        runner = MetaflowRunner(flow_path="test_flow.py")
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.PIPELINE,
            config={},
        )

        with patch("qmcp.agentframework.runners.subprocess.run", return_value=mock_result):
            result = asyncio.run(runner.run(topology, {}))

        assert result.status == "failed"
        assert result.error == "Error occurred"

    def test_timeout_run(self):
        import asyncio
        import subprocess as sp
        from unittest.mock import patch

        runner = MetaflowRunner(
            config=RunConfig(timeout_seconds=1),
            flow_path="test_flow.py",
        )
        topology = Topology(
            name="t",
            description="d",
            topology_type=TopologyType.PIPELINE,
            config={},
        )

        with patch(
            "qmcp.agentframework.runners.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="test", timeout=1),
        ):
            result = asyncio.run(runner.run(topology, {}))

        assert result.status == "failed"
        assert "timed out" in result.error


# ---------------------------------------------------------------------------
# Topology flow map
# ---------------------------------------------------------------------------


class TestTopologyFlowMap:
    """Tests for the default topology → flow path mapping."""

    def test_pipeline_mapped(self):
        assert TopologyType.PIPELINE in _TOPOLOGY_FLOW_MAP

    def test_council_mapped(self):
        assert TopologyType.COUNCIL in _TOPOLOGY_FLOW_MAP

    def test_debate_not_mapped(self):
        assert TopologyType.DEBATE not in _TOPOLOGY_FLOW_MAP
