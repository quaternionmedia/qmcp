"""Tests for qmcp.cookbook composable modules."""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from qmcp.cookbook.agent_builders import LocalLLMConfig
from qmcp.cookbook.mcp_tools import (
    EchoInput,
    ExecutorInput,
    MCPToolInvoker,
    PlannerInput,
    ReviewerInput,
    require_invocation_id,
)
from qmcp.cookbook.persistence import (
    Artifact,
    FlowPersistence,
    FlowRun,
    AgentRun,
    MCPInvocation,
    init_db,
)


# ---------------------------------------------------------------------------
# Agent builder tests
# ---------------------------------------------------------------------------


class TestLocalLLMConfig:
    """Tests for LocalLLMConfig defaults and validation."""

    def test_defaults(self):
        config = LocalLLMConfig(
            model="llama3.1",
            base_url="http://localhost:11434/v1",
        )
        assert config.api_key == "local"
        assert config.temperature == 0.3
        assert config.max_tokens == 512

    def test_custom_values(self):
        config = LocalLLMConfig(
            model="mistral",
            base_url="http://localhost:8080/v1",
            api_key="sk-test",
            temperature=0.7,
            max_tokens=1024,
        )
        assert config.model == "mistral"
        assert config.base_url == "http://localhost:8080/v1"
        assert config.api_key == "sk-test"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024


class TestBuildLocalAgent:
    """Tests for build_local_agent (mocked — no real LLM needed)."""

    def test_returns_agent(self):
        """Agent is created with correct configuration."""
        # build_local_agent imports openai lazily so the dependency stays
        # optional, but patch() resolves its target by importing it — so this
        # test needs the extra present even though the code under test does
        # not. Guard every module patched below, not just the first: the
        # 'openai' extra installs openai *and* pydantic-ai, and having only
        # one of them still fails here.
        _reason = "requires the 'openai' extra: uv sync --extra openai"
        pytest.importorskip("openai", reason=_reason)
        pytest.importorskip("pydantic_ai.providers.openai", reason=_reason)
        pytest.importorskip("pydantic_ai.models.openai", reason=_reason)

        from unittest.mock import patch, MagicMock
        from pydantic import BaseModel

        class Output(BaseModel):
            text: str

        config = LocalLLMConfig(
            model="test-model",
            base_url="http://localhost:11434/v1",
        )

        # build_local_agent uses lazy imports inside the function body,
        # so we mock at the source module level.
        mock_agent_cls = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        with patch("openai.AsyncOpenAI"), \
             patch("pydantic_ai.providers.openai.OpenAIProvider"), \
             patch("pydantic_ai.models.openai.OpenAIChatModel"), \
             patch("pydantic_ai.Agent", mock_agent_cls):

            from qmcp.cookbook.agent_builders import build_local_agent
            result = build_local_agent(config, "You are a test agent.", Output, retries=2)

        assert result is mock_agent_instance
        # Verify Agent was called with expected kwargs
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["system_prompt"].startswith("You are a test agent.")
        assert call_kwargs.kwargs["output_type"] is Output
        assert call_kwargs.kwargs["retries"] == 2


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestFlowPersistenceLifecycle:
    """Tests for FlowPersistence context manager."""

    def test_context_manager_creates_and_finishes_run(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        with FlowPersistence(db_path, "TestFlow", "run-1", {"key": "val"}) as fp:
            assert fp.flow_run_id is not None
            flow_run_id = fp.flow_run_id

            # Verify run was created
            with Session(fp.engine) as session:
                row = session.get(FlowRun, flow_run_id)
                assert row is not None
                assert row.flow_name == "TestFlow"
                assert row.run_id == "run-1"
                assert row.meta == {"key": "val"}
                assert row.finished_at is None

        # After exit, finished_at should be set
        engine = init_db(db_path)
        with Session(engine) as session:
            row = session.get(FlowRun, flow_run_id)
            assert row is not None
            assert row.finished_at is not None

    def test_agent_run_persisted(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        with FlowPersistence(db_path, "TestFlow", "run-2") as fp:
            agent = fp.agent_run("planner", "create plan", {"goal": "test"})
            assert agent.agent_name == "planner"
            assert agent.output == {"goal": "test"}

            with Session(fp.engine) as session:
                rows = session.exec(select(AgentRun)).all()
                assert len(rows) == 1
                assert rows[0].flow_run_id == fp.flow_run_id

    def test_artifact_persisted(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        with FlowPersistence(db_path, "TestFlow", "run-3") as fp:
            art = fp.artifact("plan", {"steps": [1, 2, 3]})
            assert art.kind == "plan"
            assert art.content == {"steps": [1, 2, 3]}

            with Session(fp.engine) as session:
                rows = session.exec(select(Artifact)).all()
                assert len(rows) == 1
                assert rows[0].flow_run_id == fp.flow_run_id

    def test_mcp_invocation_persisted(self, tmp_path):
        db_path = str(tmp_path / "test.db")

        with FlowPersistence(db_path, "TestFlow", "run-4") as fp:
            inv = fp.mcp_invocation(
                tool_name="executor",
                invocation_id="inv-123",
                payload={"plan": {}},
                correlation_id="corr-1",
            )
            assert inv.tool_name == "executor"
            assert inv.invocation_id == "inv-123"

            with Session(fp.engine) as session:
                rows = session.exec(select(MCPInvocation)).all()
                assert len(rows) == 1
                assert rows[0].correlation_id == "corr-1"

    def test_flow_run_id_raises_before_enter(self):
        fp = FlowPersistence("unused.db", "F", "r1")
        with pytest.raises(RuntimeError, match="not entered"):
            _ = fp.flow_run_id


# ---------------------------------------------------------------------------
# MCP tool model tests
# ---------------------------------------------------------------------------


class TestToolInputModels:
    """Tests for MCP tool input Pydantic models."""

    def test_echo_input(self):
        inp = EchoInput(message="hello")
        assert inp.model_dump() == {"message": "hello"}

    def test_planner_input_minimal(self):
        inp = PlannerInput(goal="ship it")
        assert inp.context is None

    def test_executor_input_defaults(self):
        inp = ExecutorInput(plan={"steps": []})
        assert inp.dry_run is True

    def test_reviewer_input(self):
        inp = ReviewerInput(result={"ok": True}, criteria=["risk"])
        assert inp.criteria == ["risk"]


class TestRequireInvocationId:
    """Tests for require_invocation_id helper."""

    def test_raises_on_missing(self):
        from qmcp.schemas.mcp import ToolInvokeResponse

        resp = ToolInvokeResponse(result={}, invocation_id=None)
        with pytest.raises(RuntimeError, match="missing invocation_id"):
            require_invocation_id(resp)

    def test_returns_id(self):
        from qmcp.schemas.mcp import ToolInvokeResponse

        resp = ToolInvokeResponse(result={}, invocation_id="inv-abc")
        assert require_invocation_id(resp) == "inv-abc"


class TestMCPToolInvokerWithPersistence:
    """Tests for MCPToolInvoker with FlowPersistence attached."""

    def test_invoke_persists_invocation(self, tmp_path):
        """Invoker logs MCP call to persistence when provided."""
        from unittest.mock import patch, MagicMock
        from qmcp.schemas.mcp import ToolInvokeResponse

        db_path = str(tmp_path / "test.db")

        mock_response = ToolInvokeResponse(
            result={"status": "ok"},
            invocation_id="inv-999",
        )

        with FlowPersistence(db_path, "TestFlow", "run-mcp") as fp:
            invoker = MCPToolInvoker("http://fake:3333", persistence=fp)

            with patch("qmcp.cookbook.mcp_tools.invoke_tool", return_value=mock_response):
                result = invoker.invoke(
                    "executor",
                    ExecutorInput(plan={"steps": []}, dry_run=True),
                    correlation_id="corr-test",
                    artifact_kind="mcp_exec",
                )

            assert result.invocation_id == "inv-999"

            # Verify persistence recorded the invocation
            with Session(fp.engine) as session:
                invocations = session.exec(select(MCPInvocation)).all()
                assert len(invocations) == 1
                assert invocations[0].tool_name == "executor"
                assert invocations[0].correlation_id == "corr-test"

                # Also saved as artifact
                artifacts = session.exec(
                    select(Artifact).where(Artifact.kind == "mcp_exec")
                ).all()
                assert len(artifacts) == 1


class TestMCPToolInvokerWithoutPersistence:
    """Tests for MCPToolInvoker without FlowPersistence."""

    def test_invoke_works_standalone(self):
        """Invoker works without persistence — no DB writes."""
        from unittest.mock import patch
        from qmcp.schemas.mcp import ToolInvokeResponse

        mock_response = ToolInvokeResponse(
            result={"status": "ok"},
            invocation_id="inv-standalone",
        )

        invoker = MCPToolInvoker("http://fake:3333")

        with patch("qmcp.cookbook.mcp_tools.invoke_tool", return_value=mock_response):
            result = invoker.invoke(
                "reviewer",
                ReviewerInput(result={"data": "x"}, criteria=["risk"]),
            )

        assert result.invocation_id == "inv-standalone"

    def test_invoke_raises_on_error(self):
        """Invoker raises RuntimeError when MCP tool returns an error."""
        from unittest.mock import patch
        from qmcp.schemas.mcp import ToolInvokeResponse

        mock_response = ToolInvokeResponse(
            result={},
            error="tool exploded",
            invocation_id="inv-err",
        )

        invoker = MCPToolInvoker("http://fake:3333")

        with patch("qmcp.cookbook.mcp_tools.invoke_tool", return_value=mock_response):
            with pytest.raises(RuntimeError, match="MCP reviewer failed"):
                invoker.invoke(
                    "reviewer",
                    ReviewerInput(result={}, criteria=[]),
                )


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


class TestCookbookPackageImports:
    """Verify public API is accessible from qmcp.cookbook."""

    def test_all_symbols_importable(self):
        from qmcp.cookbook import (
            LocalLLMConfig,
            build_local_agent,
            build_qmcp_agent,
            FlowPersistence,
            init_db,
            MCPToolInvoker,
            check_health,
            invoke_tool,
        )
        # Just verify they're callable / classes
        assert callable(build_local_agent)
        assert callable(build_qmcp_agent)
        assert callable(init_db)
        assert callable(check_health)
        assert callable(invoke_tool)
