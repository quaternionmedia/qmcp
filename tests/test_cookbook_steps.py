"""Tests for qmcp.cookbook.steps (AgentStep and AgentPipeline)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from sqlmodel import Session, select

from qmcp.cookbook.agent_builders import LocalLLMConfig
from qmcp.cookbook.persistence import AgentRun, Artifact, FlowPersistence
from qmcp.cookbook.steps import AgentPipeline, AgentStep, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyOutput(BaseModel):
    text: str


class DummyPlan(BaseModel):
    goal: str
    steps: list[str]


def _mock_config() -> LocalLLMConfig:
    return LocalLLMConfig(model="test", base_url="http://fake:11434/v1")


def _patch_build_local_agent(output_data: BaseModel):
    """Return a patch that makes build_local_agent return a mock agent."""
    mock_agent = MagicMock()
    mock_result = MagicMock()
    mock_result.output = output_data
    mock_agent.run_sync.return_value = mock_result
    return patch("qmcp.cookbook.steps.build_local_agent", return_value=mock_agent)


# ---------------------------------------------------------------------------
# AgentStep tests
# ---------------------------------------------------------------------------


class TestAgentStepRun:
    """Tests for AgentStep.run()."""

    def test_returns_step_result(self):
        output = DummyOutput(text="hello")

        step = AgentStep(
            name="test_step",
            system_prompt="You are a test agent.",
            output_type=DummyOutput,
        )

        with _patch_build_local_agent(output):
            result = step.run(_mock_config(), "Test prompt")

        assert isinstance(result, StepResult)
        assert result.name == "test_step"
        assert result.output == {"text": "hello"}
        assert result.mcp_invocation_id is None

    def test_persists_when_fp_provided(self, tmp_path):
        output = DummyOutput(text="persisted")
        db_path = str(tmp_path / "test.db")

        step = AgentStep(
            name="persisted_step",
            system_prompt="Persist me.",
            output_type=DummyOutput,
        )

        with FlowPersistence(db_path, "TestFlow", "run-1") as fp:
            with _patch_build_local_agent(output):
                result = step.run(_mock_config(), "Test prompt", persistence=fp)

            # Verify persistence
            with Session(fp.engine) as session:
                agent_runs = session.exec(select(AgentRun)).all()
                assert len(agent_runs) == 1
                assert agent_runs[0].agent_name == "persisted_step"

                artifacts = session.exec(select(Artifact)).all()
                assert len(artifacts) == 1
                assert artifacts[0].kind == "persisted_step"

    def test_skips_mcp_when_no_invoker(self):
        output = DummyOutput(text="no mcp")

        step = AgentStep(
            name="mcp_step",
            system_prompt="Test.",
            output_type=DummyOutput,
            mcp_tool="reviewer",
            mcp_criteria=["clarity"],
        )

        with _patch_build_local_agent(output):
            result = step.run(_mock_config(), "Prompt")

        assert result.mcp_invocation_id is None

    def test_invokes_mcp_reviewer(self):
        from qmcp.schemas.mcp import ToolInvokeResponse

        output = DummyOutput(text="reviewed")
        mock_response = ToolInvokeResponse(
            result={"status": "ok"},
            invocation_id="inv-review-1",
        )

        mock_invoker = MagicMock()
        mock_invoker.invoke.return_value = mock_response

        step = AgentStep(
            name="reviewed_step",
            system_prompt="Review me.",
            output_type=DummyOutput,
            mcp_tool="reviewer",
            mcp_criteria=["risk", "clarity"],
        )

        with _patch_build_local_agent(output):
            result = step.run(
                _mock_config(), "Prompt",
                invoker=mock_invoker,
                correlation_id="corr-1",
            )

        assert result.mcp_invocation_id == "inv-review-1"
        mock_invoker.invoke.assert_called_once()
        call_args = mock_invoker.invoke.call_args
        assert call_args.args[0] == "reviewer"

    def test_invokes_mcp_executor(self):
        from qmcp.schemas.mcp import ToolInvokeResponse

        output = DummyPlan(goal="test", steps=["a", "b"])
        mock_response = ToolInvokeResponse(
            result={"status": "ok"},
            invocation_id="inv-exec-1",
        )

        mock_invoker = MagicMock()
        mock_invoker.invoke.return_value = mock_response

        step = AgentStep(
            name="exec_step",
            system_prompt="Execute.",
            output_type=DummyPlan,
            mcp_tool="executor",
        )

        with _patch_build_local_agent(output):
            result = step.run(_mock_config(), "Prompt", invoker=mock_invoker)

        assert result.mcp_invocation_id == "inv-exec-1"
        call_args = mock_invoker.invoke.call_args
        assert call_args.args[0] == "executor"


# ---------------------------------------------------------------------------
# AgentPipeline tests
# ---------------------------------------------------------------------------


class TestAgentPipeline:
    """Tests for AgentPipeline.run()."""

    def test_returns_all_step_results(self):
        step1_output = DummyOutput(text="step1")
        step2_output = DummyOutput(text="step2")

        pipeline = AgentPipeline(steps=[
            AgentStep("first", "First agent.", DummyOutput),
            AgentStep("second", "Second agent.", DummyOutput),
        ])

        call_count = 0

        def mock_build(config, system_prompt, output_type, retries=3):
            nonlocal call_count
            agent = MagicMock()
            result = MagicMock()
            result.output = step1_output if call_count == 0 else step2_output
            agent.run_sync.return_value = result
            call_count += 1
            return agent

        with patch("qmcp.cookbook.steps.build_local_agent", side_effect=mock_build):
            results = pipeline.run(_mock_config(), "Initial prompt")

        assert "first" in results
        assert "second" in results
        assert results["first"].output == {"text": "step1"}
        assert results["second"].output == {"text": "step2"}

    def test_threads_context_to_later_steps(self):
        """Later steps receive accumulated outputs in their prompt."""
        prompts_received = []

        def mock_build(config, system_prompt, output_type, retries=3):
            agent = MagicMock()
            result = MagicMock()
            result.output = DummyOutput(text="ok")

            def capture_prompt(prompt):
                prompts_received.append(prompt)
                return result

            agent.run_sync.side_effect = capture_prompt
            return agent

        pipeline = AgentPipeline(steps=[
            AgentStep("step_a", "A.", DummyOutput),
            AgentStep("step_b", "B.", DummyOutput),
        ])

        with patch("qmcp.cookbook.steps.build_local_agent", side_effect=mock_build):
            pipeline.run(_mock_config(), "Start here")

        # First step gets just the initial prompt
        assert "Start here" in prompts_received[0]
        assert "Previous outputs" not in prompts_received[0]

        # Second step gets accumulated context
        assert "Start here" in prompts_received[1]
        assert "Previous outputs" in prompts_received[1]
        assert "step_a" in prompts_received[1]

    def test_pipeline_with_persistence(self, tmp_path):
        db_path = str(tmp_path / "pipeline.db")

        pipeline = AgentPipeline(steps=[
            AgentStep("alpha", "Alpha.", DummyOutput),
            AgentStep("beta", "Beta.", DummyOutput),
        ])

        def mock_build(config, system_prompt, output_type, retries=3):
            agent = MagicMock()
            result = MagicMock()
            result.output = DummyOutput(text="result")
            agent.run_sync.return_value = result
            return agent

        with FlowPersistence(db_path, "PipelineTest", "run-pipe") as fp:
            with patch("qmcp.cookbook.steps.build_local_agent", side_effect=mock_build):
                results = pipeline.run(_mock_config(), "Go", persistence=fp)

            with Session(fp.engine) as session:
                agent_runs = session.exec(select(AgentRun)).all()
                assert len(agent_runs) == 2
                names = {r.agent_name for r in agent_runs}
                assert names == {"alpha", "beta"}

                artifacts = session.exec(select(Artifact)).all()
                assert len(artifacts) == 2


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


class TestStepsPackageImports:
    def test_importable_from_cookbook(self):
        from qmcp.cookbook import AgentStep, AgentPipeline, StepResult
        assert callable(AgentStep)
        assert callable(AgentPipeline)
        assert callable(StepResult)
