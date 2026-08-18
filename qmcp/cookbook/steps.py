"""Composable agent step and pipeline building blocks.

Extracts the repeated "build agent → run → persist → optionally MCP review"
pattern into reusable primitives that can be composed into compound recipes.

Usage::

    from qmcp.cookbook.steps import AgentStep, AgentPipeline

    steps = AgentPipeline(steps=[
        AgentStep("summarizer", "Summarize changes.", ChangeSummary),
        AgentStep("reviewer", "Review summary.", ReviewOutput,
                  mcp_tool="reviewer", mcp_criteria=["clarity"]),
    ])
    results = steps.run(config, "Summarize: refactored auth module",
                        persistence=fp, invoker=invoker)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from qmcp.cookbook.agent_builders import LocalLLMConfig, build_local_agent
from qmcp.cookbook.mcp_tools import MCPToolInvoker, ReviewerInput, ExecutorInput

T = TypeVar("T", bound=BaseModel)


@dataclass
class StepResult:
    """Output from a single AgentStep execution."""

    name: str
    output: dict[str, Any]
    mcp_invocation_id: str | None = None


@dataclass
class AgentStep:
    """Single composable agent step in a pipeline.

    Encapsulates the full cycle: build agent, run prompt, persist output,
    and optionally invoke an MCP tool (reviewer or executor) for audit.

    Args:
        name: Step identifier used for persistence and result keys.
        system_prompt: System prompt for the agent.
        output_type: Pydantic model class for structured output.
        mcp_tool: Optional MCP tool to invoke after the agent run.
            Supported: ``"reviewer"`` or ``"executor"``.
        mcp_criteria: Criteria passed to the reviewer tool (ignored for executor).
        retries: Agent retries on validation failure.
    """

    name: str
    system_prompt: str
    output_type: type[BaseModel]
    mcp_tool: str | None = None
    mcp_criteria: list[str] = field(default_factory=list)
    retries: int = 3

    def run(
        self,
        config: LocalLLMConfig,
        prompt: str,
        persistence: Any | None = None,
        invoker: MCPToolInvoker | None = None,
        correlation_id: str | None = None,
    ) -> StepResult:
        """Execute the step: build agent, run, persist, optionally MCP review.

        Args:
            config: LLM configuration.
            prompt: User prompt for this step.
            persistence: Optional FlowPersistence for audit logging.
            invoker: Optional MCPToolInvoker for MCP tool calls.
            correlation_id: Optional correlation ID for MCP tracing.

        Returns:
            A StepResult with the agent output and optional MCP invocation ID.
        """
        agent = build_local_agent(
            config=config,
            system_prompt=self.system_prompt,
            output_type=self.output_type,
            retries=self.retries,
        )
        result = agent.run_sync(prompt)
        output = result.output.model_dump()

        if persistence is not None:
            persistence.agent_run(self.name, prompt[:200], output)
            persistence.artifact(self.name, output)

        mcp_invocation_id = None
        if self.mcp_tool and invoker is not None:
            mcp_invocation_id = self._invoke_mcp(
                invoker, output, correlation_id,
            )

        return StepResult(
            name=self.name,
            output=output,
            mcp_invocation_id=mcp_invocation_id,
        )

    def _invoke_mcp(
        self,
        invoker: MCPToolInvoker,
        output: dict[str, Any],
        correlation_id: str | None,
    ) -> str | None:
        """Invoke the configured MCP tool and return the invocation ID."""
        if self.mcp_tool == "reviewer":
            payload = ReviewerInput(
                result=output,
                criteria=self.mcp_criteria,
            )
        elif self.mcp_tool == "executor":
            payload = ExecutorInput(
                plan=output,
                dry_run=True,
            )
        else:
            return None

        mcp_result = invoker.invoke(
            self.mcp_tool,
            payload,
            correlation_id=correlation_id,
            artifact_kind=f"mcp_{self.name}_{self.mcp_tool}",
        )
        return mcp_result.invocation_id


@dataclass
class AgentPipeline:
    """Chain of AgentSteps with automatic data threading.

    Runs steps sequentially, passing accumulated outputs from previous
    steps into each subsequent prompt as JSON context.

    Usage::

        pipeline = AgentPipeline(steps=[
            AgentStep("plan", "Create a plan.", DevPlan),
            AgentStep("review", "Review the plan.", PlanReview,
                      mcp_tool="reviewer", mcp_criteria=["risk"]),
            AgentStep("refine", "Refine the plan.", RefinedPlan),
        ])
        results = pipeline.run(config, "Goal: ship QC gauntlet")
    """

    steps: list[AgentStep]

    def run(
        self,
        config: LocalLLMConfig,
        initial_prompt: str,
        persistence: Any | None = None,
        invoker: MCPToolInvoker | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, StepResult]:
        """Run all steps, threading each output into the next prompt.

        Args:
            config: LLM configuration shared by all steps.
            initial_prompt: Starting prompt for the first step.
            persistence: Optional FlowPersistence for audit logging.
            invoker: Optional MCPToolInvoker for MCP tool calls.
            correlation_id: Optional correlation ID for MCP tracing.

        Returns:
            Dict mapping step name → StepResult for each completed step.
        """
        results: dict[str, StepResult] = {}
        accumulated: dict[str, dict] = {}

        for step in self.steps:
            if accumulated:
                prompt = "\n".join([
                    initial_prompt,
                    "",
                    "Previous outputs:",
                    json.dumps(accumulated, indent=2),
                ])
            else:
                prompt = initial_prompt

            step_result = step.run(
                config=config,
                prompt=prompt,
                persistence=persistence,
                invoker=invoker,
                correlation_id=correlation_id,
            )
            results[step.name] = step_result
            accumulated[step.name] = step_result.output

        return results


__all__ = [
    "AgentStep",
    "AgentPipeline",
    "StepResult",
]
