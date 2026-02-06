"""Local agent chain flow for planning, review, and refinement.

Usage:
    # Install flow dependencies
    uv sync --extra flows

    # Start MCP server if using --use-mcp True
    uv run qmcp serve

    # Start a local OpenAI-compatible LLM (e.g., Ollama or LM Studio)
    uv run python examples/flows/local_agent_chain.py run \
        --use-mcp True \
        --goal "Ship a local QC gauntlet" \
        --context "Focus on auditability and fast feedback" \
        --llm-base-url "http://localhost:11434/v1" \
        --llm-model "llama3.1"
"""

from __future__ import annotations

import json
import os

from local_dev_db import (
    init_db,
    mark_flow_finished,
    save_agent_run,
    save_artifact,
    save_flow_run,
    save_mcp_invocation,
)
from local_llm import LocalLLMConfig, build_agent
from local_mcp import (
    ExecutorInput,
    ReviewerInput,
    check_health,
    invoke_tool,
    require_invocation_id,
)
from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field
from sqlmodel import Session


class PlanStep(BaseModel):
    """Single step in a dev plan."""

    step: int = Field(..., ge=1)
    action: str
    outcome: str


class DevPlan(BaseModel):
    """Plan produced by the planning agent."""

    goal: str
    steps: list[PlanStep]
    assumptions: list[str] = Field(default_factory=list)


class PlanReview(BaseModel):
    """Plan review output."""

    risks: list[str]
    missing_tests: list[str]
    recommendation: str


class RefinedPlan(BaseModel):
    """Refined plan after review."""

    goal: str
    steps: list[PlanStep]
    changes: list[str]


class LocalAgentChainFlow(FlowSpec):
    """Chains local LLM agents for dev planning, review, and refinement."""

    goal = Parameter("goal", help="Planning goal", required=True)
    context = Parameter("context", help="Optional planning context", default="")
    db_path = Parameter(
        "db-path",
        help="SQLite path for artifacts",
        default=os.getenv("FLOW_DB_PATH", ".qmcp_devflows.db"),
    )
    mcp_url = Parameter(
        "mcp-url",
        help="MCP server URL for tool calls",
        default=os.getenv("MCP_URL", "http://localhost:3333"),
    )
    use_mcp = Parameter(
        "use-mcp",
        help="Invoke MCP tools for audit and comparison",
        type=bool,
        default=True,
    )
    llm_base_url = Parameter(
        "llm-base-url",
        help="OpenAI-compatible base URL (local LLM)",
        default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    llm_model = Parameter(
        "llm-model",
        help="Local model name",
        default=os.getenv("LLM_MODEL", "llama3.1"),
    )
    llm_api_key = Parameter(
        "llm-api-key",
        help="API key if required",
        default=os.getenv("LLM_API_KEY", "local"),
    )

    @step
    def start(self):
        """Initialize storage for this run."""
        self.run_id = current.run_id
        engine = init_db(self.db_path)
        with Session(engine) as session:
            flow_run = save_flow_run(
                session,
                flow_name=self.__class__.__name__,
                run_id=self.run_id,
                meta={
                    "goal": self.goal,
                    "context": self.context,
                    "llm_model": self.llm_model,
                    "llm_base_url": self.llm_base_url,
                    "use_mcp": self.use_mcp,
                    "mcp_url": self.mcp_url,
                },
            )
            self.flow_run_id = flow_run.id
            if self.use_mcp:
                health = check_health(self.mcp_url)
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_health",
                    content=health,
                )

        self.mcp_invocations = []

        self.next(self.plan)

    @step
    def plan(self):
        """Generate a plan with a local planning agent."""
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        planner = build_agent(
            config=config,
            system_prompt="You are a planning agent for local dev workflows.",
            result_type=DevPlan,
        )

        prompt = "\n".join(
            [
                f"Goal: {self.goal}",
                f"Context: {self.context or 'none'}",
                "Create a concise plan with actionable steps and expected outcomes.",
            ]
        )
        result = planner.run_sync(prompt)
        self.plan_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="planner",
                input_summary=prompt,
                output=self.plan_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="plan",
                content=self.plan_output,
            )

        if self.use_mcp:
            executor_input = ExecutorInput(plan=self.plan_output, dry_run=True)
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="executor",
                payload=executor_input,
                correlation_id=f"flow-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP executor failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_executor = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "executor", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="executor",
                    invocation_id=invocation_id,
                    payload=executor_input.model_dump(),
                    correlation_id=f"flow-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_executor",
                    content=self.mcp_executor,
                )

        self.next(self.review)

    @step
    def review(self):
        """Review the plan and capture gaps."""
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        reviewer = build_agent(
            config=config,
            system_prompt="You are a critical reviewer for dev plans.",
            result_type=PlanReview,
        )

        prompt = "\n".join(
            [
                "Review the plan and identify risks, missing tests, and a recommendation.",
                f"Plan JSON:\n{json.dumps(self.plan_output, indent=2)}",
            ]
        )
        result = reviewer.run_sync(prompt)
        self.review_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="reviewer",
                input_summary="Review plan and identify gaps.",
                output=self.review_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="plan_review",
                content=self.review_output,
            )

        if self.use_mcp:
            reviewer_input = ReviewerInput(
                result=self.plan_output,
                criteria=["completeness", "correctness", "risk"],
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="reviewer",
                payload=reviewer_input,
                correlation_id=f"flow-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP reviewer failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_review = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "reviewer", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="reviewer",
                    invocation_id=invocation_id,
                    payload=reviewer_input.model_dump(),
                    correlation_id=f"flow-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_reviewer",
                    content=self.mcp_review,
                )

        self.next(self.refine)

    @step
    def refine(self):
        """Refine the plan using review feedback."""
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        refiner = build_agent(
            config=config,
            system_prompt="You refine dev plans based on review feedback.",
            result_type=RefinedPlan,
        )

        prompt = "\n".join(
            [
                "Refine the plan based on the review feedback.",
                f"Plan JSON:\n{json.dumps(self.plan_output, indent=2)}",
                f"Review JSON:\n{json.dumps(self.review_output, indent=2)}",
            ]
        )
        result = refiner.run_sync(prompt)
        self.refined_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="refiner",
                input_summary="Refine plan using review feedback.",
                output=self.refined_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="refined_plan",
                content=self.refined_output,
            )

        self.next(self.end)

    @step
    def end(self):
        """Finalize the flow and print a summary."""
        engine = init_db(self.db_path)
        with Session(engine) as session:
            mark_flow_finished(session, self.flow_run_id)

        print("Plan steps:", len(self.plan_output["steps"]))
        print("Review risks:", len(self.review_output["risks"]))
        print("Refined plan steps:", len(self.refined_output["steps"]))
        if self.mcp_invocations:
            print("MCP invocations:")
            for entry in self.mcp_invocations:
                print(f"  {entry['tool']}: {entry['invocation_id']}")


if __name__ == "__main__":
    LocalAgentChainFlow()
