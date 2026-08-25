"""Local agent chain flow for planning, review, and refinement.

Uses ``qmcp.cookbook`` modules for agent building, persistence, and MCP
tool invocation.

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

from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field

from qmcp.cookbook import FlowPersistence, LocalLLMConfig, MCPToolInvoker, build_local_agent
from qmcp.cookbook.mcp_tools import ExecutorInput, ReviewerInput


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
        default=os.getenv("MCP_URL", "http://localhost:3141"),
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

    def _llm_config(self) -> LocalLLMConfig:
        return LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )

    @step
    def start(self):
        """Initialize storage for this run."""
        self.run_id = current.run_id
        self.fp = FlowPersistence(
            db_path=self.db_path,
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
        ).__enter__()

        self.mcp_invocations: list[dict] = []

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            invoker.health()

        self.next(self.plan)

    @step
    def plan(self):
        """Generate a plan with a local planning agent."""
        planner = build_local_agent(
            config=self._llm_config(),
            system_prompt="You are a planning agent for local dev workflows.",
            output_type=DevPlan,
        )

        prompt = "\n".join([
            f"Goal: {self.goal}",
            f"Context: {self.context or 'none'}",
            "Create a concise plan with actionable steps and expected outcomes.",
        ])
        result = planner.run_sync(prompt)
        self.plan_output = result.output.model_dump()

        self.fp.agent_run("planner", prompt, self.plan_output)
        self.fp.artifact("plan", self.plan_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            executor_input = ExecutorInput(plan=self.plan_output, dry_run=True)
            mcp_result = invoker.invoke(
                "executor", executor_input,
                correlation_id=f"flow-{self.run_id}",
                artifact_kind="mcp_executor",
            )
            self.mcp_executor = mcp_result.model_dump()
            self.mcp_invocations.append({
                "tool": "executor",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.review)

    @step
    def review(self):
        """Review the plan and capture gaps."""
        reviewer = build_local_agent(
            config=self._llm_config(),
            system_prompt="You are a critical reviewer for dev plans.",
            output_type=PlanReview,
        )

        prompt = "\n".join([
            "Review the plan and identify risks, missing tests, and a recommendation.",
            f"Plan JSON:\n{json.dumps(self.plan_output, indent=2)}",
        ])
        result = reviewer.run_sync(prompt)
        self.review_output = result.output.model_dump()

        self.fp.agent_run("reviewer", "Review plan and identify gaps.", self.review_output)
        self.fp.artifact("plan_review", self.review_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            reviewer_input = ReviewerInput(
                result=self.plan_output,
                criteria=["completeness", "correctness", "risk"],
            )
            mcp_result = invoker.invoke(
                "reviewer", reviewer_input,
                correlation_id=f"flow-{self.run_id}",
                artifact_kind="mcp_reviewer",
            )
            self.mcp_review = mcp_result.model_dump()
            self.mcp_invocations.append({
                "tool": "reviewer",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.refine)

    @step
    def refine(self):
        """Refine the plan using review feedback."""
        refiner = build_local_agent(
            config=self._llm_config(),
            system_prompt="You refine dev plans based on review feedback.",
            output_type=RefinedPlan,
        )

        prompt = "\n".join([
            "Refine the plan based on the review feedback.",
            f"Plan JSON:\n{json.dumps(self.plan_output, indent=2)}",
            f"Review JSON:\n{json.dumps(self.review_output, indent=2)}",
        ])
        result = refiner.run_sync(prompt)
        self.refined_output = result.output.model_dump()

        self.fp.agent_run("refiner", "Refine plan using review feedback.", self.refined_output)
        self.fp.artifact("refined_plan", self.refined_output)

        self.next(self.end)

    @step
    def end(self):
        """Finalize the flow and print a summary."""
        self.fp.__exit__(None, None, None)

        print("Plan steps:", len(self.plan_output["steps"]))
        print("Review risks:", len(self.review_output["risks"]))
        print("Refined plan steps:", len(self.refined_output["steps"]))
        if self.mcp_invocations:
            print("MCP invocations:")
            for entry in self.mcp_invocations:
                print(f"  {entry['tool']}: {entry['invocation_id']}")


if __name__ == "__main__":
    LocalAgentChainFlow()
