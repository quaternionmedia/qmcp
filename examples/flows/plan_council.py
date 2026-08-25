"""Compound recipe: plan + mini-council deliberation + refinement.

Generates an initial plan, evaluates it through a 3-member mini-council
(strategist, sanity check, efficist), then refines the plan based on
their collective feedback.

Usage:
    uv sync --extra flows
    uv run python examples/flows/plan_council.py run \
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
from qmcp.cookbook.steps import AgentStep


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    step: int = Field(..., ge=1)
    action: str
    outcome: str


class DevPlan(BaseModel):
    goal: str
    steps: list[PlanStep]
    assumptions: list[str] = Field(default_factory=list)


class CouncilEvaluation(BaseModel):
    member: str
    verdict: str
    concerns: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class CouncilSynthesis(BaseModel):
    consensus: str
    evaluations: list[CouncilEvaluation]
    action_items: list[str] = Field(default_factory=list)


class RefinedPlan(BaseModel):
    goal: str
    steps: list[PlanStep]
    changes: list[str]


# ---------------------------------------------------------------------------
# Council member prompts
# ---------------------------------------------------------------------------

COUNCIL_MEMBERS = {
    "strategist": "You are a pragmatic strategist. Focus on feasibility, resource needs, and practical implementation order.",
    "sanity_check": "You are a sanity checker. Find gaps, contradictions, and unrealistic assumptions in plans.",
    "efficist": "You are a brutal efficist. Cut unnecessary complexity and demand the simplest path to the goal.",
}


# ---------------------------------------------------------------------------
# Metaflow flow
# ---------------------------------------------------------------------------


class PlanCouncilFlow(FlowSpec):
    """Compound pipeline: plan → mini-council evaluation → refined plan."""

    goal = Parameter("goal", help="Planning goal", required=True)
    context = Parameter("context", help="Optional planning context", default="")
    db_path = Parameter(
        "db-path", help="SQLite path for artifacts",
        default=os.getenv("FLOW_DB_PATH", ".qmcp_devflows.db"),
    )
    mcp_url = Parameter(
        "mcp-url", help="MCP server URL",
        default=os.getenv("MCP_URL", "http://localhost:3141"),
    )
    use_mcp = Parameter(
        "use-mcp", help="Invoke MCP tools", type=bool, default=True,
    )
    llm_base_url = Parameter(
        "llm-base-url", help="OpenAI-compatible base URL",
        default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    llm_model = Parameter(
        "llm-model", help="Local model name",
        default=os.getenv("LLM_MODEL", "llama3.1"),
    )
    llm_api_key = Parameter(
        "llm-api-key", help="API key if required",
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
        self.run_id = current.run_id
        self.fp = FlowPersistence(
            db_path=self.db_path,
            flow_name=self.__class__.__name__,
            run_id=self.run_id,
            meta={
                "goal": self.goal,
                "context": self.context,
                "llm_model": self.llm_model,
                "use_mcp": self.use_mcp,
            },
        ).__enter__()

        self.invoker = None
        if self.use_mcp:
            self.invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            self.invoker.health()

        self.next(self.plan)

    @step
    def plan(self):
        """Generate initial plan using AgentStep."""
        plan_step = AgentStep(
            name="planner",
            system_prompt="You are a planning agent for local dev workflows.",
            output_type=DevPlan,
            mcp_tool="executor" if self.use_mcp else None,
        )

        prompt = "\n".join([
            f"Goal: {self.goal}",
            f"Context: {self.context or 'none'}",
            "Create a concise plan with actionable steps and expected outcomes.",
        ])

        result = plan_step.run(
            config=self._llm_config(),
            prompt=prompt,
            persistence=self.fp,
            invoker=self.invoker,
            correlation_id=f"plan-council-{self.run_id}",
        )
        self.plan_output = result.output
        self.next(self.council)

    @step
    def council(self):
        """Run mini-council: 3 members evaluate the plan."""
        evaluations = []
        plan_json = json.dumps(self.plan_output, indent=2)

        for member_name, system_prompt in COUNCIL_MEMBERS.items():
            agent = build_local_agent(
                config=self._llm_config(),
                system_prompt=system_prompt,
                output_type=CouncilEvaluation,
            )

            prompt = "\n".join([
                f"Evaluate this plan as the {member_name}.",
                f"Goal: {self.goal}",
                f"Plan JSON:\n{plan_json}",
                "Provide your verdict, concerns, and suggestions.",
            ])
            result = agent.run_sync(prompt)
            evaluation = result.output.model_dump()

            self.fp.agent_run(f"council_{member_name}", prompt[:200], evaluation)
            evaluations.append(evaluation)

        # Synthesize council feedback
        synthesizer = build_local_agent(
            config=self._llm_config(),
            system_prompt="You synthesize council member evaluations into a consensus.",
            output_type=CouncilSynthesis,
        )

        synth_prompt = "\n".join([
            "Synthesize these council evaluations into a consensus with action items.",
            f"Plan: {plan_json}",
            f"Evaluations:\n{json.dumps(evaluations, indent=2)}",
        ])
        synth_result = synthesizer.run_sync(synth_prompt)
        self.council_output = synth_result.output.model_dump()

        self.fp.agent_run("council_synthesis", "Synthesize evaluations.", self.council_output)
        self.fp.artifact("council_synthesis", self.council_output)

        self.next(self.refine)

    @step
    def refine(self):
        """Refine the plan using council feedback."""
        refine_step = AgentStep(
            name="refiner",
            system_prompt="You refine dev plans based on council feedback.",
            output_type=RefinedPlan,
        )

        prompt = "\n".join([
            "Refine the plan based on the council's feedback.",
            f"Original plan:\n{json.dumps(self.plan_output, indent=2)}",
            f"Council synthesis:\n{json.dumps(self.council_output, indent=2)}",
        ])

        result = refine_step.run(
            config=self._llm_config(),
            prompt=prompt,
            persistence=self.fp,
            invoker=self.invoker,
            correlation_id=f"plan-council-{self.run_id}",
        )
        self.refined_output = result.output
        self.next(self.end)

    @step
    def end(self):
        self.fp.__exit__(None, None, None)

        print(f"Plan steps: {len(self.plan_output.get('steps', []))}")
        print(f"Council evaluations: {len(self.council_output.get('evaluations', []))}")
        print(f"Consensus: {self.council_output.get('consensus', 'N/A')}")
        print(f"Refined steps: {len(self.refined_output.get('steps', []))}")
        print(f"Changes applied: {len(self.refined_output.get('changes', []))}")


if __name__ == "__main__":
    PlanCouncilFlow()
