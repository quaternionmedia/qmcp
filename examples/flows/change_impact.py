"""Compound recipe: change impact analysis pipeline.

Runs a 4-step analysis: summarize → risk assessment → test plan → migration
guide. Demonstrates the simplest ``AgentPipeline`` usage with no MCP tools.

Usage:
    uv sync --extra flows
    uv run python examples/flows/change_impact.py run \
        --change-summary "Migrate auth from session cookies to JWT" \
        --codebase-context "Django monolith, 120 endpoints, 3 mobile clients" \
        --llm-base-url "http://localhost:11434/v1" \
        --llm-model "llama3.1"
"""

from __future__ import annotations

import os

from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field

from qmcp.cookbook import FlowPersistence, LocalLLMConfig
from qmcp.cookbook.steps import AgentPipeline, AgentStep


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ChangeSummary(BaseModel):
    themes: list[str]
    impacted_areas: list[str]
    key_changes: list[str]
    risks: list[str] = Field(default_factory=list)


class RiskItem(BaseModel):
    area: str
    risk: str
    severity: str = Field(description="low, medium, high, or critical")
    mitigation: str


class RiskAssessment(BaseModel):
    overall_risk: str
    items: list[RiskItem]


class TestCase(BaseModel):
    area: str
    test_name: str
    description: str
    priority: str = Field(description="p0, p1, p2")


class TestPlan(BaseModel):
    strategy: str
    cases: list[TestCase]
    estimated_effort: str = ""


class MigrationStep(BaseModel):
    order: int = Field(..., ge=1)
    action: str
    rollback: str


class MigrationGuide(BaseModel):
    required: bool
    summary: str
    steps: list[MigrationStep] = Field(default_factory=list)
    breaking_changes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

CHANGE_IMPACT_PIPELINE = AgentPipeline(steps=[
    AgentStep(
        name="summarizer",
        system_prompt="You summarize engineering changes, identifying themes, impacted areas, and risks.",
        output_type=ChangeSummary,
    ),
    AgentStep(
        name="risk_assessor",
        system_prompt="You assess risks from change summaries, rating severity and suggesting mitigations.",
        output_type=RiskAssessment,
    ),
    AgentStep(
        name="test_planner",
        system_prompt="You generate test plans targeting identified risks and impacted areas.",
        output_type=TestPlan,
    ),
    AgentStep(
        name="migration_guide",
        system_prompt="You produce migration guides for breaking changes. If no breaking changes exist, set required=false with a brief summary.",
        output_type=MigrationGuide,
    ),
])


# ---------------------------------------------------------------------------
# Metaflow flow
# ---------------------------------------------------------------------------


class ChangeImpactFlow(FlowSpec):
    """Compound pipeline: summarize → risk → test plan → migration guide."""

    change_summary = Parameter(
        "change-summary", help="Summary of the changes", required=True,
    )
    codebase_context = Parameter(
        "codebase-context", help="Context about the codebase", default="",
    )
    db_path = Parameter(
        "db-path", help="SQLite path for artifacts",
        default=os.getenv("FLOW_DB_PATH", ".qmcp_devflows.db"),
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
                "change_summary": self.change_summary,
                "codebase_context": self.codebase_context,
                "llm_model": self.llm_model,
            },
        ).__enter__()

        self.next(self.analyze)

    @step
    def analyze(self):
        initial_prompt = "\n".join([
            f"Change summary: {self.change_summary}",
            f"Codebase context: {self.codebase_context or 'not provided'}",
            "Analyze this change: summarize, assess risks, plan tests, produce migration guide.",
        ])

        self.pipeline_results = {
            name: sr.output
            for name, sr in CHANGE_IMPACT_PIPELINE.run(
                config=self._llm_config(),
                initial_prompt=initial_prompt,
                persistence=self.fp,
            ).items()
        }

        self.next(self.end)

    @step
    def end(self):
        self.fp.__exit__(None, None, None)

        summary = self.pipeline_results.get("summarizer", {})
        risks = self.pipeline_results.get("risk_assessor", {})
        tests = self.pipeline_results.get("test_planner", {})
        migration = self.pipeline_results.get("migration_guide", {})

        print(f"Themes: {len(summary.get('themes', []))}")
        print(f"Risk items: {len(risks.get('items', []))}")
        print(f"Overall risk: {risks.get('overall_risk', 'N/A')}")
        print(f"Test cases: {len(tests.get('cases', []))}")
        print(f"Migration required: {migration.get('required', False)}")
        print(f"Breaking changes: {len(migration.get('breaking_changes', []))}")


if __name__ == "__main__":
    ChangeImpactFlow()
