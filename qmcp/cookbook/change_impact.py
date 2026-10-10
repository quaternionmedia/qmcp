"""The change-impact pipeline: its output models and its steps, and nothing else.

The steps are `AgentStep`s -- name, prompt, output type -- and describe work;
the flow in `examples/flows/change_impact.py` is one way to execute it. They
live apart from the flow because the flow imports `metaflow` at module level,
and `import metaflow` fails on Windows (`metaflow/sidecar/sidecar_subprocess.py`
imports the POSIX-only `fcntl`).

This module imports pydantic and `qmcp.cookbook.steps`; the flow imports this.
Nothing here imports Metaflow, and a test asserts it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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
