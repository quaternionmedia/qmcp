"""The change-impact pipeline: its output models and its steps, and nothing else.

WHY THIS IS NOT IN THE FLOW FILE. It was, and that made four pure step
definitions unreachable on any machine without Metaflow. `examples/flows/change_impact.py`
imports `metaflow` at module level, and `import metaflow` fails on Windows in
`metaflow/sidecar/sidecar_subprocess.py` on `import fcntl`, which is POSIX-only
-- so importing the pipeline to read, test, plan against or map to a dossier
delta required a runtime that has nothing to do with any of those things.

The steps are `AgentStep`s: name, prompt, output type. They describe work. The
flow is one way to *execute* that work, and coupling the description to the
executor is what made the description platform-specific.

WHAT DEPENDS ON WHAT, NOW. This module imports pydantic and `qmcp.cookbook.steps`.
The flow imports this. Nothing here imports Metaflow, and a test asserts that,
because the coupling is the kind that returns quietly the next time somebody
adds a decorator.
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
