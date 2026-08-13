"""Compound recipe: QC gate + release notes pipeline.

Chains a QC gauntlet (checklist + gate) with release notes and doc update
suggestions in a single Metaflow flow using ``AgentPipeline``.

Usage:
    uv sync --extra flows
    uv run python examples/flows/qc_release.py run \
        --change-summary "Refactor metrics registry and add QC docs" \
        --target-area "metrics, docs" \
        --audience "internal" \
        --llm-base-url "http://localhost:11434/v1" \
        --llm-model "llama3.1"
"""

from __future__ import annotations

import json
import os

from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field

from qmcp.cookbook import FlowPersistence, LocalLLMConfig, MCPToolInvoker
from qmcp.cookbook.steps import AgentPipeline, AgentStep


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class QCItem(BaseModel):
    area: str
    check: str
    command: str | None = None
    expected: str | None = None


class QCChecklist(BaseModel):
    summary: str
    items: list[QCItem]
    focus_areas: list[str] = Field(default_factory=list)


class QCGate(BaseModel):
    must_pass: list[str]
    risk_flags: list[str]
    stop_ship_conditions: list[str]


class ReleaseNotes(BaseModel):
    title: str
    highlights: list[str]
    breaking_changes: list[str] = Field(default_factory=list)
    migration_notes: list[str] = Field(default_factory=list)


class DocUpdate(BaseModel):
    path: str
    reason: str
    suggested_change: str


class DocUpdatePlan(BaseModel):
    updates: list[DocUpdate]


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

QC_RELEASE_PIPELINE = AgentPipeline(steps=[
    AgentStep(
        name="qc_checklist",
        system_prompt="You create QA/QC checklists for local dev cycles.",
        output_type=QCChecklist,
        mcp_tool="reviewer",
        mcp_criteria=["coverage", "risk", "runtime"],
    ),
    AgentStep(
        name="qc_gate",
        system_prompt="You define stop-ship criteria for QC.",
        output_type=QCGate,
        mcp_tool="reviewer",
        mcp_criteria=["stop_ship", "risk_flags"],
    ),
    AgentStep(
        name="release_notes",
        system_prompt="You draft concise release notes.",
        output_type=ReleaseNotes,
        mcp_tool="reviewer",
        mcp_criteria=["clarity", "completeness"],
    ),
    AgentStep(
        name="doc_updates",
        system_prompt="You propose documentation updates for a release.",
        output_type=DocUpdatePlan,
        mcp_tool="reviewer",
        mcp_criteria=["coverage", "priority"],
    ),
])


# ---------------------------------------------------------------------------
# Metaflow flow
# ---------------------------------------------------------------------------


class QCReleaseFlow(FlowSpec):
    """Compound pipeline: QC gauntlet → release notes → doc updates."""

    change_summary = Parameter(
        "change-summary", help="Summary of changes", required=True,
    )
    target_area = Parameter(
        "target-area", help="Area impacted (comma-separated)", default="core",
    )
    audience = Parameter("audience", help="Audience for release notes", default="internal")
    db_path = Parameter(
        "db-path", help="SQLite path for artifacts",
        default=os.getenv("FLOW_DB_PATH", ".qmcp_devflows.db"),
    )
    mcp_url = Parameter(
        "mcp-url", help="MCP server URL",
        default=os.getenv("MCP_URL", "http://localhost:3333"),
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
                "change_summary": self.change_summary,
                "target_area": self.target_area,
                "audience": self.audience,
                "llm_model": self.llm_model,
                "use_mcp": self.use_mcp,
            },
        ).__enter__()

        self.invoker = None
        if self.use_mcp:
            self.invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            self.invoker.health()

        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        initial_prompt = "\n".join([
            f"Change summary: {self.change_summary}",
            f"Target area: {self.target_area}",
            f"Audience: {self.audience}",
            "Process: QC checklist → gate criteria → release notes → doc updates.",
        ])

        self.pipeline_results = {
            name: sr.output
            for name, sr in QC_RELEASE_PIPELINE.run(
                config=self._llm_config(),
                initial_prompt=initial_prompt,
                persistence=self.fp,
                invoker=self.invoker,
                correlation_id=f"qc-release-{self.run_id}",
            ).items()
        }

        self.next(self.end)

    @step
    def end(self):
        self.fp.__exit__(None, None, None)

        checklist = self.pipeline_results.get("qc_checklist", {})
        gate = self.pipeline_results.get("qc_gate", {})
        notes = self.pipeline_results.get("release_notes", {})
        docs = self.pipeline_results.get("doc_updates", {})

        print(f"QC items: {len(checklist.get('items', []))}")
        print(f"Gate must-pass: {len(gate.get('must_pass', []))}")
        print(f"Release highlights: {len(notes.get('highlights', []))}")
        print(f"Doc updates: {len(docs.get('updates', []))}")


if __name__ == "__main__":
    QCReleaseFlow()
