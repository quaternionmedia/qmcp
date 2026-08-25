"""Local release notes flow that chains LLM agents.

Uses ``qmcp.cookbook`` modules for agent building, persistence, and MCP
tool invocation.

Usage:
    uv sync --extra flows
    uv run qmcp serve
    uv run python examples/flows/local_release_notes.py run \
        --use-mcp True \
        --change-summary "Refactor metrics registry and add QC docs" \
        --audience "internal" \
        --llm-base-url "http://localhost:11434/v1" \
        --llm-model "llama3.1"
"""

from __future__ import annotations

import json
import os

from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field

from qmcp.cookbook import FlowPersistence, LocalLLMConfig, MCPToolInvoker, build_local_agent
from qmcp.cookbook.mcp_tools import ReviewerInput


class ChangeSummary(BaseModel):
    """Structured change summary."""

    themes: list[str]
    impacted_areas: list[str]
    key_changes: list[str]
    risks: list[str] = Field(default_factory=list)


class ReleaseNotes(BaseModel):
    """Release notes output."""

    title: str
    highlights: list[str]
    breaking_changes: list[str] = Field(default_factory=list)
    migration_notes: list[str] = Field(default_factory=list)


class DocUpdate(BaseModel):
    """Single documentation update suggestion."""

    path: str
    reason: str
    suggested_change: str


class DocUpdatePlan(BaseModel):
    """Documentation update plan."""

    updates: list[DocUpdate]


class LocalReleaseNotesFlow(FlowSpec):
    """Creates release notes and doc update suggestions via local LLM agents."""

    change_summary = Parameter(
        "change-summary",
        help="Summary of changes",
        required=True,
    )
    audience = Parameter("audience", help="Audience for the notes", default="internal")
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
        self.run_id = current.run_id
        self.fp = FlowPersistence(
            db_path=self.db_path,
            flow_name=self.__class__.__name__,
            run_id=self.run_id,
            meta={
                "change_summary": self.change_summary,
                "audience": self.audience,
                "llm_model": self.llm_model,
                "use_mcp": self.use_mcp,
                "mcp_url": self.mcp_url,
            },
        ).__enter__()

        self.mcp_invocations: list[dict] = []

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            invoker.health()

        self.next(self.summarize)

    @step
    def summarize(self):
        summarizer = build_local_agent(
            config=self._llm_config(),
            system_prompt="You summarize engineering changes for release notes.",
            output_type=ChangeSummary,
        )

        prompt = "\n".join([
            f"Audience: {self.audience}",
            f"Change summary: {self.change_summary}",
            "Extract themes, impacted areas, key changes, and risks.",
        ])
        result = summarizer.run_sync(prompt)
        self.summary_output = result.output.model_dump()

        self.fp.agent_run("summarizer", prompt, self.summary_output)
        self.fp.artifact("change_summary", self.summary_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            reviewer_input = ReviewerInput(
                result=self.summary_output,
                criteria=["clarity", "impact", "risk"],
            )
            mcp_result = invoker.invoke(
                "reviewer", reviewer_input,
                correlation_id=f"release-{self.run_id}",
                artifact_kind="mcp_summary_review",
            )
            self.mcp_invocations.append({
                "tool": "reviewer",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.draft_notes)

    @step
    def draft_notes(self):
        notes_agent = build_local_agent(
            config=self._llm_config(),
            system_prompt="You draft concise release notes.",
            output_type=ReleaseNotes,
        )

        prompt = "\n".join([
            f"Audience: {self.audience}",
            "Write release notes from the structured change summary.",
            f"Summary JSON:\n{json.dumps(self.summary_output, indent=2)}",
        ])
        result = notes_agent.run_sync(prompt)
        self.notes_output = result.output.model_dump()

        self.fp.agent_run("release_notes", "Draft release notes from summary.", self.notes_output)
        self.fp.artifact("release_notes", self.notes_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            reviewer_input = ReviewerInput(
                result=self.notes_output,
                criteria=["clarity", "completeness"],
            )
            mcp_result = invoker.invoke(
                "reviewer", reviewer_input,
                correlation_id=f"release-{self.run_id}",
                artifact_kind="mcp_notes_review",
            )
            self.mcp_invocations.append({
                "tool": "reviewer",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.doc_updates)

    @step
    def doc_updates(self):
        doc_agent = build_local_agent(
            config=self._llm_config(),
            system_prompt="You propose documentation updates for a release.",
            output_type=DocUpdatePlan,
        )

        prompt = "\n".join([
            "Suggest documentation updates based on release notes.",
            f"Release notes JSON:\n{json.dumps(self.notes_output, indent=2)}",
        ])
        result = doc_agent.run_sync(prompt)
        self.doc_updates_output = result.output.model_dump()

        self.fp.agent_run("doc_updates", "Suggest doc updates from release notes.", self.doc_updates_output)
        self.fp.artifact("doc_updates", self.doc_updates_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            reviewer_input = ReviewerInput(
                result=self.doc_updates_output,
                criteria=["coverage", "priority"],
            )
            mcp_result = invoker.invoke(
                "reviewer", reviewer_input,
                correlation_id=f"release-{self.run_id}",
                artifact_kind="mcp_doc_review",
            )
            self.mcp_invocations.append({
                "tool": "reviewer",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.end)

    @step
    def end(self):
        self.fp.__exit__(None, None, None)

        print("Highlights:", len(self.notes_output["highlights"]))
        print("Doc updates:", len(self.doc_updates_output["updates"]))
        if self.mcp_invocations:
            print("MCP invocations:")
            for entry in self.mcp_invocations:
                print(f"  {entry['tool']}: {entry['invocation_id']}")


if __name__ == "__main__":
    LocalReleaseNotesFlow()
