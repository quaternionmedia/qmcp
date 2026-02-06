"""Local release notes flow that chains LLM agents.

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

from local_dev_db import (
    init_db,
    mark_flow_finished,
    save_agent_run,
    save_artifact,
    save_flow_run,
    save_mcp_invocation,
)
from local_llm import LocalLLMConfig, build_agent
from local_mcp import ReviewerInput, check_health, invoke_tool, require_invocation_id
from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field
from sqlmodel import Session


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
        self.run_id = current.run_id
        engine = init_db(self.db_path)
        with Session(engine) as session:
            flow_run = save_flow_run(
                session,
                flow_name=self.__class__.__name__,
                run_id=self.run_id,
                meta={
                    "change_summary": self.change_summary,
                    "audience": self.audience,
                    "llm_model": self.llm_model,
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

        self.next(self.summarize)

    @step
    def summarize(self):
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        summarizer = build_agent(
            config=config,
            system_prompt="You summarize engineering changes for release notes.",
            result_type=ChangeSummary,
        )

        prompt = "\n".join(
            [
                f"Audience: {self.audience}",
                f"Change summary: {self.change_summary}",
                "Extract themes, impacted areas, key changes, and risks.",
            ]
        )
        result = summarizer.run_sync(prompt)
        self.summary_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="summarizer",
                input_summary=prompt,
                output=self.summary_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="change_summary",
                content=self.summary_output,
            )

        if self.use_mcp:
            reviewer_input = ReviewerInput(
                result=self.summary_output,
                criteria=["clarity", "impact", "risk"],
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="reviewer",
                payload=reviewer_input,
                correlation_id=f"release-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP reviewer failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_summary_review = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "reviewer", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="reviewer",
                    invocation_id=invocation_id,
                    payload=reviewer_input.model_dump(),
                    correlation_id=f"release-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_summary_review",
                    content=self.mcp_summary_review,
                )

        self.next(self.draft_notes)

    @step
    def draft_notes(self):
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        notes_agent = build_agent(
            config=config,
            system_prompt="You draft concise release notes.",
            result_type=ReleaseNotes,
        )

        prompt = "\n".join(
            [
                f"Audience: {self.audience}",
                "Write release notes from the structured change summary.",
                f"Summary JSON:\n{json.dumps(self.summary_output, indent=2)}",
            ]
        )
        result = notes_agent.run_sync(prompt)
        self.notes_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="release_notes",
                input_summary="Draft release notes from summary.",
                output=self.notes_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="release_notes",
                content=self.notes_output,
            )

        if self.use_mcp:
            reviewer_input = ReviewerInput(
                result=self.notes_output,
                criteria=["clarity", "completeness"],
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="reviewer",
                payload=reviewer_input,
                correlation_id=f"release-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP reviewer failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_notes_review = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "reviewer", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="reviewer",
                    invocation_id=invocation_id,
                    payload=reviewer_input.model_dump(),
                    correlation_id=f"release-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_notes_review",
                    content=self.mcp_notes_review,
                )

        self.next(self.doc_updates)

    @step
    def doc_updates(self):
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        doc_agent = build_agent(
            config=config,
            system_prompt="You propose documentation updates for a release.",
            result_type=DocUpdatePlan,
        )

        prompt = "\n".join(
            [
                "Suggest documentation updates based on release notes.",
                f"Release notes JSON:\n{json.dumps(self.notes_output, indent=2)}",
            ]
        )
        result = doc_agent.run_sync(prompt)
        self.doc_updates_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="doc_updates",
                input_summary="Suggest doc updates from release notes.",
                output=self.doc_updates_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="doc_updates",
                content=self.doc_updates_output,
            )

        if self.use_mcp:
            reviewer_input = ReviewerInput(
                result=self.doc_updates_output,
                criteria=["coverage", "priority"],
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="reviewer",
                payload=reviewer_input,
                correlation_id=f"release-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP reviewer failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_doc_review = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "reviewer", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="reviewer",
                    invocation_id=invocation_id,
                    payload=reviewer_input.model_dump(),
                    correlation_id=f"release-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_doc_review",
                    content=self.mcp_doc_review,
                )

        self.next(self.end)

    @step
    def end(self):
        engine = init_db(self.db_path)
        with Session(engine) as session:
            mark_flow_finished(session, self.flow_run_id)

        print("Highlights:", len(self.notes_output["highlights"]))
        print("Doc updates:", len(self.doc_updates_output["updates"]))
        if self.mcp_invocations:
            print("MCP invocations:")
            for entry in self.mcp_invocations:
                print(f"  {entry['tool']}: {entry['invocation_id']}")


if __name__ == "__main__":
    LocalReleaseNotesFlow()
