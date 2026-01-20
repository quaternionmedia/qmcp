"""Local QC gauntlet flow that chains multiple LLM agents.

Usage:
    uv sync --extra flows
    uv run qmcp serve
    uv run python examples/flows/local_qc_gauntlet.py run \
        --use-mcp True \
        --change-summary "Add audit fields to tool invocations" \
        --target-area "metrics, logging, db" \
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
    save_checklist_items,
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
from metaflow import FlowSpec, Parameter, step
from pydantic import BaseModel, Field
from sqlmodel import Session


class QCItem(BaseModel):
    """Single QC checklist item."""

    area: str
    check: str
    command: str | None = None
    expected: str | None = None


class QCChecklist(BaseModel):
    """Checklist output."""

    summary: str
    items: list[QCItem]
    focus_areas: list[str] = Field(default_factory=list)


class QCTask(BaseModel):
    """Task derived from checklist."""

    check: str
    command: str
    success_criteria: str


class QCTaskPlan(BaseModel):
    """Execution plan for QC tasks."""

    tasks: list[QCTask]
    ordering: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class QCGate(BaseModel):
    """Stop-ship criteria and risk flags."""

    must_pass: list[str]
    risk_flags: list[str]
    stop_ship_conditions: list[str]


class LocalQCGauntletFlow(FlowSpec):
    """Designs a QC gauntlet with chained local agents."""

    change_summary = Parameter(
        "change-summary",
        help="Summary of the change set",
        required=True,
    )
    target_area = Parameter(
        "target-area",
        help="Area impacted (comma-separated)",
        default="core",
    )
    constraints = Parameter(
        "constraints",
        help="Constraints or non-goals",
        default="Keep checks local and fast",
    )
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
        engine = init_db(self.db_path)
        with Session(engine) as session:
            flow_run = save_flow_run(
                session,
                flow_name=self.__class__.__name__,
                run_id=self.run_id,
                meta={
                    "change_summary": self.change_summary,
                    "target_area": self.target_area,
                    "constraints": self.constraints,
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

        self.next(self.draft_checklist)

    @step
    def draft_checklist(self):
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        checklist_agent = build_agent(
            config=config,
            system_prompt="You create QA/QC checklists for local dev cycles.",
            result_type=QCChecklist,
        )

        prompt = "\n".join(
            [
                f"Change summary: {self.change_summary}",
                f"Target area: {self.target_area}",
                f"Constraints: {self.constraints}",
                "Draft a QC checklist with commands and expected results when possible.",
            ]
        )
        result = checklist_agent.run_sync(prompt)
        self.checklist_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="qc_checklist",
                input_summary=prompt,
                output=self.checklist_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="qc_checklist",
                content=self.checklist_output,
            )

            item_rows = [item.model_dump() for item in result.data.items]
            save_checklist_items(session, self.flow_run_id, item_rows)

        if self.use_mcp:
            reviewer_input = ReviewerInput(
                result=self.checklist_output,
                criteria=["coverage", "risk", "runtime"],
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="reviewer",
                payload=reviewer_input,
                correlation_id=f"qc-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP reviewer failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_checklist_review = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "reviewer", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="reviewer",
                    invocation_id=invocation_id,
                    payload=reviewer_input.model_dump(),
                    correlation_id=f"qc-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_checklist_review",
                    content=self.mcp_checklist_review,
                )

        self.next(self.expand_tasks)

    @step
    def expand_tasks(self):
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        task_agent = build_agent(
            config=config,
            system_prompt="You expand QC checklists into runnable task plans.",
            result_type=QCTaskPlan,
        )

        prompt = "\n".join(
            [
                "Expand the QC checklist into runnable tasks.",
                f"Checklist JSON:\n{json.dumps(self.checklist_output, indent=2)}",
            ]
        )
        result = task_agent.run_sync(prompt)
        self.task_plan_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="qc_tasks",
                input_summary="Expand checklist into tasks.",
                output=self.task_plan_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="qc_task_plan",
                content=self.task_plan_output,
            )

        if self.use_mcp:
            plan_steps = []
            for idx, task in enumerate(self.task_plan_output["tasks"], start=1):
                plan_steps.append({"step": idx, "action": task["check"]})
            executor_input = ExecutorInput(
                plan={"goal": "QC task plan", "steps": plan_steps},
                dry_run=True,
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="executor",
                payload=executor_input,
                correlation_id=f"qc-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP executor failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_task_execution = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "executor", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="executor",
                    invocation_id=invocation_id,
                    payload=executor_input.model_dump(),
                    correlation_id=f"qc-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_task_execution",
                    content=self.mcp_task_execution,
                )

        self.next(self.gate)

    @step
    def gate(self):
        config = LocalLLMConfig(
            model=self.llm_model,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        gate_agent = build_agent(
            config=config,
            system_prompt="You define stop-ship criteria for QC.",
            result_type=QCGate,
        )

        prompt = "\n".join(
            [
                "Define must-pass checks and stop-ship conditions.",
                f"Checklist JSON:\n{json.dumps(self.checklist_output, indent=2)}",
                f"Task plan JSON:\n{json.dumps(self.task_plan_output, indent=2)}",
            ]
        )
        result = gate_agent.run_sync(prompt)
        self.gate_output = result.data.model_dump()

        engine = init_db(self.db_path)
        with Session(engine) as session:
            save_agent_run(
                session,
                flow_run_id=self.flow_run_id,
                agent_name="qc_gate",
                input_summary="Define stop-ship criteria.",
                output=self.gate_output,
            )
            save_artifact(
                session,
                flow_run_id=self.flow_run_id,
                kind="qc_gate",
                content=self.gate_output,
            )

        if self.use_mcp:
            reviewer_input = ReviewerInput(
                result=self.gate_output,
                criteria=["stop_ship", "risk_flags"],
            )
            mcp_result = invoke_tool(
                self.mcp_url,
                tool_name="reviewer",
                payload=reviewer_input,
                correlation_id=f"qc-{self.run_id}",
            )
            if mcp_result.error:
                raise RuntimeError(f"MCP reviewer failed: {mcp_result.error}")
            invocation_id = require_invocation_id(mcp_result)
            self.mcp_gate_review = mcp_result.model_dump()
            self.mcp_invocations.append({"tool": "reviewer", "invocation_id": invocation_id})

            engine = init_db(self.db_path)
            with Session(engine) as session:
                save_mcp_invocation(
                    session,
                    flow_run_id=self.flow_run_id,
                    tool_name="reviewer",
                    invocation_id=invocation_id,
                    payload=reviewer_input.model_dump(),
                    correlation_id=f"qc-{self.run_id}",
                )
                save_artifact(
                    session,
                    flow_run_id=self.flow_run_id,
                    kind="mcp_gate_review",
                    content=self.mcp_gate_review,
                )

        self.next(self.end)

    @step
    def end(self):
        engine = init_db(self.db_path)
        with Session(engine) as session:
            mark_flow_finished(session, self.flow_run_id)

        print("Checklist items:", len(self.checklist_output["items"]))
        print("Task plan tasks:", len(self.task_plan_output["tasks"]))
        print("Gate must-pass checks:", len(self.gate_output["must_pass"]))
        if self.mcp_invocations:
            print("MCP invocations:")
            for entry in self.mcp_invocations:
                print(f"  {entry['tool']}: {entry['invocation_id']}")


if __name__ == "__main__":
    LocalQCGauntletFlow()
