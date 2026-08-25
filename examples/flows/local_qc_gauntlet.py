"""Local QC gauntlet flow that chains multiple LLM agents.

Uses ``qmcp.cookbook`` modules for agent building, persistence, and MCP
tool invocation.

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

from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field

from qmcp.cookbook import FlowPersistence, LocalLLMConfig, MCPToolInvoker, build_local_agent
from qmcp.cookbook.mcp_tools import ExecutorInput, ReviewerInput


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
                "target_area": self.target_area,
                "constraints": self.constraints,
                "llm_model": self.llm_model,
                "use_mcp": self.use_mcp,
                "mcp_url": self.mcp_url,
            },
        ).__enter__()

        self.mcp_invocations: list[dict] = []

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            invoker.health()

        self.next(self.draft_checklist)

    @step
    def draft_checklist(self):
        checklist_agent = build_local_agent(
            config=self._llm_config(),
            system_prompt="You create QA/QC checklists for local dev cycles.",
            output_type=QCChecklist,
        )

        prompt = "\n".join([
            f"Change summary: {self.change_summary}",
            f"Target area: {self.target_area}",
            f"Constraints: {self.constraints}",
            "Draft a QC checklist with commands and expected results when possible.",
        ])
        result = checklist_agent.run_sync(prompt)
        self.checklist_output = result.output.model_dump()

        self.fp.agent_run("qc_checklist", prompt, self.checklist_output)
        self.fp.artifact("qc_checklist", self.checklist_output)
        self.fp.checklist_items([item.model_dump() for item in result.output.items])

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            reviewer_input = ReviewerInput(
                result=self.checklist_output,
                criteria=["coverage", "risk", "runtime"],
            )
            mcp_result = invoker.invoke(
                "reviewer", reviewer_input,
                correlation_id=f"qc-{self.run_id}",
                artifact_kind="mcp_checklist_review",
            )
            self.mcp_invocations.append({
                "tool": "reviewer",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.expand_tasks)

    @step
    def expand_tasks(self):
        task_agent = build_local_agent(
            config=self._llm_config(),
            system_prompt="You expand QC checklists into runnable task plans.",
            output_type=QCTaskPlan,
        )

        prompt = "\n".join([
            "Expand the QC checklist into runnable tasks.",
            f"Checklist JSON:\n{json.dumps(self.checklist_output, indent=2)}",
        ])
        result = task_agent.run_sync(prompt)
        self.task_plan_output = result.output.model_dump()

        self.fp.agent_run("qc_tasks", "Expand checklist into tasks.", self.task_plan_output)
        self.fp.artifact("qc_task_plan", self.task_plan_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            plan_steps = [
                {"step": idx, "action": task["check"]}
                for idx, task in enumerate(self.task_plan_output["tasks"], start=1)
            ]
            executor_input = ExecutorInput(
                plan={"goal": "QC task plan", "steps": plan_steps},
                dry_run=True,
            )
            mcp_result = invoker.invoke(
                "executor", executor_input,
                correlation_id=f"qc-{self.run_id}",
                artifact_kind="mcp_task_execution",
            )
            self.mcp_invocations.append({
                "tool": "executor",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.gate)

    @step
    def gate(self):
        gate_agent = build_local_agent(
            config=self._llm_config(),
            system_prompt="You define stop-ship criteria for QC.",
            output_type=QCGate,
        )

        prompt = "\n".join([
            "Define must-pass checks and stop-ship conditions.",
            f"Checklist JSON:\n{json.dumps(self.checklist_output, indent=2)}",
            f"Task plan JSON:\n{json.dumps(self.task_plan_output, indent=2)}",
        ])
        result = gate_agent.run_sync(prompt)
        self.gate_output = result.output.model_dump()

        self.fp.agent_run("qc_gate", "Define stop-ship criteria.", self.gate_output)
        self.fp.artifact("qc_gate", self.gate_output)

        if self.use_mcp:
            invoker = MCPToolInvoker(self.mcp_url, persistence=self.fp)
            reviewer_input = ReviewerInput(
                result=self.gate_output,
                criteria=["stop_ship", "risk_flags"],
            )
            mcp_result = invoker.invoke(
                "reviewer", reviewer_input,
                correlation_id=f"qc-{self.run_id}",
                artifact_kind="mcp_gate_review",
            )
            self.mcp_invocations.append({
                "tool": "reviewer",
                "invocation_id": mcp_result.invocation_id,
            })

        self.next(self.end)

    @step
    def end(self):
        self.fp.__exit__(None, None, None)

        print("Checklist items:", len(self.checklist_output["items"]))
        print("Task plan tasks:", len(self.task_plan_output["tasks"]))
        print("Gate must-pass checks:", len(self.gate_output["must_pass"]))
        if self.mcp_invocations:
            print("MCP invocations:")
            for entry in self.mcp_invocations:
                print(f"  {entry['tool']}: {entry['invocation_id']}")


if __name__ == "__main__":
    LocalQCGauntletFlow()
