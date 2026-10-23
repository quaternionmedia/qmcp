"""What qmcp does with the model `qmcp localmodel` stands up on this machine.

Every other test stands in for the model. These run on it: the service at
`qmcp.localmodel.ENDPOINT` serving `qmcp.localmodel.MODEL`, offline, on this
machine's GPU. The runtime asks at temperature 0, so on one model the answers
repeat run to run.

Deselected by default, as CI has no model: `uv run pytest -m local_model`.
Run that way on a machine without the model, they fail and say what is
missing rather than skip -- a skip is not a pass.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from qmcp.cookbook import AgentPipeline, AgentStep, LocalLLMConfig, MCPToolInvoker
from qmcp.integrations.agents.adapters import ollama
from qmcp.localmodel import ENDPOINT, MODEL
from qmcp.testing import throwaway_server

pytestmark = pytest.mark.local_model

CONFIG = LocalLLMConfig(model=MODEL, base_url=f"{ENDPOINT}/v1")


@pytest.fixture(scope="module", autouse=True)
def _the_model_is_here() -> None:
    missing = ollama.Runtime().ready()
    if missing:
        pytest.fail(f"Nothing ran: {missing}.", pytrace=False)


class Plan(BaseModel):
    goal: str
    steps: list[str]


class Review(BaseModel):
    approved: bool
    concerns: list[str]


class Notes(BaseModel):
    title: str
    highlights: list[str]
    breaking_changes: list[str]


@pytest.mark.parametrize(("output", "prompt"), [
    (Plan, "Goal: add a health check to a web service. Write a 3-step plan."),
    (Review, "Review this plan: 1. add /health 2. deploy 3. done"),
    (Notes, "Write release notes for: fixed a crash on startup; renamed --port to --listen"),
])
def test_a_cookbook_step_gets_structured_output_from_the_local_model(output, prompt) -> None:
    # Seen red on the model before `build_local_agent` asked for prompted
    # output: the model wrote the tool call as text, and every step failed
    # with "Exceeded maximum output retries (3)".
    step = AgentStep("step", "You are a careful engineering assistant.", output)

    result = step.run(CONFIG, prompt)

    assert output.model_validate(result.output)


def test_a_pipeline_threads_its_outputs_and_records_its_review_on_the_server(tmp_path) -> None:
    pipeline = AgentPipeline(steps=[
        AgentStep("plan", "Write a short plan with 3 steps.", Plan),
        AgentStep("review", "Review the plan in the previous outputs.", Review,
                  mcp_tool="reviewer", mcp_criteria=["risk", "clarity"]),
    ])

    with throwaway_server(tmp_path / "queue.db") as url:
        results = pipeline.run(CONFIG, "Goal: add a health check to a web service",
                               invoker=MCPToolInvoker(url), correlation_id="local-model-1")
        recorded = httpx.get(f"{url}/v1/invocations").json()["invocations"]

    assert Plan.model_validate(results["plan"].output).steps
    assert Review.model_validate(results["review"].output)
    (review,) = [row for row in recorded if row["tool_name"] == "reviewer"]
    assert review["id"] == results["review"].mcp_invocation_id
    assert (review["status"], review["correlation_id"]) == ("success", "local-model-1")
