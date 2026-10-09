"""Spoken words to a saved design: parse, consent, persist, run.

The design service is real (its own SQLite file, the real routes); only the
human queue and the model are stood in for.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp import topology_designs as designs
from qmcp.client.mcp_client import MCPClient
from qmcp.instructions import converse
from qmcp.integrations.agents import AgentOutcome
from qmcp.integrations.agents.topology_design import TopologyDesignRuntime
from qmcp.integrations.voice.vocabulary import match_topology_command, match_topology_query
from tests.test_instructions_act import PROJECT, _act, _Queue, clone, inbox  # noqa: F401


@pytest.fixture()
def service(tmp_path) -> MCPClient:
    app = FastAPI()
    designs.register(app, sessions=designs.sessions_at(tmp_path / "designs.db"),
                     project="quaternionmedia/qmcp")
    client = MCPClient("http://testserver")
    client._client = TestClient(app)
    return client


class _Model:
    prompts: list[str] = []

    def run(self, brief, on_event=None):
        self.prompts.append(brief.instruction)
        return AgentOutcome(text=f"Report {len(self.prompts)}", exit_code=0, spent=0,
                            detail={"model_calls": 1, "read": []})


def _say(words, answer, inbox, clone, service, model=None):
    """One spoken command through the real gate; returns the act and its consent."""
    command = match_topology_command(words)
    assert command is not None, words
    runtime = TopologyDesignRuntime(service, command, runtime_factory=model or _Model)
    instruction_id = inbox.record(words, project=PROJECT)
    queue = _Queue({f"instruction-{instruction_id}": answer})
    done, _ = _act(inbox, instruction_id, queue, runtime=runtime, cwd=clone)
    return done, queue.requests[f"instruction-{instruction_id}"]


def test_a_design_is_built_by_voice_persisted_and_run(inbox, clone, service):
    """Mutation: apply a change before the approve -- red, the held change
    below is found saved."""
    _Model.prompts = []

    held, consent = _say("Create topology review as pipeline in qmcp", "hold", inbox, clone, service)
    assert held.status == "refused" and "review" in consent["prompt"]
    assert service.list_topologies(project="qmcp")["count"] == 0

    for words in (
        "Create topology review as pipeline in qmcp",
        "Create component first with instruction Find direct evidence in qmcp",
        "Create component second with instruction Check the previous report in qmcp",
        "Add component first to topology review in qmcp",
        "Add component second to topology review in qmcp",
    ):
        done, _ = _say(words, "approve", inbox, clone, service)
        assert done.status == "done", (words, done.why)

    saved = service.get_topology("review", project="qmcp")
    assert [c["name"] for c in saved["config"]["components"]] == ["first", "second"]
    assert service.get_topology_component("first", project="qmcp")["instruction"] == "Find direct evidence"

    done, _ = _say("Run topology review in qmcp about Is the report supported?", "approve",
                   inbox, clone, service)
    assert done.status == "done" and done.outcome.detail["model_calls"] == 2
    assert len(_Model.prompts) == 2 and "Report 1" in _Model.prompts[1]


def test_only_read_forms_parse_as_queries():
    assert match_topology_query("Show component First in QMCP.").action == "show_component"
    assert match_topology_query("list topologies in qmcp").project == "qmcp"
    assert match_topology_query("create topology a as chain in qmcp") is None
    assert match_topology_query("list topologies") is None


def test_what_was_built_can_be_listed_and_shown_by_voice(inbox, clone, service):
    for words in (
        "Create topology review as pipeline in qmcp",
        "Create component first with instruction Find direct evidence in qmcp",
        "Add component first to topology review in qmcp",
    ):
        assert _say(words, "approve", inbox, clone, service)[0].status == "done"
    said: list[str] = []
    talk = converse.Conversation(None, None, service, object(), names=["qmcp"])
    talk.ask = lambda text, options=(): said.append(text)

    assert talk._browse("List topologies in qmcp.")
    assert talk._browse("List components in qmcp")
    assert talk._browse("Show topology review in qmcp")
    assert talk._browse("Show component first in qmcp")
    assert talk._browse("Show topology missing in qmcp")
    assert talk._browse("List topologies in vox")
    assert not talk._browse("Run topology review in qmcp about it")

    assert said == [
        "1 topology in qmcp: review, a pipeline.",
        "1 component in qmcp: first.",
        "Topology review in qmcp is a pipeline with first.",
        "Component first in qmcp: Find direct evidence",
        "No topology missing in qmcp.",
        "No topologies in vox.",
    ]
