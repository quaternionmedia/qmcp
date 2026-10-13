from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from qmcp.instructions.converse import Conversation
from qmcp.integrations.agents import AgentOutcome, Brief
from qmcp.integrations.agents.topology_design import TopologyDesignRuntime
from qmcp.integrations.voice.vocabulary import match_topology_command


def test_spoken_topology_commands_parse_explicit_project_and_arguments():
    parsed = match_topology_command(
        "Create topology review as chain of command in QMCP."
    )
    assert parsed is not None
    assert (
        parsed.action,
        parsed.project,
        parsed.name,
        parsed.kind,
    ) == ("create_topology", "qmcp", "review", "chain")
    assert match_topology_command(
        "Create component skeptic with instruction Look for counterexamples in qmcp"
    ).instruction == "Look for counterexamples"
    assert match_topology_command(
        "Set component skeptic routes API or database in topology review in qmcp"
    ).route_terms == "API or database"
    assert match_topology_command(
        "Run topology review in qmcp about Is the report supported?"
    ).prompt == "Is the report supported?"
    assert match_topology_command(
        "Create topology unsafe as mesh in qmcp"
    ) is None


class _Client:
    def __init__(self):
        self.designs = {
            "review": {
                "name": "review",
                "topology_type": "pipeline",
                "version": "1.0.0",
                "config": {
                    "components": [
                        {"name": "first", "route_terms": []},
                        {"name": "second", "route_terms": []},
                    ],
                    "compose": [],
                },
            },
        }
        self.components = {
            "first": {
                "name": "first", "description": "First pass",
                "instruction": "Find direct evidence.", "version": "1.0.0",
            },
            "second": {
                "name": "second", "description": "Second pass",
                "instruction": "Check the previous report.", "version": "1.0.0",
            },
        }

    def get_topology(self, name, project=None):
        return self.designs[name]

    def get_topology_component(self, name, project=None):
        return self.components[name]

    def update_topology(self, name, changes, project=None):
        self.designs[name].update(changes)
        return self.designs[name]


class _Runtime:
    def __init__(self, prompts):
        self.prompts = prompts

    def run(self, brief: Brief, on_event=None) -> AgentOutcome:
        self.prompts.append(brief.instruction)
        return AgentOutcome(
            text=f"Report {len(self.prompts)}",
            exit_code=0,
            spent=0,
            detail={"model_calls": 1, "read": ["read_file(README.md)"]},
        )


def _request(action="run", **overrides):
    values = {
        "action": action, "project": "qmcp", "name": "review", "kind": "",
        "component": "", "child": "", "route_terms": "", "instruction": "",
        "prompt": "Check the docs",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pipeline_voice_run_is_bounded_and_passes_prior_report_explicitly(tmp_path):
    prompts = []
    runtime = TopologyDesignRuntime(
        _Client(), _request(), runtime_factory=lambda: _Runtime(prompts)
    )
    outcome = runtime.run(Brief("spoken instruction", tmp_path, "qmcp"))
    assert outcome.succeeded
    assert outcome.detail["model_calls"] == 2
    assert "Earlier component reports" in prompts[1]
    assert "first: Report 1" in prompts[1]


def test_delegation_refuses_an_unmatched_task_before_consent():
    """Refused while the run is prepared, so nobody is asked to approve a run
    that would route nothing, and no model is called.

    Mutation: drop the routing check in `_refuse_before_consent` and the
    runtime is built.
    """
    client = _Client()
    client.designs["review"]["topology_type"] = "delegation"
    client.designs["review"]["config"]["components"] = [
        {"name": "first", "route_terms": ["security review"]}
    ]
    prompts = []
    with pytest.raises(ValueError, match="would not route the task"):
        TopologyDesignRuntime(client, _request(), runtime_factory=lambda: _Runtime(prompts))
    assert prompts == []


def test_design_change_after_approval_request_is_refused_before_model_call(tmp_path):
    client = _Client()
    prompts = []
    runtime = TopologyDesignRuntime(
        client, _request(), runtime_factory=lambda: _Runtime(prompts)
    )
    client.components["first"]["instruction"] = "Changed after approval."
    outcome = runtime.run(Brief("spoken instruction", tmp_path, "qmcp"))
    assert not outcome.succeeded
    assert "changed after approval" in outcome.text
    assert outcome.detail["model_calls"] == 0
    assert prompts == []


def test_component_routing_is_an_approved_persisted_design_edit():
    client = _Client()
    client.designs["review"]["topology_type"] = "delegation"
    request = _request(
        action="route_component", component="first", route_terms="security or auth"
    )
    runtime = TopologyDesignRuntime(client, request)
    outcome = runtime.run(Brief("spoken instruction", Path(".")))
    assert outcome.succeeded
    assert client.designs["review"]["config"]["components"][0]["route_terms"] == [
        "security", "auth",
    ]
    assert "security or auth" in runtime.command


def test_component_edits_ignore_the_case_it_was_spoken_in():
    client = _Client()
    client.designs["review"]["topology_type"] = "delegation"
    routed = TopologyDesignRuntime(client, _request(
        action="route_component", component="FIRST", route_terms="auth",
    )).run(Brief("spoken instruction", Path(".")))
    assert routed.succeeded
    assert client.designs["review"]["config"]["components"][0]["route_terms"] == ["auth"]

    removed = TopologyDesignRuntime(client, _request(
        action="remove_component", component="First",
    )).run(Brief("spoken instruction", Path(".")))
    assert removed.succeeded
    assert [c["name"] for c in client.designs["review"]["config"]["components"]] == ["second"]


def test_composed_designs_execute_in_order_with_shared_report_context(tmp_path):
    client = _Client()
    client.designs["parent"] = {
        "name": "parent",
        "topology_type": "compound",
        "version": "1.0.0",
        "config": {
            "components": [],
            "compose": ["review"],
        },
    }
    prompts = []
    runtime = TopologyDesignRuntime(
        client, _request(name="parent"), runtime_factory=lambda: _Runtime(prompts)
    )
    outcome = runtime.run(Brief("spoken instruction", tmp_path, "qmcp"))
    assert outcome.succeeded
    assert outcome.detail["model_calls"] == 2
    assert "first: Report 1" in prompts[1]
    assert "second: Report 2" in outcome.text


def test_crosscheck_runs_its_composed_designs_after_checking(monkeypatch, tmp_path):
    from qmcp.integrations.agents import crosscheck

    client = _Client()
    client.designs["review"]["topology_type"] = "crosscheck"
    client.designs["review"]["config"]["compose"] = ["child"]
    client.designs["child"] = {
        "name": "child",
        "topology_type": "pipeline",
        "version": "1.0.0",
        "config": {
            "components": [{"name": "first", "route_terms": []}],
            "compose": [],
        },
    }
    received = {}

    class FakeCrossCheck:
        def __init__(self, prompt, runtime_factory, *, perspectives, checker_count):
            received["checker_count"] = checker_count

        def run(self, brief, on_event=None):
            return AgentOutcome(
                text="Cross-check report",
                exit_code=0,
                spent=0,
                detail={"model_calls": 4},
            )

    monkeypatch.setattr(crosscheck, "CrossCheckRuntime", FakeCrossCheck)
    prompts = []
    runtime = TopologyDesignRuntime(
        client, _request(), runtime_factory=lambda: _Runtime(prompts)
    )

    outcome = runtime.run(Brief("spoken instruction", tmp_path, "qmcp"))

    assert outcome.succeeded
    assert received["checker_count"] == 2
    assert outcome.detail["model_calls"] == 5
    assert len(prompts) == 1
    assert "Cross-check report" in prompts[0]


def test_empty_non_crosscheck_design_is_refused_instead_of_succeeding_empty(tmp_path):
    client = _Client()
    client.designs["review"]["topology_type"] = "council"
    client.designs["review"]["config"] = {"components": [], "compose": []}

    try:
        TopologyDesignRuntime(
            client, _request(), runtime_factory=lambda: _Runtime([])
        )
    except ValueError as exc:
        assert "no components or composed designs" in str(exc)
    else:
        raise AssertionError("an empty Council must not be accepted as runnable")


def test_an_advisory_council_runs_and_reports(tmp_path):
    """`arbiter_can_override` false selects the advisory declaration, which
    the voice runner takes.

    Mutation: judge the council with `plane.refuses(kind, prompt)` alone, or
    leave `arbiter_can_override` out of the council's supported options, and
    this is refused.
    """
    client = _Client()
    client.designs["review"]["topology_type"] = "council"
    client.designs["review"]["config"]["arbiter_can_override"] = False
    prompts = []
    runtime = TopologyDesignRuntime(client, _request(),
                                    runtime_factory=lambda: _Runtime(prompts))
    outcome = runtime.run(Brief("spoken instruction", tmp_path, "qmcp"))
    assert outcome.succeeded
    assert len(prompts) >= 2 and outcome.detail["model_calls"] == len(prompts)


def test_a_deciding_council_is_refused_before_consent():
    """The default council's arbiter takes the final decision, which the plane
    refuses outright, so nobody is asked to approve its run.

    Mutation: drop the `refusal` check in `_refuse_before_consent` and the
    message no longer names the arbiter.
    """
    client = _Client()
    client.designs["review"]["topology_type"] = "council"
    client.designs["review"]["config"]["arbiter_can_override"] = True
    prompts = []
    with pytest.raises(ValueError, match="refused: council is refused here.*arbiter"):
        TopologyDesignRuntime(client, _request(), runtime_factory=lambda: _Runtime(prompts))
    assert prompts == []


def test_conversation_routes_saved_topology_command_through_consent_gate(monkeypatch, tmp_path):
    observed = {}

    class Client:
        def get_instruction(self, instruction_id):
            return {
                "id": instruction_id,
                "status": "done",
                "project": "qmcp",
                "outcome_text": "Created topology review.",
            }

    def fake_act(instruction_id, runtime, budget, **kwargs):
        observed["runtime"] = runtime
        observed["budget"] = budget.authorised
        observed["cwd"] = kwargs["cwd"]
        return SimpleNamespace(why="", ran=True, carried=())

    monkeypatch.setattr("qmcp.instructions.act.act", fake_act)
    monkeypatch.setattr(
        Conversation, "_clone_for", lambda self, project, instruction_id: tmp_path
    )
    conversation = Conversation(
        stt=object(), tts=SimpleNamespace(speak=lambda text: None), client=Client(),
        runtime=object(), names=["qmcp"]
    )
    conversation._act_on({
        "id": "instruction-1",
        "project": "qmcp",
        "text": "Create topology review as pipeline in qmcp",
    })
    assert isinstance(observed["runtime"], TopologyDesignRuntime)
    assert observed["runtime"].request.action == "create_topology"
    assert observed["budget"] == 1
    assert observed["cwd"] == tmp_path


def test_the_conversation_answers_a_listing_without_recording_anything():
    """Asked in the standing conversation, a listing is said back and nothing
    is recorded or acted on.

    Mutation: drop the `_browse` call from `Conversation.run` and the listing
    is taken as an instruction.
    """
    from tests.test_spoken_iteration import _Client as SpokenClient, _talk

    class Client(SpokenClient):
        def list_topologies(self, project=None):
            return {"topologies": [{"name": "review", "topology_type": "pipeline"}]}

    ended, tts, client, acted = _talk("List topologies in qmcp.", "stop", client=Client())

    assert "1 topology in qmcp: review, a pipeline." in tts.spoken
    assert ended.turns == [] and acted == [] and client.created == []
