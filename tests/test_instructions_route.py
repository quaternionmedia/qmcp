"""`/v1/instructions`: recorded, listed, read; and the spoken route, which starts
a command and never runs an instruction.

The roster is the real one, read from `governance/qm`, so the names used are
two the organisation's own workspace carries. No process is started: the
launcher is stood in for, as `tests/test_voice_route.py` does.
"""

from __future__ import annotations

import pytest

from qmcp.instructions import RULE_NONE, RULE_ONE, RULE_SEVERAL, RULE_STATED, roster_names

NAMED, OTHER = "qmcp", "dossier"

pytestmark = pytest.mark.skipif(
    not {NAMED, OTHER} <= set(roster_names()),
    reason="governance/qm is not checked out, so there is no roster to resolve against",
)


class _Process:
    def __init__(self, argv, stdout, lines):
        self.argv = argv
        self.code = None
        stdout.write("\n".join(lines) + "\n")

    def poll(self):
        return self.code


@pytest.fixture
def launched(client, monkeypatch):
    """Every process either voice route starts, in order."""
    started: list[_Process] = []

    def popen(argv, stdout, stderr, env):
        process = _Process(argv, stdout, ["  [=] abc  qmcp  voice"])
        started.append(process)
        return process

    monkeypatch.setattr(client.app.state.voice_runs, "_popen", popen)
    return started


def _record(client, text, **extra):
    response = client.post("/v1/instructions", json={"text": text, **extra})
    assert response.status_code == 201, response.text
    return response.json()


# --- recording -----------------------------------------------------------------


def test_an_instruction_naming_one_project_is_recorded_against_it(client):
    """Mutation: write `status=InstructionStatus.UNRESOLVED` regardless -- red."""
    row = _record(client, f"Deploy {NAMED} to the pi.")

    assert row["status"] == "recorded"
    assert row["project"] == NAMED
    assert row["source"] == "typed"
    assert row["detail"] == {"candidates": [NAMED], "rule": RULE_ONE}
    assert row["id"] and row["created_at"] == row["updated_at"]


def test_an_instruction_naming_none_is_unresolved(client):
    """Mutation: default `project` to the first roster name -- red."""
    row = _record(client, "Rotate the logs.")

    assert row["status"] == "unresolved"
    assert row["project"] is None
    assert row["detail"] == {"candidates": [], "rule": RULE_NONE}


def test_an_instruction_naming_several_is_unresolved_with_the_candidates(client):
    row = _record(client, f"Move the vectors from {OTHER} into {NAMED}.")

    assert row["status"] == "unresolved"
    assert sorted(row["detail"]["candidates"]) == sorted([NAMED, OTHER])
    assert row["detail"]["rule"] == RULE_SEVERAL


def test_a_stated_project_is_recorded_as_stated(client):
    row = _record(client, f"Deploy {NAMED}.", project=OTHER, source="page")

    assert row["project"] == OTHER
    assert row["source"] == "page"
    assert row["detail"]["rule"] == RULE_STATED


def test_what_was_heard_travels_with_a_spoken_instruction(client):
    row = _record(client, f"Deploy {NAMED}.", source="voice",
                  heard=[f"Deploy {NAMED}.", "record"])

    assert row["detail"]["heard"] == [f"Deploy {NAMED}.", "record"]


def test_an_empty_instruction_is_refused(client):
    assert client.post("/v1/instructions", json={"text": ""}).status_code == 422
    assert client.post("/v1/instructions", json={"text": "x", "source": "sms"}).status_code == 422


# --- listing and reading ---------------------------------------------------------


def test_the_listing_is_newest_first_and_filters_by_status(client):
    """Mutation: order by `created_at` ascending -- red on the first id."""
    first = _record(client, f"Deploy {NAMED}.")
    second = _record(client, "Rotate the logs.")
    third = _record(client, f"Tag {OTHER}.")

    listed = client.get("/v1/instructions").json()
    assert listed["count"] == 3
    assert [row["id"] for row in listed["instructions"]] == [third["id"], second["id"], first["id"]]

    unresolved = client.get("/v1/instructions", params={"status": "unresolved"}).json()
    assert [row["id"] for row in unresolved["instructions"]] == [second["id"]]
    assert client.get("/v1/instructions", params={"status": "running"}).status_code == 422


def test_one_instruction_is_read_back_with_its_evidence(client):
    row = _record(client, f"Deploy {NAMED}.")

    found = client.get(f"/v1/instructions/{row['id']}")

    assert found.status_code == 200
    assert found.json() == row


def test_an_unknown_instruction_is_a_404(client):
    assert client.get("/v1/instructions/nobody").status_code == 404


# --- the spoken route ------------------------------------------------------------


def test_the_spoken_route_runs_the_command_in_a_process_of_its_own(client, launched):
    """Mutation: drop `"--voice"` from argv -- red."""
    response = client.post("/v1/instructions/voice", headers={"host": "localhost:3000"})

    assert response.status_code == 202, response.text
    assert response.json()["kind"] == "instruction"
    assert response.json()["request_id"] is None
    argv = launched[0].argv
    assert argv[1:5] == ["-m", "qmcp", "instruct", "--voice"]
    assert argv[argv.index("--engine") + 1] == "joe"
    # The socket the request arrived on, which is what the child must call.
    assert argv[argv.index("--base-url") + 1] == "http://testserver:80"


def test_the_status_says_how_the_dialog_ended(client, launched):
    client.post("/v1/instructions/voice")
    status = client.get("/v1/instructions/voice").json()
    assert status["running"] is True and status["kind"] == "instruction"

    launched[0].code = 0
    status = client.get("/v1/instructions/voice").json()
    assert status["running"] is False
    assert status["exit_code"] == 0
    assert status["output"] == ["  [=] abc  qmcp  voice"]


def test_one_conversation_at_a_time_across_both_kinds(client, launched):
    """There is one microphone. An approval being asked blocks an instruction,
    and an instruction being taken blocks an approval. Mutation: give the
    inbox a `VoiceRuns()` of its own in `qmcp.server` -- red on the second
    assertion."""
    client.post("/v1/human/requests", json={
        "id": "demo", "request_type": "approval", "prompt": "Launch?",
        "options": ["approve", "hold"], "timeout_seconds": 3600})
    assert client.post("/v1/human/requests/demo/voice").status_code == 202

    refused = client.post("/v1/instructions/voice")
    assert refused.status_code == 409
    assert "already running, for 'demo'" in refused.json()["detail"]
    assert client.get("/v1/instructions/voice").json()["kind"] == "approval"

    launched[0].code = 0
    assert client.post("/v1/instructions/voice").status_code == 202
    second = client.post("/v1/instructions/voice")
    assert second.status_code == 409
    assert "for 'instruction'" in second.json()["detail"]
    assert len(launched) == 2


def test_recording_starts_nothing(client, launched):
    """The whole point. Mutation: have `record_instruction` call `runs.start`
    -- red."""
    _record(client, f"Deploy {NAMED}.")
    _record(client, "Rotate the logs.")

    assert launched == []
    assert client.get("/v1/instructions/voice").json()["running"] is False


def test_the_routes_are_not_served_off_loopback(monkeypatch):
    """An instruction is a person's own words, and the spoken route makes this
    machine listen. Mutation: register the inbox outside the `is_loopback`
    branch in `qmcp.server` -- red."""
    import qmcp.server

    class OffLoopback:
        host = "0.0.0.0"
        port = 3141
        debug = False
        database_url = "sqlite+aiosqlite:///:memory:"
        log_level = "WARNING"
        voice_engine = "joe"
        voice_engine_url = None

    monkeypatch.setattr(qmcp.server, "get_settings", lambda: OffLoopback())
    app = qmcp.server.create_app()

    assert not [r for r in app.routes if "instructions" in getattr(r, "path", "")]
