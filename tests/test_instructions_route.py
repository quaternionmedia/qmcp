"""`/v1/instructions`: recorded, listed, read; and the spoken route, which starts
a command and never runs an instruction.

The roster is handed to the routes, so these run whatever the governance
submodule lists and whether it is checked out; `qmcp cookbook instruct` is
where the real roster is read. No process is started: the launcher is stood in
for, as `tests/test_voice_route.py` does.
"""

from __future__ import annotations

import pytest

from qmcp.instructions import RULE_NONE, RULE_ONE, RULE_SEVERAL, RULE_STATED

NAMED, OTHER = "qmcp", "dossier"
# `alpha` is on no real roster, so a route reading the checkout's instead of
# this one is told apart from one reading this one.
NAMES = (NAMED, "vox", OTHER, "alpha")


@pytest.fixture(autouse=True)
def _roster(monkeypatch):
    """The routes read the module's `roster_names` at each record, so the
    stand-in reaches them without rebuilding the app."""
    monkeypatch.setattr("qmcp.instructions.service.roster_names", lambda: NAMES)


def test_the_routes_read_the_roster_at_each_record(client):
    """Mutation: bind `roster_names` as `register`'s default argument -- red,
    the checkout's roster has no `alpha` and the row is unresolved."""
    row = _record(client, "Deploy alpha.")

    assert row["status"] == "recorded"
    assert row["project"] == "alpha"


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
    """Whitespace alone is refused with the empty string. Mutation: drop the
    validator and keep `min_length=1` -- red on the spaces, 201 unresolved."""
    assert client.post("/v1/instructions", json={"text": ""}).status_code == 422
    assert client.post("/v1/instructions", json={"text": "   "}).status_code == 422
    assert client.post("/v1/instructions", json={"text": "x", "source": "sms"}).status_code == 422


def test_a_blank_project_is_nothing_stated(client):
    """`project: ""` is a client meaning none, not a project called nothing.
    Mutation: `if project is not None` alone in `resolve` -- red, recorded
    against `''` as `stated`."""
    row = _record(client, "Rotate the logs.", project="")

    assert row["status"] == "unresolved"
    assert row["project"] is None
    assert row["detail"]["rule"] == RULE_NONE
    listed = client.get("/v1/instructions", params={"status": "unresolved"}).json()
    assert [r["id"] for r in listed["instructions"]] == [row["id"]]


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


# --- the act route ---------------------------------------------------------------


def test_the_act_route_runs_the_command_with_nothing_defaulted(client, launched):
    """Mutation: leave `--runtime` out of argv -- red; the command would refuse
    and the page would read a 202 as an act under way."""
    row = _record(client, f"Deploy {NAMED}.")

    response = client.post(f"/v1/instructions/{row['id']}/act",
                           json={"runtime": "scripted", "budget": 1, "cwd": "/work/qmcp"},
                           headers={"host": "localhost:3000"})

    assert response.status_code == 202, response.text
    assert response.json()["kind"] == "act"
    assert response.json()["request_id"] == row["id"]
    argv = launched[0].argv
    assert argv[1:6] == ["-m", "qmcp", "instructions", "act", row["id"]]
    assert argv[argv.index("--runtime") + 1] == "scripted"
    assert argv[argv.index("--budget") + 1] == "1"
    assert argv[argv.index("--cwd") + 1] == "/work/qmcp"
    assert argv[argv.index("--base-url") + 1] == "http://testserver:80"
    assert "--voice" not in argv
    assert client.get("/v1/instructions/voice").json()["kind"] == "act"


def test_the_act_route_asks_by_voice_only_when_told(client, launched):
    row = _record(client, f"Deploy {NAMED}.")

    client.post(f"/v1/instructions/{row['id']}/act", json={"runtime": "scripted", "voice": True})

    argv = launched[0].argv
    assert "--voice" in argv and argv[argv.index("--engine") + 1] == "joe"
    assert argv[argv.index("--budget") + 1] == "0"


def test_the_act_route_requires_a_runtime_and_a_row(client, launched):
    """A runtime the registry does not name is refused here, before anything
    starts: the command would refuse it too, but after a 202 a page reads as
    an act under way. Mutation: drop the `runtime` validator -- red on
    `nobody`, 202 and a process."""
    row = _record(client, f"Deploy {NAMED}.")

    assert client.post(f"/v1/instructions/{row['id']}/act", json={}).status_code == 422
    assert client.post(f"/v1/instructions/{row['id']}/act",
                       json={"runtime": "scripted", "budget": -1}).status_code == 422
    for name in ("nobody", " "):
        refused = client.post(f"/v1/instructions/{row['id']}/act", json={"runtime": name})
        assert refused.status_code == 422, refused.text
        assert "is not a runtime" in refused.text and "scripted" in refused.text
    assert client.post("/v1/instructions/nobody/act", json={"runtime": "scripted"}).status_code == 404
    assert launched == []


def test_a_blank_cwd_is_no_clone(client, launched):
    """A page that sends the field empty has named no clone. Mutation: drop
    the `cwd` validator -- red, `--cwd` is passed with spaces."""
    row = _record(client, f"Deploy {NAMED}.")

    response = client.post(f"/v1/instructions/{row['id']}/act",
                           json={"runtime": "scripted", "cwd": "   "})

    assert response.status_code == 202, response.text
    assert "--cwd" not in launched[0].argv


def test_an_act_is_one_at_a_time_with_the_conversations(client, launched):
    """Mutation: give the act route a tracker of its own -- red on the 409."""
    row = _record(client, f"Deploy {NAMED}.")
    assert client.post(f"/v1/instructions/{row['id']}/act", json={"runtime": "scripted"}).status_code == 202

    again = client.post(f"/v1/instructions/{row['id']}/act", json={"runtime": "scripted"})
    spoken = client.post("/v1/instructions/voice")

    assert again.status_code == 409 and f"for '{row['id']}'" in again.json()["detail"]
    assert spoken.status_code == 409
    launched[0].code = 0
    assert client.post("/v1/instructions/voice").status_code == 202


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


def test_both_spoken_routes_hand_the_engine_url_on_when_one_is_configured(client, tmp_path):
    """The routes as `qmcp serve` registers them with `QMCP_VOICE_ENGINE_URL`
    set, over the fixture's database. Mutation: drop either `--engine-url`
    branch -- red, that child talks to the adapter's default port rather than
    the engine the server was told about."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from qmcp.instructions.service import register
    from qmcp.integrations.voice.service import VoiceRuns

    started: list[_Process] = []

    def popen(argv, stdout, stderr, env):
        started.append(_Process(argv, stdout, []))
        started[-1].code = 0  # finished at once, so the second route may start
        return started[-1]

    app = FastAPI()
    register(app, VoiceRuns(log_dir=tmp_path, popen=popen), engine="joe",
             engine_url="http://127.0.0.1:8001", names=lambda: NAMES)
    served = TestClient(app)
    row = _record(served, f"Deploy {NAMED}.")

    assert served.post("/v1/instructions/voice").status_code == 202
    assert served.post(f"/v1/instructions/{row['id']}/act",
                       json={"runtime": "scripted", "voice": True}).status_code == 202

    told = [run.argv[run.argv.index("--engine-url") + 1] for run in started]
    assert told == ["http://127.0.0.1:8001", "http://127.0.0.1:8001"]
