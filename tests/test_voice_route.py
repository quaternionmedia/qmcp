"""`POST /v1/human/requests/{id}/voice` and `GET /v1/human/voice`.

No process is started and nothing speaks: the launcher is stood in for, and
it writes what `qmcp human voice` would print to the log the route reads.
"""

from datetime import datetime, timedelta

import pytest

import qmcp.integrations.voice.service as service


class _Process:
    """A launched conversation whose end the test decides."""

    def __init__(self, argv, stdout, lines):
        self.argv = argv
        self.code = None
        stdout.write("\n".join(lines) + "\n")

    def poll(self):
        return self.code


@pytest.fixture
def launched(client, monkeypatch):
    """Every process the route starts, in order."""
    started: list[_Process] = []
    lines = ["  demo answered 'approve' (by voice)."]

    def popen(argv, stdout, stderr, env):
        process = _Process(argv, stdout, lines)
        started.append(process)
        return process

    monkeypatch.setattr(client.app.state.voice_runs, "_popen", popen)
    return started


def _queue(client, request_id="demo", timeout_seconds=3600):
    response = client.post("/v1/human/requests", json={
        "id": request_id, "request_type": "approval", "prompt": "Launch?",
        "options": ["approve", "hold"], "timeout_seconds": timeout_seconds,
    })
    assert response.status_code == 201, response.text


def test_a_pending_request_is_asked_by_running_the_command(client, launched):
    _queue(client)

    # Through joe's dev proxy the Host header names the page, not this server.
    response = client.post("/v1/human/requests/demo/voice", headers={"host": "localhost:3000"})

    assert response.status_code == 202, response.text
    assert response.json()["request_id"] == "demo"
    argv = launched[0].argv
    assert argv[1:6] == ["-m", "qmcp", "human", "voice", "demo"]
    assert argv[argv.index("--engine") + 1] == "joe"
    # The socket the request arrived on, which is what the child must call.
    assert argv[argv.index("--base-url") + 1] == "http://testserver:80"


def test_the_status_says_how_the_conversation_ended(client, launched):
    _queue(client)
    client.post("/v1/human/requests/demo/voice")

    assert client.get("/v1/human/voice").json()["running"] is True

    launched[0].code = 0
    status = client.get("/v1/human/voice").json()
    assert status["running"] is False
    assert status["exit_code"] == 0
    assert status["output"] == ["  demo answered 'approve' (by voice)."]


def test_one_conversation_at_a_time(client, launched):
    _queue(client, "first")
    _queue(client, "second")
    client.post("/v1/human/requests/first/voice")

    response = client.post("/v1/human/requests/second/voice")

    assert response.status_code == 409
    assert "already running, for 'first'" in response.json()["detail"]
    assert len(launched) == 1

    launched[0].code = 0
    assert client.post("/v1/human/requests/second/voice").status_code == 202


def test_an_unknown_request_is_not_asked(client, launched):
    assert client.post("/v1/human/requests/nobody/voice").status_code == 404
    assert launched == []


def test_an_answered_request_is_not_asked_again(client, launched):
    _queue(client)
    client.post("/v1/human/responses", json={"request_id": "demo", "response": "hold"})

    response = client.post("/v1/human/requests/demo/voice")

    assert response.status_code == 409
    assert "not waiting" in response.json()["detail"]
    assert launched == []


def test_an_expired_request_is_not_asked(client, launched, monkeypatch):
    _queue(client, timeout_seconds=60)

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(hours=2)

    monkeypatch.setattr(service, "datetime", Later)
    response = client.post("/v1/human/requests/demo/voice")

    assert response.status_code == 409
    assert "expired" in response.json()["detail"]
    assert launched == []


def test_the_routes_are_not_served_off_loopback(monkeypatch):
    """Speaking and listening happen on this machine; nobody elsewhere may
    start them."""
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

    assert not [r for r in app.routes if "voice" in getattr(r, "path", "")]


def test_a_real_launch_reports_the_command_s_own_words(client):
    """The command line is a runnable command, not just a list that looks
    right. The child's preflight finds no server at the test client's address
    and stops before anything is spoken, and its words reach the status."""
    import time

    _queue(client)
    assert client.post("/v1/human/requests/demo/voice").status_code == 202

    deadline = time.monotonic() + 90
    status = client.get("/v1/human/voice").json()
    while status["running"] and time.monotonic() < deadline:
        time.sleep(0.25)
        status = client.get("/v1/human/voice").json()

    assert status["running"] is False, status
    assert status["exit_code"] != 0
    assert any("No qmcp server answers at http://testserver:80" in line
               for line in status["output"]), status["output"]
