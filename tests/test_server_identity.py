"""A server says which one it is, on /health and in a run-file, and binds port 0 on request.

`qmcp/instance.py` holds the corpus's monitoring-seam decisions 4 and 5: two
clones answer `/health` alike unless each says which it is, and a session
that dials a port it did not start must be able to tell. Each mutation named
below was seen red.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

import qmcp.cli as cli
import qmcp.config
import qmcp.db.engine
import qmcp.server
from qmcp.config import Settings
from qmcp.instance import database_fingerprint


def test_health_says_which_server_this_is(client) -> None:
    # Mutation seen red: the identity left out of /health -- the port and the
    # fingerprint were missing.
    body = client.get("/health").json()

    assert body["status"] == "healthy"
    assert body["port"] == 3333
    assert body["pid"] == os.getpid()
    assert body["database_fingerprint"] == database_fingerprint(
        qmcp.server.get_settings().database_url)
    assert body["started_at"]
    assert "database" not in body and "host" not in body


def test_one_database_has_one_fingerprint_however_it_is_named(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    relative = database_fingerprint("sqlite+aiosqlite:///./queue.db")
    absolute = database_fingerprint(f"sqlite+aiosqlite:///{(tmp_path / 'queue.db').as_posix()}")

    assert relative == absolute
    assert relative != database_fingerprint("sqlite+aiosqlite:///./another.db")


def _serving(monkeypatch, settings: Settings) -> qmcp.server.FastAPI:
    import qmcp.db.engine as engine

    engine._engine = None
    for module in (qmcp.config, qmcp.db.engine, qmcp.server):
        monkeypatch.setattr(module, "get_settings", lambda: settings)
    return qmcp.server.create_app()


def test_a_serving_process_leaves_a_run_file_and_takes_it_away(tmp_path, monkeypatch) -> None:
    # Mutation seen red: `remove_run_file` dropped from the shutdown -- the file
    # outlived the server.
    run_dir = tmp_path / "run"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{(tmp_path / 'q.db').as_posix()}",
                        port=4711, run_file=True, run_dir=str(run_dir), session="review")
    app = _serving(monkeypatch, settings)

    with TestClient(app) as served:
        (run_file,) = run_dir.glob("qmcp-*.json")
        recorded = json.loads(run_file.read_text(encoding="utf-8"))
        stated = served.get("/health").json()

    assert not run_file.exists()
    assert recorded["database"] == str((tmp_path / "q.db").resolve())
    for key in ("pid", "port", "started_at", "database_fingerprint", "session"):
        assert recorded[key] == stated[key], key
    assert (recorded["port"], recorded["session"]) == (4711, "review")
    qmcp.db.engine._engine = None


def test_a_server_not_started_by_serve_leaves_no_run_file(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    settings = Settings(database_url=f"sqlite+aiosqlite:///{(tmp_path / 'q.db').as_posix()}",
                        run_dir=str(run_dir))
    with TestClient(_serving(monkeypatch, settings)):
        pass

    assert not run_dir.exists()
    qmcp.db.engine._engine = None


@pytest.fixture
def serve_environment(monkeypatch):
    """`_run_server` writes these to os.environ; registering them first means
    monkeypatch puts back whatever was there."""
    for name, value in (("QMCP_HOST", "127.0.0.1"), ("QMCP_PORT", "3141"),
                        ("QMCP_RUN_FILE", "0")):
        monkeypatch.setenv(name, value)


def test_port_zero_binds_a_port_the_system_picks_and_tells_the_app(
        monkeypatch, serve_environment) -> None:
    # Mutation seen red: `port or settings.port` again -- port 0 read as unset,
    # and the server was handed 3141.
    served: dict[str, object] = {}

    def run(self, sockets=None):
        served["port"] = sockets[0].getsockname()[1]
        sockets[0].close()

    monkeypatch.setattr(cli.uvicorn.Server, "run", run)

    result = CliRunner().invoke(cli.cli, ["serve", "--port", "0"])

    assert result.exit_code == 0, result.output
    assert served["port"] not in (0, 3141)
    assert os.environ["QMCP_PORT"] == str(served["port"])
    assert os.environ["QMCP_RUN_FILE"] == "1"
    assert f"Starting QMCP server on 127.0.0.1:{served['port']}" in result.output


def test_port_zero_with_reload_is_refused(serve_environment) -> None:
    result = CliRunner().invoke(cli.cli, ["serve", "--port", "0", "--reload"])

    assert result.exit_code == 2
    assert "--reload restarts on a fixed port" in result.output


@pytest.mark.parametrize(("stated", "why"), [
    ({"database_fingerprint": "0000000000000000"}, "it serves another database"),
    ({}, "it does not say which database it serves"),
])
def test_a_server_over_another_database_is_not_borrowed(monkeypatch, stated, why) -> None:
    # Mutation seen red: `_foreign_server` returning None -- the command used a
    # server whose database it had not checked.
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json={"status": "healthy", **stated}))

    assert cli._foreign_server("http://127.0.0.1:1/health") == why


def test_this_checkouts_own_server_is_used(monkeypatch) -> None:
    own = database_fingerprint(cli.get_settings().database_url)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(
        200, json={"status": "healthy", "database_fingerprint": own}))

    assert cli._foreign_server("http://127.0.0.1:1/health") is None
