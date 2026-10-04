"""`qmcp serve --converse`: the conversation started with the server through the
voice tracker, its command line, its end at shutdown, and the two commands.

No process is launched: the tracker is given a stand-in for `Popen`, and the
server command a stand-in for running the server.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import qmcp.cli as cli
from qmcp.integrations.voice.service import VoiceRuns
from qmcp.server import conversation_argv, start_conversation


def _settings(**over):
    base = dict(converse_runtime="local", converse_clones=None, converse_wake=None,
                converse_synth="pyttsx3", host="127.0.0.1", port=3141, voice_engine="joe",
                voice_engine_url=None)
    return SimpleNamespace(**{**base, **over})


class _Process:
    def __init__(self):
        self.terminated = self.killed = False
        self.code = None

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("converse", timeout)

    def kill(self):
        self.killed = True
        self.code = -9


def test_the_command_line_is_converse_against_this_server_and_the_engine():
    """Mutation: drop `--synth` -- red, a check run on an empty machine would speak."""
    argv = conversation_argv(_settings(converse_clones="/repos", converse_wake="computer",
                                       converse_synth="recording",
                                       voice_engine_url="http://127.0.0.1:8001"))

    assert argv[1:4] == ["-m", "qmcp", "converse"]
    for pair in (["--runtime", "local"], ["--base-url", "http://127.0.0.1:3141"],
                 ["--engine", "joe"], ["--engine-url", "http://127.0.0.1:8001"],
                 ["--clones", "/repos"], ["--wake", "computer"], ["--synth", "recording"]):
        i = argv.index(pair[0])
        assert argv[i:i + 2] == pair


def test_the_conversation_starts_through_the_tracker_the_buttons_share(tmp_path):
    """Mutation: start it with a process of its own -- red, the page's buttons
    could open the microphone a second time."""
    launched: list[list[str]] = []

    def popen(argv, **kw):
        launched.append(argv)
        return _Process()

    runs = VoiceRuns(log_dir=tmp_path, popen=popen)
    app = SimpleNamespace(state=SimpleNamespace(voice_runs=runs))

    assert start_conversation(app, _settings()) is True
    assert "converse" in launched[0]
    assert runs.status()["kind"] == "conversation" and runs.running() == "conversation"
    with pytest.raises(RuntimeError, match="conversation"):
        runs.start(["anything"], kind="approval")


def test_no_conversation_without_a_runtime_or_off_loopback(tmp_path):
    runs = VoiceRuns(log_dir=tmp_path, popen=lambda argv, **kw: _Process())

    assert start_conversation(SimpleNamespace(state=SimpleNamespace(voice_runs=runs)),
                              _settings(converse_runtime=None)) is False
    assert start_conversation(SimpleNamespace(state=SimpleNamespace()), _settings()) is False
    assert runs.status()["running"] is False


def test_stop_asks_then_makes_the_conversation_end(tmp_path):
    """Mutation: terminate without the kill fallback -- red, a conversation
    that ignores the request outlives the server."""
    process = _Process()
    runs = VoiceRuns(log_dir=tmp_path, popen=lambda argv, **kw: process)
    runs.start(["converse"], kind="conversation")

    runs.stop(timeout=0.01)

    assert process.terminated and process.killed
    runs.stop()  # nothing running: nothing to do


def test_the_server_ends_its_conversation_when_it_stops(monkeypatch):
    """The lifespan starts it after the database and stops it on shutdown.
    Mutation: drop the stop at shutdown -- red."""
    import uuid

    from fastapi.testclient import TestClient

    import qmcp.db.engine
    import qmcp.server as server

    # The database the shared `client` fixture uses: in memory, and this test's own.
    memory = SimpleNamespace(database_url=f"sqlite+aiosqlite:///file:converse_{uuid.uuid4().hex}"
                                          "?mode=memory&cache=shared&uri=true", debug=False)
    monkeypatch.setattr(qmcp.db.engine, "_engine", None)
    monkeypatch.setattr(qmcp.db.engine, "get_settings", lambda: memory)
    stopped: list[bool] = []
    started: list = []

    def start(app, settings):
        started.append((app, settings))
        return True

    monkeypatch.setattr(server, "start_conversation", start)
    app = server.create_app()
    app.state.voice_runs = SimpleNamespace(stop=lambda: stopped.append(True))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        # Started once, at startup, with the running app; not yet stopped.
        assert [a for a, _ in started] == [app]
        assert stopped == []

    assert stopped == [True]


# --- the commands ------------------------------------------------------------------------


def test_serve_converse_needs_a_runtime_and_hands_the_server_its_settings(monkeypatch):
    """Mutation: skip the cache clear -- red, the server reads the settings
    from before the flag."""
    from qmcp.config import get_settings

    # The command writes these into the process environment, as it must for the
    # server it starts; set-then-delete registers each so the test's teardown
    # removes what the command wrote, and later tests build servers without a
    # conversation.
    for name in ("QMCP_CONVERSE_RUNTIME", "QMCP_CONVERSE_CLONES", "QMCP_CONVERSE_WAKE",
                 "QMCP_AGENT_RUNTIME"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    seen: list = []
    monkeypatch.setattr(cli, "_run_server",
                        lambda host, port, reload: seen.append(get_settings()))

    missing = CliRunner().invoke(cli.cli, ["serve", "--converse"])
    unknown = CliRunner().invoke(cli.cli, ["serve", "--converse", "--runtime", "nobody"])
    alone = CliRunner().invoke(cli.cli, ["serve", "--runtime", "local"])
    ok = CliRunner().invoke(cli.cli, ["serve", "--converse", "--runtime", "scripted",
                                      "--wake", "computer"])

    assert missing.exit_code != 0 and "--converse needs --runtime" in missing.output
    assert unknown.exit_code != 0 and "'nobody' is not a runtime" in unknown.output
    assert alone.exit_code != 0 and "read only with --converse" in alone.output
    assert ok.exit_code == 0, ok.output
    assert seen[-1].converse_runtime == "scripted" and seen[-1].converse_wake == "computer"
    get_settings.cache_clear()


def test_converse_needs_a_runtime(monkeypatch):
    monkeypatch.delenv("QMCP_AGENT_RUNTIME", raising=False)

    result = CliRunner().invoke(cli.cli, ["converse"])

    assert result.exit_code != 0 and "--runtime is required" in result.output


def test_converse_waits_for_the_servers_says_what_is_missing_and_talks(monkeypatch):
    """The command's own wiring: the engine URL it talks to, the synthesizer
    `--synth` picks, the wait, a runtime that is not ready said aloud, and the
    conversation run. Mutation: ignore `--synth recording` -- red, a check
    would speak through the speakers."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock

    from qmcp.instructions import converse as converse_module
    from qmcp.integrations.agents.scripted import ScriptedRuntime

    made: dict = {}

    class STT:
        def __init__(self, url, contract=None):
            made["engine"] = url

        def close(self):
            made["closed"] = True

    class Recording:
        def __init__(self, *a, **k):
            made["synth"] = "recording"

    class Aloud:
        def __init__(self, *a, **k):
            made["synth"] = "aloud"

    fake_vox, fake_tts = ModuleType("vox"), ModuleType("vox.tts")
    fake_vox.HttpSTT, fake_tts.RecordingTTS = STT, Recording
    fake_joe = ModuleType("vox.adapters.joe")
    fake_joe.JOE, fake_joe.DEFAULT_URL = MagicMock(name="EngineContract"), "http://127.0.0.1:8000"
    fake_pyttsx3 = ModuleType("vox.adapters.pyttsx3")
    fake_pyttsx3.Pyttsx3TTS = Aloud
    for name, module in [("vox", fake_vox), ("vox.tts", fake_tts),
                         ("vox.adapters", ModuleType("vox.adapters")),
                         ("vox.adapters.joe", fake_joe), ("vox.adapters.pyttsx3", fake_pyttsx3)]:
        monkeypatch.setitem(sys.modules, name, module)

    class Unready(ScriptedRuntime):
        def ready(self):
            return "the model is not served"

    said: list[str] = []
    monkeypatch.setattr("qmcp.integrations.agents.runtime_named", lambda name, **kw: Unready())
    monkeypatch.setattr(converse_module, "wait_for",
                        lambda client, stt, echo: made.setdefault("waited", True))
    monkeypatch.setattr(converse_module.Conversation, "say", lambda self, text: said.append(text))
    monkeypatch.setattr(converse_module.Conversation, "run",
                        lambda self: converse_module.Ended(reason="told to stop"))
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: MagicMock())

    result = CliRunner().invoke(cli.cli, ["converse", "--runtime", "local", "--synth",
                                          "recording", "--engine-url", "http://127.0.0.1:8001"])

    assert result.exit_code == 0, result.output
    assert made == {"engine": "http://127.0.0.1:8001", "synth": "recording", "waited": True,
                    "closed": True}
    assert said == ["The local runtime is not ready: the model is not served."]
    assert "ended: told to stop; 0 instruction(s), 0 question(s) answered" in result.output
