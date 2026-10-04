"""Tests for `qmcp human voice`, the CLI wiring for VoiceApprovalLoop."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import qmcp.cli as cli
from qmcp.client.mcp_client import HumanRequest


class _FakeSTT:
    """Returns each transcript in order, repeating the last once exhausted."""

    def __init__(self, transcripts: list[str]):
        self._transcripts = list(transcripts)
        self.calls = 0

    def listen(self, duration: float = 5.0) -> tuple[str, str]:
        text = self._transcripts[min(self.calls, len(self._transcripts) - 1)]
        self.calls += 1
        return text, f"capture_{self.calls}.wav"


class _FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str, out_path: str | None = None) -> str:
        self.spoken.append(text)
        return out_path or "spoken.wav"


# Captured before the autouse stub below replaces it for the dialog tests.
_REAL_PREFLIGHT = cli._voice_preflight


@pytest.fixture(autouse=True)
def _servers_up(monkeypatch):
    """The dialog tests stand in for both servers; the preflight has its own tests."""
    monkeypatch.setattr(cli, "_voice_preflight", lambda *a, **k: None)


def _install_fake_vox(monkeypatch, stt, tts) -> None:
    """`qmcp human voice` imports `vox` lazily; stand it in without installing it.

    Three modules, because vox states the contract in one place and names the
    engine in another: the client and the adapters are separate imports, and
    standing in for only one of them would not exercise the lookup the CLI
    actually performs.
    """
    fake_vox = ModuleType("vox")
    fake_vox.HttpSTT = lambda *a, **k: stt

    fake_contract = MagicMock(name="EngineContract")
    fake_adapters = ModuleType("vox.adapters")
    fake_adapters.JOE = fake_contract

    fake_joe = ModuleType("vox.adapters.joe")
    fake_joe.JOE = fake_contract
    fake_joe.DEFAULT_URL = "http://127.0.0.1:8000"

    fake_pyttsx3 = ModuleType("vox.adapters.pyttsx3")
    fake_pyttsx3.Pyttsx3TTS = lambda *a, **k: tts

    for name, module in [
        ("vox", fake_vox),
        ("vox.adapters", fake_adapters),
        ("vox.adapters.joe", fake_joe),
        ("vox.adapters.pyttsx3", fake_pyttsx3),
    ]:
        monkeypatch.setitem(sys.modules, name, module)


def _pending_request(request_id: str = "demo-1") -> HumanRequest:
    return HumanRequest(
        id=request_id,
        request_type="approval",
        prompt="Deploy?",
        status="pending",
        created_at="now",
        options=["approve", "reject"],
    )


def test_human_voice_answers_the_given_request(monkeypatch):
    stt = _FakeSTT(["yes"])
    tts = _FakeTTS()
    _install_fake_vox(monkeypatch, stt, tts)

    request = _pending_request()
    fake_client = MagicMock()
    fake_client.get_human_request.return_value = (request, None)
    fake_client.submit_human_response.return_value = MagicMock(response="approve")
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: fake_client)

    result = CliRunner().invoke(cli.cli, ["human", "voice", "demo-1"])

    assert result.exit_code == 0, result.output
    assert "demo-1 answered 'approve' (by voice)" in result.output
    fake_client.submit_human_response.assert_called_once_with(
        request_id="demo-1", response="approve", responded_by="vox"
    )
    assert tts.spoken[0] == "Deploy? Approve or reject?"


def test_human_voice_without_request_id_uses_oldest_pending(monkeypatch):
    stt = _FakeSTT(["no"])
    tts = _FakeTTS()
    _install_fake_vox(monkeypatch, stt, tts)

    request = _pending_request("demo-3")
    fake_client = MagicMock()
    fake_client.list_human_requests.return_value = [request]
    fake_client.get_human_request.return_value = (request, None)
    fake_client.submit_human_response.return_value = MagicMock(response="reject")
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: fake_client)

    result = CliRunner().invoke(cli.cli, ["human", "voice"])

    assert result.exit_code == 0, result.output
    fake_client.get_human_request.assert_called_once_with("demo-3")
    assert "answered 'reject'" in result.output
    # The server lists newest first unless asked; "oldest" has to be asked for.
    assert fake_client.list_human_requests.call_args.kwargs["oldest_first"] is True


def test_human_voice_without_request_id_and_nothing_pending(monkeypatch):
    stt = _FakeSTT(["yes"])
    tts = _FakeTTS()
    _install_fake_vox(monkeypatch, stt, tts)

    fake_client = MagicMock()
    fake_client.list_human_requests.return_value = []
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: fake_client)

    result = CliRunner().invoke(cli.cli, ["human", "voice"])

    assert result.exit_code == 0, result.output
    assert "Nothing is waiting" in result.output
    assert stt.calls == 0


def test_human_voice_forever_answers_then_stops_on_interrupt(monkeypatch):
    stt = _FakeSTT(["yes"])
    tts = _FakeTTS()
    _install_fake_vox(monkeypatch, stt, tts)

    request = _pending_request("demo-2")
    fake_client = MagicMock()
    # one pending request, then none left -- the empty poll is what triggers the
    # sleep that we interrupt below, so run_forever doesn't spin for real.
    fake_client.list_human_requests.side_effect = [[request], []]
    fake_client.get_human_request.return_value = (request, None)
    fake_client.submit_human_response.return_value = MagicMock(response="approve")
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        "qmcp.integrations.voice.adapter.time.sleep", MagicMock(side_effect=KeyboardInterrupt)
    )

    result = CliRunner().invoke(cli.cli, ["human", "voice", "--forever"])

    assert result.exit_code == 0, result.output
    assert "Stopped." in result.output
    fake_client.submit_human_response.assert_called_once()


def test_human_voice_forever_names_what_it_asked_and_left_pending(monkeypatch):
    """Nobody answered: the request is asked once, left pending, and named
    when the loop stops, so the person returning knows what went unheard."""
    stt = _FakeSTT([""])
    tts = _FakeTTS()
    _install_fake_vox(monkeypatch, stt, tts)

    request = _pending_request("demo-4")
    fake_client = MagicMock()
    # Still pending on the second look; the loop passes over it and sleeps.
    fake_client.list_human_requests.side_effect = [[request], [request]]
    fake_client.get_human_request.return_value = (request, None)
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: fake_client)
    monkeypatch.setattr(
        "qmcp.integrations.voice.adapter.time.sleep", MagicMock(side_effect=KeyboardInterrupt)
    )

    result = CliRunner().invoke(cli.cli, ["human", "voice", "--forever"])

    assert result.exit_code == 0, result.output
    assert "still pending: demo-4" in result.output
    fake_client.submit_human_response.assert_not_called()
    assert sum(s.startswith("Deploy?") for s in tts.spoken) == 1


def test_human_voice_without_vox_installed_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "vox", None)  # forces ImportError on `import vox`

    result = CliRunner().invoke(cli.cli, ["human", "voice", "demo-1"])

    assert result.exit_code != 0
    assert "vox is not importable" in result.output
    assert "git submodule update --init vendor/vox" in result.output
    # The one declared form, and what a failed sync means while a server runs.
    assert "`uv run qmcp human voice` again" in result.output
    assert "os error 32" in result.output


def test_serve_refuses_to_race_a_healthy_server(monkeypatch):
    """A port already answering as qmcp is information, not a bind traceback."""
    monkeypatch.setattr(cli, "_qmcp_already_serving", lambda host, port: "0.1.0")
    started = MagicMock()
    monkeypatch.setattr(cli.uvicorn, "run", started)

    result = CliRunner().invoke(cli.cli, ["serve"])

    assert result.exit_code != 0
    assert "already serves" in result.output
    assert "second server is not needed" in result.output
    started.assert_not_called()


def test_serve_starts_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(cli, "_qmcp_already_serving", lambda host, port: None)
    started = MagicMock()
    monkeypatch.setattr(cli.uvicorn, "run", started)

    result = CliRunner().invoke(cli.cli, ["serve"])

    assert result.exit_code == 0
    started.assert_called_once()


def test_serve_started_the_declared_way_names_no_other_way(monkeypatch):
    """`uv run qmcp serve` is the declared form. It used to print a note sending
    the person to a second one, and the second one was then typed wrongly --
    `python qmcp serve` -- which is how the standard-library shadowing that
    `test_entry_points.py` guards was found."""
    monkeypatch.setattr(cli, "_qmcp_already_serving", lambda host, port: None)
    monkeypatch.setattr(cli.uvicorn, "run", MagicMock())
    monkeypatch.setattr(cli.sys, "argv", [r"C:\x\.venv\Scripts\qmcp.exe", "serve"])

    result = CliRunner().invoke(cli.cli, ["serve"])

    assert result.exit_code == 0
    assert "python -m" not in result.output


def test_human_voice_names_a_shadowing_directory(monkeypatch):
    """The failure that recurred: vox resolving to a bare directory. The CLI
    names the directory rather than recommending a reinstall that cannot help."""
    shadow = ModuleType("vox")
    shadow.__path__ = ["C:/somewhere/qmcp/vox"]  # a namespace package has no __file__
    monkeypatch.setitem(sys.modules, "vox", shadow)

    result = CliRunner().invoke(cli.cli, ["human", "voice", "demo-1"])

    assert result.exit_code != 0
    assert "vox is shadowed" in result.output
    assert "C:/somewhere/qmcp/vox" in result.output


# --- the preflight: both servers up before anything is spoken -----------------


def _contract():
    from vox.contract import EngineContract

    return EngineContract()


def _refuse_urls(monkeypatch, down: str):
    import httpx

    def fake_get(url, timeout=None):
        if down in url:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)


def test_preflight_names_the_command_when_qmcp_is_down(monkeypatch):
    _refuse_urls(monkeypatch, "3141")
    with pytest.raises(SystemExit) as exc:
        _REAL_PREFLIGHT("http://127.0.0.1:3141", "http://127.0.0.1:8000", _contract(), "joe")
    assert "No qmcp server answers at http://127.0.0.1:3141" in str(exc.value)
    assert "uv run qmcp serve" in str(exc.value)


def test_preflight_names_the_command_when_the_engine_is_down(monkeypatch):
    _refuse_urls(monkeypatch, "8000")
    with pytest.raises(SystemExit) as exc:
        _REAL_PREFLIGHT("http://127.0.0.1:3141", "http://127.0.0.1:8000", _contract(), "joe")
    assert "No speech engine answers at http://127.0.0.1:8000" in str(exc.value)
    assert "uv run joe voice setup" in str(exc.value)
    assert "uv run joe backend" in str(exc.value)


def test_preflight_passes_when_both_answer(monkeypatch):
    _refuse_urls(monkeypatch, "nowhere")
    _REAL_PREFLIGHT("http://127.0.0.1:3141", "http://127.0.0.1:8000", _contract(), "joe")
