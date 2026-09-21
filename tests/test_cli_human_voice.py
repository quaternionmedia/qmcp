"""Tests for `qmcp human voice`, the CLI wiring for VoiceApprovalLoop."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

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


def _install_fake_vox(monkeypatch, stt, tts) -> None:
    """`qmcp human voice` imports `vox` lazily; stand it in without installing it."""
    fake_vox = ModuleType("vox")
    fake_vox.JoeSTT = lambda *a, **k: stt
    fake_vox.Pyttsx3TTS = lambda *a, **k: tts
    monkeypatch.setitem(sys.modules, "vox", fake_vox)


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
    assert tts.spoken[0] == "Deploy?"


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


def test_human_voice_without_vox_installed_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "vox", None)  # forces ImportError on `import vox`

    result = CliRunner().invoke(cli.cli, ["human", "voice", "demo-1"])

    assert result.exit_code != 0
    assert "vox is not installed" in result.output
