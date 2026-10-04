"""`qmcp cookbook voice`: the voice HITL check, offline and live.

The offline form is itself an integration test -- a real qmcp server, vox's
deterministic engine, real HTTP on both sides -- so these tests run it and
then make sure it can fail: a check that cannot report a broken loop is a
green row standing where a reader believes something is checked.
"""

import os

from click.testing import CliRunner

import qmcp.config
import qmcp.db.engine as db_engine
from qmcp.cli import cli
from qmcp.integrations.voice import adapter
from qmcp.integrations.voice.check import (
    OFFLINE_CASES,
    Case,
    run_live,
    run_offline,
    throwaway_server,
)


def test_the_offline_check_passes_through_the_real_path():
    result = CliRunner().invoke(cli, ["cookbook", "voice"])

    assert result.exit_code == 0, result.output
    assert f"{len(OFFLINE_CASES)} of {len(OFFLINE_CASES)} ended as scripted." in result.output
    assert 'heard "banana"' in result.output and "nothing recorded" in result.output


def test_the_offline_check_fails_when_the_loop_guesses(monkeypatch):
    """A loop that records the first option for anything it cannot match is
    the failure the dialog exists to prevent. The check must say so."""
    monkeypatch.setattr(adapter, "match_option", lambda text, options: options[0])

    result = CliRunner().invoke(cli, ["cookbook", "voice"])

    assert result.exit_code == 1, result.output
    assert '[FAIL] heard "banana"' in result.output
    assert "recorded 'approve', expected None" in result.output


def test_the_offline_check_records_an_open_answer_through_the_real_path():
    """The open-question case: two transcripts over the engine contract, the
    first read back and recorded on the second. Seen red with `_ask_open`
    returning `heard.strip()` in place of the read-back: the row turns
    `[FAIL]` for the missing re-ask though the recorded answer is right."""
    result = CliRunner().invoke(cli, ["cookbook", "voice"])

    assert result.exit_code == 0, result.output
    row = '[ok]   heard "release candidate", "agree" recorded "release candidate" by vox'
    assert row in result.output


def test_the_offline_check_fails_when_an_open_answer_is_recorded_unconfirmed(monkeypatch):
    """A loop that records the first transcript without reading it back
    records what the engine misheard. The check must say so, even though
    what was recorded is the scripted answer."""
    monkeypatch.setattr(
        adapter.VoiceApprovalLoop, "_ask_open",
        lambda self, prompt: (self.tts.speak(prompt), self.stt.listen()[0])[1],
    )

    result = CliRunner().invoke(cli, ["cookbook", "voice"])

    assert result.exit_code == 1, result.output
    assert '[FAIL] heard "release candidate", "agree"' in result.output
    assert "no turn beginning 'I heard: release candidate. Say agree or again.'" in result.output


def test_the_offline_check_fails_when_the_prompt_omits_the_options(monkeypatch):
    monkeypatch.setattr(adapter, "say_options", lambda options: "")

    lines: list[str] = []
    assert run_offline(echo=lines.append) is False
    assert any("prompt was" in line for line in lines)


def test_the_configured_queue_is_left_alone(monkeypatch):
    """The process's own database engine is swapped out for the run and put
    back. A sentinel stands in for it: with no engine created beforehand, a
    missing restore and a correct one both leave None, and the check cannot
    tell them apart."""
    before_url = os.environ.get("QMCP_DATABASE_URL")
    before_setting = qmcp.config.get_settings().database_url
    sentinel = object()
    monkeypatch.setattr(db_engine, "_engine", sentinel)

    assert run_offline(echo=lambda line: None, cases=OFFLINE_CASES[:1])

    assert os.environ.get("QMCP_DATABASE_URL") == before_url
    assert qmcp.config.get_settings().database_url == before_setting
    assert db_engine._engine is sentinel


def _live(tmp_path, heard: str) -> tuple[bool, list[str], object]:
    """run_live against a throwaway server and vox's engine scripted to hear `heard`."""
    from vox import HttpSTT
    from vox.adapters.joe import JOE
    from vox.engine import EngineState
    from vox.engine import serve as serve_engine
    from vox.tts import RecordingTTS

    from qmcp.client import MCPClient

    lines: list[str] = []
    with throwaway_server(tmp_path / "queue.db") as url, MCPClient(base_url=url) as client:
        state = EngineState(audio_dirs=[tmp_path], microphone="test", heard=heard, contract=JOE)
        with serve_engine(state) as (engine_url, _), HttpSTT(engine_url, contract=JOE) as stt:
            ok = run_live(client, stt, RecordingTTS(out_dir=str(tmp_path / "spoken")),
                          echo=lines.append, max_retries=0, listen_duration=1.0)
        queued = next(line for line in lines if line.startswith("queued:")).split()[1]
        _, response = client.get_human_request(queued)
    return ok, lines, response


def test_live_records_the_spoken_option(tmp_path):
    ok, lines, response = _live(tmp_path, "approve")

    assert ok is True
    assert response.response == "approve" and response.responded_by == "vox"
    assert 'asking:  "Voice check. Say approve or hold."' in lines


def test_live_records_nothing_for_an_unclear_answer(tmp_path):
    ok, lines, response = _live(tmp_path, "banana")

    assert ok is False
    assert response is None
    assert any(line.startswith("nothing recorded") for line in lines)


def test_a_case_expecting_nothing_fails_when_something_is_recorded():
    """The verdict reads what the queue holds, not what the loop returned."""
    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=(Case("Yes.", None),)) is False
    assert "recorded 'approve', expected None" in lines[-2]
