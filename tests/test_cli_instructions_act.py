"""`qmcp instructions act`: the runtime is required, the inbox is written, and
`--voice` answers the consent in the command.

The queue is stood in for as `tests/test_cli_instruct.py` stands in for the
client; the inbox is a database made for the test, reached through the
module's `configured_rows`; the runtime is `scripted`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import qmcp.cli as cli
from qmcp.db.models import Instruction, InstructionSource, InstructionStatus
from qmcp.instructions import act as act_module
from qmcp.instructions.act import rows_at
from tests.test_instructions_act import _Queue, _STT, _TTS


@pytest.fixture
def inbox(tmp_path, monkeypatch):
    rows = rows_at(tmp_path / "inbox.db")
    monkeypatch.setattr(act_module, "configured_rows", lambda: rows)
    monkeypatch.setattr(act_module, "archive_sources", lambda: [])
    with rows() as session:
        row = Instruction(id="row-1", text="Deploy qmcp to the pi.", project="qmcp",
                          source=InstructionSource.TYPED, status=InstructionStatus.RECORDED)
        session.add(row)
        session.commit()
    return rows


@pytest.fixture
def queue(monkeypatch):
    queue = _Queue()
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: queue)
    monkeypatch.setattr(cli, "_voice_preflight", lambda *a, **k: None)
    # The command sleeps between listings; the stand-in answers instead.
    monkeypatch.setattr(act_module, "_sleep", queue.settle)
    return queue


@pytest.fixture
def clone(tmp_path):
    path = tmp_path / "qmcp"
    path.mkdir()
    return path


def _status(rows, instruction_id="row-1"):
    with rows() as session:
        return session.get(Instruction, instruction_id).status


def test_the_runtime_is_required_and_says_what_exists(inbox, queue, monkeypatch):
    """Mutation: default `--runtime` to `scripted` -- red; the command would
    report a run that did nothing."""
    monkeypatch.delenv("QMCP_AGENT_RUNTIME", raising=False)

    result = CliRunner().invoke(cli.cli, ["instructions", "act", "row-1"])

    assert result.exit_code != 0
    assert "--runtime is required" in result.output
    assert "scripted" in result.output and "QMCP_AGENT_RUNTIME" in result.output
    assert queue.requests == {} and _status(inbox) == InstructionStatus.RECORDED


def test_an_unknown_runtime_is_refused(inbox, queue):
    result = CliRunner().invoke(cli.cli, ["instructions", "act", "row-1", "--runtime", "nobody"])

    assert result.exit_code != 0 and "'nobody' is not a runtime" in result.output


def test_the_environment_can_name_the_runtime(inbox, queue, clone, monkeypatch):
    monkeypatch.setenv("QMCP_AGENT_RUNTIME", "scripted")

    result = CliRunner().invoke(cli.cli, ["instructions", "act", "row-1", "--cwd", str(clone)])

    assert result.exit_code == 0, result.output
    assert "Nothing was spent" in result.output and "--budget 1" in result.output
    assert queue.requests == {}


def test_a_consented_act_runs_and_prints_the_record(inbox, queue, clone):
    queue.script = {"instruction-row-1": "approve"}

    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1", "--cwd", str(clone)])

    assert result.exit_code == 0, result.output
    assert "row-1  done  stages: instruction > clone > budget > ask > answer > run > record" in result.output
    assert "consent: instruction-row-1  answered: 'approve'" in result.output
    assert "[asking] instruction-row-1" in result.output and "[acting]" in result.output
    assert "1 of 1 authorised" in result.output
    assert _status(inbox) == InstructionStatus.DONE


def test_a_held_act_runs_nothing_and_exits_clean(inbox, queue, clone):
    queue.script = {"instruction-row-1": "hold"}

    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1", "--cwd", str(clone)])

    assert result.exit_code == 0, result.output
    assert "row-1  refused" in result.output and "nothing ran" in result.output
    assert _status(inbox) == InstructionStatus.REFUSED


def test_a_failed_run_exits_non_zero(inbox, queue, clone, monkeypatch):
    from qmcp.integrations.agents.scripted import ScriptedRuntime

    monkeypatch.setattr("qmcp.integrations.agents.runtime_named",
                        lambda name, **kw: ScriptedRuntime(text="boom", exit_code=3))
    queue.script = {"instruction-row-1": "approve"}

    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1", "--cwd", str(clone)])

    assert result.exit_code == 1, result.output
    assert "row-1  failed" in result.output and "exit 3" in result.output


def test_an_unknown_instruction_points_at_the_list(inbox, queue, clone):
    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "nobody", "--runtime", "scripted", "--cwd", str(clone)])

    assert result.exit_code != 0
    assert "not found" in result.output and "instructions list" in result.output


def test_voice_answers_the_consent_in_the_command(inbox, queue, clone, monkeypatch):
    """The same loop `qmcp human voice` runs, on the request just created.
    Mutation: drop `stt`/`tts` from the `act` call -- red, the consent is
    never spoken and expires."""
    stt, tts = _STT("approve"), _TTS()
    fake_vox = ModuleType("vox")
    fake_vox.HttpSTT = lambda *a, **k: stt
    fake_joe = ModuleType("vox.adapters.joe")
    fake_joe.JOE = MagicMock(name="EngineContract")
    fake_joe.DEFAULT_URL = "http://127.0.0.1:8000"
    fake_pyttsx3 = ModuleType("vox.adapters.pyttsx3")
    fake_pyttsx3.Pyttsx3TTS = lambda *a, **k: tts
    for name, module in [("vox", fake_vox), ("vox.adapters", ModuleType("vox.adapters")),
                         ("vox.adapters.joe", fake_joe), ("vox.adapters.pyttsx3", fake_pyttsx3)]:
        monkeypatch.setitem(sys.modules, name, module)
    queue.script = {"instruction-row-1": None}  # nobody else answers

    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1",
        "--cwd", str(clone), "--voice"])

    assert result.exit_code == 0, result.output
    assert tts.spoken[0].startswith("Act on the instruction: Deploy qmcp to the pi.")
    assert tts.spoken[0].endswith("Say approve or hold.")
    assert "answered: 'approve'" in result.output
    assert _status(inbox) == InstructionStatus.DONE


def _fake_vox(monkeypatch, stt, tts):
    fake_vox = ModuleType("vox")
    fake_vox.HttpSTT = lambda *a, **k: stt
    fake_joe = ModuleType("vox.adapters.joe")
    fake_joe.JOE = MagicMock(name="EngineContract")
    fake_joe.DEFAULT_URL = "http://127.0.0.1:8000"
    fake_pyttsx3 = ModuleType("vox.adapters.pyttsx3")
    fake_pyttsx3.Pyttsx3TTS = lambda *a, **k: tts
    for name, module in [("vox", fake_vox), ("vox.adapters", ModuleType("vox.adapters")),
                         ("vox.adapters.joe", fake_joe), ("vox.adapters.pyttsx3", fake_pyttsx3)]:
        monkeypatch.setitem(sys.modules, name, module)


def test_with_voice_the_run_is_announced_and_its_outcome_said_last(inbox, queue, clone, monkeypatch):
    """Mutation: drop the closing `say` -- red, the last thing said is the
    running line; drop the `acting` branch of `show` -- red, approve is met
    with silence until the run ends."""
    stt, tts = _STT("approve"), _TTS()
    _fake_vox(monkeypatch, stt, tts)
    queue.script = {"instruction-row-1": None}

    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1",
        "--cwd", str(clone), "--voice"])

    assert result.exit_code == 0, result.output
    assert "Approved. Running in qmcp." in tts.spoken
    assert tts.spoken[-1] == "Done in qmcp. done, as scripted."
    assert "said: Done in qmcp. done, as scripted." in result.output


def test_without_voice_the_summary_is_printed_and_nothing_is_said(inbox, queue, clone):
    queue.script = {"instruction-row-1": "hold"}

    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1", "--cwd", str(clone)])

    assert result.exit_code == 0, result.output
    assert "summary: Held. Nothing ran for: Deploy qmcp to the pi." in result.output


def test_a_refusal_before_the_ask_is_summarised_as_nothing_ran(inbox, queue, tmp_path):
    """Mutation: drop `why=done.why` -- red, the row reads as merely recorded."""
    result = CliRunner().invoke(cli.cli, [
        "instructions", "act", "row-1", "--runtime", "scripted", "--budget", "1",
        "--cwd", str(tmp_path / "nowhere")])

    assert result.exit_code == 0, result.output
    assert "summary: Nothing was asked and nothing ran in qmcp." in result.output


def test_say_prints_the_summary_of_a_row_and_speaks_it_on_request(monkeypatch):
    """Mutation: speak without `--speak` -- red."""
    row = {"id": "row-1", "text": "Add a health check.", "project": "qmcp", "status": "done",
           "outcome_text": "Added it. Two files.", "exit_code": 0}
    client = MagicMock()
    client.get_instruction.return_value = row
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: client)
    tts = _TTS()
    _fake_vox(monkeypatch, _STT(""), tts)

    quiet = CliRunner().invoke(cli.cli, ["instructions", "say", "row-1"])
    aloud = CliRunner().invoke(cli.cli, ["instructions", "say", "row-1", "--speak"])

    assert quiet.exit_code == 0 and aloud.exit_code == 0, quiet.output + aloud.output
    assert quiet.output.strip() == "Done in qmcp. Added it. The rest is on the record."
    assert tts.spoken == ["Done in qmcp. Added it. The rest is on the record."]


def test_say_names_a_row_that_does_not_exist(monkeypatch):
    from qmcp.client import MCPClientError

    client = MagicMock()
    client.get_instruction.side_effect = MCPClientError("Instruction 'nobody' not found")
    monkeypatch.setattr("qmcp.client.MCPClient", lambda *a, **k: client)

    result = CliRunner().invoke(cli.cli, ["instructions", "say", "nobody"])

    assert result.exit_code != 0 and "not found" in result.output


def test_the_list_filters_by_every_status(queue):
    """`--status` is read from the model's vocabulary. Mutation: list the two
    inbox words in the `Choice` -- red on `asking`."""
    queue.list_instructions = lambda status=None, limit=50: []
    for value in (s.value for s in InstructionStatus):
        result = CliRunner().invoke(cli.cli, ["instructions", "list", "--status", value])
        assert result.exit_code == 0, result.output
