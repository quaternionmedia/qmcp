"""`qmcp cookbook instruct`: the spoken-instruction check, and that it can fail.

The check is itself an integration test -- a real qmcp server, vox's
deterministic engine, real HTTP on both sides -- so these tests run it and then
break the dialog under it: a check that cannot report a broken dialog is a green
row standing where a reader believes something is checked.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qmcp.cli import cli
from qmcp.instructions import RULE_ONE, RULE_STATED, check, dialog, roster_names
from qmcp.instructions.check import OFFLINE_CASES, Case, run_offline

pytestmark = pytest.mark.skipif(
    not {check.NAMED, check.OTHER} <= set(roster_names()),
    reason="governance/qm is not checked out, so there is no roster to resolve against",
)


def test_the_offline_check_passes_through_the_real_path():
    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 0, result.output
    assert f"{len(OFFLINE_CASES)} of {len(OFFLINE_CASES)} ended as scripted." in result.output
    assert "recorded for dossier" in result.output and "recorded for qmcp" in result.output


def test_the_offline_check_fails_when_a_project_is_picked_by_position(monkeypatch):
    """A dialog that takes the first candidate for anything it cannot match
    is the guess the inbox exists to refuse. The check must say so."""
    monkeypatch.setattr(dialog, "match_option", lambda text, options: options[0])

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    # The closed choice took the first candidate, and so did the confirmation:
    # `again` read as `record`, and the first take was recorded.
    assert "recorded project 'dossier', expected 'qmcp'" in result.output
    assert "recorded text 'Deploy qmcp.', expected 'Deploy qmcp to the pi.'" in result.output


def test_the_offline_check_fails_when_the_long_take_drops_the_pause(monkeypatch):
    """The parameter the stack exists for. A dialog that listens at the
    engine's default pause ends an instruction mid-sentence, and nothing
    about the recorded row would show it."""
    original = dialog.InstructionDialog._listen

    def without_pause(self, *, long, hint=None):
        self.pause_ms = None
        return original(self, long=long, hint=hint)

    monkeypatch.setattr(dialog.InstructionDialog, "_listen", without_pause)

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("carried pause [None], expected 1500" in line for line in lines)


def test_the_offline_check_fails_when_the_confirmation_carries_the_pause(monkeypatch):
    """The other half of the same claim: a word in answer gets the engine's
    own pause. Mutation: `if False:` for the check on `state.pauses[1]` in
    `run_offline` -- red, the check stays green with the pause on every take."""
    original = dialog.InstructionDialog._listen

    def long_everywhere(self, *, long, hint=None):
        return original(self, long=True)

    monkeypatch.setattr(dialog.InstructionDialog, "_listen", long_everywhere)

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("the confirmation carried pause 1500" in line for line in lines)


def test_the_offline_check_fails_when_the_row_is_not_a_spoken_one(monkeypatch):
    """A dialog recording what it heard as typed would pass every other check.
    Mutation: drop the `source` check from `run_offline` -- red."""
    from qmcp.client.mcp_client import MCPClient

    original = MCPClient.create_instruction

    def as_typed(self, text, source="typed", project=None, heard=None):
        return original(self, text, source="typed", project=project, heard=heard)

    monkeypatch.setattr(MCPClient, "create_instruction", as_typed)

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("source 'typed', expected 'voice'" in line for line in lines)


def test_the_offline_check_fails_when_nothing_is_read_back(monkeypatch):
    monkeypatch.setattr(dialog, "say_options", lambda options: "")

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("never read back" in line for line in lines)


def test_a_case_expecting_one_project_fails_when_another_is_recorded():
    """A script whose expectation the inbox does not meet is reported as such."""
    lines: list[str] = []
    wrong = Case(("Deploy qmcp to the pi.", "record"), "Deploy qmcp to the pi.", check.OTHER,
                 RULE_ONE)

    assert run_offline(echo=lines.append, cases=(wrong,)) is False
    assert "recorded project 'qmcp', expected 'dossier'" in lines[-2]


def test_the_row_says_how_the_project_was_settled():
    """A text that named the project is read by the server and the row says
    so; a project the person chose is `stated`. Mutation: have the dialog
    send `found.project` whatever settled it -- red, the first case's row
    says `stated`; drop the `rule` check from `run_offline` -- green here, so
    the wrong expectation below is what shows the check reads it."""
    lines: list[str] = []
    wrong = Case(("Deploy qmcp to the pi.", "record"), "Deploy qmcp to the pi.", check.NAMED,
                 RULE_STATED)

    assert run_offline(echo=lines.append, cases=(wrong,)) is False
    assert f"rule {RULE_ONE!r}, expected 'stated'" in lines[-2]


def test_the_verdict_reads_the_inbox_and_not_the_dialogs_return(monkeypatch):
    """What the dialog returns is the POST's response; the verdict reads the row
    again with a GET, so a server that records one thing and returns another is
    caught. The GET is swapped here to return a different project, and the
    check must report it. Mutation: `recorded = row` in `run_offline` -- green
    on every other test in this file, because POST and GET carry the same row
    when nothing is swapped, and red here, because the swapped GET is never
    consulted."""
    from qmcp.client.mcp_client import MCPClient

    original = MCPClient.get_instruction

    def swapped(self, instruction_id):
        found = original(self, instruction_id)
        return {**found, "project": check.OTHER}

    monkeypatch.setattr(MCPClient, "get_instruction", swapped)

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("recorded project 'dossier', expected 'qmcp'" in line for line in lines)


# --- the loop --------------------------------------------------------------------------


def test_the_loop_goes_the_whole_way_and_prints_the_conversation():
    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 0, result.output
    assert f"{len(check.LOOP_CASES)} of {len(check.LOOP_CASES)} loops ended as scripted." in result.output
    assert "heard     approve" in result.output and "heard     hold" in result.output
    assert "said      Done in qmcp. Added a health route" in result.output
    assert "said      Held. Nothing ran for: Rotate the qmcp logs." in result.output


def test_the_loop_fails_when_the_wrong_answer_runs(monkeypatch):
    """A loop that ran on `hold` and not on `approve` is the defect the gate
    exists to prevent, and the check must turn red on it."""
    monkeypatch.setattr("qmcp.instructions.act.APPROVE", "hold")

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    assert "the runtime ran 1 time(s), expected 0" in result.output
    assert "the runtime ran 0 time(s), expected 1" in result.output


def test_the_loop_fails_when_the_summary_is_not_said_back(monkeypatch):
    """Mutation of the check's subject: a summary that says only 'Done.'"""
    monkeypatch.setattr("qmcp.instructions.spoken.summarise", lambda row, why="": "Done.")

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    assert "said 'Done.', expected it to begin 'Done in qmcp. Added a health route'" in result.output


def test_the_loop_fails_when_the_panel_is_told_recorded(monkeypatch):
    """`recorded` shows on the panel as a person's answer accepted."""
    from qmcp.instructions import spoken

    def recorded(text, tts, stt=None):
        spoken.announce(stt, "speaking", text)
        tts.speak(text)
        spoken.announce(stt, "recorded", text)

    monkeypatch.setattr(spoken, "say", recorded)

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    assert "the panel was last told" in result.output


def test_the_loop_fails_when_the_consent_does_not_say_the_instruction(monkeypatch):
    """The person at the gate may not be the person who spoke the instruction."""
    monkeypatch.setattr("qmcp.instructions.act.consent_prompt",
                        lambda row, clone, runtime, budget, carried=0, spoken=False: "Act on the instruction?")

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    assert "the consent was never asked aloud with the instruction" in result.output


def test_the_loop_fails_when_the_run_fails(monkeypatch):
    from qmcp.integrations.agents import scripted

    real = scripted.ScriptedRuntime
    monkeypatch.setattr(scripted, "ScriptedRuntime",
                        lambda text="", **kw: real(text=text, exit_code=3, **kw))

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    assert "status 'failed', expected 'done'" in result.output


def test_the_loop_fails_when_something_is_said_after_the_summary(monkeypatch):
    from qmcp.instructions import spoken

    real = spoken.say

    def chatty(text, tts, stt=None):
        real(text, tts, stt)
        if text.startswith(("Done", "Held")):
            tts.speak("Anything else?")

    monkeypatch.setattr(spoken, "say", chatty)

    result = CliRunner().invoke(cli, ["cookbook", "instruct"])

    assert result.exit_code == 1, result.output
    assert "the summary was not the last thing said" in result.output


def test_a_loop_case_expecting_a_run_fails_when_nothing_ran():
    """The verdict reads the runtime's calls, not the case's expectation."""
    lines: list[str] = []
    case = check.Loop(("Rotate the qmcp logs.", "record", "hold"), "refused", True, "Held.")

    assert check.run_loop(echo=lines.append, cases=(case,)) is False
    assert any("the runtime ran 0 time(s), expected 1" in line for line in lines)


# --- the continuity, on a runtime ------------------------------------------------------
#
# Run on `scripted`, which spends nothing, so the path qmcp owns -- the clone
# remembered, the history carried, the summary said -- is checked without a
# model. The same command with `--runtime local` runs it on the model.


def test_continuity_on_a_runtime_is_shown_end_to_end():
    result = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "scripted"])

    assert result.exit_code == 0, result.output
    assert "carrying 1 earlier instruction(s) from qmcp's record" in result.output
    assert "carried   instruction 1 and what it found, from qmcp's record" in result.output
    assert "ran in the remembered clone, told what the first found" in result.output
    assert "The second instruction was carried out knowing what the first found." in result.output


def test_continuity_fails_when_nothing_is_carried(monkeypatch):
    """Mutation of the subject: the act hands the runtime no history."""
    monkeypatch.setattr("qmcp.instructions.act.history", lambda *a, **k: ())

    result = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "scripted"])

    assert result.exit_code == 1, result.output
    assert "carried (), expected the first instruction" in result.output


def test_continuity_fails_when_the_clone_is_not_remembered(monkeypatch):
    monkeypatch.setattr("qmcp.instructions.act.last_clone", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "scripted"])

    assert result.exit_code == 1, result.output
    assert "the runtime never ran: no clone for 'qmcp' in qmcp's record yet" in result.output


def test_continuity_refuses_a_clone_that_is_no_rostered_project(tmp_path):
    from qmcp.integrations.agents.scripted import ScriptedRuntime

    lines: list[str] = []
    clone = tmp_path / "not-a-project"
    clone.mkdir()

    assert check.run_continuity(echo=lines.append, runtime=ScriptedRuntime(), clone=clone) is False
    assert "'not-a-project' is not on the roster" in lines[0]


def test_a_runtime_that_is_not_ready_runs_nothing(monkeypatch):
    """Mutation: skip `ready()` -- red, the demonstration would start and fail
    one timeout later."""
    from qmcp.integrations.agents.scripted import ScriptedRuntime

    class Unready(ScriptedRuntime):
        def ready(self):
            return "the model is not served"

    monkeypatch.setattr("qmcp.integrations.agents.runtime_named", lambda name, **kw: Unready())

    result = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "local"])

    assert result.exit_code != 0
    assert "Nothing ran: the model is not served." in result.output
    assert "instruction 1" not in result.output


def test_the_demonstration_options_are_refused_without_a_runtime(tmp_path):
    clone = CliRunner().invoke(cli, ["cookbook", "instruct", "--clone", str(tmp_path)])
    unknown = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "nobody"])

    assert clone.exit_code != 0 and "read only with --runtime" in clone.output
    assert unknown.exit_code != 0 and "'nobody' is not a runtime" in unknown.output


@pytest.mark.parametrize(("rule", "message"), [
    ("passed as --cwd", "not remembered"),
    ("the clone the project's last act ran in, from qmcp's record", "not the one passed"),
])
def test_continuity_fails_when_a_clone_was_chosen_by_the_wrong_rule(monkeypatch, rule, message):
    """The first instruction's clone must be the one passed and the second's
    the remembered one; a run in the right directory for the wrong reason is
    not continuity."""
    from pathlib import Path

    from qmcp.instructions import act as act_module

    monkeypatch.setattr(act_module, "clone_for",
                        lambda rows, instruction_id, project, cwd, cwd_rule="": act_module.Clone(
                            cwd=Path(__file__).resolve().parent.parent, rule=rule))

    result = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "scripted"])

    assert result.exit_code == 1, result.output
    assert message in result.output


def test_continuity_fails_when_something_is_said_after_the_summary(monkeypatch):
    from qmcp.instructions import spoken

    real = spoken.say

    def chatty(text, tts, stt=None):
        real(text, tts, stt)
        if text.startswith("Done"):
            tts.speak("Anything else?")

    monkeypatch.setattr(spoken, "say", chatty)

    result = CliRunner().invoke(cli, ["cookbook", "instruct", "--runtime", "scripted"])

    assert result.exit_code == 1, result.output
    assert "the summary was not the last thing said" in result.output


# --- the conversation ---------------------------------------------------------------------


def test_one_spoken_session_runs_with_nothing_typed():
    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 0, result.output
    assert "said   Ready. What should be done?" in result.output
    assert "answered: agent-question -> approve" in result.output
    assert "carrying 1 earlier instruction(s) from qmcp's record" in result.output
    assert "said   Stopping. Start the server again to talk." in result.output
    assert "[ok]   one spoken session" in result.output


def test_the_session_fails_when_the_waiting_question_is_not_asked(monkeypatch):
    from qmcp.instructions.converse import Conversation

    monkeypatch.setattr(Conversation, "_ask_waiting", lambda self, ended: None)

    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 1, result.output
    assert "the waiting question was not asked and answered by voice" in result.output


def test_the_session_fails_when_the_second_instruction_is_not_told_the_first(monkeypatch):
    monkeypatch.setattr("qmcp.instructions.act.history", lambda *a, **k: ())

    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 1, result.output
    assert "the second instruction was not told what the first found" in result.output


def test_the_session_fails_when_the_first_clone_is_not_found_by_name(monkeypatch):
    from qmcp.instructions.converse import Conversation

    monkeypatch.setattr(Conversation, "_clone_for", lambda self, project, iid: None)

    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 1, result.output
    assert "expected two done" in result.output


def test_the_session_fails_when_it_does_not_stop_when_told(monkeypatch):
    monkeypatch.setattr("qmcp.instructions.converse.STOP", ())

    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 1, result.output
    assert "not on being told to stop" in result.output


def test_a_session_on_a_runtime_that_is_not_ready_runs_nothing(monkeypatch):
    from qmcp.integrations.agents.scripted import ScriptedRuntime

    class Unready(ScriptedRuntime):
        def ready(self):
            return "the model is not served"

    monkeypatch.setattr("qmcp.integrations.agents.runtime_named", lambda name, **kw: Unready())

    result = CliRunner().invoke(cli, ["cookbook", "converse", "--runtime", "local"])

    assert result.exit_code != 0 and "Nothing ran: the model is not served." in result.output


def test_the_session_fails_when_no_question_hints_its_options(monkeypatch):
    """The check reads the hints off the engine, over the wire."""
    from qmcp.instructions import dialog as dialog_module
    from qmcp.integrations.voice import adapter

    original = adapter.listen_for

    def unhinted(stt, duration, *, pause_ms=None, hint=None):
        return original(stt, duration, pause_ms=pause_ms)

    monkeypatch.setattr(adapter, "listen_for", unhinted)
    monkeypatch.setattr(dialog_module, "listen_for", unhinted)

    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 1, result.output
    assert "no listen was hinted 'approve, hold'" in result.output


def test_the_session_fails_when_repeat_is_not_honoured(monkeypatch):
    from qmcp.instructions import dialog as dialog_module

    monkeypatch.setattr(dialog_module, "asks_repeat", lambda text: False)

    result = CliRunner().invoke(cli, ["cookbook", "converse"])

    assert result.exit_code == 1, result.output
    assert "the first read-back was said 1 time(s)" in result.output
