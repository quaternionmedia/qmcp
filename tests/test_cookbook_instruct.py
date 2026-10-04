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
from qmcp.instructions import check, dialog, roster_names
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

    def without_pause(self, *, long):
        self.pause_ms = None
        return original(self, long=long)

    monkeypatch.setattr(dialog.InstructionDialog, "_listen", without_pause)

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("carried pause [None], expected 1500" in line for line in lines)


def test_the_offline_check_fails_when_nothing_is_read_back(monkeypatch):
    monkeypatch.setattr(dialog, "say_options", lambda options: "")

    lines: list[str] = []
    assert run_offline(echo=lines.append, cases=OFFLINE_CASES[:1]) is False
    assert any("never read back" in line for line in lines)


def test_a_case_expecting_one_project_fails_when_another_is_recorded():
    """A script whose expectation the inbox does not meet is reported as such."""
    lines: list[str] = []
    wrong = Case(("Deploy qmcp to the pi.", "record"), "Deploy qmcp to the pi.", check.OTHER)

    assert run_offline(echo=lines.append, cases=(wrong,)) is False
    assert "recorded project 'qmcp', expected 'dossier'" in lines[-2]


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
