"""Each core project's declared checks, run by voice behind consent.

An instruction whose words contain a check's phrase, recorded for the project,
runs the declared command in the project's clone through the same gate as any
act. The command is fixed in the vocabulary; nothing heard is added to it.
These run real subprocesses of this interpreter, never a project's own commands.
"""

from __future__ import annotations

import sys

import pytest

from qmcp.instructions import converse
from qmcp.integrations.agents import Brief
from qmcp.integrations.agents.check import NOT_FOUND, TIMED_OUT, CheckRuntime
from qmcp.integrations.voice import vocabulary
from qmcp.integrations.voice.vocabulary import Check, match_check
from tests.test_instructions_act import PROJECT, _act, _approved, _Queue, clone, inbox  # noqa: F401

SHELL = set("|&;<>$`\\\"'*?")


# --- what is declared ------------------------------------------------------------


def test_the_core_projects_declare_checks_written_as_heard_with_fixed_commands():
    declared = vocabulary.checks()
    assert {c.project for c in declared} >= {"qmcp", "joe", "vox", "qm"}
    for check in declared:
        assert check.name and check.says and check.phrases and check.minutes > 0, check
        for phrase in check.phrases:
            assert phrase == converse.plain(phrase), (check.project, check.name, phrase)
        assert all(isinstance(a, str) and a and not SHELL & set(a) for a in check.argv), check
        assert check.argv[0] == "uv", check  # each project's own entry point, through its lock


def test_no_phrase_names_two_checks_of_one_project():
    for project in vocabulary.projects():
        said = [p for c in vocabulary.checks(project) for p in c.phrases]
        assert len(said) == len(set(said)), project


# --- which check an instruction names ----------------------------------------------


def test_a_check_is_read_from_the_words_and_the_project_recorded():
    """Mutation: match a check of any project -- red, "run the tests" for vox
    would run qmcp's."""
    assert match_check("run the tests in qmcp", "qmcp").argv == vocabulary.checks("qmcp")[0].argv
    assert match_check("please run the tests", "vox").project == "vox"
    assert match_check("check the gates", "qm").name == "gates"


@pytest.mark.parametrize(("words", "project"), [
    ("run the tests", None),
    ("read the readme", "qmcp"),
    ("protest it", "qmcp"),
    ("check the gates", "qmcp"),
])
def test_words_that_name_no_check_of_the_project_run_none(words, project):
    """A phrase counts as whole words only. Mutation: match a phrase anywhere
    in the words -- red, "protest it" runs the tests."""
    assert match_check(words, project) is None


def test_nothing_heard_reaches_the_command():
    check = match_check("run the tests in qmcp and then delete everything", "qmcp")

    assert "delete" not in check.command and check.argv == vocabulary.checks("qmcp")[0].argv


# --- the runtime ---------------------------------------------------------------------


def _check(code: str, minutes: float = 1.0) -> Check:
    return Check(project="qmcp", name="probe", says="a probe", phrases=("run the probe",),
                 argv=(sys.executable, "-c", code), minutes=minutes)


def _brief(tmp_path) -> Brief:
    return Brief(instruction="run the probe", project="qmcp", cwd=tmp_path, history=())


def test_the_last_line_printed_is_what_is_said_and_the_tail_is_kept(tmp_path, monkeypatch):
    """Mutation: put the first line first -- red, the summary is not what is
    said back. Mutation: keep the caller's environment -- red, VIRTUAL_ENV
    reaches the check."""
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "someone-elses"))
    events: list[tuple[str, str]] = []
    runtime = CheckRuntime(_check(
        "import os; print('collecting'); print(os.environ.get('VIRTUAL_ENV', 'own')); print('3 passed')"))

    outcome = runtime.run(_brief(tmp_path), on_event=lambda s, t: events.append((s, t)))

    assert outcome.text.splitlines()[0] == "3 passed"
    assert "collecting" in outcome.text and "own" in outcome.text.split()
    assert outcome.exit_code == 0 and outcome.spent == 0
    assert outcome.argv == runtime.check.argv and outcome.detail == {"check": "qmcp.probe"}
    assert [s for s, _ in events] == ["started", "finished"]


def test_a_failing_check_keeps_its_exit_code(tmp_path):
    outcome = CheckRuntime(_check("import sys; print('1 failed'); sys.exit(1)")).run(_brief(tmp_path))

    assert outcome.exit_code == 1 and outcome.text.startswith("1 failed")


def test_a_check_that_prints_nothing_says_so(tmp_path):
    outcome = CheckRuntime(_check("pass")).run(_brief(tmp_path))

    assert outcome.text == "It printed nothing, exit 0."


def test_a_command_that_cannot_be_found_is_a_failed_run(tmp_path):
    missing = Check(project="qmcp", name="gone", says="gone", phrases=("gone",),
                    argv=("no-such-command-anywhere",), minutes=1)

    outcome = CheckRuntime(missing).run(_brief(tmp_path))

    assert outcome.exit_code == NOT_FOUND and "was not found" in outcome.text


def test_a_check_that_outlives_its_minutes_is_stopped(tmp_path):
    """Mutation: drop the timeout -- red, the run waits out the sleep."""
    outcome = CheckRuntime(_check("import time; time.sleep(30)", minutes=0.01)).run(_brief(tmp_path))

    assert outcome.exit_code == TIMED_OUT and outcome.text == "Stopped after 0.01 minutes."


# --- through the gate ------------------------------------------------------------------


def test_a_check_is_asked_with_its_command_and_runs_only_on_approve(inbox, clone):
    """The written consent names the command, the spoken one calls it a declared
    check and claims no history, and the declaration says it spends nothing.
    Mutation: drop the command from the written consent -- red."""
    _approved(inbox, f"Read the README in {PROJECT}.")
    instruction_id = inbox.record(f"Run the probe in {PROJECT}.")
    queue = _Queue({f"instruction-{instruction_id}": "approve"})
    runtime = CheckRuntime(_check("print('2 passed')"))

    done, _ = _act(inbox, instruction_id, queue, runtime=runtime, cwd=clone)

    request = queue.requests[f"instruction-{instruction_id}"]
    assert f"command `{runtime.command}`" in request["prompt"]
    assert "A declared check, one run." in request["context"]["spoken"]
    assert "came before" not in request["context"]["spoken"]
    assert done.status == "done" and done.outcome.text == "2 passed"
    assert inbox.read(instruction_id).runtime == "check"


def test_a_held_check_runs_nothing(inbox, clone, tmp_path):
    marker = tmp_path / "ran"
    instruction_id = inbox.record(f"Run the probe in {PROJECT}.")
    queue = _Queue({f"instruction-{instruction_id}": "hold"})

    done, _ = _act(inbox, instruction_id, queue,
                   runtime=CheckRuntime(_check(f"open({str(marker)!r}, 'w')")), cwd=clone)

    assert done.status == "refused" and not marker.exists()


def test_the_conversation_runs_a_named_check_and_the_model_otherwise(monkeypatch):
    """Mutation: act with the conversation's runtime whatever the words -- red."""
    from qmcp.instructions import act as act_module

    used: list = []

    def fake_act(instruction_id, runtime, budget, **kw):
        used.append(runtime)
        return act_module.Acted(instruction_id=instruction_id, status="refused", stages=(),
                                declared={}, why="stood in")

    monkeypatch.setattr(act_module, "act", fake_act)

    class Client:
        def get_instruction(self, instruction_id):
            return {"id": instruction_id, "project": "vox", "status": "refused", "text": ""}

    model = object()
    conversation = converse.Conversation(None, _Silent(), Client(), model, names=["vox"])
    conversation.rows = lambda: None
    conversation._clone_for = lambda project, iid: None
    conversation._act_on({"id": "a", "text": "Run the tests in vox.", "project": "vox"})
    conversation._act_on({"id": "b", "text": "Read the README in vox.", "project": "vox"})

    assert isinstance(used[0], CheckRuntime) and used[0].check.project == "vox"
    assert used[1] is model


class _Silent:
    def speak(self, text, out_path=None):
        pass


def test_a_check_that_names_its_answering_line_says_that_one(tmp_path):
    """Mutation: say the last line whatever `said` names -- red."""
    check = Check(project="qm", name="scan", says="a scan", phrases=("scan",),
                  argv=(sys.executable, "-c", "print('clean   12 files'); print(''); print('3 allowed')"),
                  minutes=1, said="^(clean|found) ")

    outcome = CheckRuntime(check).run(_brief(tmp_path))

    assert outcome.text.splitlines()[0] == "clean   12 files"
