"""The agent runtime contract, its registry, the scripted runtime and the one adapter.

No test here launches a tool. The adapter's command line is a pure function,
and its process is injected, so what is asserted is the exact argv and how
the output is read -- which is what a record of an act says ran.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from qmcp.integrations import agents
from qmcp.integrations.agents import SCRIPTED, AgentOutcome, runtime_class, runtime_named, runtime_names
from qmcp.integrations.agents.adapters import claudecode
from qmcp.integrations.agents.scripted import ScriptedRuntime

PACKAGE = Path(agents.__file__).resolve().parent


# --- the registry ---------------------------------------------------------------


def test_the_registry_names_the_adapters_and_scripted():
    """Mutation: drop `NAME` from the adapter module -- red, the adapter is
    not discovered."""
    names = runtime_names()

    assert names[-1] == SCRIPTED
    assert claudecode.NAME in names
    assert runtime_class(claudecode.NAME) is claudecode.Runtime
    assert runtime_class(SCRIPTED) is ScriptedRuntime
    assert isinstance(runtime_named(SCRIPTED, text="x"), ScriptedRuntime)


def test_an_unknown_runtime_is_refused_naming_what_exists():
    with pytest.raises(KeyError, match=f"{claudecode.NAME}.*{SCRIPTED}"):
        runtime_class("nobody")


def test_the_contract_and_the_service_name_no_product():
    """A product appears in its adapter and nowhere else: the contract, the
    scripted runtime, the act and the routes stay free of it, as `qmcp.governed`
    is of a model. Mutation: write the adapter's name into the contract's
    docstring -- red."""
    from qmcp.instructions import act, service

    product_words = {word for name in agents.adapter_names() for word in re.split(r"[-_]", name)}
    product_words -= {"code"}  # the generic half of a name is not the product
    for path in (PACKAGE / "__init__.py", PACKAGE / "scripted.py",
                 Path(act.__file__), Path(service.__file__)):
        text = path.read_text(encoding="utf-8").lower()
        for word in product_words:
            assert not re.search(rf"\b{re.escape(word)}\b", text), (
                f"{path.name} names {word!r}; a product belongs in its adapter")


# --- the scripted runtime -------------------------------------------------------


def test_the_scripted_runtime_answers_as_configured_and_spends_nothing(tmp_path):
    """Mutation: report `spent=unknown(...)` -- red; a run that called nothing
    knows it called nothing."""
    runtime = ScriptedRuntime(text="all done", exit_code=3)
    events: list[tuple[str, str]] = []

    outcome = runtime.run("Deploy.", tmp_path, resume="s-1", on_event=lambda s, t: events.append((s, t)))

    assert isinstance(outcome, AgentOutcome)
    assert (outcome.text, outcome.exit_code, outcome.succeeded) == ("all done", 3, False)
    assert outcome.spent == 0
    assert outcome.session_ref == "s-1"
    assert runtime.calls == [{"instruction": "Deploy.", "cwd": str(tmp_path), "resume": "s-1"}]
    assert [state for state, _ in events] == ["started", "finished"]


def test_the_scripted_runtime_can_name_the_session_it_leaves(tmp_path):
    runtime = ScriptedRuntime(session_ref="s-new")
    assert runtime.run("x", tmp_path, resume="s-old").session_ref == "s-new"
    assert ScriptedRuntime().run("x", tmp_path).session_ref is None


# --- the adapter ------------------------------------------------------------------


def test_the_adapters_argv_is_non_interactive_and_resumes_when_asked():
    """Mutation: drop `-p` -- red; the tool would open a prompt and wait."""
    fresh = claudecode.argv_for("Deploy qmcp to the pi.")
    resumed = claudecode.argv_for("Deploy qmcp to the pi.", resume="abc-123")

    assert fresh == ["claude", "-p", "Deploy qmcp to the pi.", "--output-format", "json"]
    assert resumed == fresh + ["--resume", "abc-123"]
    assert claudecode.argv_for("x", executable="/opt/bin/claude")[0] == "/opt/bin/claude"


def test_the_adapter_reads_the_tools_json_and_keeps_anything_else_as_text():
    """Mutation: return `0` for a missing `num_turns` -- red; a count nobody
    took is not a count of zero."""
    text, session, spent = claudecode.read_output(
        '{"result": "pinned", "session_id": "s-9", "num_turns": 4}', asked_to_resume="s-1")
    assert (text, session, spent) == ("pinned", "s-9", 4)

    text, session, spent = claudecode.read_output("plain words", asked_to_resume="s-1")
    assert (text, session) == ("plain words", "s-1")
    assert "unknown" in spent

    text, session, spent = claudecode.read_output('{"result": "r"}', asked_to_resume=None)
    assert (text, session) == ("r", None)
    assert "unknown" in spent


class _Process:
    def __init__(self, stdout: str, returncode: int):
        self._stdout, self.returncode = stdout, returncode

    def communicate(self):
        return self._stdout, ""


def test_the_adapter_runs_the_argv_in_the_clone_through_the_injected_popen(tmp_path):
    """Mutation: pass `cwd=None` to popen -- red; the agent would run wherever
    the command was issued rather than in the project's clone."""
    launched: list[dict] = []

    def popen(argv, **kw):
        launched.append({"argv": argv, **kw})
        return _Process('{"result": "done", "session_id": "s-2", "num_turns": 2}', 0)

    ticks = iter([10.0, 12.5])
    runtime = claudecode.Runtime(popen=popen, clock=lambda: next(ticks))
    events: list[tuple[str, str]] = []

    outcome = runtime.run("Pin the vectors.", tmp_path, resume="s-1",
                          on_event=lambda s, t: events.append((s, t)))

    assert launched[0]["argv"] == claudecode.argv_for("Pin the vectors.", "s-1")
    assert launched[0]["cwd"] == str(tmp_path)
    assert launched[0]["stdout"] is subprocess.PIPE
    assert outcome.text == "done" and outcome.exit_code == 0
    assert outcome.session_ref == "s-2" and outcome.spent == 2
    assert outcome.elapsed_seconds == 2.5
    assert outcome.argv == tuple(launched[0]["argv"])
    assert events[0][0] == "started" and events[-1] == ("finished", "0")


def test_a_failed_run_keeps_its_exit_code_and_output(tmp_path):
    runtime = claudecode.Runtime(popen=lambda argv, **kw: _Process("boom", 2))

    outcome = runtime.run("x", tmp_path)

    assert outcome.exit_code == 2 and not outcome.succeeded
    assert outcome.text == "boom"
    assert "unknown" in outcome.spent
