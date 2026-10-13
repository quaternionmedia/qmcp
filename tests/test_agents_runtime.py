"""The agent runtime contract: the brief, the registry, the scripted runtime and
the command-line adapter.

No test here launches a tool or calls a model. The command-line adapter's argv
is a pure function and its process is injected; the local model's adapter is
tested in `tests/test_agents_local.py` over a stand-in transport. What is
asserted is what a record of an act says ran, and that every runtime is handed
the same words for the same brief.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import httpx
import pytest

from qmcp.integrations import agents
from qmcp.integrations.agents import (
    OUTCOME_CHARS,
    SCRIPTED,
    AgentOutcome,
    Brief,
    Turn,
    runtime_class,
    runtime_named,
    runtime_names,
)
from qmcp.integrations.agents.adapters import claudecode, ollama
from qmcp.integrations.agents.scripted import ScriptedRuntime

PACKAGE = Path(agents.__file__).resolve().parent


def _brief(tmp_path, history=()):
    return Brief(instruction="Which file says what qmcp is?", cwd=tmp_path, project="qmcp",
                 history=tuple(history))


FIRST = Turn(id="t-1", instruction="Find the README.", status="done",
             outcome="README.md, at the root.", runtime="local", at="2026-10-04T09:00:00")
SECOND = Turn(id="t-2", instruction="Read it.", status="failed",
              outcome="The file could not be read.", runtime="claude-code", at="2026-10-04T09:05:00")


# --- the brief -------------------------------------------------------------------


def test_the_brief_carries_the_history_oldest_first_and_ends_with_the_instruction(tmp_path):
    """Mutation: render the history newest first -- red; drop the runtime from
    a turn -- red."""
    prompt = _brief(tmp_path, [FIRST, SECOND]).prompt()

    assert prompt.index("Find the README.") < prompt.index("Read it.")
    assert "1. 2026-10-04 09:00 asked: Find the README." in prompt
    assert "   done by local: README.md, at the root." in prompt
    assert "   failed by claude-code: The file could not be read." in prompt
    assert f"in the project qmcp, in the directory {tmp_path}" in prompt
    assert prompt.rstrip().endswith("Answer in a few sentences, and start with the answer.")
    assert "\nNow: Which file says what qmcp is?\n" in prompt


def test_a_brief_with_no_history_says_so(tmp_path):
    """Mutation: drop the no-history sentence -- red; a model told nothing
    might invent earlier work."""
    assert "Nothing has been asked in this project before." in _brief(tmp_path).prompt()


def test_a_long_outcome_is_cut_at_a_word_boundary(tmp_path):
    """Mutation: cut at `OUTCOME_CHARS` exactly -- red, a word is broken."""
    long = Turn(id="t", instruction="x", status="done", outcome="word " * 400)

    line = [ln for ln in _brief(tmp_path, [long]).prompt().splitlines() if "done" in ln][0]

    assert line.endswith("word ...")
    assert len(line) < OUTCOME_CHARS + 40


# --- the registry ---------------------------------------------------------------


def test_the_registry_names_both_adapters_and_scripted():
    """Mutation: drop `NAME` from an adapter module -- red, it is not discovered."""
    names = runtime_names()

    assert names[-1] == SCRIPTED
    assert {claudecode.NAME, ollama.NAME} <= set(names)
    assert runtime_class(claudecode.NAME) is claudecode.Runtime
    assert runtime_class(ollama.NAME) is ollama.Runtime
    assert runtime_class(SCRIPTED) is ScriptedRuntime
    assert isinstance(runtime_named(SCRIPTED, text="x"), ScriptedRuntime)


def test_an_unknown_runtime_is_refused_naming_what_exists():
    with pytest.raises(KeyError, match=f"{claudecode.NAME}.*{ollama.NAME}.*{SCRIPTED}"):
        runtime_class("nobody")


def test_the_contract_and_the_service_name_no_product():
    """A product appears in its adapter and nowhere else: the contract, the
    scripted runtime, the continuity, the act and the routes stay free of it,
    as `qmcp.governed` is of a model. Mutation: write an adapter's `PRODUCT`
    into the contract's docstring -- red."""
    from qmcp.instructions import act, continuity, service

    words = {word.lower() for module in (claudecode, ollama)
             for word in module.PRODUCT.split()} - {"code"}  # the generic half
    for path in (PACKAGE / "__init__.py", PACKAGE / "scripted.py", Path(continuity.__file__),
                 Path(act.__file__), Path(service.__file__)):
        text = path.read_text(encoding="utf-8").lower()
        for word in words:
            assert not re.search(rf"\b{re.escape(word)}\b", text), (
                f"{path.name} names {word!r}; a product belongs in its adapter")


# --- the scripted runtime -------------------------------------------------------


def test_the_scripted_runtime_answers_as_configured_keeps_the_brief_and_spends_nothing(tmp_path):
    """Mutation: report `spent=unknown(...)` -- red; a run that called nothing
    knows it called nothing."""
    runtime = ScriptedRuntime(text="all done", exit_code=3)
    events: list[tuple[str, str]] = []
    brief = _brief(tmp_path, [FIRST])

    outcome = runtime.run(brief, on_event=lambda s, t: events.append((s, t)))

    assert isinstance(outcome, AgentOutcome)
    assert (outcome.text, outcome.exit_code, outcome.succeeded) == ("all done", 3, False)
    assert outcome.spent == 0
    assert runtime.briefs == [brief]
    assert runtime.calls == [{"instruction": brief.instruction, "cwd": str(tmp_path)}]
    assert [state for state, _ in events] == ["started", "finished"]


# --- the command-line adapter --------------------------------------------------------


def test_the_adapters_argv_is_non_interactive_and_never_resumes():
    """Continuity is in the prompt, not in a session of the tool's own.
    Mutation: drop `-p` -- red; the tool would open a prompt and wait."""
    argv = claudecode.argv_for("the brief")

    assert argv == ["claude", "-p", "the brief", "--output-format", "json"]
    assert "--resume" not in argv and "--continue" not in argv
    assert claudecode.argv_for("x", executable="/opt/bin/claude")[0] == "/opt/bin/claude"


def test_the_adapter_reads_the_tools_json_and_keeps_anything_else_as_text():
    """Mutation: return `0` for a missing `num_turns` -- red; a count nobody
    took is not a count of zero."""
    assert claudecode.read_output('{"result": "pinned", "session_id": "s-9", "num_turns": 4}') == (
        "pinned", 4)

    text, spent = claudecode.read_output("plain words")
    assert text == "plain words" and "unknown" in spent

    text, spent = claudecode.read_output('{"result": "r"}')
    assert text == "r" and "unknown" in spent


class _Process:
    def __init__(self, stdout: str, returncode: int):
        self._stdout, self.returncode = stdout, returncode

    def communicate(self):
        return self._stdout, ""


def test_the_adapter_runs_the_rendered_brief_in_the_clone(tmp_path):
    """Mutation: pass `brief.instruction` rather than `brief.prompt()` -- red,
    the history never reaches the tool; pass `cwd=None` to popen -- red, it
    would run wherever the command was issued."""
    launched: list[dict] = []

    def popen(argv, **kw):
        launched.append({"argv": argv, **kw})
        return _Process('{"result": "done", "num_turns": 2}', 0)

    ticks = iter([10.0, 12.5])
    runtime = claudecode.Runtime(popen=popen, clock=lambda: next(ticks))
    events: list[tuple[str, str]] = []
    brief = _brief(tmp_path, [FIRST])

    outcome = runtime.run(brief, on_event=lambda s, t: events.append((s, t)))

    assert launched[0]["argv"] == claudecode.argv_for(brief.prompt())
    assert "README.md, at the root." in launched[0]["argv"][2]
    assert launched[0]["cwd"] == str(tmp_path)
    assert launched[0]["stdout"] is subprocess.PIPE
    assert outcome.text == "done" and outcome.exit_code == 0 and outcome.spent == 2
    assert outcome.elapsed_seconds == 2.5
    assert outcome.argv == tuple(launched[0]["argv"])
    assert events[0] == ("started", "claude -p, with 1 earlier instruction(s) from qmcp's record")
    assert events[-1] == ("finished", "0")


def test_a_failed_run_keeps_its_exit_code_and_output(tmp_path):
    outcome = claudecode.Runtime(popen=lambda argv, **kw: _Process("boom", 2)).run(_brief(tmp_path))

    assert outcome.exit_code == 2 and not outcome.succeeded
    assert outcome.text == "boom" and "unknown" in outcome.spent


def test_a_missing_executable_is_a_failed_run_that_says_so(tmp_path):
    """Mutation: let `FileNotFoundError` out -- red; the act would record the
    exception's repr rather than what to do about it."""
    def popen(argv, **kw):
        raise FileNotFoundError(argv[0])

    outcome = claudecode.Runtime(popen=popen).run(_brief(tmp_path))

    assert outcome.exit_code == 127 and outcome.spent == 0
    assert "not on PATH" in outcome.text


# --- the same words for every runtime ---------------------------------------------------


def test_every_runtime_is_handed_the_same_prompt_for_the_same_brief(tmp_path):
    """The abstraction this package exists for: the command-line adapter and
    the local model read identical words, so either can carry the next
    instruction. Mutation: have either adapter render its own prompt -- red."""
    brief = _brief(tmp_path, [FIRST, SECOND])
    seen: dict[str, str] = {}

    def popen(argv, **kw):
        seen["claude-code"] = argv[2]
        return _Process('{"result": "x", "num_turns": 1}', 0)

    def chat(request):
        seen["local"] = json.loads(request.content)["messages"][1]["content"]
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "x"}})

    claudecode.Runtime(popen=popen).run(brief)
    ollama.Runtime(client=httpx.Client(transport=httpx.MockTransport(chat))).run(brief)

    assert seen["claude-code"] == seen["local"] == brief.prompt()


def test_output_whose_result_is_not_text_is_kept_whole():
    """Mutation: return `result` whatever its type -- red, a number is said back."""
    stdout = '{"result": 5, "num_turns": 1}'

    assert claudecode.read_output(stdout) == (stdout, 1)
