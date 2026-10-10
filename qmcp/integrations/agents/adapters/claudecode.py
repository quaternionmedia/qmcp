"""Claude Code's command line, run non-interactively in a clone, behind the same contract.

    claude -p "<the brief qmcp rendered>" --output-format json

This adapter is the one place the product is named. It is chosen at the
command line with `--runtime claude-code`, and nothing in the contract or the
service imports it.

The prompt is `Brief.prompt()`, the rendering every runtime reads. No session
of the tool's own is resumed, so this runtime and the local one are
interchangeable from one instruction to the next.

`argv_for` builds the arguments as a pure function, and `Runtime.run` hands
them to the `popen` it was built with, so no test launches the tool and the
record shows the exact command that ran. No permission flag is passed: the
tool's own permission rules decide what it may do.

`num_turns` is reported as `spent` where the tool prints it; what a turn cost
is the tool's own accounting. The session id it prints is not kept. Output
that is not the expected JSON is kept as text with `spent` unknown, never
zero.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from typing import Any

from qmcp.integrations.agents import AgentOutcome, Brief, OnEvent
from qmcp.spend import unknown

NAME = "claude-code"
PRODUCT = "Claude Code"
EXECUTABLE = "claude"


def argv_for(prompt: str, executable: str = EXECUTABLE) -> list[str]:
    """The exact command, as a list. Pure, so a test can read it."""
    return [executable, "-p", prompt, "--output-format", "json"]


def read_output(stdout: str) -> tuple[str, int | dict[str, str]]:
    """(text, spent) from what the tool printed.

    The JSON form carries `result` and `num_turns`. Anything else is kept as
    the text it is: the run happened, and a parser that reported a count it
    did not read would be inventing it.
    """
    try:
        document: Any = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        document = None
    if not isinstance(document, dict):
        return stdout, unknown("the output was not the JSON this adapter reads")
    text = document.get("result")
    if not isinstance(text, str):
        text = stdout
    turns = document.get("num_turns")
    spent = turns if isinstance(turns, int) and turns >= 0 else unknown(
        "the output carried no turn count")
    return text, spent


class Runtime:
    """Claude Code, one non-interactive run per brief."""

    name = NAME

    def __init__(self, executable: str = EXECUTABLE,
                 popen: Callable[..., Any] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.executable = executable
        self.popen = popen
        self.clock = clock

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        argv = argv_for(brief.prompt(), self.executable)
        if on_event:
            on_event("started", f"{self.executable} -p, with {len(brief.history)} earlier"
                                " instruction(s) from qmcp's record")
        started = self.clock()
        try:
            process = self.popen(argv, cwd=str(brief.cwd), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace")
        except FileNotFoundError:
            return AgentOutcome(
                text=(f"{self.executable} is not on PATH, so nothing ran. Install the"
                      f" command line, or choose another runtime."),
                exit_code=127, spent=0, argv=tuple(argv))
        stdout, _ = process.communicate()
        elapsed = self.clock() - started
        text, spent = read_output(stdout or "")
        if on_event:
            on_event("finished", str(process.returncode))
        return AgentOutcome(text=text, exit_code=int(process.returncode),
                            elapsed_seconds=elapsed, spent=spent, argv=tuple(argv))
