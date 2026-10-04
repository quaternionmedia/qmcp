"""Claude Code's command line, run non-interactively in a clone.

    claude -p "<instruction>" --output-format json [--resume <session id>]

**THIS IS THE ONE PLACE THE PRODUCT IS NAMED**, as `qmcp.threads.claudecode`
is for reading its sessions: an adapter, chosen at the command line with
`--runtime claude-code`, and nothing in the contract or the service imports it.

**THE ARGUMENTS ARE BUILT BY A PURE FUNCTION AND THE PROCESS IS INJECTED.**
`argv_for` is what the tests assert; `Runtime.run` hands that list to whatever
`popen` it was built with, so no test launches the tool, and a reader of the
record sees the exact command that ran.

**A SESSION IS RESUMED WHEN THE ARCHIVE NAMES ONE.** `--resume` continues the
session the thread archive found for the project, so an instruction lands in
the conversation that was already working there rather than in a fresh one
that has to rediscover the checkout. The JSON the tool prints carries the id of
the session it ended in, resumed or new, and that is the `session_ref` the
record keeps for the next instruction.

WHAT IS NOT READ FROM THE OUTPUT. `num_turns` is reported as `spent` where the
tool prints it, because that is the count it offers; what one turn cost in
calls or money is the tool's own accounting and is not restated here. Output
that is not the JSON this expects is kept as text, with `spent` unknown and the
session reference left as it was asked -- a run that happened and cannot be
read is not reported as one that spent nothing.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qmcp.integrations.agents import AgentOutcome, OnEvent
from qmcp.spend import unknown

NAME = "claude-code"
EXECUTABLE = "claude"


def argv_for(instruction: str, resume: str | None = None,
             executable: str = EXECUTABLE) -> list[str]:
    """The exact command, as a list. Pure, so a test can read it."""
    argv = [executable, "-p", instruction, "--output-format", "json"]
    if resume:
        argv += ["--resume", resume]
    return argv


def read_output(stdout: str, asked_to_resume: str | None) -> tuple[str, str | None, int | dict[str, str]]:
    """(text, session_ref, spent) from what the tool printed.

    The JSON form carries `result`, `session_id` and `num_turns`. Anything
    else is kept as the text it is: the run happened, and a parser that
    reported a session or a count it did not read would be inventing both.
    """
    try:
        document: Any = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        document = None
    if not isinstance(document, dict):
        return stdout, asked_to_resume, unknown("the output was not the JSON this adapter reads")
    text = document.get("result")
    if not isinstance(text, str):
        text = stdout
    session = document.get("session_id") or asked_to_resume
    turns = document.get("num_turns")
    spent = turns if isinstance(turns, int) and turns >= 0 else unknown(
        "the output carried no turn count")
    return text, (str(session) if session else None), spent


class Runtime:
    """Claude Code, one non-interactive run per instruction."""

    name = NAME

    def __init__(self, executable: str = EXECUTABLE,
                 popen: Callable[..., Any] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.executable = executable
        self.popen = popen
        self.clock = clock

    def run(self, instruction: str, cwd: Path, resume: str | None = None,
            on_event: OnEvent | None = None) -> AgentOutcome:
        argv = argv_for(instruction, resume, self.executable)
        if on_event:
            on_event("started", " ".join(argv[:2]) + (f" --resume {resume}" if resume else ""))
        started = self.clock()
        process = self.popen(argv, cwd=str(cwd), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace")
        stdout, _ = process.communicate()
        elapsed = self.clock() - started
        text, session, spent = read_output(stdout or "", resume)
        if on_event:
            on_event("finished", str(process.returncode))
        return AgentOutcome(text=text, exit_code=int(process.returncode),
                            session_ref=session, elapsed_seconds=elapsed,
                            spent=spent, argv=tuple(argv))
