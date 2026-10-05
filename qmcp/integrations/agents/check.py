"""A runtime that runs one of a project's declared checks, and spends nothing.

A check is a command the project declares (`qmcp vocabulary`, `projects.*.checks`):
its test suite, an offline loop, a scan. An instruction whose words name one is
acted on by this runtime instead of a model, through the same gate as any act --
recorded, consent asked, run only on approve, the outcome kept on the row and
said back. It runs the declared `argv` in the project's clone and nothing else:
no shell, and no word of the instruction added to the command.

The outcome's first line is the last line the command printed -- a test suite's
summary -- or the last matching the check's `said` where the last line printed
is a note rather than the verdict, so the sentence said back is the one that answers;
the output's tail follows it on the record. A command that cannot be found, or
outlives its minutes, ends as a failed run that says so.
"""

from __future__ import annotations

import os
import re
import subprocess
import time

from qmcp.integrations.agents import AgentOutcome, Brief, OnEvent

CHECK = "check"
# How much of a check's output is kept on the row, from its end.
TAIL_LINES = 40
# Exit codes for a check that could not run, or was stopped, in the shell's own
# convention, so a row reads the same as a terminal would.
NOT_FOUND, TIMED_OUT = 127, 124


class CheckRuntime:
    """Runs one declared check in the brief's directory."""

    name = CHECK
    # A check runs on this machine and calls nothing that is paid for.
    would_spend = 0
    # The project's earlier instructions change nothing a check does, so its
    # consent does not say it carries them.
    uses_history = False

    def __init__(self, check) -> None:
        self.check = check
        self.command = check.command

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        argv = list(self.check.argv)
        # A command run from inside another project's environment would be
        # pointed at it; each check runs in its own project's.
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        if on_event:
            on_event("started", f"{self.command} in {brief.cwd}")
        began = time.monotonic()
        try:
            done = subprocess.run(argv, cwd=brief.cwd, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=self.check.minutes * 60)
            code, printed = done.returncode, (done.stdout or "") + (done.stderr or "")
            lines = [line.strip() for line in printed.splitlines() if line.strip()]
            answering = [line for line in lines
                         if self.check.said and re.search(self.check.said, line)]
            last = (answering or lines or [f"It printed nothing, exit {code}."])[-1]
        except FileNotFoundError:
            code, printed = NOT_FOUND, ""
            last = f"{argv[0]} was not found on this machine."
        except subprocess.TimeoutExpired as stopped:
            code = TIMED_OUT
            printed = (stopped.stdout or "") if isinstance(stopped.stdout, str) else ""
            last = f"Stopped after {self.check.minutes:g} minutes."
        if on_event:
            on_event("finished", str(code))
        tail = "\n".join(printed.splitlines()[-TAIL_LINES:])
        text = last if tail.strip() in ("", last) else f"{last}\n\n{tail}"
        return AgentOutcome(text=text, exit_code=code,
                            elapsed_seconds=round(time.monotonic() - began, 2), spent=0,
                            argv=tuple(argv),
                            detail={"check": f"{self.check.project}.{self.check.name}"})
