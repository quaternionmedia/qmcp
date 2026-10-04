"""A runtime that does what its script says and spends nothing.

For tests, walkthroughs and the offline check. It answers with the text and
exit code it was built with, keeps every call it was asked to make, and reports
`spent` as zero -- a real count here, because nothing was called. It is a
runtime in every respect the contract names, so the path from consent to
record is exercised end to end without an agent, a model or a bill.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qmcp.integrations.agents import SCRIPTED, AgentOutcome, OnEvent


class ScriptedRuntime:
    """Answers as configured; records what it was asked."""

    name = SCRIPTED

    def __init__(self, text: str = "done, as scripted", exit_code: int = 0,
                 session_ref: str | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.text = text
        self.exit_code = exit_code
        # What a later run would resume. None by default: a scripted run
        # leaves no session behind, and a runtime handed `resume` it cannot
        # honour says so by returning what it was given.
        self.session_ref = session_ref
        self.clock = clock
        # Every call, in order: the instruction, the directory and the
        # session it was asked to continue. A test asserts on this to show
        # what ran and, more often, that nothing did.
        self.calls: list[dict[str, str | None]] = []

    def run(self, instruction: str, cwd: Path, resume: str | None = None,
            on_event: OnEvent | None = None) -> AgentOutcome:
        self.calls.append({"instruction": instruction, "cwd": str(cwd), "resume": resume})
        if on_event:
            on_event("started", f"{SCRIPTED} in {cwd}")
            on_event("finished", str(self.exit_code))
        return AgentOutcome(
            text=self.text, exit_code=self.exit_code,
            session_ref=self.session_ref if self.session_ref is not None else resume,
            elapsed_seconds=0.0, spent=0, argv=(SCRIPTED,))

