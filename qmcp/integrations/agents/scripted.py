"""A runtime that does what its script says and spends nothing.

For tests, walkthroughs and the offline check. It answers with the text and
exit code it was built with, keeps every brief it was given, and reports
`spent` as zero -- a real count here, because nothing was called. It is a
runtime in every respect the contract names, so the path from consent to
record is exercised end to end without an agent, a model or a bill, and the
brief it keeps is how a test sees what continuity qmcp handed over.
"""

from __future__ import annotations

from qmcp.integrations.agents import SCRIPTED, AgentOutcome, Brief, OnEvent


class ScriptedRuntime:
    """Answers as configured; records what it was given."""

    name = SCRIPTED

    def __init__(self, text: str = "done, as scripted", exit_code: int = 0) -> None:
        self.text = text
        self.exit_code = exit_code
        # Every brief, in order. A test asserts on this to show what ran, what
        # history it was handed, and, more often, that nothing ran at all.
        self.briefs: list[Brief] = []

    @property
    def calls(self) -> list[dict[str, str]]:
        """The briefs as plain values: the instruction and the directory."""
        return [{"instruction": b.instruction, "cwd": str(b.cwd)} for b in self.briefs]

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        self.briefs.append(brief)
        if on_event:
            on_event("started", f"{SCRIPTED} in {brief.cwd}")
            on_event("finished", str(self.exit_code))
        return AgentOutcome(text=self.text, exit_code=self.exit_code,
                            elapsed_seconds=0.0, spent=0, argv=(SCRIPTED,))
