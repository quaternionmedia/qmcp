"""The agent runtime behind an instruction, as a contract that names no product.

    uv run qmcp instructions act <id> --runtime <name>

**CONTINUITY COMES FROM QMCP, NOT THE MODEL.** A runtime remembers nothing
between runs. Everything it knows about the project's history arrives in the
`Brief` qmcp hands it: the instruction, the project, the clone, and the
project's earlier instructions and outcomes, read from qmcp's own record
(`qmcp.instructions.continuity`). `Brief.prompt()` renders that once, the same
way for every runtime, so swapping one runtime for another loses nothing --
the history is in the record, not in a conversation some tool keeps. No
runtime is asked to resume a session of its own, and none is trusted to.

**AN AGENT RUNTIME IS A DEPLOYMENT DECISION, SO THE CONTRACT NAMES NONE.**
`qmcp.instructions.act` takes an instruction a person recorded and has a
runtime carry it out in the project's clone. Which runtime is a choice
somebody makes at the command line, the way `vox.adapters` leaves the speech
engine to the person running the loop and `qmcp.localmodel` is the one module
allowed to name a model. So this package holds the contract and a registry,
and a product appears only in `qmcp.integrations.agents.adapters.<product>`,
one module per runtime, discovered by name rather than imported here. `local`,
the model `qmcp.localmodel` stands up on this machine, is one; a coding
assistant's command line is another, behind the same contract and given the
same brief.

**THE CONTRACT IS ONE CALL.** `AgentRuntime.run(brief, on_event)` returns an
`AgentOutcome`: the text that came back, the exit code, how long it took, what
it spent -- as a count where the runtime reports one and `unknown` with a
reason where it does not, never zero for "nobody counted" (`qmcp.spend`) --
and whatever else the runtime can say about the run, kept as `detail`.

**`scripted` IS FOR TESTS AND THE OFFLINE CHECK, AND IS NEVER THE DEFAULT AT
THE COMMAND LINE.** It answers with a configured text and exit code, records
the brief it was given, and spends nothing. The command requires `--runtime`
or `QMCP_AGENT_RUNTIME` and says so, because a default that ran a real agent
would be a run nobody chose and a default that ran the scripted one would
report a run that did nothing.

WHAT THIS CANNOT DO. Stop a runtime that spends more than it says, or one that
does not say: the outcome carries the runtime's own report, and
`qmcp.spend`'s budget counts runs, not the calls a run makes. Nor can it make
a runtime read-only: what a runtime may touch is the adapter's to state.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qmcp.spend import unknown

OnEvent = Callable[[str, str], None]
"""Told what the runtime is doing, as `(state, text)`: `started` with the
runtime's shape, `output` with a line, `finished` with the exit code. The same
shape the voice loop announces in, so whatever shows one can show the other."""

# The runtime that spends nothing and exists for tests and checks. Registered
# by name here rather than discovered, because it is not an adapter for any
# product and does not live beside them.
SCRIPTED = "scripted"

# How much of each earlier outcome a brief carries. Enough for the gist of
# what was found; the whole text stays on its row.
OUTCOME_CHARS = 600


@dataclass(frozen=True)
class Turn:
    """One earlier instruction in the project, as qmcp's record holds it."""

    id: str
    instruction: str
    status: str
    outcome: str = ""
    runtime: str | None = None
    at: str | None = None


@dataclass(frozen=True)
class Brief:
    """Everything a runtime is given for one run, assembled by qmcp."""

    instruction: str
    cwd: Path
    project: str | None = None
    history: tuple[Turn, ...] = ()
    """The project's earlier instructions that ran, oldest first."""

    def prompt(self) -> str:
        """The brief as one prompt, rendered here so every runtime reads the
        same words. It starts with the place and the history and ends with the
        instruction, and asks for the answer first, because the first sentence
        is what is said back aloud."""
        where = f"the project {self.project}" if self.project else "this project"
        lines = [f"You are working in {where}, in the directory {self.cwd}."]
        if self.history:
            lines += ["", "What has been asked in this project so far, from qmcp's record,"
                          " oldest first:"]
            for number, turn in enumerate(self.history, start=1):
                when = f"{turn.at[:16].replace('T', ' ')} " if turn.at else ""
                lines.append(f"{number}. {when}asked: {turn.instruction}")
                outcome = " ".join(turn.outcome.split())
                if len(outcome) > OUTCOME_CHARS:
                    outcome = outcome[:OUTCOME_CHARS].rsplit(" ", 1)[0] + " ..."
                by = f" by {turn.runtime}" if turn.runtime else ""
                lines.append(f"   {turn.status}{by}: {outcome or '(nothing reported)'}")
        else:
            lines += ["", "Nothing has been asked in this project before."]
        lines += ["", f"Now: {self.instruction}", "",
                  "Answer in a few sentences, and start with the answer."]
        return "\n".join(lines)


@dataclass(frozen=True)
class AgentOutcome:
    """What one run of a runtime did.

    `spent` is the runtime's own count of the paid calls it made, or `unknown`
    with a reason. It is reported beside the budget the act declared and never
    folded into it: the budget counts runs a person authorised, and what a run
    cost is what the runtime says it cost.
    """

    text: str
    exit_code: int
    elapsed_seconds: float = 0.0
    spent: int | dict[str, str] = field(default_factory=lambda: unknown(
        "the runtime did not report what it spent"))
    argv: tuple[str, ...] = ()
    """The command the runtime ran, where it ran one, so a record of the act
    says what was executed and not only what came back."""

    detail: dict[str, Any] = field(default_factory=dict)
    """What else the runtime can say about the run -- the model, the calls it
    made, the files it read -- kept on the row beside the outcome."""

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class AgentRuntime(Protocol):
    """One agent, able to carry out a brief in its directory."""

    name: str

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        """Carry out `brief` in `brief.cwd`.

        Blocks until the agent is done. Everything a caller is told about the
        run is in the outcome; `on_event` is for showing progress while it
        happens and may be left out.
        """
        ...


def adapter_names() -> tuple[str, ...]:
    """Every runtime a product adapter provides, by the name it declares.

    Read from the `adapters` package so this module never names one: a module
    there exports `NAME` and `Runtime`, and that is the whole registration.
    """
    from qmcp.integrations.agents import adapters

    found = []
    for module in pkgutil.iter_modules(adapters.__path__):
        loaded = importlib.import_module(f"{adapters.__name__}.{module.name}")
        name = getattr(loaded, "NAME", None)
        if name and hasattr(loaded, "Runtime"):
            found.append(str(name))
    return tuple(sorted(found))


def runtime_names() -> tuple[str, ...]:
    """Every name `--runtime` accepts: the adapters, and `scripted`."""
    return (*adapter_names(), SCRIPTED)


def runtime_class(name: str) -> type:
    """The class behind a runtime name, or `KeyError` naming what exists."""
    if name == SCRIPTED:
        from qmcp.integrations.agents.scripted import ScriptedRuntime

        return ScriptedRuntime
    from qmcp.integrations.agents import adapters

    for module in pkgutil.iter_modules(adapters.__path__):
        loaded = importlib.import_module(f"{adapters.__name__}.{module.name}")
        if getattr(loaded, "NAME", None) == name and hasattr(loaded, "Runtime"):
            return loaded.Runtime
    raise KeyError(
        f"{name!r} is not a runtime. The names are: {', '.join(runtime_names())}.")


def runtime_named(name: str, **options: Any) -> AgentRuntime:
    """An instance of the runtime `name`, built with `options`."""
    return runtime_class(name)(**options)


__all__ = [
    "OUTCOME_CHARS",
    "SCRIPTED",
    "AgentOutcome",
    "AgentRuntime",
    "Brief",
    "OnEvent",
    "Turn",
    "adapter_names",
    "runtime_class",
    "runtime_named",
    "runtime_names",
]
