"""The agent runtime behind an instruction, as a contract that names no product.

    uv run qmcp instructions act <id> --runtime <name>

**AN AGENT RUNTIME IS A DEPLOYMENT DECISION, SO THE CONTRACT NAMES NONE.**
`qmcp.instructions.act` takes an instruction a person recorded and has an
agent carry it out in the project's clone. Which agent is a choice somebody
makes at the command line, the way `vox.adapters` leaves the speech engine to
the person running the loop and `qmcp.localmodel` is the one module allowed
to name a model. So this package holds the contract and a registry, and a
product appears only in `qmcp.integrations.agents.adapters.<product>`, one
module per runtime, discovered by name rather than imported here.

**THE CONTRACT IS ONE CALL.** `AgentRuntime.run(instruction, cwd, resume,
on_event)` returns an `AgentOutcome`, and that is the whole surface: the text
that came back, the exit code, a reference to the session the runtime left
behind so a later instruction can continue it, how long it took, and what it
spent -- as a count where the runtime reports one and `unknown` with a reason
where it does not, never zero for "nobody counted" (`qmcp.spend`).

**`scripted` IS FOR TESTS AND THE OFFLINE CHECK, AND IS NEVER THE DEFAULT AT
THE COMMAND LINE.** It answers with a configured text and exit code, records
what it was asked to run, and spends nothing. The command requires `--runtime`
or `QMCP_AGENT_RUNTIME` and says so, because a default that ran a real agent
would be a paid call nobody chose and a default that ran the scripted one
would report a run that did nothing.

WHAT THIS CANNOT DO. Stop a runtime that spends more than it says, or one that
does not say: the outcome carries the runtime's own report, and
`qmcp.spend`'s budget counts runs, not the calls a run makes. Nor can it
resume a session a runtime does not know how to; `resume` is a request, and a
runtime that cannot honour it starts fresh and says so in `session_ref`.
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
command's shape, `output` with a line, `finished` with the exit code. The
same shape the voice loop announces in, so whatever shows one can show the
other."""

# The runtime that spends nothing and exists for tests and checks. Registered
# by name here rather than discovered, because it is not an adapter for any
# product and does not live beside them.
SCRIPTED = "scripted"


@dataclass(frozen=True)
class AgentOutcome:
    """What one run of a runtime did.

    `spent` is the runtime's own count of the calls it made, or `unknown` with
    a reason. It is reported beside the budget the act declared and never
    folded into it: the budget counts runs a person authorised, and what a run
    cost is what the runtime says it cost.
    """

    text: str
    exit_code: int
    session_ref: str | None = None
    """What a later run passes as `resume` to continue this one, or None when
    the runtime left nothing to continue."""

    elapsed_seconds: float = 0.0
    spent: int | dict[str, str] = field(default_factory=lambda: unknown(
        "the runtime did not report what it spent"))
    argv: tuple[str, ...] = ()
    """The command the runtime ran, where it ran one, so a record of the act
    says what was executed and not only what came back."""

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class AgentRuntime(Protocol):
    """One agent, able to carry out an instruction in a directory."""

    name: str

    def run(self, instruction: str, cwd: Path, resume: str | None = None,
            on_event: OnEvent | None = None) -> AgentOutcome:
        """Carry out `instruction` in `cwd`, continuing `resume` where given.

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
    "SCRIPTED",
    "AgentOutcome",
    "AgentRuntime",
    "OnEvent",
    "adapter_names",
    "runtime_class",
    "runtime_named",
    "runtime_names",
]
