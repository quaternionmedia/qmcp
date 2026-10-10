"""Acting on an instruction: declare what it would spend, ask, and run only on approve.

    uv run qmcp instructions act <id> --runtime <name> [--budget N] [--cwd PATH] [--voice]

Nothing runs before a person says approve. The runtime is budgeted before it
runs and recorded after, with the consent in front of it, and every runtime is
asked. The budget counts runs and defaults to zero, which declares and stops;
`qmcp.spend.declare`'s declaration is written on every path. The consent
expires after `CONSENT_SECONDS`, and the row then reads `unanswered`.

`docs/integrations/voice.md`, *Acting on an instruction*, explains the brief, the clone, the consent
and how the act waits.

What this cannot do: stop a runtime once it is running, know what it will
spend before it reports, or tell that the clone is the one the person meant --
the prompt names it, and the person at the gate is the check.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qmcp.db.models import Instruction, InstructionStatus
from qmcp.instructions.continuity import history, last_clone
from qmcp.integrations.agents import AgentOutcome, AgentRuntime, Brief, OnEvent
from qmcp.spend import Budget, declare, unknown

Rows = Callable[[], Any]
"""A callable returning a context manager that yields a `sqlmodel.Session` over
the server database, committed by the caller -- what `rows_at` builds, and
what the configured database is wrapped in by the command."""

# The stages, declared once, so a reader of a result sees how far an act got
# without reading `act`. `run` is the one with no halting guarantee.
STAGES: tuple[str, ...] = ("instruction", "clone", "budget", "ask", "answer", "run", "record")

OPTIONS = ["approve", "hold"]
APPROVE, HOLD = OPTIONS

# How long a consent waits before nobody is taken to have answered: long
# enough to walk to the machine, short enough that a request nobody saw does
# not sit answerable for a day.
CONSENT_SECONDS = 600

# The rules that can choose a clone, reported as data beside it.
RULE_CWD = "passed as --cwd"
RULE_RECORD = "the clone the project's last act ran in, from qmcp's record"
RULE_SIBLING = "a clone named for the project, beside the others"

# How the pending listing is read: pages of the server's own cap, so a queue
# longer than one page is still searched to its end.
PAGE = 500


@dataclass(frozen=True)
class Clone:
    """Where an instruction is carried out, and what chose it."""

    cwd: Path
    rule: str

    def detail(self) -> dict[str, Any]:
        return {"cwd": str(self.cwd), "rule": self.rule}


@dataclass(frozen=True)
class Acted:
    """What one act did, whether or not it ran anything.

    Every field that can be populated is, on every path: a refused act
    reports the stages it reached and the spend it declared as a run does.
    `status` is the row's status afterwards, which a refusal before the ask
    leaves as it found it.
    """

    instruction_id: str
    status: str
    stages: tuple[str, ...]
    declared: dict[str, Any]
    why: str = ""
    request_id: str | None = None
    cwd: str | None = None
    carried: tuple[str, ...] = ()
    """The ids of the earlier instructions the brief carried."""
    answer: str | None = None
    outcome: AgentOutcome | None = None

    @property
    def ran(self) -> bool:
        """Whether the runtime was reached. Not the same as `done`."""
        return "run" in self.stages


class NoSuchInstruction(LookupError):
    """The id names no row."""


def rows_at(path: Path) -> Rows:
    """Sessions over one SQLite file, with the tables created.

    For a walkthrough or a test; the configured database is somebody's inbox.
    Synchronous, because the act is a command that waits, not a route.
    """
    from sqlmodel import Session, SQLModel, create_engine

    import qmcp.db.models  # noqa: F401  -- registers every server table

    engine = create_engine(f"sqlite:///{Path(path).as_posix()}")
    SQLModel.metadata.create_all(engine)
    return lambda: Session(engine)


def configured_rows() -> Rows:
    """Sessions over the database the settings name. Raises when they name no file."""
    from sqlmodel import Session, create_engine

    from qmcp.config import get_settings
    from qmcp.db.paths import database_file

    url = get_settings().database_url
    found = database_file(url)
    if found is None:
        raise RuntimeError(f"{url}: names no file on disk, so there is no inbox to act on.")
    engine = create_engine(f"sqlite:///{found.as_posix()}")
    return lambda: Session(engine)


def clone_for(rows: Rows, instruction_id: str, project: str | None,
              cwd: str | Path | None, cwd_rule: str = RULE_CWD) -> Clone | None:
    """`cwd` when given, under the rule that chose it; else the clone the
    project's last act ran in; else None."""
    if cwd is not None:
        return Clone(cwd=Path(cwd), rule=cwd_rule)
    remembered = last_clone(rows, project, before=instruction_id)
    return Clone(cwd=remembered, rule=RULE_RECORD) if remembered is not None else None


def consent_prompt(row: Instruction, clone: Clone, runtime: str, budget: Budget,
                   carried: int = 0, spoken: bool = False, command: str | None = None) -> str:
    """What the person at the gate is asked, in full, because they may not be
    the person who recorded the instruction.

    `spoken` is the form said aloud: a few plain words -- where, what, how,
    how many runs in words, and whether it carries what came before. The
    clone is not named: an absolute path read character by character kept a
    consent talking for half a minute. The written prompt keeps everything,
    and names the `command` a runtime that runs one will run.
    """
    if spoken:
        from qmcp.integrations.voice.adapter import counted

        how = {"local": "The local model", "scripted": "A script",
               "check": "A declared check",
               "topology-crosscheck": "The local cross-check topology"}.get(
                   runtime, f"Runtime {runtime}")
        remembers = ", with what came before" if carried else ""
        asked = row.text.strip()
        asked += "" if asked.endswith((".", "?", "!")) else "."
        command_note = f" The command is {command}." if command else ""
        return (f"Run in {row.project or 'no project'}: {asked} "
                f"{how}, {counted(budget.authorised, 'run')}{remembers}."
                f"{command_note}")
    history_note = (f", carrying {carried} earlier instruction(s) from qmcp's record"
                    if carried else "")
    command_note = f", command `{command}`" if command else ""
    return (f"Act on the instruction: {row.text} "
            f"Project {row.project or 'unresolved'}, clone {clone.cwd}, "
            f"runtime {runtime}{command_note}, budget {budget.authorised} run(s){history_note}.")


def _now() -> datetime:
    # Naive UTC, as the rest of this database keeps time.
    return datetime.now(UTC).replace(tzinfo=None)


def _sleep(seconds: float) -> None:
    """The wait between listings; what a test of the command stands in for."""
    time.sleep(seconds)


def _is_pending(client: Any, request_id: str) -> bool:
    """Whether the pending listing still carries the request. Reads nothing else."""
    offset = 0
    while True:
        page = client.list_human_requests(status_filter="pending", limit=PAGE, offset=offset)
        if any(r.id == request_id for r in page):
            return True
        if len(page) < PAGE:
            return False
        offset += PAGE


def act(instruction_id: str, runtime: AgentRuntime, budget: Budget, *, client: Any,
        cwd: str | Path | None = None, cwd_rule: str = RULE_CWD,
        rows: Rows | None = None, stt: Any = None, tts: Any = None,
        on_event: OnEvent | None = None, poll_interval: float = 1.0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        consent_seconds: int = CONSENT_SECONDS) -> Acted:
    """One instruction through the gate, and through the runtime only on approve.

    `client` is the human queue (`qmcp.client.MCPClient` or anything with its
    `create_human_request`, `list_human_requests` and `get_human_request`).
    `rows` is the inbox, defaulting to the configured database; it is also the
    record the clone and the history are read from. `stt` and `tts` together
    answer the consent by voice in this process, through the approval loop a
    `qmcp human voice` would run; without them the act waits for the answer to
    arrive from anywhere. `sleep` and `clock` are the wait and the seconds it
    is measured in; a test hands in both, so a consent nobody answers ends in
    its own time rather than the wall's.

    Returns an `Acted` on every path a caller can provoke. Raises
    `NoSuchInstruction` for an id that names no row, because there is nothing
    to record a refusal against.
    """
    rows = rows or configured_rows()
    # Resolved at the call rather than bound as a default, so a test of the
    # command -- which passes no `sleep` -- can stand in for the wait.
    sleep = sleep if sleep is not None else _sleep
    reached: list[str] = ["instruction"]
    with rows() as session:
        row = session.get(Instruction, instruction_id)
        if row is None:
            raise NoSuchInstruction(f"Instruction '{instruction_id}' not found")
        text, project, status = row.text, row.project, row.status.value

    # --- the clone -----------------------------------------------------------
    reached.append("clone")
    clone = clone_for(rows, instruction_id, project, cwd, cwd_rule)
    if clone is None or not clone.cwd.is_dir():
        reason = (f"no clone for {project!r} in qmcp's record yet"
                  if project else "the instruction's project is unresolved")
        if clone is not None:
            reason = f"{clone.cwd} is not a directory"
        return Acted(instruction_id=instruction_id, status=status, stages=tuple(reached),
                     declared=declare(budget, unknown("nothing was resolved to run in")),
                     why=(f"{reason}; pass --cwd <path to the project's clone>, and it is"
                          " remembered for the project's later instructions."))
    turns = history(rows, project, before=instruction_id)
    carried = tuple(turn.id for turn in turns)

    # --- the budget ----------------------------------------------------------
    reached.append("budget")
    runtime_name = getattr(runtime, "name", type(runtime).__name__)
    budget.service = budget.service or runtime_name
    would_need = getattr(runtime, "would_spend", None)
    if would_need is None:
        would_need = unknown("an agent run makes as many calls as it needs; "
                             "the runtime reports what it made afterwards")
    if budget.free:
        return Acted(instruction_id=instruction_id, status=status, stages=tuple(reached),
                     declared=declare(budget, would_need), cwd=str(clone.cwd),
                     carried=carried,
                     why=("issued against 0 runs, so nothing was asked and nothing ran. "
                          "Re-issue with --budget 1 to ask consent for one run."))

    # --- the ask --------------------------------------------------------------
    reached.append("ask")
    from qmcp.client import HumanRequestConflictError

    request_id, attempt = f"instruction-{instruction_id}", 1
    command = getattr(runtime, "command", None)
    told = len(carried) if getattr(runtime, "uses_history", True) else 0
    prompt = consent_prompt(row, clone, runtime_name, budget, carried=told, command=command)
    declared = declare(budget, would_need)
    while True:
        try:
            client.create_human_request(
                request_id=request_id, request_type="approval", prompt=prompt,
                options=list(OPTIONS), timeout_seconds=consent_seconds,
                context={"instruction_id": instruction_id, "project": project,
                         "cwd": str(clone.cwd), "runtime": runtime_name,
                         "carried": list(carried), "spend": declared,
                         "spoken": consent_prompt(row, clone, runtime_name, budget,
                                                  carried=told, spoken=True, command=command)})
            break
        except HumanRequestConflictError:
            # Acted on before: the earlier consent stands as its own record,
            # and this one is numbered beside it rather than overwriting it.
            attempt += 1
            request_id = f"instruction-{instruction_id}-{attempt}"
    _update(rows, instruction_id, status=InstructionStatus.ASKING,
            consent_request_id=request_id, runtime=runtime_name, cwd=str(clone.cwd),
            declared=declared, detail={"clone": clone.detail(), "continuity": list(carried)})
    if on_event:
        on_event("asking", request_id)

    if stt is not None and tts is not None:
        from qmcp.integrations.voice.adapter import UnclearResponse, VoiceApprovalLoop

        try:
            VoiceApprovalLoop(stt=stt, tts=tts, client=client).run_once(request_id)
        except UnclearResponse:
            # Nothing was guessed. The request is still pending and still
            # answerable from anywhere, so the wait below goes on.
            pass

    # --- the answer -------------------------------------------------------------
    reached.append("answer")
    deadline = clock() + consent_seconds + poll_interval * 2
    while _is_pending(client, request_id):
        if clock() > deadline:
            break
        sleep(poll_interval)
    _, response = client.get_human_request(request_id)
    answer = response.response if response is not None else None

    if answer != APPROVE:
        ended = InstructionStatus.REFUSED if answer is not None else InstructionStatus.UNANSWERED
        _update(rows, instruction_id, status=ended, declared=declared, acted_at=_now())
        return Acted(instruction_id=instruction_id, status=ended.value, stages=tuple(reached),
                     declared=declared, request_id=request_id, cwd=str(clone.cwd),
                     carried=carried, answer=answer,
                     why=(f"the consent was answered {answer!r}; nothing ran" if answer
                          else "nobody answered the consent before it expired; nothing ran"))

    # --- the run ------------------------------------------------------------------
    _update(rows, instruction_id, status=InstructionStatus.CONSENTED)
    reached.append("run")
    budget.spend(1)
    _update(rows, instruction_id, status=InstructionStatus.ACTING)
    if on_event:
        on_event("acting", str(clone.cwd))
    brief = Brief(instruction=text, cwd=clone.cwd, project=project, history=turns)
    try:
        outcome = runtime.run(brief, on_event=on_event)
    except Exception as exc:  # noqa: BLE001 -- a runtime that raised is a failed run, recorded
        outcome = AgentOutcome(text=f"{type(exc).__name__}: {exc}", exit_code=-1,
                               spent=unknown("the runtime raised before reporting"))

    # --- the record ----------------------------------------------------------------
    reached.append("record")
    declared = declare(budget, outcome.spent)
    ended = InstructionStatus.DONE if outcome.succeeded else InstructionStatus.FAILED
    _update(rows, instruction_id, status=ended, declared=declared, acted_at=_now(),
            outcome_text=outcome.text, exit_code=outcome.exit_code,
            detail={"outcome": {"elapsed_seconds": outcome.elapsed_seconds,
                                "argv": list(outcome.argv), "spent": outcome.spent,
                                **outcome.detail}})
    return Acted(instruction_id=instruction_id, status=ended.value, stages=tuple(reached),
                 declared=declared, request_id=request_id, cwd=str(clone.cwd),
                 carried=carried, answer=answer, outcome=outcome)


def _update(rows: Rows, instruction_id: str, *, detail: dict | None = None,
            **fields: Any) -> None:
    """Write `fields` onto the row and move `updated_at`. `detail` is merged key
    by key, never replaced: the evidence for the project stays beside the
    evidence for the clone, the continuity and the outcome."""
    with rows() as session:
        row = session.get(Instruction, instruction_id)
        for name, value in fields.items():
            setattr(row, name, value)
        if detail:
            row.detail = {**(row.detail or {}), **detail}
        row.updated_at = _now()
        session.add(row)
        session.commit()


__all__ = [
    "APPROVE",
    "CONSENT_SECONDS",
    "HOLD",
    "OPTIONS",
    "RULE_CWD",
    "RULE_RECORD",
    "RULE_SIBLING",
    "STAGES",
    "Acted",
    "Clone",
    "NoSuchInstruction",
    "act",
    "clone_for",
    "configured_rows",
    "consent_prompt",
    "rows_at",
]
