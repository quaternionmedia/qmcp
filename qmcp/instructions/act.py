"""Acting on an instruction: declare what it would spend, ask, and run only on approve.

    uv run qmcp instructions act <id> --runtime <name> [--budget N] [--cwd PATH] [--voice]

**NOTHING RUNS BEFORE A PERSON SAYS APPROVE, AND NOTHING RUNS ON ANY OTHER
ANSWER.** The inbox records an instruction and stops (`qmcp.instructions`).
This is the step after it, and it is shaped by `qmcp.governed`: a fixed
sequence of stages with exactly one that has no halting guarantee -- the agent
-- budgeted before it and recorded after it, with the human gate in front of it
rather than behind. `governance/qm/records/DRAFT-no-unattended-spending.md`
is the rule. The command is issued by a person (clause 1), it states the number
of runs it may make before it asks (clause 2), zero is its default and a real
count (clause 3), and the consent it asks for is for this run and does not
carry (clause 5).

**THE CLONE COMES FROM THE ARCHIVE, SO THE WORK CONTINUES WHERE IT WAS.** The
thread archive knows which checkout each session worked in and which session
it was (`qmcp.threads.claudecode` reads both). The most recently active thread
about the instruction's project whose checkout still exists on disk names the
clone and the session to resume, so an instruction lands in the conversation
that was already doing the project's work rather than in a fresh one that has
to rediscover it. **An explicit `--cwd` wins, and the archive is not read**: a
path the person typed is what they meant, and it starts a fresh session there,
so leaving it out is how to continue the archive's. With neither -- a project
nobody has worked on here, or an instruction whose project is unresolved --
the act refuses, leaves the row as it was, and says what to pass. A thread is
about a project when `qmcp.threads.consolidate.about` reads
it so, or when its checkout is a directory named for the project; the rule
that chose the clone is kept in the row's `detail`, as the rule that chose
the project is.

**THE CONSENT IS THE EXISTING VOICE APPROVAL.** The request put on the human
queue is an ordinary approval with the options `approve` and `hold`, so it is
answered wherever approvals are: `qmcp human voice`, `qmcp human respond`, a
page, or by voice in this process when the command is given `--voice`. Its
prompt says the instruction, the project, the clone, the runtime and the
declared budget, because the person answering may not be the person who
recorded it, and it expires after `CONSENT_SECONDS`: a consent nobody gave
within that is `unanswered`, and the row says so.

**THE WAIT READS THE PENDING LISTING AND NOTHING ELSE.** `AGENTS.md` records
that reading one request expires it when its time has passed, and the listing
applies no expiry. So the wait watches the listing until the request leaves
it -- answered, or past its expiry -- and reads the request itself exactly
once, afterwards, to learn which. Polling the single route would be a wait
that expires the gate it is waiting on. Behind the listing is a deadline on
the act's own clock, two polls past the expiry, for a server whose listing
never drops the request: the act then records `unanswered` and runs nothing,
while the request may still sit pending there, answerable by anyone.

**THE DECLARATION IS WRITTEN ON EVERY PATH**, as `qmcp.governed.Outcome`
carries one on every path: a refusal before anything was asked, a hold, an
expiry and a run all leave `qmcp.spend.declare` on the row, so a reader can
tell which happened without inferring it from which fields are empty. The
budget counts runs; what a run cost in calls is the runtime's own report and
is kept beside it, never folded into it.

WHAT THIS CANNOT DO. Stop a runtime once it is running, or know what it will
spend before it does -- `would_need` is unknown until the runtime reports, and
a count it does not report stays unknown rather than becoming zero. Nor can it
tell that the clone the archive named is the one the person meant: the prompt
says which, and the person at the gate is the check.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qmcp.db.models import Instruction, InstructionStatus
from qmcp.integrations.agents import AgentOutcome, AgentRuntime, OnEvent
from qmcp.spend import Budget, declare, unknown
from qmcp.threads.consolidate import about

Rows = Callable[[], Any]
"""A callable returning a context manager that yields a `sqlmodel.Session` over
the server database, committed by the caller -- what `rows_at` builds, and
what the configured database is wrapped in by the command."""

# The stages, declared once, so a reader of a result sees how far an act got
# without reading `act`. `run` is the one with no halting guarantee.
STAGES: tuple[str, ...] = ("instruction", "clone", "budget", "ask", "answer", "run", "record")

OPTIONS = ["approve", "hold"]
APPROVE, HOLD = OPTIONS

# How long a consent waits before nobody is taken to have answered. Ten
# minutes: long enough to walk to the machine, short enough that a request
# nobody saw does not sit answerable for a day.
CONSENT_SECONDS = 600

# The rules `clone_for` can report, as data, beside the clone it chose.
RULE_ARCHIVE = "the checkout of the most recently active thread about the project"
RULE_NAMED_DIR = "the checkout is a directory named for the project"
RULE_CWD = "passed as --cwd"

# How the pending listing is read: pages of the server's own cap, so a queue
# longer than one page is still searched to its end.
PAGE = 500


@dataclass(frozen=True)
class Clone:
    """Where an instruction is carried out, and what chose it."""

    cwd: Path
    session_ref: str | None
    rule: str
    thread_id: str | None = None
    last_at: str | None = None

    def detail(self) -> dict[str, Any]:
        return {"cwd": str(self.cwd), "session_ref": self.session_ref,
                "rule": self.rule, "thread": self.thread_id, "last_at": self.last_at}


@dataclass(frozen=True)
class Acted:
    """What one act did, whether or not it ran anything.

    **EVERY FIELD THAT CAN BE POPULATED IS, ON EVERY PATH.** A refused act
    reports the stages it reached and the spend it declared exactly as a run
    does; `status` is the row's status afterwards, which a refusal before the
    ask leaves as it found it.
    """

    instruction_id: str
    status: str
    stages: tuple[str, ...]
    declared: dict[str, Any]
    why: str = ""
    request_id: str | None = None
    cwd: str | None = None
    session_ref: str | None = None
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


def archive_sources() -> list[Any]:
    """The configured archive stores, as the thread routes read them.

    Every store, including the web exports that know nothing of checkouts:
    `clone_for` reads only those that carry a `context`, and never fetches
    the rest.
    """
    from qmcp.threads.cache import DEFAULT_ROOT
    from qmcp.threads.service import sources_for

    return list(sources_for(DEFAULT_ROOT).values())


def clone_for(project: str | None, sources: Iterable[Any],
              exists: Callable[[Path], bool] = Path.is_dir) -> Clone | None:
    """The clone the archive names for `project`, or None.

    The most recently active thread about the project whose checkout still
    exists. Threads are read, spending nothing, from every source handed in
    that carries a `context` -- what `qmcp.threads.claudecode` keeps per
    thread -- and a source without one is never fetched, because it has
    nothing to say about checkouts. A thread is about the project when its
    checkout is a directory named for it (exactly, ignoring case: a sibling
    carrying the name as a prefix is another checkout), or by
    `consolidate.about`'s rule for this project alone -- named in the title,
    or in at least two turns. The roster is not read here, so the survey
    reading `qmcp threads consolidate` makes across the whole roster is not
    made, and a thread that surveyed the workspace counts as about each
    project it named.
    """
    if not project:
        return None
    short = project.rsplit("/", 1)[-1]
    found: list[Clone] = []
    for source in sources:
        if not hasattr(source, "context"):
            continue
        threads = source.fetch([], Budget())
        contexts = source.context or {}
        for thread in threads:
            context = contexts.get(thread.id) or {}
            cwd = context.get("cwd")
            if not cwd:
                continue
            path = Path(cwd)
            if path.name.lower() == short.lower():
                rule = RULE_NAMED_DIR
            elif short in about(thread, [short]).projects:
                rule = RULE_ARCHIVE
            else:
                continue
            if not exists(path):
                continue
            found.append(Clone(cwd=path, session_ref=context.get("session"), rule=rule,
                               thread_id=thread.id, last_at=context.get("last_at")))
    if not found:
        return None
    # Newest activity first. ISO timestamps order as text; a thread with no
    # timestamp sorts last, since nothing says it was recent.
    found.sort(key=lambda c: c.last_at or "", reverse=True)
    return found[0]


def consent_prompt(row: Instruction, clone: Clone, runtime: str, budget: Budget) -> str:
    """What the person at the gate is asked, in full, because they may not be
    the person who recorded the instruction."""
    return (f"Act on the instruction: {row.text} "
            f"Project {row.project or 'unresolved'}, clone {clone.cwd}, "
            f"runtime {runtime}, budget {budget.authorised} run(s)"
            + (f", continuing session {clone.session_ref}." if clone.session_ref else "."))


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
        cwd: str | Path | None = None, rows: Rows | None = None,
        sources: Iterable[Any] | None = None, stt: Any = None, tts: Any = None,
        on_event: OnEvent | None = None, poll_interval: float = 1.0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        consent_seconds: int = CONSENT_SECONDS) -> Acted:
    """One instruction through the gate, and through the runtime only on approve.

    `client` is the human queue (`qmcp.client.MCPClient` or anything with its
    `create_human_request`, `list_human_requests` and `get_human_request`).
    `rows` is the inbox, defaulting to the configured database. `sources` is
    the archive, defaulting to the configured stores. `stt` and `tts` together
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
    if cwd is not None:
        clone = Clone(cwd=Path(cwd), session_ref=None, rule=RULE_CWD)
    else:
        clone = clone_for(project, archive_sources() if sources is None else sources)
    if clone is None or not clone.cwd.is_dir():
        reason = (f"no checkout for {project!r} in the thread archive"
                  if project else "the instruction's project is unresolved")
        if clone is not None:
            reason = f"{clone.cwd} is not a directory"
        return Acted(instruction_id=instruction_id, status=status, stages=tuple(reached),
                     declared=declare(budget, unknown("nothing was resolved to run in")),
                     why=f"{reason}; pass --cwd <path to the project's clone>.")

    # --- the budget ----------------------------------------------------------
    reached.append("budget")
    runtime_name = getattr(runtime, "name", type(runtime).__name__)
    budget.service = budget.service or runtime_name
    would_need = unknown("an agent run makes as many calls as it needs; "
                         "the runtime reports what it made afterwards")
    if budget.free:
        return Acted(instruction_id=instruction_id, status=status, stages=tuple(reached),
                     declared=declare(budget, would_need), cwd=str(clone.cwd),
                     session_ref=clone.session_ref,
                     why=("issued against 0 runs, so nothing was asked and nothing ran. "
                          "Re-issue with --budget 1 to ask consent for one run."))

    # --- the ask --------------------------------------------------------------
    reached.append("ask")
    from qmcp.client import HumanRequestConflictError

    request_id, attempt = f"instruction-{instruction_id}", 1
    prompt = consent_prompt(row, clone, runtime_name, budget)
    declared = declare(budget, would_need)
    while True:
        try:
            client.create_human_request(
                request_id=request_id, request_type="approval", prompt=prompt,
                options=list(OPTIONS), timeout_seconds=consent_seconds,
                context={"instruction_id": instruction_id, "project": project,
                         "cwd": str(clone.cwd), "runtime": runtime_name,
                         "session_ref": clone.session_ref, "spend": declared})
            break
        except HumanRequestConflictError:
            # Acted on before: the earlier consent stands as its own record,
            # and this one is numbered beside it rather than overwriting it.
            attempt += 1
            request_id = f"instruction-{instruction_id}-{attempt}"
    _update(rows, instruction_id, status=InstructionStatus.ASKING,
            consent_request_id=request_id, runtime=runtime_name, cwd=str(clone.cwd),
            session_ref=clone.session_ref, declared=declared,
            detail_clone=clone.detail())
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
                     session_ref=clone.session_ref, answer=answer,
                     why=(f"the consent was answered {answer!r}; nothing ran" if answer
                          else "nobody answered the consent before it expired; nothing ran"))

    # --- the run ------------------------------------------------------------------
    _update(rows, instruction_id, status=InstructionStatus.CONSENTED)
    reached.append("run")
    budget.spend(1)
    _update(rows, instruction_id, status=InstructionStatus.ACTING)
    if on_event:
        on_event("acting", str(clone.cwd))
    try:
        outcome = runtime.run(text, clone.cwd, resume=clone.session_ref, on_event=on_event)
    except Exception as exc:  # noqa: BLE001 -- a runtime that raised is a failed run, recorded
        outcome = AgentOutcome(text=f"{type(exc).__name__}: {exc}", exit_code=-1,
                               session_ref=clone.session_ref,
                               spent=unknown("the runtime raised before reporting"))

    # --- the record ----------------------------------------------------------------
    reached.append("record")
    declared = declare(budget, outcome.spent)
    ended = InstructionStatus.DONE if outcome.succeeded else InstructionStatus.FAILED
    _update(rows, instruction_id, status=ended, declared=declared, acted_at=_now(),
            outcome_text=outcome.text, exit_code=outcome.exit_code,
            session_ref=outcome.session_ref or clone.session_ref,
            detail_outcome={"elapsed_seconds": outcome.elapsed_seconds,
                            "argv": list(outcome.argv), "spent": outcome.spent})
    return Acted(instruction_id=instruction_id, status=ended.value, stages=tuple(reached),
                 declared=declared, request_id=request_id, cwd=str(clone.cwd),
                 session_ref=outcome.session_ref or clone.session_ref, answer=answer,
                 outcome=outcome)


def _update(rows: Rows, instruction_id: str, *, detail_clone: dict | None = None,
            detail_outcome: dict | None = None, **fields: Any) -> None:
    """Write `fields` onto the row and move `updated_at`. `detail` is merged,
    never replaced: the evidence for the project stays beside the evidence
    for the clone and the outcome."""
    with rows() as session:
        row = session.get(Instruction, instruction_id)
        for name, value in fields.items():
            setattr(row, name, value)
        if detail_clone is not None or detail_outcome is not None:
            detail = dict(row.detail or {})
            if detail_clone is not None:
                detail["clone"] = detail_clone
            if detail_outcome is not None:
                detail["outcome"] = detail_outcome
            row.detail = detail
        row.updated_at = _now()
        session.add(row)
        session.commit()


__all__ = [
    "APPROVE",
    "CONSENT_SECONDS",
    "HOLD",
    "OPTIONS",
    "RULE_ARCHIVE",
    "RULE_CWD",
    "RULE_NAMED_DIR",
    "STAGES",
    "Acted",
    "Clone",
    "NoSuchInstruction",
    "act",
    "archive_sources",
    "clone_for",
    "configured_rows",
    "consent_prompt",
    "rows_at",
]
