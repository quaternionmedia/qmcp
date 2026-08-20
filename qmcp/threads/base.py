"""The contract every thread source implements, and what it refuses to assume.

A thread is a conversation with an assistant. This pulls them and turns what
they *produced* into units of work. It is deliberately not a transcript
importer.

FOUR DECISIONS THIS OBEYS, EACH FROM A RECORD, EACH VISIBLE IN THE SHAPE HERE.

**A thread is mostly steps.** A hundred turns producing three decisions. The
three are the deltas; the turns are not. `DRAFT-granularity-is-a-perspective.md`
is why the question "is a turn a delta" had no answer: it was missing its
subject. So a source emits from a named perspective, and every payload says
which -- a receiver that had to guess would flatten the thread into its own
level.

**`survey` may not spend, ever.** It is the free-first pass from
`DRAFT-no-unattended-spending.md` 3: establish what the paid work would cost
without spending to find out. A source that cannot establish it for free says
`unknown` with a reason rather than guessing a number or quietly making a call.
`survey` takes no budget, which is the enforcement -- there is nothing for it to
spend against.

**Zero is a real count and never a sentinel.** `available=0` means there is
genuinely nothing there. `available=unknown(...)` means nobody could look. The
two are different claims and the second must never render as the first.

**Nothing here is scheduled.** Every method that spends takes a `Budget` a
person issued for one command. There is no retry, no backfill, and no method
that decides to make a second call.

WHAT A SOURCE STILL OWES ITS READER. Whether extraction is honest. This module
can hold a source to the shape of the answer and cannot tell whether the
decisions it pulled out of a thread are the decisions the thread made. That is
the part a person reads, and it is why `Decision` carries the turns it came
from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qmcp.spend import Budget, unknown

# Kept in step with `dossier`'s lifecycle by the payload contract rather than by
# an import. A decision somebody has recorded and nobody has acted on is
# `brainstorm`: noticing is not deciding, which is the same rule the self-check
# follows for a failing gate.
NOTICED = "brainstorm"

# What a delta produced from a thread points back at. `link_type` values are the
# consumer's own vocabulary; `thread` and `turn` are new here and are what let a
# reader get from a unit of work to the conversation that produced it.
THREAD_LINK = "thread"
TURN_LINK = "turn"


@dataclass(frozen=True)
class Turn:
    """One message in a thread.

    A turn is a step, not a unit of work. It is here so a decision can point at
    where it came from, which is the only way a reader can check the extraction
    rather than trusting it.
    """

    id: str
    role: str
    at: str | None = None
    text: str = ""


@dataclass(frozen=True)
class Thread:
    """One conversation, as pulled."""

    id: str
    title: str | None = None
    started_at: str | None = None
    url: str | None = None
    turns: tuple[Turn, ...] = ()

    @property
    def partial(self) -> bool:
        """True when the turns are an excerpt rather than the whole thread.

        A thread pulled under a budget that ran out is partial, and a consumer
        counting turns would otherwise report the size of the excerpt and call
        it the conversation -- the same trap the harness payload names about
        its own `recent` list.
        """
        return self._partial

    _partial: bool = False


@dataclass(frozen=True)
class Decision:
    """Something a thread settled, which is what becomes a unit of work.

    `from_turns` is not decoration. A source claiming a thread produced a
    decision is making a claim, and the turns are what a person checks it
    against. A decision citing no turn is refused rather than stored.
    """

    name: str
    title: str
    summary: str = ""
    from_turns: tuple[str, ...] = ()
    delta_type: str = "chore"
    priority: str = "medium"


@dataclass
class Survey:
    """What a source could establish about itself without spending anything.

    `available` and `would_need` are separate because they answer different
    questions: how many threads exist, and how many paid calls pulling them
    would take. Either may be `unknown`, and a source that guesses one to avoid
    saying so is the failure this type exists to make awkward.
    """

    source: str
    available: int | dict[str, str] = field(
        default_factory=lambda: unknown("nothing established"))
    would_need: int | dict[str, str] = field(
        default_factory=lambda: unknown("nothing established"))
    note: str | None = None

    @property
    def established(self) -> bool:
        """True when both figures are numbers somebody could act on."""
        return isinstance(self.available, int) and isinstance(self.would_need, int)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "available": self.available,
            "would_need": self.would_need,
            "note": self.note,
        }


def to_delta(decision: Decision, thread: Thread, *, project: str,
             perspective: str) -> dict[str, Any]:
    """One decision as a delta payload, naming the perspective it speaks from.

    PURE, AND SHARED BY EVERY SOURCE. What differs between assistants is how a
    thread is fetched and how a decision is recognised; what a delta looks like
    does not, and a source free to shape its own payload is a source whose rows
    a consumer has to special-case.

    The perspective is required rather than defaulted. A default here would be a
    silent claim about level -- `DRAFT-granularity-is-a-perspective.md` 2 -- and
    the whole reason a thread's turns can stay steps is that somebody said at
    what level this source speaks.
    """
    if not perspective:
        raise ValueError(
            "a delta from a thread names its perspective. Without one a "
            "receiver does not know at what level it was told this, and will "
            "flatten it into its own."
        )
    if not decision.from_turns:
        raise ValueError(
            f"{decision.name!r} cites no turn. A decision a source cannot point "
            f"at is one a person cannot check, and an unfalsifiable row is "
            f"worse on a board than an absent one."
        )

    links: list[dict[str, Any]] = [
        {"link_type": "address", "target_id": None,
         "target_name": f"{project}/delta/{decision.name}"},
        {"link_type": THREAD_LINK, "target_id": None,
         "target_name": thread.url or thread.id},
    ]
    links += [
        {"link_type": TURN_LINK, "target_id": None, "target_name": turn}
        for turn in decision.from_turns
    ]

    return {
        "schema": 1,
        "project": project,
        # The field the granularity record requires. Carried beside the row
        # rather than inside it, because it is not a `ProjectDelta` column: it
        # describes the claim, not the work.
        "perspective": perspective,
        "delta": {
            "name": decision.name,
            "title": decision.title,
            "description": decision.summary,
            # Never past `brainstorm` from an extraction. A source recognising a
            # decision has noticed something; it has not established that
            # anybody acted, and a row opened at `planning` would assert they
            # had.
            "phase": NOTICED,
            "delta_type": decision.delta_type,
            "priority": decision.priority,
        },
        "links": links,
    }


class ThreadSource(ABC):
    """One assistant's threads, behind a contract that keeps the spending honest.

    A subclass implements three things and inherits the fourth. `to_delta` is
    not overridable in spirit: a source that shapes its own payload is a source
    whose rows every consumer special-cases.
    """

    #: Short name, used in messages and in `Survey.source`.
    name: str = ""

    #: At what level this source's deltas speak. Required, and there is no
    #: sensible default -- see `to_delta`.
    perspective: str = ""

    #: `owner/repo` the emitted deltas belong to. A thread belongs to no
    #: repository, so a person says which one owns the work it produced.
    #: `plans/qmpm-standardisations.md` 1 has the open question about whether a
    #: delta may span owners; until that is decided, somebody chooses.
    project: str = ""

    def __init__(self, *, project: str = "", perspective: str = "") -> None:
        self.project = project or self.project
        self.perspective = perspective or self.perspective

    # --- the free pass ------------------------------------------------------

    @abstractmethod
    def survey(self) -> Survey:
        """What is here, established without spending anything.

        MUST NOT MAKE A PAID CALL. It takes no budget, so there is nothing for
        it to spend against, and a source that reaches around that is breaking
        the record rather than this signature.

        A source whose listing is itself metered returns `unknown` with that as
        the reason. Saying "I cannot tell you for free" is a useful answer; a
        guessed number is not.
        """

    # --- the passes that spend ----------------------------------------------

    @abstractmethod
    def fetch(self, ids: list[str], budget: Budget) -> list[Thread]:
        """Pull the named threads, spending against `budget`.

        Every call goes through `budget.spend()` before it is made. A budget
        that runs out stops the pull; the threads already pulled are returned
        and the ones that were not are simply absent, which is why `Thread`
        carries `partial` rather than leaving a consumer to infer completeness.
        """

    @abstractmethod
    def decisions(self, thread: Thread, budget: Budget) -> list[Decision]:
        """What this thread settled.

        Takes a budget because recognising a decision may itself be a paid
        call. A source that does it with a heuristic simply never spends, and
        that is a difference a reader should be able to see rather than assume.
        """

    # --- inherited, and the same for every source ---------------------------

    def deltas(self, thread: Thread, budget: Budget) -> list[dict[str, Any]]:
        """This thread's decisions, as payloads a control panel ingests."""
        return [
            to_delta(decision, thread, project=self.project,
                     perspective=self.perspective)
            for decision in self.decisions(thread, budget)
        ]

    def describe(self) -> str:
        """What this source is and what it would cost, for a person deciding."""
        survey = self.survey()
        lines = [f"  {self.name or type(self).__name__}"]
        if self.perspective:
            lines.append(f"    speaks from   {self.perspective}")
        if self.project:
            lines.append(f"    deltas go to  {self.project}")

        for label, value in (("threads", survey.available),
                             ("paid calls", survey.would_need)):
            if isinstance(value, dict) and "unknown" in value:
                lines.append(f"    {label:<13} unknown -- {value['unknown']}")
            else:
                lines.append(f"    {label:<13} {value}")
        if survey.note:
            lines.append(f"    {survey.note}")
        if not survey.established:
            lines.append("    Nothing was spent establishing this, and an "
                         "unknown is not a zero.")
        return "\n".join(lines)
