"""The contract every thread source implements.

A thread is a conversation with an assistant. A source pulls threads and turns
what they settled into delta payloads; the turns themselves are not deltas.

- Every payload names the perspective it speaks from, and a source has no
  default perspective.
- `survey` takes no budget and makes no paid call. A figure it cannot
  establish for free is `unknown` with a reason.
- `0` is a count; `unknown(...)` means nothing was counted.
- Every method that spends takes a `Budget` issued for one command. There is
  no retry, no backfill and no second call.

`docs/threads.md` describes the contract at length.

What this cannot do: tell whether the decisions a source extracted are the
ones the thread made. `Decision` carries the turns it came from so a person
can check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qmcp.spend import Budget, unknown

# `dossier`'s lifecycle phase for something noticed and not acted on, kept in
# step by the payload contract rather than by an import. The self-check opens a
# failing gate at the same phase.
NOTICED = "brainstorm"

# What a delta produced from a thread points back at. `thread` and `turn` are
# `link_type` values this module adds to the consumer's vocabulary; they lead
# from a unit of work to the conversation that produced it.
THREAD_LINK = "thread"
TURN_LINK = "turn"

# A thread is itself a delta, not an address kind of its own: it is addressed
# `<owner>/<repo>/delta/thread-<id>`, and each decision it settled is `part-of`
# it.
THREAD_TYPE = "thread"
PART_OF = "part-of"


@dataclass(frozen=True)
class Turn:
    """One message in a thread.

    A turn is a step, not a unit of work. A decision points at the turns it
    came from, so a reader can check the extraction.
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

        A thread pulled under a budget that ran out is partial; its turn count
        is the size of the excerpt, not of the conversation.
        """
        return self._partial

    _partial: bool = False


@dataclass(frozen=True)
class Decision:
    """Something a thread settled, which is what becomes a unit of work.

    `from_turns` names the turns a person checks the decision against. A
    decision citing no turn is refused by `to_delta`.
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

    `available` is how many threads exist; `would_need` is how many paid calls
    pulling them would take. Either may be `unknown` with a reason.
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

    Pure, and shared by every source: sources differ in how they fetch a
    thread and recognise a decision, never in the payload's shape.

    Raises `ValueError` when `perspective` is empty
    (`DRAFT-granularity-is-a-perspective.md` 2) or the decision cites no turn.
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
        # Required by the granularity record. Beside the row rather than in
        # it: it describes the claim, and is not a `ProjectDelta` column.
        "perspective": perspective,
        "delta": {
            "name": decision.name,
            "title": decision.title,
            "description": decision.summary,
            # Never past `brainstorm` from an extraction: recognising a
            # decision does not establish that anybody acted on it.
            "phase": NOTICED,
            "delta_type": decision.delta_type,
            "priority": decision.priority,
        },
        "links": links,
    }


def thread_name(thread: Thread) -> str:
    """The delta name for a thread itself.

    Prefixed with `thread-`, because thread ids and decision names share one
    delta namespace and must not collide.
    """
    return f"thread-{thread.id}"


def to_thread_delta(thread: Thread, *, project: str,
                    perspective: str) -> dict[str, Any]:
    """The conversation itself, as a delta.

    Its phase is `brainstorm`, like anything else a detector noticed. Nothing
    here can establish that a thread was acted on; a later phase is set by a
    person.
    """
    if not perspective:
        raise ValueError("a thread delta names the perspective it speaks from")

    return {
        "schema": 1,
        "project": project,
        "perspective": perspective,
        "delta": {
            "name": thread_name(thread),
            "title": thread.title or f"Thread {thread.id}",
            "description": (thread.url or ""),
            "phase": NOTICED,
            "delta_type": THREAD_TYPE,
            "priority": "medium",
        },
        "links": [
            {"link_type": "address", "target_id": None,
             "target_name": f"{project}/delta/{thread_name(thread)}"},
            {"link_type": THREAD_LINK, "target_id": None,
             "target_name": thread.url or thread.id},
        ],
    }


def relations_for(thread: Thread, decisions: list[Decision], *,
                  project: str) -> list[dict[str, str]]:
    """Each decision is `part-of` the thread that settled it.

    Stated explicitly, because a consumer must not infer containment from two
    rows sharing a prefix (`DRAFT-deltas-compose.md` 5).
    """
    whole = f"{project}/delta/{thread_name(thread)}"
    return [
        {"source": f"{project}/delta/{decision.name}",
         "relation": PART_OF,
         "target": whole}
        for decision in decisions
    ]


class ThreadSource(ABC):
    """One assistant's threads, behind a contract that keeps the spending honest.

    A subclass implements `survey`, `fetch` and `decisions`, and inherits
    `deltas` and `relations`, which build every payload through `to_delta`.
    """

    #: Short name, used in messages and in `Survey.source`.
    name: str = ""

    #: At what level this source's deltas speak. Required, and there is no
    #: sensible default -- see `to_delta`.
    perspective: str = ""

    #: `owner/repo` the emitted deltas belong to. A thread belongs to no
    #: repository, so the source names the one that owns the work it produced.
    #: Whether a delta may span owners is open in
    #: `plans/qmpm-standardisations.md` 1.
    project: str = ""

    def __init__(self, *, project: str = "", perspective: str = "") -> None:
        self.project = project or self.project
        self.perspective = perspective or self.perspective

    # --- the free pass ------------------------------------------------------

    @abstractmethod
    def survey(self) -> Survey:
        """What is here, established without spending anything.

        Makes no paid call: it takes no budget, and
        `DRAFT-no-unattended-spending.md` 3 forbids reaching around that.

        A source whose listing is itself metered returns `unknown` with that as
        the reason, never a guessed number.
        """

    # --- the passes that spend ----------------------------------------------

    @abstractmethod
    def fetch(self, ids: list[str], budget: Budget) -> list[Thread]:
        """Pull the named threads, spending against `budget`.

        Every call goes through `budget.spend()` before it is made. A budget
        that runs out stops the pull: the threads already pulled are returned,
        the rest are absent, and a thread cut short has `partial` set.
        """

    @abstractmethod
    def decisions(self, thread: Thread, budget: Budget) -> list[Decision]:
        """What this thread settled.

        Takes a budget because recognising a decision may itself be a paid
        call. A source that recognises decisions with a heuristic never spends.
        """

    # --- inherited, and the same for every source ---------------------------

    def deltas(self, thread: Thread, budget: Budget) -> list[dict[str, Any]]:
        """The thread, and what it settled, as payloads a control panel ingests.

        The thread comes first and is always emitted, so a thread that settled
        nothing still appears.
        """
        found = self.decisions(thread, budget)
        return [to_thread_delta(thread, project=self.project,
                                perspective=self.perspective)] + [
            to_delta(decision, thread, project=self.project,
                     perspective=self.perspective)
            for decision in found
        ]

    def relations(self, thread: Thread, budget: Budget) -> list[dict[str, str]]:
        """How what this thread settled hangs off the thread itself."""
        return relations_for(thread, self.decisions(thread, budget),
                             project=self.project)

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
