"""The one door a model's output comes through, and what it is when it arrives.

    uv run qmcp topology show governed --level 2

**THE PIPELINE IS DETERMINISTIC AND ONE BOX INSIDE IT IS NOT.** That asymmetry
is the whole design. `governance/qm/PRINCIPLES.md` P17 asks for the black box's
surface to be minimised and metered at one seam, and this module is that seam:
a fixed sequence of total stages with exactly one call to something that has no
halting guarantee, budgeted before it and bounded after it.

**IT ENDS AT THE HUMAN GATE AND DOES NOT PASS IT.** What comes back is a draft.
There is no function here that turns a draft into a decision, because answering
a question in the human-in-the-loop queue is one of the acts
`governance/qm/ci/attested-registry.yaml` reserves for a person. `queued()`
builds the payload that puts a draft in front of somebody; posting it and
answering it are two different acts and only the first is a machine's.

**THE STAGES ARE DECLARED ONCE AND THE DRAWING IS DERIVED FROM THEM.** `STAGES`
is what `run` walks and what `view` draws, so the picture cannot describe a
pipeline the code does not run -- P12, and the drift this organisation keeps
finding whenever a behaviour is described in a second place beside the code.

**NO VENDOR APPEARS HERE.** The model is a callable the caller passes in. A
module that imported one would be this repository choosing a supplier on behalf
of every reader of the corpus, which `governance/qm` forbids and
`qmcp.localmodel` is the declared exception to.

WHAT THIS CANNOT DO, STATED SO NOBODY READS MORE INTO IT:

- **It cannot stop a slow call.** `Bound.seconds` is measured and reported,
  never enforced: interrupting a callable this module does not own is not
  something Python offers for free, and a guard that claimed to would be a
  green check standing where a reader believes something is enforced.
- **It cannot stop a caller reading a draft and acting on it.** It can only
  make sure the thing they read says what it is.
- **It cannot govern a paid call made somewhere else.** Like `qmcp.spend`, it
  is what a module that spends must pass through, and a surface added without
  passing through it is the failure this depends on people not committing.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable

from qmcp.spend import Budget, Refused, declare
from qmcp.topology_view import (Arrow, Box, FEEDBACK, FLOWS, GATE, INPUT,
                                LEVELS, OUTPUT, REFUSAL, View, WORKER)

# What a run ended as. Three, and they are exhaustive: `run` returns one of
# these or raises, and it raises only for a caller's own mistake.
REFUSED = "refused"
"""The budget would have been exceeded. Nothing was called."""

STOPPED = "stopped"
"""A bound fired. Something was called and what came back is not usable."""

DRAFTED = "drafted"
"""One call was made and a draft came back."""

STATES = (REFUSED, STOPPED, DRAFTED)

# The default ceiling on what one call may return. Generous on purpose: the
# bound exists to catch a runaway, not to trim an answer, and a bound that
# fires on ordinary work teaches people to raise it without reading it.
MAX_CHARS = 200_000


@dataclass(frozen=True)
class Bound:
    """A ceiling on one call, and which parts of it are enforced.

    **`chars` IS ENFORCED HERE AND `seconds` IS NOT**, and the difference is
    named in the field rather than left in a docstring somebody may not reach.
    Both are reported. P17's second obligation is that a bound which fires is
    reported and never absorbed, and reporting an unenforced measurement is
    within that; claiming to enforce it would not be.
    """

    chars: int = MAX_CHARS
    seconds: float | None = None
    """The elapsed time this call was expected to stay under, or `None` for no
    expectation. **Measured and reported. Never enforced** -- nothing here
    interrupts the call."""

    def __post_init__(self) -> None:
        if self.chars < 1:
            raise ValueError("a bound of fewer than one character admits nothing")
        if self.seconds is not None and self.seconds <= 0:
            raise ValueError("an expectation of zero seconds is not one")


@dataclass(frozen=True)
class Request:
    """What is being asked, and who asked. Never edited by this module."""

    text: str
    purpose: str
    """Why this call is being made, in a person's words. Carried into the human
    queue, where somebody who did not issue it has to judge it."""

    issued_by: str = ""
    """Who issued the command. Empty is honest; a default naming a machine
    would not be."""

    @property
    def fingerprint(self) -> str:
        """A stable name for this request, for correlating without quoting it.

        Content-addressed rather than counted or timed, so the same request
        made twice on two machines has the same name and nothing here needs a
        clock or a sequence to be reproducible.
        """
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return f"governed-{digest[:16]}"


@dataclass(frozen=True)
class Outcome:
    """What one run of the pipeline did, whether or not it called anything.

    **EVERY FIELD IS POPULATED ON EVERY PATH.** A refused run reports the
    stages it reached and the spend it declared exactly as a drafted one does,
    because a consumer telling those apart by which fields are empty is the
    substitution `qmcp.spend` was written to refuse.
    """

    state: str
    request: Request
    stages: tuple[str, ...]
    """Which stages ran, in order. This is the metering: a reader can see how
    far a run got without reading `run`."""

    declared: dict[str, Any]
    """`qmcp.spend.declare` for this run, on every path."""

    draft: str = ""
    """What came back. Empty unless `state` is `DRAFTED`."""

    why: str = ""
    """Why a refused or stopped run ended where it did. Empty when drafted."""

    elapsed: float | None = None
    """Seconds the call took, or `None` if nothing was called."""

    over_bound: bool = False
    """Whether `Bound.seconds` was exceeded. **Reported, not acted on** -- a
    drafted run may carry this and still be drafted, and that is the honest
    report rather than a refusal nothing enforced."""

    @property
    def called(self) -> bool:
        """Whether the black box was reached. Not the same as `drafted`."""
        return "model" in self.stages

    @property
    def drafted(self) -> bool:
        return self.state == DRAFTED


@dataclass(frozen=True)
class Stage:
    """One step, and what a window should call it."""

    id: str
    label: str
    kind: str
    note: str = ""


# The stages, declared once. `run` walks this and `view` draws it, so the two
# cannot disagree -- there is nothing for them to disagree about.
STAGES: tuple[Stage, ...] = (
    Stage("in", "the request", INPUT,
          "carried verbatim; nothing in this module edits it"),
    Stage("budget", "what may be spent", GATE,
          "checked before the call. A refusal after the money is gone is a report"),
    Stage("model", "the black box", WORKER,
          "one call, and the only stage with no halting guarantee"),
    Stage("bound", "what came back", GATE,
          "size enforced here; elapsed measured and reported, never enforced"),
    Stage("draft", "a draft", OUTPUT,
          "never a decision, whatever it says about itself"),
    Stage("queue", "a person", GATE,
          "the pipeline ends here. Answering is a person's by constitution"),
)

# Where a run can end other than at the gate. Drawn, because a path that exists
# and is not on the picture is a path a reader does not know about.
_ENDS: tuple[Stage, ...] = (
    Stage("refused", "refused", OUTPUT, "the budget would have been exceeded"),
    Stage("stopped", "stopped", OUTPUT,
          "a bound fired; what came back is not usable"),
)


def run(request: Request, budget: Budget, call: Callable[[str], str], *,
        bound: Bound | None = None,
        clock: Callable[[], float] = time.monotonic) -> Outcome:
    """One request through the seam, ending at the human gate.

    Returns an `Outcome` on every path a caller can provoke. `Refused` from
    `qmcp.spend` is caught here and turned into a refused outcome rather than
    propagating: it is raised there so that a caller cannot ignore it, and it
    is caught here because at this seam a refusal is an ordinary result that
    the picture already draws.

    `clock` is injected so a test can establish what an elapsed report says
    without waiting for it, which is the only reason this module knows the time
    at all.
    """
    bound = bound or Bound()
    reached: list[str] = ["in"]
    would_need = 1

    reached.append("budget")
    try:
        budget.spend(1)
    except Refused as refusal:
        return Outcome(state=REFUSED, request=request, stages=tuple(reached),
                       declared=declare(budget, would_need), why=str(refusal))

    reached.append("model")
    started = clock()
    answer = call(request.text)
    elapsed = clock() - started

    reached.append("bound")
    over = bound.seconds is not None and elapsed > bound.seconds
    if not isinstance(answer, str):
        return Outcome(
            state=STOPPED, request=request, stages=tuple(reached),
            declared=declare(budget, would_need), elapsed=elapsed,
            over_bound=over,
            why=(f"the call returned {type(answer).__name__}, not text. A call "
                 f"was made and paid for, which is why this is a stopped run "
                 f"rather than a refused one"))
    if len(answer) > bound.chars:
        return Outcome(
            state=STOPPED, request=request, stages=tuple(reached),
            declared=declare(budget, would_need), elapsed=elapsed,
            over_bound=over,
            why=(f"{len(answer)} characters came back against a bound of "
                 f"{bound.chars}. The call was made and is not being reported "
                 f"as free"))

    reached.extend(("draft", "queue"))
    return Outcome(state=DRAFTED, request=request, stages=tuple(reached),
                   declared=declare(budget, would_need), draft=answer,
                   elapsed=elapsed, over_bound=over)


def queued(outcome: Outcome, *, timeout_seconds: int = 3600) -> dict[str, Any]:
    """The payload that puts one draft in front of a person.

    Shaped for `POST /v1/human/requests`. **Building it is not posting it, and
    posting it is not answering it** -- this returns a dict and does nothing,
    so that putting work in front of somebody stays a command somebody issued.

    A refused or stopped run is queued too, and says so. Somebody deciding
    whether to authorise a paid retry needs the refusal in front of them more
    than they need the drafts that succeeded.
    """
    if outcome.state not in STATES:
        raise ValueError(f"{outcome.state!r} is not one of {STATES}")

    return {
        "id": f"{outcome.request.fingerprint}:{outcome.state}",
        "request_type": "approval",
        "prompt": outcome.request.purpose,
        "options": ["accept", "reject"],
        "timeout_seconds": timeout_seconds,
        "context": {
            # Said outright rather than left to the reader of a text field.
            "this_is_a_draft": True,
            "state": outcome.state,
            "stages": list(outcome.stages),
            "draft": outcome.draft,
            "why": outcome.why,
            "elapsed": outcome.elapsed,
            "over_bound": outcome.over_bound,
            "issued_by": outcome.request.issued_by,
            "spend": outcome.declared,
        },
    }


def view(level: int = FLOWS) -> View:
    """The seam, at one level, built from `STAGES` rather than beside it.

    Folded down from the flow the same way `qmcp.topology_view.view_of` folds
    the vocabulary's shapes, so a reader comparing this with a delegation
    topology is comparing two things assembled by the same rule.
    """
    if level not in LEVELS:
        raise ValueError(f"level is one of {LEVELS}, not {level!r}")

    boxes = tuple(Box(s.id, s.label, s.kind, s.note) for s in (*STAGES, *_ENDS))
    arrows = tuple(
        [Arrow(a.id, b.id) for a, b in zip(STAGES, STAGES[1:])]
        + [Arrow("budget", "refused", "over budget", REFUSAL),
           Arrow("bound", "stopped", "bound fired", REFUSAL),
           Arrow("queue", "in", "re-issued by a person", FEEDBACK)])
    caption = ("one call to a black box, budgeted before and bounded after, "
               "ending at a person")

    if level == 0:
        inside = len(boxes) - 2
        boxes = (Box("in", "the request", INPUT),
                 Box("box", "governed", WORKER, f"{inside} parts inside"),
                 Box("queue", "a person", GATE, "the pipeline ends here"))
        arrows = (Arrow("in", "box"), Arrow("box", "queue"))
    elif level == 1:
        arrows = ()

    return View(topology="governed", level=level, boxes=boxes, arrows=arrows,
                caption=caption,
                # Declared, not discovered. This shape exists to make one paid
                # call, and it ends at a gate rather than at a decision.
                spends=True, writes=False, decides=False)
