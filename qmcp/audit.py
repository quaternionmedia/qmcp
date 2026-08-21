"""Answering questions about what ran, from records rather than from inference.

    uv run qmcp audit models --since 2026-08-21
    uv run qmcp audit in-flight

**DETERMINISTIC MEANS TWO RUNS OVER THE SAME ROWS GIVE THE SAME ANSWER, AND
THAT IS A STRONGER CLAIM THAN IT SOUNDS.** Three things break it and all three
are avoided here rather than mitigated:

*The clock.* "Today" is not a fact about the world; it is a fact about a
timezone. `created_at` is written in UTC, so a window is a UTC half-open
interval stated in the answer -- never `date.today()`, which gives two different
answers either side of midnight in two different places, both correct and
neither reproducible.

*The default.* A window nobody named would be filled in from the wall clock, so
the same query run twice returns different rows. `Window` has no default: a
caller states it, and `Window.day()` exists to make stating it easy rather than
to make it optional.

*The silent omission.* An invocation that recorded no model is not evidence of a
model that ran unrecorded, and it is not evidence of nothing running. It is
`unknown`, it is counted, and it appears in the answer beside the models that
are named.

**WHAT CANNOT BE ANSWERED TODAY, SAID PLAINLY.** `tool_invocations` has no model
column and `agent_types` is empty, so "which models ran" currently answers
`unknown` for every row. That is the honest state of the record rather than a
defect in the query, and `record_model` is the path that makes it answerable
going forward. A module that inferred a model from a tool name would be
answering a question nobody's data supports.

**THE PAIR IS THE BRIDGE TEST.** Both sides can compute the same figures -- the
harness from its own tables, the panel from what crosses the seam. Agreement is
evidence the bridge carries what it claims to; disagreement is a finding, and
`compare()` reports it as one rather than picking a winner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

UNKNOWN = "unknown"

# Where a model would be recorded if anybody recorded one. Read in this order,
# first hit wins, and nothing is inferred from a tool's name.
MODEL_KEYS = ("model", "model_name", "model_id")


@dataclass(frozen=True)
class Window:
    """A half-open UTC interval, `[start, end)`.

    Half-open so that consecutive windows neither overlap nor drop a row. A
    closed interval double-counts anything landing exactly on a boundary, which
    is rare, real, and invisible until two windows disagree by one.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("a window is stated in UTC, not in local time")
        if self.end <= self.start:
            raise ValueError("a window ends after it starts")

    @classmethod
    def day(cls, on: date) -> "Window":
        """One UTC day. The caller names the day; nothing reads a clock."""
        start = datetime(on.year, on.month, on.day, tzinfo=timezone.utc)
        return cls(start=start, end=start + timedelta(days=1))

    def holds(self, when: datetime | None) -> bool:
        if when is None:
            return False
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return self.start <= when < self.end

    def describe(self) -> str:
        return (f"{self.start.strftime('%Y-%m-%dT%H:%MZ')} to "
                f"{self.end.strftime('%Y-%m-%dT%H:%MZ')}")


@dataclass(frozen=True)
class ModelRun:
    """One model, and what it did inside the window."""

    model: str
    invocations: int
    first: datetime | None = None
    last: datetime | None = None
    failures: int = 0

    @property
    def is_unknown(self) -> bool:
        return self.model == UNKNOWN


@dataclass
class ModelReport:
    """What ran, over a window somebody named."""

    window: Window
    rows_read: int
    runs: list[ModelRun] = field(default_factory=list)

    @property
    def named(self) -> list[ModelRun]:
        return [r for r in self.runs if not r.is_unknown]

    @property
    def unrecorded(self) -> int:
        return sum(r.invocations for r in self.runs if r.is_unknown)

    def summary(self) -> str:
        """The window first, then what was read, then what is unknown.

        The window leads because a figure without one is not an answer to
        anything, and the unknown count is never omitted -- a report of two
        models that silently dropped fifty-three unrecorded invocations would
        be worse than no report.
        """
        lines = [f"{self.window.describe()}: {self.rows_read} invocation(s) read"]
        if not self.runs:
            lines.append("  nothing ran in this window")
            return "\n".join(lines)
        for run in sorted(self.runs, key=lambda r: (-r.invocations, r.model)):
            mark = "  (no model recorded)" if run.is_unknown else ""
            fail = f", {run.failures} failed" if run.failures else ""
            lines.append(f"  {run.model:<28} {run.invocations}{fail}{mark}")
        if self.unrecorded:
            lines.append(f"  -- {self.unrecorded} of {self.rows_read} recorded no "
                         f"model, so which model ran is unknown for those")
        return "\n".join(lines)


def model_of(params: Any) -> str:
    """The model an invocation recorded, or `unknown`.

    **NOTHING IS INFERRED FROM A TOOL NAME.** `planner` is a tool, not a model,
    and mapping one to the other would be this module inventing the fact it was
    asked to report.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            return UNKNOWN
    if not isinstance(params, dict):
        return UNKNOWN
    for key in MODEL_KEYS:
        found = params.get(key)
        if isinstance(found, str) and found.strip():
            return found.strip()
    return UNKNOWN


def models_run(rows: Iterable[Any], window: Window) -> ModelReport:
    """Which models ran inside `window`, from rows as recorded.

    Takes rows rather than a session so the same function answers for the
    harness's own table and for anything that crosses the seam -- and so this
    is testable without a database, which is what makes the determinism
    checkable rather than asserted.
    """
    counts: dict[str, list[Any]] = {}
    read = 0
    for row in rows:
        when = _when(row)
        if not window.holds(when):
            continue
        read += 1
        model = model_of(_get(row, "input_params"))
        counts.setdefault(model, []).append((when, _failed(row)))

    runs = []
    for model, entries in counts.items():
        times = sorted(w for w, _ in entries if w is not None)
        runs.append(ModelRun(
            model=model,
            invocations=len(entries),
            first=times[0] if times else None,
            last=times[-1] if times else None,
            failures=sum(1 for _, bad in entries if bad),
        ))
    return ModelReport(window=window, rows_read=read, runs=runs)


@dataclass(frozen=True)
class InFlight:
    """One delta, and the harness it is moving through."""

    delta: str
    phase: str
    harness: str
    since: datetime | None = None

    @property
    def harness_is_known(self) -> bool:
        return self.harness != UNKNOWN


@dataclass
class FlightReport:
    """What is in flight, and through what."""

    at: datetime
    open_phases: tuple[str, ...]
    items: list[InFlight] = field(default_factory=list)

    def by_harness(self) -> dict[str, list[InFlight]]:
        found: dict[str, list[InFlight]] = {}
        for item in self.items:
            found.setdefault(item.harness, []).append(item)
        return found

    @property
    def unattributed(self) -> list[InFlight]:
        """In flight, through a harness nothing recorded. Not zero, not none."""
        return [i for i in self.items if not i.harness_is_known]

    def summary(self) -> str:
        lines = [f"as at {self.at.strftime('%Y-%m-%dT%H:%MZ')}: "
                 f"{len(self.items)} delta(s) in flight",
                 f"  open phases: {', '.join(self.open_phases)}"]
        for harness, items in sorted(self.by_harness().items(),
                                     key=lambda kv: (-len(kv[1]), kv[0])):
            mark = "  (no harness recorded)" if harness == UNKNOWN else ""
            lines.append(f"  {harness:<28} {len(items)}{mark}")
        return "\n".join(lines)


# Phases a delta is still moving through. `complete` and `abandoned` are not in
# flight, and the list is stated rather than derived from "not complete" so that
# a phase added later is a decision somebody makes here rather than a silent
# reclassification.
OPEN_PHASES = ("brainstorm", "planning", "implementation", "review",
               "documentation")


def in_flight(rows: Iterable[Any], at: datetime,
              open_phases: tuple[str, ...] = OPEN_PHASES) -> FlightReport:
    """Deltas still moving, and through which harness.

    `at` is passed rather than read from a clock, for the same reason `Window`
    has no default: an answer that depends on when it was asked cannot be
    checked against itself.
    """
    items = []
    for row in rows:
        phase = str(_get(row, "phase") or UNKNOWN).lower()
        if phase not in open_phases:
            continue
        items.append(InFlight(
            delta=str(_get(row, "name") or _get(row, "delta") or "?"),
            phase=phase,
            harness=str(_get(row, "harness") or UNKNOWN),
            since=_when(row),
        ))
    return FlightReport(at=at, open_phases=open_phases, items=items)


@dataclass(frozen=True)
class Disagreement:
    """One figure the two sides read differently."""

    figure: str
    harness_says: Any
    panel_says: Any


def compare(harness: dict[str, Any], panel: dict[str, Any]) -> list[Disagreement]:
    """Where the two sides of the pair disagree about the same question.

    **THIS IS THE BRIDGE TEST, AND IT PICKS NO WINNER.** Both sides count from
    what they hold; agreement is evidence the seam carried what it claimed to.
    A disagreement is a finding about the bridge, and deciding which side is
    right needs somebody who can look at both -- exactly the split that made
    the panel report an absent archive while the harness served 203 threads.

    A figure only one side reports is a disagreement too: a bridge that drops a
    field is a bridge that lost it, and reading the shared keys alone would
    report that as agreement.

    **SO SCOPE THE COMPARISON TO WHAT THE ROUTE CLAIMS TO CARRY.** That rule
    makes any figure one side simply does not send look like a loss. The first
    bridge run compared the harness's invocation count against a threads route
    and reported a disagreement -- correctly by the rule, and wrongly about the
    world, because `/v1/threads` never promised invocations. Pass the carried
    figures, not everything each side happens to know.
    """
    found = []
    for key in sorted(set(harness) | set(panel)):
        left = harness.get(key, "not reported")
        right = panel.get(key, "not reported")
        if left != right:
            found.append(Disagreement(key, left, right))
    return found


def _get(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)


def _when(row: Any) -> datetime | None:
    when = _get(row, "created_at") or _get(row, "invoked_at")
    if isinstance(when, str):
        try:
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(when, datetime) and when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when if isinstance(when, datetime) else None


def _failed(row: Any) -> bool:
    status = str(_get(row, "status") or "").lower()
    if status:
        return "fail" in status or "error" in status
    return bool(_get(row, "error"))


def record_model(params: dict[str, Any] | None, model: str | None) -> dict[str, Any]:
    """Put the model where `model_of` will find it, so today's answer is
    `unknown` and tomorrow's is not.

    **THIS IS THE WHOLE REMEDY, AND IT IS DELIBERATELY SMALL.** The reason
    "which models ran" cannot be answered is not that the query is weak; it is
    that fifty-five invocations recorded no model. Adding a column would be a
    migration and a second place for the fact to live -- `input_params` is
    already JSON, already written on every invocation, and already the thing
    `model_of` reads.

    `None` is left absent rather than written as the string "unknown". An
    absent key and a recorded "unknown" both read as unknown today, and only
    the first stays true if somebody later learns which model it was.
    """
    out = dict(params or {})
    if model and model.strip():
        out[MODEL_KEYS[0]] = model.strip()
    return out
