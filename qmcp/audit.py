"""Answering questions about what ran, from records rather than from inference.

    uv run qmcp audit models --since 2026-08-21
    uv run qmcp audit in-flight

Two runs over the same rows give the same answer:

- a window is a UTC half-open interval, stated in the answer, because
  `created_at` is written in UTC; nothing reads `date.today()`;
- `Window` has no default: a caller states it, and `Window.day()` makes that
  easy;
- an invocation that recorded no model is `unknown`, counted, and shown beside
  the models that are named.

`tool_invocations` has no model column. A model is read from the invocation's
`input_params`, where `record_model` puts it; nothing is inferred from a tool's
name.

`compare()` is the bridge test: the harness and the panel each compute the
same figures, and a disagreement is reported as a finding, with no winner
picked. `docs/self-checks.md` describes it.
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

    Nothing is inferred from a tool name: `planner` is a tool, not a model.
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

    Picks no winner: agreement is evidence the seam carried what it claims
    to, and a disagreement is a finding about the bridge for somebody who can
    look at both sides.

    A figure only one side reports is a disagreement too, shown as
    "not reported". So pass only the figures the route claims to carry, not
    everything each side knows.
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
    """Put the model in `params` where `model_of` will find it.

    The model lives in `input_params`, which is JSON written on every
    invocation, rather than in a column of its own.

    `None` is left absent rather than written as the string "unknown", so a
    model learned later can still be recorded truthfully.
    """
    out = dict(params or {})
    if model and model.strip():
        out[MODEL_KEYS[0]] = model.strip()
    return out
