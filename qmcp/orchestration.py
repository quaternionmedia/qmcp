"""What each topology would do, declared before anything runs one.

    uv run qmcp orchestration plane

Each registered topology has one `Capability` in `PLANE`: its status,
whether running it spends money, writes to a repository or decides, why, and
what a caller must supply before it can run. A capability is declared, never
discovered by running the shape.

A status is one of three. `RUNS`: implemented, and safe to point at ordinary
work. `BRAINSTORM`: a designed shape whose `run` still raises -- a proposal,
not a runtime. `REFUSED`: a shape whose design performs an act this
organisation reserves for a person.

`governance/qm/ci/attested-registry.yaml` lists those acts: ratifying a
record, cutting a tag, closing a delta as complete, answering a question in the
human queue and authorising a paid call among them. A topology that decides is
refused when pointed at one, and a topology that only reports is not, so the
refusal belongs to the pairing of shape and act. `refuses` answers for a
pairing.

`OPTIONS` declares a shape that does something else under one setting, such
as the advisory council.

What this cannot do: stop a topology whose declaration is wrong. A declaration
is checked against the registry by `undeclared()` and `stubs()`, and against
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from qmcp.agentframework.models.enums import TopologyType
from qmcp.agentframework.topologies import BaseTopology, TopologyRegistry, topology

RUNS = "runs"
"""Implemented, and safe to point at ordinary work."""

BRAINSTORM = "brainstorm"
"""A shape somebody designed and nobody has built. Not debt: a proposal."""

REFUSED = "refused"
"""Its shape performs an act this organisation reserves for a person."""

# The acts, as `governance/qm/ci/attested-registry.yaml` lists them, restated
# here. The registry is the authority.
ATTESTED = (
    "ratify a record",
    "cut a version tag",
    "apply the main ruleset",
    "close a delta as complete",
    "answer a question in the human-in-the-loop queue",
    "authorise a paid call",
    "request an export of your own data from a service",
)


@dataclass(frozen=True)
class Need:
    """One thing a topology must have before it can be run, and what supplies it.

    Every need names its remedy, so a reader told that a shape cannot run is
    also told what would make it run.
    """

    key: str
    """What is missing. One of `NEEDS`."""

    because: str
    """What the topology cannot do without it, in a person's words."""

    supplied_by: str
    """The command or the thing that provides it."""


# What a topology can be short of. Named rather than written as free text so
# two shapes needing the same thing say so identically.
BUILD = "build"
BUDGET = "budget"
WORKERS = "workers"
MODEL = "model"
PERSON = "person"

NEEDS = (BUILD, BUDGET, WORKERS, MODEL, PERSON)


@dataclass(frozen=True)
class Capability:
    """What one topology would do if somebody ran it."""

    topology: TopologyType
    status: str
    spends: bool
    writes: bool
    decides: bool
    """Whether the shape ends in a machine choosing rather than reporting.
    This is the property that makes a topology unsuitable for an attested act,
    and it is separate from `status` because a deciding shape is fine pointed
    at a question nobody's constitution reserves."""

    why: str

    needs: tuple[Need, ...] = ()
    """What must be supplied before this shape could run.

    **Separate from `status`, and the two answer different questions.**
    `status` says whether anybody has built it; `needs` says what a built one
    still wants from the caller. A `RUNS` topology with an unmet need is not
    broken -- it is waiting, and the difference is what a reader needs in
    order to act.
    """

    voice_runnable: bool = False
    """A saved design of this shape runs through the voice runner,
    `qmcp.integrations.agents.topology_design`: bounded, read-only, behind a
    consent. Separate from `status`, which is about the framework class."""

    voice_spends: bool = False
    voice_writes: bool = False
    voice_decides: bool = False
    """What the voice runner's run of this shape does. The runner refuses a
    shape whose voice run would spend, write or decide."""

    @property
    def can_run(self) -> bool:
        return self.status == RUNS


PLANE: tuple[Capability, ...] = (
    Capability(
        TopologyType.PIPELINE, BRAINSTORM, spends=False, writes=False,
        decides=False,
        why="stages in sequence. The registered class is a stub. "
            "`qmcp.feedback` and `intake` are working pipelines that do not "
            "register the type: `TopologyRegistry` keeps one class per type, "
            "and import order would decide which one it kept",
        needs=(Need(BUILD,
                    "the registered class is a stub whose `run` raises",
                    "somebody writes it; `qmcp.feedback` and `intake` are "
                    "working pipelines to start from"),),
        voice_runnable=True),
    Capability(
        TopologyType.DELEGATION, RUNS, spends=False, writes=False, decides=False,
        why="route each unit of work to the worker registered for its shape. "
            "`qmcp.sweep` routes its parsers and questions this way, so the "
            "mix follows the work rather than a setting",
        needs=(Need(WORKERS,
                    "it routes each unit of work to the worker registered for "
                    "its shape, and an unregistered shape has nowhere to go",
                    "pass `workers` to `delegate`; an unrouted shape is "
                    "reported, never dropped"),),
        voice_runnable=True),
    Capability(
        TopologyType.CROSS_CHECK, RUNS, spends=False, writes=False, decides=False,
        why="several independent checkers on one claim, and a consensus that "
            "is reported rather than acted on. Reports; does not decide",
        needs=(Need(WORKERS,
                    "a consensus of one checker is not a consensus",
                    "pass more than one checker to `cross_check`"),),
        voice_runnable=True),
    Capability(
        TopologyType.ENSEMBLE, BRAINSTORM, spends=True, writes=False,
        decides=False,
        why="several workers on the same item, answers combined. Unbuilt. "
            "Every run spends -- N answers to one question -- so it needs a "
            "budget as well as a runtime",
        needs=(Need(BUILD, "nobody has written it", "somebody writes it"),
               Need(BUDGET,
                    "N answers to one question is N paid calls, and the "
                    "default budget is zero",
                    "issue it against an authorised budget; consent is an "
                    "amount rather than a category"),),
        voice_runnable=True),
    Capability(
        TopologyType.DEBATE, BRAINSTORM, spends=True, writes=False, decides=True,
        why="positions argued to a conclusion, for a question with no right "
            "answer. Unbuilt. It decides, so it is not for an attested act",
        needs=(Need(BUILD, "nobody has written it", "somebody writes it"),
               Need(BUDGET, "positions are argued by paid calls",
                    "issue it against an authorised budget"),),
        voice_runnable=True),
    Capability(
        TopologyType.CHAIN_OF_COMMAND, BRAINSTORM, spends=True, writes=False,
        decides=True,
        why="escalation up a hierarchy, ending in something choosing. "
            "Unbuilt",
        needs=(Need(BUILD, "nobody has written it", "somebody writes it"),
               Need(BUDGET, "each escalation is a paid call",
                    "issue it against an authorised budget"),),
        voice_runnable=True),
    Capability(
        TopologyType.COMPOUND, BRAINSTORM, spends=True, writes=False,
        decides=False,
        why="topologies composed of topologies. It runs usefully once more "
            "than one of its parts runs",
        needs=(Need(BUILD,
                    "it composes topologies, and more than one of its parts "
                    "has to run first",
                    "build the parts first"),),
        voice_runnable=True),
    Capability(
        TopologyType.COUNCIL, REFUSED, spends=True, writes=False, decides=True,
        why="its default config gives the arbiter the final decision when "
            "consensus fails, so the shape adjudicates. A council deciding "
            "whether to ratify would be a machine performing an act "
            "`ci/attested-registry.yaml` reserves for a person. "
            "`arbiter_can_override` false is the advisory council, declared in "
            "`OPTIONS`",
        needs=(Need(PERSON,
                    "its arbiter takes the final decision when consensus "
                    "fails, and that is an act the constitution reserves",
                    "nothing supplies this: a person decides, and the shape is "
                    "refused here rather than made safe"),)),
)


@dataclass(frozen=True)
class Option:
    """One configuration under which a shape would do something else.

    An option is a second declaration for one setting. A design that does
    not select it is read against the kind's row in `PLANE`.
    """

    setting: str
    """The configuration key, as the kind's config class names it."""

    value: Any
    """The value that selects this declaration. Compared by type as well as
    by value, so `0` does not select an option declared for `False`."""

    capability: Capability


ADVISORY_COUNCIL = Capability(
    TopologyType.COUNCIL, BRAINSTORM, spends=True, writes=False, decides=True,
    why="the same perspectives with `arbiter_can_override` false: when they "
        "do not reach consensus the arbiter synthesizes what each said and "
        "the council reports a split, so no member and no arbiter takes the "
        "final decision. A consensus is still a conclusion a machine reached, "
        "so like `debate` it is refused an attested act; pointed at an "
        "ordinary question it is a proposal waiting for a runtime",
    needs=(Need(BUILD, "the registered class is a stub whose `run` raises",
                "somebody writes it"),
           Need(BUDGET, "each perspective is a paid call",
                "issue it against an authorised budget"),),
    voice_runnable=True)

OPTIONS: tuple[Option, ...] = (
    Option("arbiter_can_override", False, ADVISORY_COUNCIL),
)


def selected(kind: TopologyType,
             config: dict[str, Any] | None) -> Option | None:
    """The option this configuration selects for this kind, or None."""
    for option in OPTIONS:
        if option.capability.topology != kind or not config:
            continue
        value = config.get(option.setting)
        if type(value) is type(option.value) and value == option.value:
            return option
    return None


def capability_for(kind: TopologyType,
                   config: dict[str, Any] | None = None) -> Capability | None:
    """What this kind would do under this configuration.

    An option's declaration when the configuration selects one, otherwise
    the kind's row in `PLANE`. A configuration that omits the setting is the
    kind's default, so it reads the row.
    """
    option = selected(kind, config)
    return option.capability if option else by_type().get(kind)


def unmet(capability: Capability, *, built: bool | None = None,
          budget: int = 0, workers: int = 0, model: bool = False
          ) -> tuple[Need, ...]:
    """What this shape is still short of, given what the caller has.

    `status` and `needs` answer different questions, and both are asked: a
    shape nobody has built is short of a build whatever else the caller
    brings, and a built one can still be short of a budget.

    `built` overrides the declared status, so a caller who has written a
    runtime for a `BRAINSTORM` shape can ask what else it wants.
    """
    have_build = capability.can_run if built is None else built
    short = []
    for need in capability.needs:
        if need.key == BUILD and have_build:
            continue
        if need.key == BUDGET and budget > 0:
            continue
        if need.key == WORKERS and workers > 1:
            continue
        if need.key == MODEL and model:
            continue
        # PERSON is never supplied: a shape whose need is a person's judgement
        # is not made runnable by an argument.
        short.append(need)
    return tuple(short)


def runnable_now(**have) -> list[TopologyType]:
    """Every shape a caller with `have` could run right now."""
    return [c.topology for c in PLANE if not unmet(c, **have)]


def by_type() -> dict[TopologyType, Capability]:
    return {c.topology: c for c in PLANE}


def undeclared() -> list[str]:
    """Registered topologies with no capability, and declarations for nothing.

    Both directions: a registered topology with no declaration would run with
    its cost and its authority unstated, and a declaration with no registered
    topology describes nothing.
    """
    registered = set(TopologyRegistry._topologies)
    declared = set(by_type())
    found = []
    for missing in sorted(t.value for t in registered - declared):
        found.append(f"{missing}: registered, no capability declared")
    for extra in sorted(t.value for t in declared - registered):
        found.append(f"{extra}: declared, but nothing registers it")
    return found


def stubs() -> list[str]:
    """Registered topologies whose `run` is still the base class's.

    A `RUNS` declaration is a claim about a reachable class, so this checks
    it. `TopologyRegistry` keeps one class per type, and when two classes
    register one type the import order decides which is kept.
    """
    found = []
    for kind, cls in sorted(TopologyRegistry._topologies.items(),
                            key=lambda kv: kv[0].value):
        if cls.run is BaseTopology.run:
            found.append(kind.value)
    return found


def unregistered_types() -> list[str]:
    """Names in the vocabulary that no topology implements.

    `mesh`, `star` and `ring` are in `TopologyType` with no class and no
    config. They are reported, not removed.
    """
    registered = set(TopologyRegistry._topologies)
    return sorted(t.value for t in TopologyType if t not in registered)


def refuses(kind: TopologyType, act: str,
            config: dict[str, Any] | None = None) -> str | None:
    """Why this pairing is refused, or None.

    The refusal belongs to the pairing. A deciding topology is allowed a
    question nobody's constitution reserves, and a reporting topology is
    allowed an attested act, because reporting on an act is not performing
    it. The answer names both.

    `config` is the design's configuration. It changes the answer only where
    it selects one of `OPTIONS`, and the answer then names that setting.
    """
    option = selected(kind, config)
    found = capability_for(kind, config)
    name = (f"{kind.value} with {option.setting}={option.value!r}"
            if option else kind.value)
    if found is None:
        return f"{name} has no declared capability, so nothing knows what it would do"
    if found.status == REFUSED:
        return f"{name} is refused here: {found.why}"
    if act in ATTESTED and found.decides:
        return (f"{name} decides, and {act!r} is a person's by "
                f"constitution -- a machine performing it changes what it "
                f"asserts")
    return None


def _rendered(entry: Capability, label: str) -> list[str]:
    marks = []
    if entry.spends:
        marks.append("spends")
    if entry.writes:
        marks.append("writes")
    if entry.decides:
        marks.append("decides")
    if entry.voice_runnable:
        marks.append("saved-design voice runner")
    return [f"{entry.status.upper():<10} {label:<12}"
            f"{'  [' + ', '.join(marks) + ']' if marks else ''}",
            f"           {entry.why}"]


def render() -> str:
    """The plane, for somebody deciding what to point at what."""
    lines = ["what each topology would do, before anything runs one", ""]
    for entry in PLANE:
        lines += _rendered(entry, entry.topology.value)
    if OPTIONS:
        lines += ["", "and under one setting, something else:"]
        for option in OPTIONS:
            lines += _rendered(option.capability,
                               f"{option.capability.topology.value} with "
                               f"{option.setting}={option.value!r}")
    inert = stubs()
    claiming = [c.topology.value for c in PLANE
                if c.can_run and c.topology.value in inert]
    lines += ["", f"registered but still inheriting the stub `run`: "
                  f"{', '.join(inert) if inert else 'none'}"]
    if claiming:
        lines += [f"  and claiming to run anyway: {', '.join(claiming)}"]

    drift = undeclared()
    if drift:
        lines += ["", "declaration drift:"] + [f"  - {d}" for d in drift]
    absent = unregistered_types()
    if absent:
        lines += ["", f"in the vocabulary, implemented by nothing: "
                      f"{', '.join(absent)}"]
    return "\n".join(lines)


# --- the two shapes that run ---------------------------------------------------


@dataclass(frozen=True)
class Routed:
    """One unit of work and the worker that took it."""

    item: Any
    worker: str
    result: Any = None
    taken: bool = True


@topology
class DelegationTopology(BaseTopology):
    """Route each unit of work to the worker registered for its shape.

    A unit whose shape has no worker is returned with `taken=False` and
    named, never dropped, so a run that left work untouched says so.
    `qmcp.sweep` routes its parsers and questions this way.
    """

    topology_type = TopologyType.DELEGATION

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        if args or kwargs:
            super().__init__(*args, **kwargs)

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        routed = delegate(input_data.get("items") or (),
                          input_data.get("workers") or {},
                          shape_of=input_data.get("shape_of"))
        return {
            "routed": len([r for r in routed if r.taken]),
            "unrouted": [r.worker for r in routed if not r.taken],
        }


def delegate(items: Iterable[Any], workers: dict[str, Callable[[Any], Any]],
             shape_of: Callable[[Any], str] | None = None) -> list[Routed]:
    """Hand each item to the worker for its shape."""
    read = shape_of or (lambda item: item.get("shape", "unknown"))
    out: list[Routed] = []
    for item in items:
        shape = read(item)
        worker = workers.get(shape)
        if worker is None:
            out.append(Routed(item, f"no worker for {shape!r}", taken=False))
            continue
        try:
            out.append(Routed(item, shape, worker(item)))
        except Exception as exc:                  # noqa: BLE001
            out.append(Routed(item, shape, f"{type(exc).__name__}: {exc}"))
    return out


@dataclass(frozen=True)
class Checked:
    """One claim, and what independent checkers said about it."""

    claim: Any
    verdicts: tuple[bool, ...]
    reasons: tuple[str, ...] = ()

    @property
    def agreed(self) -> int:
        return sum(1 for v in self.verdicts if v)

    @property
    def unanimous(self) -> bool:
        return bool(self.verdicts) and all(self.verdicts)

    @property
    def majority(self) -> bool:
        return bool(self.verdicts) and self.agreed * 2 > len(self.verdicts)

    @property
    def is_split(self) -> bool:
        """Checkers disagreed. A finding in itself, and not a failure."""
        return bool(self.verdicts) and 0 < self.agreed < len(self.verdicts)


@topology
class CrossCheckTopology(BaseTopology):
    """Several independent checkers on one claim.

    It reports and does not decide: the consensus is a count, and what to do
    about a split is a person's.

    The checkers are independent. A checker that saw another's reasoning
    would not be a second opinion.
    """

    topology_type = TopologyType.CROSS_CHECK

    def __init__(self, *args, **kwargs) -> None:  # noqa: D107
        if args or kwargs:
            super().__init__(*args, **kwargs)

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        found = cross_check(input_data["claim"], input_data["checkers"])
        return {
            "agreed": found.agreed,
            "of": len(found.verdicts),
            "unanimous": found.unanimous,
            "split": found.is_split,
        }


def cross_check(claim: Any,
                checkers: Iterable[Callable[[Any], Any]]) -> Checked:
    """Ask each checker independently. A checker that raises is a `False`.

    A raising checker counts against rather than being dropped: a check that
    could not be made is not agreement.
    """
    verdicts: list[bool] = []
    reasons: list[str] = []
    for checker in checkers:
        try:
            answer = checker(claim)
        except Exception as exc:                  # noqa: BLE001
            verdicts.append(False)
            reasons.append(f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(answer, tuple) and len(answer) == 2:
            verdicts.append(bool(answer[0]))
            reasons.append(str(answer[1]))
        else:
            verdicts.append(bool(answer))
            reasons.append("")
    return Checked(claim=claim, verdicts=tuple(verdicts),
                   reasons=tuple(reasons))
