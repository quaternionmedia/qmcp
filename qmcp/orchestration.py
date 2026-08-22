"""What each topology would do, declared before anything runs one.

    uv run qmcp orchestration plane

**THE STUBS ARE BRAINSTORMS, AND THIS SAYS SO IN THE ONE PLACE A READER LOOKS.**
Seven topologies are registered with a schema, a config class and a `run` that
raises. Read as code that is a to-do list; read as design they are somebody's
considered catalogue of collaboration shapes, written before anybody needed
them. Neither reading is served by silence: `BRAINSTORM` names them as
proposals, so nobody mistakes a deliberate blank for an oversight and nobody
mistakes the registry for a runtime.

**AND SOME OF THEM MUST NOT RUN HERE, WHICH IS A DIFFERENT STATE AGAIN.**
`governance/qm/ci/attested-registry.yaml` names seven acts that are a person's
by constitution -- ratifying a record, cutting a tag, closing a delta as
complete, answering a question in the human queue, authorising a paid call.
Those are not acts a machine performs badly; they are acts that *change what
they assert* when a machine performs them. A topology pointed at one is refused,
and the refusal is a property of the pairing rather than of the topology.

**CAPABILITY IS DECLARED, NOT DISCOVERED.** Every entry states whether running
it spends money, whether it writes to a repository, and which attested acts its
shape would naturally perform. A plane that worked those out by running
something would have already done the thing it was deciding about.

WHAT THIS CANNOT DO. Stop a topology that lies in its declaration. The
declaration is a claim by whoever wrote the entry, checked against the registry
by `undeclared()` and against nothing else.
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

# The acts, from `governance/qm/ci/attested-registry.yaml`. Restated because
# this repository's corpus pin predates that file -- `undeclared()` reports the
# drift rather than this module pretending the list is authoritative.
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

    @property
    def can_run(self) -> bool:
        return self.status == RUNS


PLANE: tuple[Capability, ...] = (
    Capability(
        TopologyType.PIPELINE, BRAINSTORM, spends=False, writes=False,
        decides=False,
        why="stages in sequence. The registered class is still a stub, and the "
            "concrete pipelines -- `qmcp.feedback` for the self-checks, "
            "`intake` for an export -- run without claiming the type. They "
            "cannot claim it safely: `TopologyRegistry` is keyed by type and "
            "replaces silently, so a second class claiming PIPELINE wins or "
            "loses by import order rather than colliding"),
    Capability(
        TopologyType.DELEGATION, RUNS, spends=False, writes=False, decides=False,
        why="route each unit of work to the worker registered for its shape. "
            "`qmcp.sweep` already had this shape before it had this name: nine "
            "parsers and fifteen questions, and the mix follows the work "
            "rather than a setting"),
    Capability(
        TopologyType.CROSS_CHECK, RUNS, spends=False, writes=False, decides=False,
        why="several independent checkers on one claim, and a consensus that "
            "is reported rather than acted on. Reports; does not decide"),
    Capability(
        TopologyType.ENSEMBLE, BRAINSTORM, spends=True, writes=False,
        decides=False,
        why="several workers on the same item, answers combined. Plausible and "
            "unbuilt. It spends by construction -- N answers to one question -- "
            "so the first version needs a budget before it needs a runtime"),
    Capability(
        TopologyType.DEBATE, BRAINSTORM, spends=True, writes=False, decides=True,
        why="positions argued to a conclusion. A good shape for a question with "
            "no right answer, and unbuilt. It decides, so it is not for an "
            "attested act"),
    Capability(
        TopologyType.CHAIN_OF_COMMAND, BRAINSTORM, spends=True, writes=False,
        decides=True,
        why="escalation up a hierarchy. Unbuilt, and the escalation terminates "
            "in something choosing"),
    Capability(
        TopologyType.COMPOUND, BRAINSTORM, spends=True, writes=False,
        decides=False,
        why="topologies composed of topologies. Cannot usefully run until more "
            "than one of its parts does, which is an ordering fact rather than "
            "a judgement about the shape"),
    Capability(
        TopologyType.COUNCIL, REFUSED, spends=True, writes=False, decides=True,
        why="its config gives the arbiter the final decision when consensus "
            "fails -- 'Council manager who facilitates and makes final "
            "decisions'. That is adjudication by construction. Deliberation is "
            "welcome here and the deciding is not: a council that reached a "
            "verdict on whether to ratify would be a machine performing an act "
            "`ci/attested-registry.yaml` reserves for a person, and the verdict "
            "would be indistinguishable from one somebody made"),
)


def by_type() -> dict[TopologyType, Capability]:
    return {c.topology: c for c in PLANE}


def undeclared() -> list[str]:
    """Registered topologies with no capability, and declarations for nothing.

    Both directions. A topology somebody registers and nobody declares would
    run with its cost and its authority unstated, which is the whole failure
    this module exists against; a declaration for a topology nobody registered
    describes something that is not there.
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

    **THE CHECK THAT CAUGHT THE PLANE LYING.** A declaration of `RUNS` is a
    claim about a class, and the class is reachable, so the claim is checkable.
    It said the pipeline ran; the registry held the stub, because two classes
    claimed one type and the winner depended on what had been imported.
    """
    found = []
    for kind, cls in sorted(TopologyRegistry._topologies.items(),
                            key=lambda kv: kv[0].value):
        if cls.run is BaseTopology.run:
            found.append(kind.value)
    return found


def unregistered_types() -> list[str]:
    """Names in the vocabulary that no topology implements.

    Reported rather than removed. `mesh`, `star` and `ring` are in
    `TopologyType` with no class and no config -- which is a smaller brainstorm
    than the seven with schemas, and still somebody's intent.
    """
    registered = set(TopologyRegistry._topologies)
    return sorted(t.value for t in TopologyType if t not in registered)


def refuses(kind: TopologyType, act: str) -> str | None:
    """Why this pairing is refused, or None.

    THE REFUSAL IS A PROPERTY OF THE PAIRING. A deciding topology is fine
    pointed at a question nobody's constitution reserves, and a reporting
    topology is fine pointed at an attested act because reporting is not
    performing. It is the combination that is refused, so both are named in
    the answer.
    """
    found = by_type().get(kind)
    if found is None:
        return f"{kind.value} has no declared capability, so nothing knows what it would do"
    if found.status == REFUSED:
        return f"{kind.value} is refused here: {found.why}"
    if act in ATTESTED and found.decides:
        return (f"{kind.value} decides, and {act!r} is a person's by "
                f"constitution -- a machine performing it changes what it "
                f"asserts")
    return None


def render() -> str:
    """The plane, for somebody deciding what to point at what."""
    lines = ["what each topology would do, before anything runs one", ""]
    for entry in PLANE:
        marks = []
        if entry.spends:
            marks.append("spends")
        if entry.writes:
            marks.append("writes")
        if entry.decides:
            marks.append("decides")
        lines.append(f"{entry.status.upper():<10} {entry.topology.value:<12}"
                     f"{'  [' + ', '.join(marks) + ']' if marks else ''}")
        lines.append(f"           {entry.why}")
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


# --- the two shapes this work actually needed ---------------------------------


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

    **THIS IS THE SHAPE `qmcp.sweep` ALREADY WAS.** Nine parsers and fifteen
    questions, chosen by the work rather than by a setting. Generalised here so
    the manager knows the shape by name, and so a second caller does not write
    a third dispatcher.

    A unit whose shape has no worker is `taken=False` and named. That is the
    property that matters: dropping it silently would leave a run looking
    finished with two thirds of its work untouched.
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

    **IT REPORTS AND DOES NOT DECIDE.** The consensus is a count, and what to
    do about a split is a person's. A cross-check that closed the question
    would be the deciding shape wearing a reporting name -- and every mutation
    this pair has run today was a cross-check of exactly this form: break the
    thing, see whether the guard says so.

    `independent` is not decoration. Checkers that saw each other's reasoning
    would be one checker with extra steps, which is the failure mode of every
    panel that agrees too easily.
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
    could not be made is not agreement, and treating it as absent would let a
    broken checker quietly raise everybody else's share of the vote.
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
