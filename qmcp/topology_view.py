"""One description of a topology, for any number of windows onto it.

    uv run qmcp topology gallery
    uv run qmcp topology show pipeline --level 2

A `View` is boxes and arrows with kinds and notes. It holds no coordinates,
glyphs, colours or widths, so every window draws from the same description:
`dossier` draws it as text in a panel and `codecarto` as a graph in a page.

A view is one topology at one level. Level 0 is the black box: what goes in
and what comes out. Level 1 is the parts. Level 2 is the flow between them.

Boxes carry a `kind` and a `note`; arrows carry a kind that tells flow from
feedback from refusal. A window may show less than the description carries --
a terminal collapses some of it to a border style -- and never more: it draws
no dimension the description does not hold.

The plane's declarations are part of the view. A topology that spends, writes
or decides shows it, and a refused one is drawn refused rather than omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qmcp.agentframework.models.enums import TopologyType
from qmcp.orchestration import REFUSED, RUNS, by_type

# What a box is, which is as much as any renderer needs to choose a shape.
INPUT = "input"
WORKER = "worker"
GATE = "gate"
"""A point where something is decided or approved. A person's, unless stated."""

STORE = "store"
OUTPUT = "output"

# What an arrow means. A renderer may draw all three alike; it may not merge
# them, or a refusal would read as a path that is taken.
FLOW = "flow"
FEEDBACK = "feedback"
REFUSAL = "refusal"

BLACK_BOX, PARTS, FLOWS = 0, 1, 2
LEVELS = (BLACK_BOX, PARTS, FLOWS)


@dataclass(frozen=True)
class Box:
    """One node. No position, no size, no colour."""

    id: str
    label: str
    kind: str = WORKER
    note: str = ""
    """What a window with room may show. A terminal will drop it."""

    count: int | None = None
    """How many of this box there are, when that is known and varies -- nine
    parsers, six questions. `None` is unknown and is not one."""


@dataclass(frozen=True)
class Arrow:
    """One edge. `label` is what crosses, not how it is drawn."""

    frm: str
    to: str
    label: str = ""
    kind: str = FLOW

    weight: float | None = None
    """How strong this edge is, 0 to 1, or `None` for unmeasured.

    **`None` IS NOT ZERO AND A WINDOW MUST NOT DRAW IT AS ONE.** A hairline for
    an unmeasured edge asserts weakness nobody established; a hairline for a
    measured 0.01 reports one. The two need different renderings, which is why
    this is optional rather than defaulted to a number.

    A shape's own arrows carry `None`: "stage to stage" in a pipeline has no
    strength, it is just the shape. Weight appears when a view is built over
    real relations -- a thread to the projects it is about, a delta to the
    repositories it touches -- where the number was measured from something.
    """

    basis: str = ""
    """What the weight was read from, so the line is arguable rather than
    authoritative. Empty when there is no weight."""


@dataclass(frozen=True)
class View:
    """One topology at one level, ready for any window."""

    topology: str
    level: int
    boxes: tuple[Box, ...]
    arrows: tuple[Arrow, ...]
    caption: str = ""
    spends: bool = False
    writes: bool = False
    decides: bool = False
    status: str = RUNS

    @property
    def is_refused(self) -> bool:
        return self.status == REFUSED

    @property
    def marks(self) -> tuple[str, ...]:
        """The plane's declarations, for a window to show beside the shape."""
        found = []
        if self.spends:
            found.append("spends")
        if self.writes:
            found.append("writes")
        if self.decides:
            found.append("decides")
        return tuple(found)

    def box(self, box_id: str) -> Box | None:
        return next((b for b in self.boxes if b.id == box_id), None)


# The shapes, at level 2. Levels 0 and 1 are derived from these, so a black
# box cannot disagree with the flow it summarises.
_SHAPES: dict[TopologyType, dict[str, Any]] = {
    TopologyType.PIPELINE: {
        "caption": "stages in sequence, each one's output the next one's input",
        "boxes": [Box("in", "work", INPUT), Box("s1", "stage", WORKER),
                  Box("s2", "stage", WORKER), Box("s3", "stage", WORKER),
                  Box("out", "result", OUTPUT)],
        "arrows": [Arrow("in", "s1"), Arrow("s1", "s2"), Arrow("s2", "s3"),
                   Arrow("s3", "out")],
    },
    TopologyType.DELEGATION: {
        "caption": "each unit of work to the worker registered for its shape",
        "boxes": [Box("in", "work", INPUT),
                  Box("route", "route by shape", GATE),
                  Box("w1", "worker", WORKER, "one per shape"),
                  Box("w2", "worker", WORKER),
                  Box("none", "no worker", OUTPUT, "named, never dropped"),
                  Box("out", "result", OUTPUT)],
        "arrows": [Arrow("in", "route"), Arrow("route", "w1", "shape a"),
                   Arrow("route", "w2", "shape b"),
                   Arrow("route", "none", "unregistered", REFUSAL),
                   Arrow("w1", "out"), Arrow("w2", "out")],
    },
    TopologyType.CROSS_CHECK: {
        "caption": "independent checkers on one claim; the count is reported",
        "boxes": [Box("in", "claim", INPUT),
                  Box("c1", "checker", WORKER, "independent"),
                  Box("c2", "checker", WORKER, "independent"),
                  Box("c3", "checker", WORKER, "independent"),
                  Box("tally", "count", OUTPUT, "reported, not resolved")],
        "arrows": [Arrow("in", "c1"), Arrow("in", "c2"), Arrow("in", "c3"),
                   Arrow("c1", "tally"), Arrow("c2", "tally"),
                   Arrow("c3", "tally")],
    },
    TopologyType.ENSEMBLE: {
        "caption": "several workers on one item, answers combined",
        "boxes": [Box("in", "item", INPUT), Box("w1", "worker", WORKER),
                  Box("w2", "worker", WORKER),
                  Box("agg", "combine", WORKER),
                  Box("out", "answer", OUTPUT)],
        "arrows": [Arrow("in", "w1"), Arrow("in", "w2"), Arrow("w1", "agg"),
                   Arrow("w2", "agg"), Arrow("agg", "out")],
    },
    TopologyType.DEBATE: {
        "caption": "positions argued to a conclusion",
        "boxes": [Box("in", "question", INPUT),
                  Box("a", "position", WORKER), Box("b", "position", WORKER),
                  Box("judge", "conclude", GATE, "this shape decides"),
                  Box("out", "conclusion", OUTPUT)],
        "arrows": [Arrow("in", "a"), Arrow("in", "b"),
                   Arrow("a", "b", "rebut", FEEDBACK),
                   Arrow("b", "a", "rebut", FEEDBACK),
                   Arrow("a", "judge"), Arrow("b", "judge"),
                   Arrow("judge", "out")],
    },
    TopologyType.CHAIN_OF_COMMAND: {
        "caption": "escalation up a hierarchy until something decides",
        "boxes": [Box("in", "work", INPUT), Box("l1", "first", WORKER),
                  Box("l2", "escalation", WORKER),
                  Box("top", "decide", GATE, "this shape decides"),
                  Box("out", "result", OUTPUT)],
        "arrows": [Arrow("in", "l1"), Arrow("l1", "l2", "cannot resolve"),
                   Arrow("l2", "top", "cannot resolve"), Arrow("l1", "out"),
                   Arrow("top", "out")],
    },
    TopologyType.COMPOUND: {
        "caption": "topologies composed of topologies",
        "boxes": [Box("in", "work", INPUT),
                  Box("t1", "topology", WORKER, "any shape"),
                  Box("t2", "topology", WORKER, "any shape"),
                  Box("out", "result", OUTPUT)],
        "arrows": [Arrow("in", "t1"), Arrow("t1", "t2"), Arrow("t2", "out")],
    },
    TopologyType.COUNCIL: {
        "caption": "perspectives deliberate; the arbiter decides when they cannot",
        "boxes": [Box("in", "issue", INPUT),
                  Box("members", "perspectives", WORKER,
                      "storyteller, dreamer, strategist, and six more", count=9),
                  Box("arbiter", "arbiter", GATE,
                      "makes the final decision -- refused here"),
                  Box("out", "verdict", OUTPUT)],
        "arrows": [Arrow("in", "members"),
                   Arrow("members", "members", "deliberate", FEEDBACK),
                   Arrow("members", "arbiter"),
                   Arrow("arbiter", "out", "decides", REFUSAL)],
    },
}


def view_of(kind: TopologyType, level: int = FLOWS) -> View:
    """One topology at one level.

    Levels 0 and 1 are folded down from the level-2 shape rather than written
    separately, so a black box cannot disagree with the flow it summarises.
    """
    if level not in LEVELS:
        raise ValueError(f"level is one of {LEVELS}, not {level!r}")

    shape = _SHAPES.get(kind)
    capability = by_type().get(kind)
    if shape is None:
        # Declared in the vocabulary, drawn by nothing: shown as an empty
        # black box rather than omitted, so the name stays visible.
        return View(topology=kind.value, level=BLACK_BOX,
                    boxes=(Box("in", "work", INPUT),
                           Box("box", kind.value, WORKER,
                               "no shape described for this name"),
                           Box("out", "result", OUTPUT)),
                    arrows=(Arrow("in", "box"), Arrow("box", "out")),
                    caption="in the vocabulary; no shape described",
                    status=capability.status if capability else RUNS)

    boxes = tuple(shape["boxes"])
    arrows = tuple(shape["arrows"])

    if level == BLACK_BOX:
        inputs = [b for b in boxes if b.kind == INPUT]
        outputs = [b for b in boxes if b.kind == OUTPUT]
        middle = Box("box", kind.value, WORKER,
                     f"{len(boxes) - len(inputs) - len(outputs)} parts inside")
        boxes = (*inputs, middle, *outputs)
        arrows = tuple(
            [Arrow(b.id, "box") for b in inputs]
            + [Arrow("box", b.id) for b in outputs])
    elif level == PARTS:
        # The parts, and nothing about the order they run in.
        arrows = ()

    return View(
        topology=kind.value, level=level, boxes=boxes, arrows=arrows,
        caption=shape["caption"],
        spends=bool(capability and capability.spends),
        writes=bool(capability and capability.writes),
        decides=bool(capability and capability.decides),
        status=capability.status if capability else RUNS,
    )


def gallery(level: int = BLACK_BOX) -> list[View]:
    """Every core topology at one level, in the plane's order.

    Refused and unbuilt shapes are included, marked as such.
    """
    return [view_of(kind, level) for kind in _SHAPES]


def as_payload(view: View) -> dict[str, Any]:
    """The view as data, for a window on the other side of the seam.

    Flat and JSON-shaped: windows read it over HTTP and import nothing from
    this module.
    """
    return {
        "topology": view.topology,
        "level": view.level,
        "caption": view.caption,
        "status": view.status,
        "marks": list(view.marks),
        "boxes": [{"id": b.id, "label": b.label, "kind": b.kind,
                   "note": b.note, "count": b.count} for b in view.boxes],
        "arrows": [{"from": a.frm, "to": a.to, "label": a.label,
                    "kind": a.kind, "weight": a.weight, "basis": a.basis}
                   for a in view.arrows],
    }


def _subject_address(subject: str, relations: list[dict[str, Any]]) -> str:
    """`<owner>/<subject>`, with the owner read off the relations.

    Returns "" when no relation names an owner: the subject is then a name
    with no known owner, and no owner is guessed.
    """
    if "/" in subject:
        return subject
    owners = {
        str(relation.get("target") or "").split("/", 1)[0]
        for relation in relations
        if "/" in str(relation.get("target") or "")
    }
    owners.discard("")
    return f"{owners.pop()}/{subject}" if len(owners) == 1 else ""


def from_relations(subject: str, relations: list[dict[str, Any]],
                   caption: str = "") -> View:
    """A view of one thing and what it is related to, with the edges weighted.

    Each relation carries a `weight` and the `evidence` it was read from, both
    measured by `qmcp.threads.consolidate`; this arranges them into boxes and
    arrows and changes neither. A relation with no weight produces an arrow
    with `weight=None`, which a window draws differently from a weak one: an
    unmeasured edge is not a negligible one.

    There is one box per address, not one per relation: relations reaching the
    same address are evidence about one thing
    (`records/DRAFT-a-route-is-an-address.md`). The arrows are not merged, so
    each relation keeps its own weight and basis.
    """
    # The subject carries its own address, like every relation target. Its
    # owner is read from the relations, which are resolved addresses.
    subject_address = _subject_address(subject, relations)
    boxes = [Box("subject", subject, INPUT, note=subject_address)]
    arrows = []
    box_of: dict[str, str] = {}
    for index, relation in enumerate(relations):
        target = str(relation.get("target") or f"?{index}")
        box_id = box_of.get(target)
        if box_id is None:
            # The tail of an address is what fits on a line; the whole address
            # is the note, so nothing is lost to abbreviation.
            box_id = f"r{len(box_of)}"
            box_of[target] = box_id
            boxes.append(Box(box_id, target.rsplit("/", 1)[-1], WORKER,
                             note=target))
        weight = relation.get("weight")
        evidence = (relation.get("evidence") or [{}])[0]
        arrows.append(Arrow(
            "subject", box_id,
            label=str(relation.get("relation") or ""),
            weight=None if weight is None else float(weight),
            basis=str(evidence.get("basis") or ""),
        ))
    return View(topology=subject, level=FLOWS, boxes=tuple(boxes),
                arrows=tuple(arrows),
                caption=caption or f"{len(relations)} relation(s), weighted")


# --- which visual channel carries which data axis ------------------------------
#
# Each channel is declared here, so every window maps it to the same data axis
# and a line's thickness means the same thing in each.
#
# `measured` is its own axis, not the bottom of `strength`: an unmeasured edge
# is an absence of evidence, not evidence of absence. It therefore has a
# different channel -- style and colour rather than width -- and a window that
# runs out of channels drops one and says so rather than folding two axes into
# one.


@dataclass(frozen=True)
class Channel:
    """One visual channel, and the single data axis it carries."""

    channel: str
    axis: str
    scale: str
    """`continuous` or `categorical`. A categorical axis on a continuous
    channel invents an ordering; the reverse throws a magnitude away."""

    values: tuple[str, ...] = ()
    note: str = ""


ENCODING: tuple[Channel, ...] = (
    Channel(
        channel="line_weight", axis="strength", scale="continuous",
        note="0 to 1, and only where a weight was measured. A line has no "
             "width for an unmeasured edge -- it is drawn by style instead"),
    Channel(
        channel="line_style", axis="measured", scale="categorical",
        values=("measured", "unmeasured"),
        note="the axis that must never be folded into strength. Unmeasured is "
             "not weak; it is unlooked-at, and a reader has to be able to tell"),
    Channel(
        channel="line_colour", axis="relation_kind", scale="categorical",
        values=(FLOW, FEEDBACK, REFUSAL),
        note="what the edge is, which is not how strong it is. A refusal is "
             "drawn a refusal however heavily travelled the path it forbids"),
    Channel(
        channel="node_shape", axis="box_kind", scale="categorical",
        values=(INPUT, WORKER, GATE, STORE, OUTPUT),
        note="a gate is a gate at any size"),
)


def encoding_payload() -> list[dict[str, Any]]:
    """The mapping, for a window to read before it draws anything."""
    return [{"channel": c.channel, "axis": c.axis, "scale": c.scale,
             "values": list(c.values), "note": c.note} for c in ENCODING]


def channels_for(axis: str) -> list[str]:
    """Every channel declared to carry one axis."""
    return [c.channel for c in ENCODING if c.axis == axis]
