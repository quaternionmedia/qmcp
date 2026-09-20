"""The orchestration plane, over HTTP, for a window that must not decide.

    GET /v1/orchestration/plane      what every shape would do, declared
    GET /v1/orchestration/runnable   what each shape is short of, given a hand

**THE PLANE WAS CLI-ONLY, AND A WINDOW CANNOT SHELL OUT.**
`uv run qmcp orchestration plane` printed everything `qmcp.orchestration`
declares, and the front end growing a topology designer had no way to ask the
same question.
A window that answered it itself -- by hard-coding which shapes spend, which
decide, which are refused -- would be a second copy of this organisation's
rule, kept current by nobody. So the window asks, and this is what answers.

**IT SERVES THE DECLARATION AND ADDS NOTHING TO IT.** Every field here is read
off `PLANE`, `NEEDS`, `ATTESTED` and the drift reports the module already
computes. Nothing is derived that the module does not state, because a plane
served over HTTP that disagreed with the one printed in a terminal would be two
answers to one question.

**SAFE TO SERVE ANYWHERE.** These routes name nobody: a capability is a claim
about a shape, not about a person or a conversation. They are registered
beside the topology shapes, off loopback as well as on it.

**`person` IS NEVER SUPPLIED, AND THERE IS NO PARAMETER FOR IT.** `runnable`
takes workers, a budget, a model and a build, which are things a caller can
have. A shape whose need is a person's judgement is not made runnable by a
query string, and a parameter that let it be would be the refusal quietly
undone. `qmcp.orchestration.unmet` holds the same line.

WHAT THIS CANNOT DO. Say whether a shape whose needs are all met will work.
`unmet` reads declarations and runs nothing; the invocation record is where a
run reports.
"""

from __future__ import annotations

from typing import Any

from qmcp import orchestration as plane


def need_payload(need: plane.Need) -> dict[str, Any]:
    """One need as data: what is missing, why it matters, what supplies it."""
    return {"key": need.key, "because": need.because,
            "supplied_by": need.supplied_by}


def capability_payload(capability: plane.Capability) -> dict[str, Any]:
    """One capability as data, every declared field carried."""
    return {
        "topology": capability.topology.value,
        "status": capability.status,
        "can_run": capability.can_run,
        "spends": capability.spends,
        "writes": capability.writes,
        "decides": capability.decides,
        "why": capability.why,
        "needs": [need_payload(need) for need in capability.needs],
    }


def plane_payload() -> dict[str, Any]:
    """The whole plane as one document.

    The drift reports ride along because a window showing the plane should
    show where the plane and the registry disagree -- a declaration for a
    shape nothing registers is a picture of something that is not there.
    """
    return {
        "schema": 1,
        "capabilities": [capability_payload(c) for c in plane.PLANE],
        "statuses": [plane.RUNS, plane.BRAINSTORM, plane.REFUSED],
        "needs": list(plane.NEEDS),
        "attested": list(plane.ATTESTED),
        "drift": {
            "undeclared": plane.undeclared(),
            "stubs": plane.stubs(),
            "unregistered_types": plane.unregistered_types(),
        },
        "reading": {
            "declared_not_discovered": (
                "every field is a claim by whoever wrote the entry, checked "
                "against the registry by `drift.undeclared` and against "
                "nothing else. `uv run qmcp orchestration plane` prints the "
                "same declaration."),
        },
    }


def runnable_payload(*, workers: int = 0, budget: int = 0,
                     model: bool = False,
                     built: bool | None = None) -> dict[str, Any]:
    """What a caller with this hand could run, and what each shape still wants.

    `runnable` and `shapes` answer at two grains: the first is the list a
    window can offer, the second is why everything else is not on it.
    """
    have = {"workers": workers, "budget": budget, "model": model}
    if built is not None:
        have["built"] = built
    runnable = [kind.value for kind in plane.runnable_now(**have)]
    shapes = []
    for capability in plane.PLANE:
        short = plane.unmet(capability, **have)
        shapes.append({
            "topology": capability.topology.value,
            "status": capability.status,
            "runnable": not short,
            "unmet": [need_payload(need) for need in short],
        })
    return {
        "schema": 1,
        "have": {"workers": workers, "budget": budget, "model": model,
                 "built": built},
        "runnable": runnable,
        "shapes": shapes,
        "never_supplied": {
            "key": plane.PERSON,
            "because": ("a shape whose need is a person's judgement is not "
                        "made runnable by a parameter, so there is none"),
        },
    }


def register(app: Any) -> None:
    """Attach the plane routes. Safe to serve anywhere.

    Takes the app rather than creating one, for the reason
    `qmcp.topology_service.register` gives: what may leave the machine is
    decided in `create_app`, once.
    """
    from fastapi import Query

    @app.get("/v1/orchestration/plane")
    async def orchestration_plane() -> dict[str, Any]:
        """Every shape's declared capability, the vocabularies, and the drift."""
        return plane_payload()

    @app.get("/v1/orchestration/runnable")
    async def orchestration_runnable(
        workers: int = Query(0, ge=0),
        budget: int = Query(0, ge=0),
        model: bool = Query(False),
        built: bool | None = Query(None),
    ) -> dict[str, Any]:
        """What this hand could run now, and what each shape is short of.

        No `person` parameter, on purpose. See the module docstring.
        """
        return runnable_payload(workers=workers, budget=budget, model=model,
                                built=built)
