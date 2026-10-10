"""The orchestration plane, over HTTP, for a window that must not decide.

    GET /v1/orchestration/plane      what every shape would do, declared
    GET /v1/orchestration/runnable   what each shape is short of, given a hand

Every field is read from `qmcp.orchestration` -- `PLANE`, `OPTIONS`, `NEEDS`,
`ATTESTED` and its drift reports -- and nothing is derived here, so the served
plane and `uv run qmcp orchestration plane` give one answer. A window asks
these routes rather than keeping its own copy of which shapes spend, decide or
are refused.

The routes name nobody -- a capability is a claim about a shape -- so they are
served off loopback as well as on it.

`runnable` takes workers, a budget, a model and a build. There is no parameter
for `person`: a shape whose need is a person's judgement is never made
runnable by a request, as in `qmcp.orchestration.unmet`.

What this cannot do: say whether a shape whose needs are all met will work.
`unmet` reads declarations and runs nothing; a run reports in the invocation
record.
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

    The drift reports are included so a window can show where the plane and
    the registry disagree.

    `options` are the second declarations one setting selects. A window
    offering a council offers the advisory one from here, not from a copy.
    """
    return {
        "schema": 1,
        "capabilities": [capability_payload(c) for c in plane.PLANE],
        "options": [{"setting": o.setting, "value": o.value,
                     "capability": capability_payload(o.capability)}
                    for o in plane.OPTIONS],
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
