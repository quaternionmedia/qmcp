"""The topology, over HTTP, for whatever wants to draw it.

**THE FRONT ENDS HAD NOTHING TO FETCH.** `qmcp.topology_view` could build a view
and hand it to anything that imported it, and both front ends are repositories
that must not import it. So a demo could prove the two renderings agreed by
running the harness's code in a subprocess, while nothing at either front end's
port could obtain a topology at all. The contract was tested and the seam was
not deployed, and those look identical from inside the demo.

**TWO CLASSES OF ROUTE, AND THE DIFFERENCE IS PERSONAL DATA.**

- The *shapes* -- `/v1/topology`, `/v1/topology/encoding`,
  `/v1/topology/shape/{kind}` -- are this harness's own vocabulary. A
  delegation topology looks the same on every machine and names nobody, and so
  does `governed`, the seam a model is called through. They are served wherever
  the server is bound.
- The *readings* -- `/v1/topology/relations/{subject}` -- are derived from the
  thread archive, which holds a person's conversations. They carry project
  addresses, turn counts and what somebody talked about, and they are
  registered **only on loopback**, exactly as `qmcp.threads.service` is. Off
  loopback the route does not exist rather than refusing, because a 403 tells
  a caller the archive is here.

That split is the whole reason this module takes the app rather than creating
one: the decision about what may leave the machine is already made in
`create_app`, and a second module inventing its own answer is how two answers
start to differ.

**IT SERVES A DOCUMENT, NOT A PICTURE.** Every route returns the same flat
payload `as_payload` produces, plus the encoding that says which visual channel
carries which data axis. What a front end does with it is the front end's
business -- one draws a terminal and one draws a graph, and neither is more
correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmcp import topology_view as tv

# The subject of a reading is a repository name. Anything longer than this is
# not one, and refusing early keeps a hostile path out of the survey loop.
MAX_SUBJECT = 100

# What a survey may spend. Zero: these sources read files already on the disk.
# Named rather than defaulted, because `records/DRAFT-no-unattended-spending.md`
# says an amount is consented to rather than a category.
SURVEY_BUDGET = 0


def _gallery() -> list[Any]:
    """Every shape a caller can ask for, at black-box level.

    The vocabulary's shapes, and then `governed` -- which is not one of them.
    `TopologyType` is a catalogue of collaboration patterns and the governed
    seam is this organisation's own pipeline, so it is listed beside them
    rather than added to the enum: a name in that vocabulary is a name every
    consumer of the agent framework inherits.
    """
    from qmcp import governed

    return [*tv.gallery(), governed.view(level=tv.BLACK_BOX)]


def _views() -> dict[str, Any]:
    """Every topology this harness knows, at black-box level."""
    return {
        "schema": 1,
        "level": tv.BLACK_BOX,
        "topologies": [
            {"topology": view.topology, "caption": view.caption,
             "status": view.status,
             "boxes": len(view.boxes), "arrows": len(view.arrows)}
            for view in _gallery()
        ],
        "encoding": tv.encoding_payload(),
    }


def register(app: Any) -> None:
    """Attach the topology shapes. Safe to serve anywhere.

    Takes the app rather than creating one, so the topology is served by the
    process that already serves everything else -- one thing to start, one
    port.
    """
    from fastapi import HTTPException, Query

    @app.get("/v1/topology")
    async def list_topologies() -> dict[str, Any]:
        """Every topology, and the encoding a window must read before drawing."""
        return _views()

    @app.get("/v1/topology/encoding")
    async def encoding() -> dict[str, Any]:
        """Which visual channel carries which data axis.

        Served separately because a window should be able to check the mapping
        it is honouring without fetching a view it does not want. A front end
        that hard-coded the mapping would keep drawing an old contract, and the
        picture would be confidently wrong.
        """
        return {"schema": 1, "encoding": tv.encoding_payload()}

    @app.get("/v1/topology/shape/{kind}")
    async def one_topology(
        kind: str,
        level: int = Query(tv.FLOWS, ge=0, le=2),
    ) -> dict[str, Any]:
        """One topology as a payload, at the requested resolution.

        `/shape/` sits in the path so a topology can never be mistaken for the
        `relations` route below -- one is this harness's vocabulary and the
        other is a person's conversations, and a route that could be confused
        for the other is the wrong shape for that boundary.
        """
        from qmcp import governed
        from qmcp.agentframework.models.enums import TopologyType

        if kind == "governed":
            # Not in `TopologyType`, and served here because a front end
            # drawing the shapes should not need a second route to draw the
            # one shape that calls a model.
            view = governed.view(level=level)
        else:
            try:
                wanted = TopologyType(kind)
            except ValueError:
                raise HTTPException(
                    status_code=404,
                    detail=(f"no topology named {kind!r}. "
                            f"`GET /v1/topology` lists them."))
            view = tv.view_of(wanted, level=level)
        return {"schema": 1, "payload": tv.as_payload(view),
                "encoding": tv.encoding_payload(), "source": "topology"}


def register_readings(app: Any, root: Path) -> None:
    """Attach the archive-derived readings. **Loopback only.**

    Separate from `register` so the caller cannot serve these by accident. The
    two are registered at different times in `create_app` for the same reason
    the thread routes are: what may leave this machine is one decision, taken
    in one place.
    """
    from fastapi import HTTPException, Query

    @app.get("/v1/topology/relations/{subject}")
    async def relations_for_subject(
        subject: str,
        min_share: float | None = Query(None, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        """What the archive says one project is related to, weighted.

        Every arrow carries the weight `qmcp.threads.consolidate` measured and
        the basis it was read from. **A relation nobody measured arrives with a
        null weight and must stay that way** -- filling it in would turn
        "nobody looked" into "negligible", and a window cannot recover the
        difference once this end has lost it.
        """
        if not subject or len(subject) > MAX_SUBJECT:
            raise HTTPException(status_code=400,
                                detail="a subject is a repository name")

        relations, surveyed = _survey(root, subject, min_share)
        if not surveyed:
            raise HTTPException(
                status_code=404,
                detail=("no readable thread archive. `uv run qmcp threads "
                        "index --write` builds one. An absent archive is an "
                        "absent answer rather than a subject with no "
                        "relations."))

        view = tv.from_relations(
            subject, relations,
            caption=f"what the archive says about {subject}")
        return {"schema": 1, "payload": tv.as_payload(view),
                "encoding": tv.encoding_payload(),
                "source": "thread archive", "surveyed": surveyed,
                "relations": len(relations)}


def _survey(root: Path, subject: str,
            min_share: float | None = None) -> tuple[list[dict], int]:
    """Every relation the archive states about `subject`, and threads read.

    Returns the count separately because zero relations and zero threads are
    different answers: the first says the archive was read and this subject is
    not in it, the second says there was nothing to read.
    """
    from qmcp.spend import Budget
    from qmcp.threads import consolidate
    from qmcp.threads.chatgpt import ChatGPTThreads
    from qmcp.threads.claude import ClaudeThreads

    corpus = Path("governance") / "qm"
    if not (corpus / "ci" / "workspace.yaml").is_file():
        return [], 0
    names = consolidate.roster(corpus)

    threads: list[Any] = []
    for source_class in (ClaudeThreads, ChatGPTThreads):
        try:
            threads.extend(
                source_class(root=root).fetch([], Budget(authorised=SURVEY_BUDGET)))
        except Exception:                          # noqa: BLE001
            continue

    found = []
    for thread in threads:
        reading = consolidate.about(thread, names, min_share=min_share)
        for relation in consolidate.relations_for(thread, reading,
                                                  project_of=dict(names)):
            if subject in str(relation.get("source", "")) \
               or subject in str(relation.get("target", "")):
                found.append(relation)
    return found, len(threads)
