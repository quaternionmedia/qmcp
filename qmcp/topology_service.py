"""The topology, over HTTP, for whatever wants to draw it.

Two classes of route, divided by personal data:

- The *shapes* -- `/v1/topology`, `/v1/topology/encoding`,
  `/v1/topology/shape/{kind}`, `/v1/topology/schema/{kind}` -- are this
  harness's own vocabulary: the collaboration shapes, `governed` (the seam a
  model is called through), and the schema of each shape's configuration
  class. They name nobody and are served wherever the server is bound.
- The *readings* -- `/v1/topology/relations/{subject}` -- are derived from the
  thread archive, which holds a person's conversations, and carry project
  addresses, turn counts and topics. They are registered only on loopback, as
  `qmcp.threads.service` is. Off loopback the route does not exist, so a
  response reveals nothing about whether an archive is there.

`register` and `register_readings` take the app rather than creating one, so
what may leave the machine is decided once, in `create_app`.

Every route returns the flat payload `qmcp.topology_view.as_payload` produces,
with the encoding that says which visual channel carries which data axis. How
a window draws it is the window's.
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

    The vocabulary's shapes, then `governed`, listed beside them rather than
    added to `TopologyType`: a name in that enum is inherited by every
    consumer of the agent framework.
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

        Served on its own, so a window can check the mapping it draws with
        without fetching a view.
        """
        return {"schema": 1, "encoding": tv.encoding_payload()}

    @app.get("/v1/topology/shape/{kind}")
    async def one_topology(
        kind: str,
        level: int = Query(tv.FLOWS, ge=0, le=2),
    ) -> dict[str, Any]:
        """One topology as a payload, at the requested resolution.

        `/shape/` is in the path so a shape cannot be mistaken for the
        `relations` route, which reads a person's conversations.
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

    @app.get("/v1/topology/schema/{kind}")
    async def configuration_schema(kind: str) -> dict[str, Any]:
        """The JSON schema of one shape's configuration class.

        What a designer's form is built from, so the form cannot offer a field
        the class does not hold. The class is the one
        `Topology.get_typed_config` validates a saved design through, read
        from the same table, so a config that fits the schema fits the store.

        `governed` is a 404 with a reason rather than an empty schema: it is a
        seam, not a configurable shape.
        """
        from qmcp.agentframework.models.entities.topologies import config_class_for
        from qmcp.agentframework.models.enums import TopologyType
        from qmcp.orchestration import by_type

        kinds = [t.value for t in TopologyType if config_class_for(t) is not None]
        if kind == "governed":
            raise HTTPException(
                status_code=404,
                detail=("governed is a seam, not a configurable shape: it has "
                        "no config class. `GET /v1/topology/shape/governed` "
                        "draws it; `GET /v1/topology/schema/{kind}` answers "
                        f"for {', '.join(kinds)}."))
        try:
            wanted = TopologyType(kind)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=(f"no topology named {kind!r}. Kinds with a "
                        f"configuration: {', '.join(kinds)}."))
        config_class = config_class_for(wanted)
        if config_class is None:
            raise HTTPException(
                status_code=404,
                detail=(f"{kind} is in the vocabulary and has no config class. "
                        f"Kinds with a configuration: {', '.join(kinds)}."))
        capability = by_type().get(wanted)
        return {
            "schema": 1,
            "topology": wanted.value,
            "config_class": config_class.__name__,
            "json_schema": config_class.model_json_schema(),
            "status": capability.status if capability else None,
        }


def register_readings(app: Any, root: Path) -> None:
    """Attach the archive-derived readings. **Loopback only.**

    Separate from `register`, and called by `create_app` only on a loopback
    bind, as the thread routes are.
    """
    from fastapi import HTTPException, Query

    @app.get("/v1/topology/relations/{subject}")
    async def relations_for_subject(
        subject: str,
        min_share: float | None = Query(None, ge=0.0, le=1.0),
    ) -> dict[str, Any]:
        """What the archive says one project is related to, weighted.

        Every arrow carries the weight `qmcp.threads.consolidate` measured and
        the basis it was read from. A relation nobody measured has a null
        weight, and a window keeps it null: an unmeasured edge is not a
        negligible one.
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
