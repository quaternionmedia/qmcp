"""Saved topology designs, over HTTP, with the plane's verdict on each one.

    POST /v1/topologies          save a design, validated through its kind's class
    GET  /v1/topologies          every saved design
    GET  /v1/topologies/{ref}    one, by id or by name
    PUT  /v1/topologies/{ref}    change its description, config or version

**THE TABLE EXISTED AND NOTHING SERVED IT.** `Topology` has been a table since
the agent framework was written, `qmcp council create` builds a row and prints
it, and no route read or wrote one. A front end growing a designer had nowhere
to put a design except its own storage -- and a design held by the window is a
design the harness has never seen, so nothing could tell the designer what the
plane thinks of it.

**DESIGNING IS NOT AN ACT. RUNNING IS.** A refused shape can be saved here.
`council` is refused by `qmcp.orchestration` because its arbiter decides, and
the refusal is of the *run*: drawing the shape, describing it, keeping it, are
none of them the act `governance/qm/ci/attested-registry.yaml` reserves. So a
saved council comes back with the plane's refusal beside it and a sentence
saying the design is kept and the run is what is refused. Refusing the save
would hide the rule at the moment somebody was choosing a shape, which is the
same reasoning `qmcp.topology_view.gallery` gives for drawing `council` rather
than dropping it.

**THE WINDOW ADDS NO GOVERNANCE, SO EVERY VERDICT HERE IS THE PLANE'S.** The
`capability` block on every row is read from `qmcp.orchestration` -- status,
what the shape spends, writes or decides, what it needs, and whether it is
refused -- and this module invents none of it. A window that computed its own
would be a second copy of the rule.

**A CONFIG IS VALIDATED THROUGH THE CLASS ITS KIND DECLARES**, the same one
`Topology.get_typed_config` reads a saved row through, and what is stored is
what that class accepted with its defaults filled. So what a window reads back
is the shape as the harness reads it, and a design that saved will load.

**EVERY ROW CARRIES AN ADDRESS**, `<owner>/<repo>/topology/<name>`, built by
`qmcp.addresses` from the repository's own identity. When the identity cannot
be established the address is absent with the reason beside it, never guessed:
`qmcp.identity` says why a guessed owner is worse than none.

**SAFE TO SERVE ANYWHERE.** A design is a shape and a configuration. It names
no person and holds no conversation, so these routes are registered beside
the topology shapes wherever the server is bound.

WHAT THIS CANNOT DO. Delete. There is no `DELETE` route: a design somebody
saved is a record, and removing one is a decision this module has no way to
know was somebody's. Nor run: nothing here executes a topology, and a saved
design with every need met is still a drawing until a command runs it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlmodel import select

from qmcp import identity
from qmcp import orchestration as plane
from qmcp.addresses import topology_address
from qmcp.agentframework.models.base import utc_now
from qmcp.agentframework.models.entities.topologies import (
    Topology,
    config_class_for,
)
from qmcp.agentframework.models.enums import TopologyType
from qmcp.orchestration_service import need_payload

# What `PUT` may change. The name is the design's identity and its address;
# the kind is what the config was validated against. Changing either would be
# a different design wearing an existing address.
MUTABLE = ("description", "config", "version")

Sessions = Callable[[], Any]
"""A callable returning an async context manager that yields a session,
committed on exit -- what `qmcp.db.get_session` is."""


def sessions_at(path: Path) -> Sessions:
    """A session factory over one SQLite file, with the tables created.

    For a walkthrough or a test. The configured database holds somebody's
    human-in-the-loop queue, and `AGENTS.md` records what happens to another
    session's work when a test writes into it.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel, create_engine

    import qmcp.db.models  # noqa: F401  -- registers every server table
    from qmcp.db.engine import session_scope

    posix = Path(path).as_posix()
    SQLModel.metadata.create_all(create_engine(f"sqlite:///{posix}"))
    return partial(session_scope, create_async_engine(f"sqlite+aiosqlite:///{posix}"))


def kinds() -> list[str]:
    """Every kind a design may declare: the vocabulary, with a config class."""
    return [t.value for t in TopologyType if config_class_for(t) is not None]


def capability_block(kind: TopologyType, act: str = "") -> dict[str, Any]:
    """The plane's verdict on one kind, and on one pairing when `act` is given.

    `refusal` is `qmcp.orchestration.refuses` verbatim: a sentence, or None.
    When a kind has no declared capability the block says so through the
    refusal rather than through empty fields -- `needs` is None, not `[]`,
    because nobody has declared what the shape wants.
    """
    capability = plane.by_type().get(kind)
    refusal = plane.refuses(kind, act)
    block: dict[str, Any] = {
        "declared": capability is not None,
        "status": capability.status if capability else None,
        "spends": capability.spends if capability else None,
        "writes": capability.writes if capability else None,
        "decides": capability.decides if capability else None,
        "why": capability.why if capability else None,
        "needs": ([need_payload(n) for n in capability.needs]
                  if capability else None),
        "refusal": refusal,
    }
    if refusal:
        block["saved_anyway"] = (
            "designing a shape is not an act; running it is. The design is "
            "kept, and the run is what this harness refuses.")
    return block


def _stamp(when: datetime | None) -> str | None:
    """A timestamp as text, UTC either way.

    SQLite stores a naive datetime, so a row just written reports `+00:00`
    and the same row read back reports nothing -- two spellings of one
    instant, and a window comparing them would see a change nobody made.
    `qmcp.server` treats a stored naive datetime as UTC for the same reason.
    """
    if when is None:
        return None
    return (when if when.tzinfo else when.replace(tzinfo=UTC)).isoformat()


def row_payload(row: Topology, *, project: str | None = None,
                act: str = "") -> dict[str, Any]:
    """One saved design as data, addressed, with the plane's verdict."""
    project = project or identity.this_project()
    payload: dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "topology_type": TopologyType(row.topology_type).value,
        "version": row.version,
        "config": row.config,
        "created_at": _stamp(row.created_at),
        "updated_at": _stamp(row.updated_at),
        "capability": capability_block(TopologyType(row.topology_type), act),
    }
    if identity.is_known(project):
        payload["address"] = topology_address(row.name, project)
    else:
        payload["address"] = None
        payload["address_unknown"] = (
            "this checkout has no origin remote and QMCP_PROJECT is unset, so "
            "the owner is not known. An address built on a guess would join "
            "this design to somebody else's organisation.")
    return payload


def _validation_detail(where: str, error: ValidationError) -> dict[str, Any]:
    """Pydantic's errors, as a reason a window can show beside the field."""
    return {
        "where": where,
        "errors": [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in error.errors(include_url=False)
        ],
    }


def validated_config(kind: TopologyType, config: Any) -> dict[str, Any]:
    """`config` through the kind's class, defaults filled. Raises `ValidationError`.

    Raises `ValueError` for a kind with no class, which the vocabulary does not
    currently contain; the branch exists so a new enum member without a class
    is a refusal rather than a row nothing can load.
    """
    config_class = config_class_for(kind)
    if config_class is None:
        raise ValueError(f"{kind.value} has no config class")
    return config_class.model_validate(config or {}).model_dump(mode="json")


def register(app: Any, sessions: Sessions | None = None,
             project: str | None = None) -> None:
    """Attach the design routes. Safe to serve anywhere.

    `sessions` defaults to the configured database, `qmcp.db.get_session`.
    `project` defaults to the repository's own identity, read per request so
    an environment set after import is honoured.
    """
    from fastapi import HTTPException, Query

    if sessions is None:
        from qmcp.db import get_session
        sessions = get_session

    async def _find(session: Any, ref: str) -> Topology | None:
        """By id when the reference is an ASCII integer, then by name.

        A name may legally be all digits, so a digit reference tries the id
        first and the name second; the id wins a collision, and that order is
        the rule rather than an accident of the query.

        **`isdigit()` IS NOT `int()`'s TEST.** `'²'.isdigit()` (superscript
        two) is True and `int('²')` raises, so a saveable name made of
        such characters turned every read of it into a 500. The gate is the
        conversion itself: an id is what `int` accepts and nothing wider.
        """
        try:
            wanted = int(ref) if ref.isascii() and ref.isdigit() else None
        except ValueError:
            wanted = None
        if wanted is not None:
            found = (await session.execute(
                select(Topology).where(Topology.id == wanted))).scalar_one_or_none()
            if found is not None:
                return found
        return (await session.execute(
            select(Topology).where(Topology.name == ref.lower()))).scalar_one_or_none()

    @app.post("/v1/topologies", status_code=201)
    async def save_design(body: dict[str, Any]) -> dict[str, Any]:
        """Save one design, validated through its kind's configuration class.

        422 carries the validation detail, so a window can put the message
        beside the field. 409 is a name already taken: the name is the
        address, and two designs at one address would be one design with two
        histories.
        """
        raw_kind = body.get("topology_type")
        try:
            kind = TopologyType(raw_kind)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail={"where": "topology_type",
                        "errors": [{"loc": ["topology_type"],
                                    "msg": (f"no topology named {raw_kind!r}. "
                                            f"Kinds: {', '.join(kinds())}"),
                                    "type": "unknown_kind"}]})
        try:
            config = validated_config(kind, body.get("config"))
        except ValidationError as error:
            raise HTTPException(status_code=422,
                                detail=_validation_detail("config", error))
        except ValueError as error:
            raise HTTPException(status_code=422,
                                detail={"where": "topology_type",
                                        "errors": [{"loc": ["topology_type"],
                                                    "msg": str(error),
                                                    "type": "no_config_class"}]})
        try:
            # `model_validate`, because a table model skips validation on
            # construction -- `Topology(name="Bad Name!")` would store the
            # bad name.
            row = Topology.model_validate({
                "name": body.get("name"),
                "description": body.get("description"),
                "topology_type": kind,
                "version": body.get("version", "1.0.0"),
                "config": config,
            })
        except ValidationError as error:
            raise HTTPException(status_code=422,
                                detail=_validation_detail("design", error))

        async with sessions() as session:
            taken = (await session.execute(
                select(Topology).where(Topology.name == row.name))).scalar_one_or_none()
            if taken is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(f"a design named {row.name!r} exists, id {taken.id}. "
                            f"`PUT /v1/topologies/{row.name}` changes it."))
            session.add(row)
            await session.flush()
            return {"schema": 1, **row_payload(row, project=project)}

    @app.get("/v1/topologies")
    async def list_designs() -> dict[str, Any]:
        """Every saved design, oldest first, each addressed and judged."""
        async with sessions() as session:
            rows = (await session.execute(
                select(Topology).order_by(Topology.created_at, Topology.id)
            )).scalars().all()
            return {
                "schema": 1,
                "count": len(rows),
                "topologies": [row_payload(r, project=project) for r in rows],
            }

    @app.get("/v1/topologies/{ref}")
    async def one_design(
        ref: str,
        act: str = Query("", description=(
            "an act to judge the pairing against; empty judges the shape alone")),
    ) -> dict[str, Any]:
        """One design by id or by name, with the plane's verdict.

        `act` lets a window ask what the plane would say about pointing this
        design at a particular act, which is `qmcp.orchestration.refuses`'
        second argument. A deciding shape is refused an attested act and
        allowed an ordinary one, and only the pairing knows which.
        """
        async with sessions() as session:
            row = await _find(session, ref)
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=(f"no design {ref!r}, by id or by name. "
                            f"`GET /v1/topologies` lists them."))
            return {"schema": 1, **row_payload(row, project=project, act=act)}

    @app.put("/v1/topologies/{ref}")
    async def change_design(ref: str, body: dict[str, Any]) -> dict[str, Any]:
        """Change a design's description, config or version. Name and kind stay.

        A new config is validated through the same class the save was, and
        `updated_at` moves. A body that names none of the mutable fields is a
        400: an empty change that returned 200 would look like one.
        """
        changes = {key: body[key] for key in MUTABLE if key in body}
        if not changes:
            raise HTTPException(
                status_code=400,
                detail=(f"nothing to change. One or more of "
                        f"{', '.join(MUTABLE)}; name and topology_type are "
                        f"fixed, because they are the address and the class "
                        f"the config was validated against."))
        async with sessions() as session:
            row = await _find(session, ref)
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=(f"no design {ref!r}, by id or by name. "
                            f"`GET /v1/topologies` lists them."))
            kind = TopologyType(row.topology_type)
            if "config" in changes:
                try:
                    changes["config"] = validated_config(kind, changes["config"])
                except ValidationError as error:
                    raise HTTPException(status_code=422,
                                        detail=_validation_detail("config", error))
            try:
                Topology.model_validate({
                    "name": row.name, "topology_type": kind,
                    "description": changes.get("description", row.description),
                    "version": changes.get("version", row.version),
                    "config": changes.get("config", row.config),
                })
            except ValidationError as error:
                raise HTTPException(status_code=422,
                                    detail=_validation_detail("design", error))
            for key, value in changes.items():
                setattr(row, key, value)
            row.updated_at = utc_now()
            session.add(row)
            await session.flush()
            return {"schema": 1, "changed": sorted(changes),
                    **row_payload(row, project=project)}
