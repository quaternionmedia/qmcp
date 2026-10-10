"""qmcp's own view of what it has run, addressed so another system can join it.

    qmcp dashboard              # the view, rendered for a terminal
    qmcp dashboard --json       # the same data, for something else to render

Every row carries an address, `<owner>/<repo>/invocation/<id>`, so dossier can
point at it, a delta can link to it and `qm divergence` can compare it.

It reads the database directly and read-only, not the HTTP API, so it answers
when the server is down.

The human queue crosses as rows: each question is addressed
`<owner>/<repo>/ask/<id>` and carries its prompt, its options and its answer if
it has one.

A count that could not be taken, because its table is absent, is
`{"unknown": "<reason>"}` in the payload, never `0`.

What this cannot see:

  * anything the database does not record: a tool that failed before its
    invocation row was written leaves nothing, so the count of what ran is a
    floor;
  * whether a result was any good: `status` is what the server recorded;
  * dossier's half of any row. This addresses its own data so the two can be
    joined, and does not join them or judge between them
    (`records/DRAFT-a-disagreement-is-a-delta.md`).

`docs/self-checks.md` describes the dashboard beside the harness's other views
of itself.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from qmcp import identity
from qmcp.addresses import ask_address, invocation_address

# Derived from the git remote by `qmcp.identity`, so a fork's dashboard reports
# the fork's project.
DEFAULT_PROJECT = identity.this_project()
RECENT = 10
"""How many invocations the payload carries. A window onto a log."""

QUEUE = 50
"""How many queue rows the payload carries.

Separate from `RECENT`: truncating a log window loses history, truncating a
work list loses work. The payload states `queue_total` beside the rows.
"""

SCHEMA = 2
"""The payload version a reader checks.

Additive keys such as `queue_total` and `queue_shown` do not move it:
`dossier.harness.check_schema` requires an exact match, and a reader at this
version is correct without them.
"""

# The tables this reads. Named so the payload can say which one was missing
# rather than reporting a number nobody took.
INVOCATIONS = "tool_invocations"
HUMAN_REQUESTS = "human_requests"
HUMAN_RESPONSES = "human_responses"


def unknown(table: str) -> dict:
    """A count that could not be taken, and why."""
    return {"unknown": f"no {table} table in this database"}


@dataclass(frozen=True)
class Invocation:
    """One recorded tool call, with the name other systems can use for it."""

    id: str
    address: str
    tool_name: str
    status: str
    duration_ms: int | None
    created_at: str
    error: str | None = None


@dataclass(frozen=True)
class Waiting:
    """One question put to a person, and the answer if it has arrived.

    `answered_with` is None while it is outstanding. Answered rows are kept:
    what was asked and what was said are the audit trail.
    """

    id: str
    address: str
    request_type: str
    prompt: str
    options: list[str]
    status: str
    created_at: str
    answered_with: str | None = None
    answered_by: str | None = None
    answered_at: str | None = None

    @property
    def outstanding(self) -> bool:
        return self.answered_with is None


@dataclass(frozen=True)
class View:
    """Everything the dashboard shows, separated from how it is shown.

    Renderers read a `View` and query nothing, so the query lives in one
    place.
    """

    project: str
    database: Path
    total: int | None = None
    by_tool: dict[str, int] = field(default_factory=dict)
    by_status: dict[str, int] = field(default_factory=dict)
    recent: list[Invocation] = field(default_factory=list)
    human_requests: int | None = None
    human_responses: int | None = None
    waiting: list[Waiting] = field(default_factory=list)
    queue_total: int = 0
    """How many rows the queue holds, before any cap. Never derived from
    `len(waiting)`, which is the number that survived the cap."""

    tables: int = 0

    # Every table this view wanted and did not find. Empty is the healthy case.
    missing: tuple[str, ...] = ()

    @property
    def failures(self) -> int | None:
        """How many invocations were not successful, or None if none were counted.

        None rather than 0: with no invocations table nothing was classified.
        """
        if self.total is None:
            return None
        return sum(count for status, count in self.by_status.items()
                   if status.lower() not in ("success", "pending"))


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _count(connection: sqlite3.Connection, table: str) -> int | None:
    """Row count, or None when the table is not there.

    Does not raise, so the dashboard stays available when a table the code
    expects is missing. None is not zero: the caller turns it into
    `{"unknown": ...}` on the way out.
    """
    if not _table_exists(connection, table):
        return None
    return connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]


def _waiting(connection: sqlite3.Connection, project: str,
             limit: int) -> tuple[list[Waiting], int]:
    """The human queue, outstanding first, each row addressed, and its size.

    The size is returned beside the capped rows, not inferred from them.
    Outstanding rows come first, the order a person acts in; answered rows
    are kept as the audit trail.

    One left join rather than two queries, so a response arriving between
    queries cannot leave its request reading as outstanding.
    """
    if not _table_exists(connection, HUMAN_REQUESTS):
        # No queue table: no rows and a total of zero, distinct from a capped
        # queue.
        return [], 0
    responses = _table_exists(connection, HUMAN_RESPONSES)
    rows = connection.execute(
        f"""SELECT r.id, r.request_type, r.prompt, r.options, r.status,
                   r.created_at,
                   {'p.response, p.responded_by, p.created_at' if responses
                    else 'NULL, NULL, NULL'} AS answered
            FROM "{HUMAN_REQUESTS}" r
            {f'LEFT JOIN "{HUMAN_RESPONSES}" p ON p.request_id = r.id'
             if responses else ''}
            ORDER BY r.created_at"""
    ).fetchall()

    queue = []
    for row in rows:
        raw = row["options"]
        try:
            options = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except json.JSONDecodeError:
            options = []
        answered = row[6] if responses else None
        queue.append(Waiting(
            id=str(row["id"]),
            address=ask_address(str(row["id"]), project),
            request_type=str(row["request_type"]),
            prompt=str(row["prompt"]),
            options=list(options or []),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            answered_with=None if answered is None else str(answered),
            answered_by=None if not responses or row[7] is None else str(row[7]),
            answered_at=None if not responses or row[8] is None else str(row[8]),
        ))
    queue.sort(key=lambda item: (not item.outstanding, item.created_at))
    return queue[:limit], len(queue)


def build(database: Path, project: str = DEFAULT_PROJECT,
          recent: int = RECENT) -> View:
    """Read the database and return what the dashboard shows."""
    if not database.is_file():
        raise FileNotFoundError(f"{database}: no database to read.")

    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = len([
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ])
        missing = tuple(
            name for name in (INVOCATIONS, HUMAN_REQUESTS, HUMAN_RESPONSES)
            if not _table_exists(connection, name)
        )
        if INVOCATIONS in missing:
            queue, queue_total = _waiting(connection, project, QUEUE)
            return View(project=project, database=database, tables=tables,
                        human_requests=_count(connection, HUMAN_REQUESTS),
                        human_responses=_count(connection, HUMAN_RESPONSES),
                        waiting=queue, queue_total=queue_total,
                        missing=missing)

        rows = list(connection.execute(
            "SELECT id, tool_name, status, duration_ms, created_at, error "
            "FROM tool_invocations ORDER BY created_at DESC"
        ))
        queue, queue_total = _waiting(connection, project, QUEUE)
        return View(
            project=project,
            database=database,
            total=len(rows),
            by_tool=dict(Counter(r["tool_name"] for r in rows).most_common()),
            by_status=dict(Counter(str(r["status"]) for r in rows).most_common()),
            recent=[
                Invocation(
                    id=str(r["id"]),
                    address=invocation_address(r["id"], project),
                    tool_name=r["tool_name"],
                    status=str(r["status"]),
                    duration_ms=r["duration_ms"],
                    created_at=str(r["created_at"]),
                    error=r["error"],
                )
                for r in rows[:recent]
            ],
            human_requests=_count(connection, HUMAN_REQUESTS),
            human_responses=_count(connection, HUMAN_RESPONSES),
            waiting=queue,
            queue_total=queue_total,
            tables=tables,
            missing=missing,
        )
    finally:
        connection.close()


def to_dict(view: View) -> dict:
    """The view as data, for a renderer that is not this one.

    A count of None becomes `{"unknown": "<reason>"}`, which a consumer must
    not read as zero.
    """
    def count(value: int | None, table: str):
        return unknown(table) if value is None else value

    return {
        "schema": SCHEMA,
        "project": view.project,
        "database": str(view.database),
        "totals": {
            "invocations": count(view.total, INVOCATIONS),
            "failures": count(view.failures, INVOCATIONS),
            "human_requests": count(view.human_requests, HUMAN_REQUESTS),
            "human_responses": count(view.human_responses, HUMAN_RESPONSES),
            "tables": view.tables,
        },
        "missing_tables": list(view.missing),
        "waiting": [
            {
                "address": item.address,
                "id": item.id,
                "request_type": item.request_type,
                "prompt": item.prompt,
                "options": list(item.options),
                "status": item.status,
                "created_at": item.created_at,
                "answered_with": item.answered_with,
                "answered_by": item.answered_by,
                "answered_at": item.answered_at,
            }
            for item in view.waiting
        ],
        # Both stated, so a reader comparing them learns that rows were
        # dropped.
        "queue_shown": len(view.waiting),
        "queue_total": view.queue_total,
        "by_tool": view.by_tool,
        "by_status": view.by_status,
        "recent": [
            {
                "address": row.address,
                "tool_name": row.tool_name,
                "status": row.status,
                "duration_ms": row.duration_ms,
                "created_at": row.created_at,
                "error": row.error,
            }
            for row in view.recent
        ],
    }


def _topologies() -> list[str]:
    """The topologies this harness can serve.

    Read from `qmcp.topology_view`, not over HTTP, so the dashboard answers
    when the server is down.

    It reports what this build can serve, not what is reachable now;
    `uv run qm dashboard` knows whether anything is listening.
    """
    try:
        from qmcp import topology_view

        return [view.topology for view in topology_view.gallery()]
    except Exception:                              # noqa: BLE001
        return []


def render(view: View) -> str:
    """The terminal view. Reads the `View` and queries nothing."""
    def shown(value: int | None) -> str:
        return "unknown" if value is None else str(value)

    out = [
        f"qmcp  {view.project}",
        f"{view.database}",
        "",
        f"  invocations   {shown(view.total)}"
        + (f"   ({view.failures} not successful)" if view.failures else ""),
        f"  human loop    {shown(view.human_requests)} request(s), "
        f"{shown(view.human_responses)} response(s)"
        + (f"   -- {len([w for w in view.waiting if w.outstanding])} outstanding"
           if view.waiting else ""),
        f"  tables        {view.tables}",
        f"  topologies    {len(_topologies())} shape(s) served at "
        f"/v1/topology",
        "",
    ]

    outstanding = [item for item in view.waiting if item.outstanding]
    if outstanding:
        out.append("  waiting on a person")
        for item in outstanding:
            out.append(f"    {item.address}")
            out.append(f"      {item.prompt}")
            if item.options:
                out.append(f"      options: {', '.join(item.options)}")
        out.append("")

    if view.missing:
        out += [
            "  This database is missing " + ", ".join(view.missing) + ".",
            "  The counts above are unknown rather than zero: nobody took them.",
            "  A table count does not tell this apart from an idle harness -- a",
            "  database of unrelated tables counts like any other.",
            "",
        ]
    elif not view.total:
        out += [
            "  Nothing has been invoked against this database.",
            "  Every table this reads is present, so this is an idle harness",
            "  rather than an unreadable one.",
            "",
        ]
    else:
        out.append("  by tool")
        for name, count in view.by_tool.items():
            out.append(f"    {name:<16} {count}")
        out.append("")
        out.append("  by status")
        for name, count in view.by_status.items():
            out.append(f"    {name:<16} {count}")
        out.append("")
        out.append(f"  most recent {len(view.recent)}")
        for row in view.recent:
            out.append(f"    {row.address}")
            out.append(f"      {row.tool_name:<12} {row.status:<10} "
                       f"{row.duration_ms if row.duration_ms is not None else '-'}ms   "
                       f"{row.created_at[:19]}")
            if row.error:
                out.append(f"      error: {row.error[:70]}")
        out.append("")

    out += [
        "Every row carries an address, so another system can name the same row.",
        "This view holds no opinion about dossier's half of it: a disagreement "
        "between",
        "two views is a delta to resolve, not a winner to pick.",
    ]
    return "\n".join(out)
