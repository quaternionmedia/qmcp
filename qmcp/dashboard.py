"""qmcp's own view of what it has run, addressed so another system can join it.

    qmcp dashboard              # the view, rendered for a terminal
    qmcp dashboard --json       # the same data, for something else to render

WHY qmcp NEEDS ONE AT ALL. dossier has a dashboard; qmcp had an API and a log.
Two views of one dataset cannot be put beside each other when only one of them
is a view, and "read the JSON from `/v1/invocations`" is not a view -- it is the
thing a view is made from.

EVERY ROW CARRIES AN ADDRESS. An invocation is a bare UUID, which nothing
outside this database can name. As
`quaternionmedia/qmcp/invocation/<id>` it is a row dossier can point at, a delta
can link to, and `qm divergence` can compare. That is the whole reason this is
more than a pretty print of a table.

WHAT IT READS. The database, directly and read-only -- not the HTTP API. A
dashboard that required the server to be up could not tell you why the server is
down, which is when somebody wants it most.

WHAT IT CANNOT SEE.

  * Anything the database does not record. A tool that failed before its
    invocation row was written leaves nothing here, and the count of what ran
    is therefore a floor.
  * Whether a result was any good. `status` is what the server recorded, and a
    successful call returning nonsense is a success here.
  * dossier's half of any row. This addresses its own data so the two *can* be
    joined; it does not do the joining, and it holds no opinion about which
    view is right -- that is `records/DRAFT-a-disagreement-is-a-delta.md`.

THE HUMAN QUEUE CROSSES AS ROWS, NOT AS A COUNT. Schema 2 carried
`human_requests: 1` and nothing else, so a control panel could say that one
thing was waiting on a person and never which thing -- and a count is not
something anybody can answer. Each question now crosses addressed, as
`<owner>/<repo>/ask/<id>`, with its prompt, its options, and its answer if it
has one.

A COUNT NOBODY COULD TAKE IS `unknown`, NEVER ZERO. When a table this reads is
absent, the payload carries `{"unknown": "<reason>"}` in place of the number.
Schema 1 emitted `0`, which made a database missing its tables indistinguishable
from a harness that had run nothing -- and the consumer stored the zero as fact.
The docstring on the terminal renderer said the table counts told them apart;
they do not, because a database holding two unrelated tables reports a count
like any other. The convention is the corpus's own, stated in
`harness-status.json`'s reading block: unknown is a value, it says why, and it
is not zero, not empty and not compliant.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from qmcp.addresses import ask_address, invocation_address

DEFAULT_PROJECT = "quaternionmedia/qmcp"
RECENT = 10
SCHEMA = 2

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

    `answered_with` is None while it is outstanding. A row that carries an
    answer is kept rather than dropped: what was asked and what was said are
    the audit trail, and a queue that forgets answered questions cannot show
    anybody why a thing was decided.
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

    A renderer that queried would be a second place the query lives, and the
    two would drift the first time one was fixed.
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
    tables: int = 0

    # Every table this view wanted and did not find. Empty is the healthy case.
    missing: tuple[str, ...] = ()

    @property
    def failures(self) -> int | None:
        """How many invocations were not successful, or None if none were counted.

        None rather than 0: with no invocations table there is nothing to
        classify, and reporting no failures would be a clean bill of health
        issued without an examination.
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

    Not raising is right and was never the question: this database has been in
    states where a table the code expects does not exist, and a dashboard that
    raised then would be unavailable exactly when it was needed to explain why.

    Returning zero was the error. Zero is an answer, and the honest report is
    that nobody could take the count -- which the caller turns into
    `{"unknown": ...}` on the way out.
    """
    if not _table_exists(connection, table):
        return None
    return connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]


def _waiting(connection: sqlite3.Connection, project: str,
             limit: int) -> list[Waiting]:
    """The human queue, outstanding first, each row addressed.

    Outstanding first because that is the order a person acts in. Answered rows
    are kept because what was asked and what was said is the audit trail.

    A left join rather than two queries: a request whose response arrived
    between them would otherwise read as outstanding, which is the wrong
    answer in the direction that matters.
    """
    if not _table_exists(connection, HUMAN_REQUESTS):
        return []
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
    return queue[:limit]


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
            return View(project=project, database=database, tables=tables,
                        human_requests=_count(connection, HUMAN_REQUESTS),
                        human_responses=_count(connection, HUMAN_RESPONSES),
                        waiting=_waiting(connection, project, recent),
                        missing=missing)

        rows = list(connection.execute(
            "SELECT id, tool_name, status, duration_ms, created_at, error "
            "FROM tool_invocations ORDER BY created_at DESC"
        ))
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
            waiting=_waiting(connection, project, recent),
            tables=tables,
            missing=missing,
        )
    finally:
        connection.close()


def to_dict(view: View) -> dict:
    """The view as data, for a renderer that is not this one.

    A count of None becomes `{"unknown": "<reason>"}`. A consumer that treats
    that as zero is making a claim this side declined to make.
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
