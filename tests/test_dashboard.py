"""qmcp's dashboard: what it shows, and what it refuses to pretend.

Every test builds its own database FROM THIS PROJECT'S OWN MODEL METADATA, and
inserts through the models rather than through hand-written SQL. That is the
difference between testing the dashboard and testing a second, private idea of
what the schema is. The hand-written version of this file agreed with the models
by coincidence and used two status strings the enum does not contain -- `ERROR`
and, on the consumer's side, `FAILURE` -- while both suites stayed green.

The dashboard reads a file directly rather than the HTTP API, so nothing here
needs a server, which is the same property that lets the real one explain why
the server is down.

THE TESTS WORTH READING ARE THE DEGRADED ONES. A dashboard is easy to write so
that it works on healthy data and raises on everything else, and the states it
has to survive here are real: this database has been missing a table the code
expected, and has been empty.

AN EMPTY DASHBOARD AND A BROKEN ONE ARE NOT TOLD APART BY THE TABLE COUNT. This
file used to say they were. A database holding two unrelated tables reports a
table count like any other, so the count distinguishes nothing a consumer can
act on. What tells them apart is that a count nobody could take is `unknown`
with a reason, and never zero.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

from qmcp import dashboard
from qmcp.dashboard import DEFAULT_PROJECT, SCHEMA, build, render, to_dict
from qmcp.db.models import InvocationStatus, ToolInvocation

ROOT = Path(__file__).resolve().parent.parent
VECTORS = ROOT / "governance" / "qm" / "project-seed" / "harness-payload-vectors.json"


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def make_db(path: Path, rows: list[tuple] | None = None,
            with_tables: bool = True) -> Path:
    """A database built the way the server builds one.

    `SQLModel.metadata.create_all` rather than `CREATE TABLE`: a model that
    renames a column now breaks the dashboard's query here, which is the whole
    point. The previous version wrote the DDL by hand and would have kept
    passing against a schema the server no longer produces.
    """
    if with_tables:
        engine = create_engine(f"sqlite:///{path.as_posix()}")
        SQLModel.metadata.create_all(engine)
        with Session(engine) as session:
            for row in rows or []:
                identifier, tool, status, duration, created, error = row
                session.add(ToolInvocation(
                    id=identifier, tool_name=tool, status=status,
                    duration_ms=duration, created_at=at(created), error=error))
            session.commit()
    else:
        connection = sqlite3.connect(str(path))
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
        connection.commit()
        connection.close()
    return path


# The statuses are enum members, not strings. `ERROR` stood here and is not a
# value InvocationStatus has ever held.
SAMPLE = [
    ("id-1", "planner", InvocationStatus.SUCCESS, 5, "2026-08-18T10:00:00", None),
    ("id-2", "executor", InvocationStatus.SUCCESS, 7, "2026-08-18T10:00:01", None),
    ("id-3", "reviewer", InvocationStatus.FAILED, 2, "2026-08-18T10:00:02", "it broke"),
]


# --- every row is addressable ------------------------------------------------


def test_every_row_carries_an_address(tmp_path):
    """A bare UUID names nothing outside this database. The address is the
    whole reason this is more than a print of a table."""
    view = build(make_db(tmp_path / "a.db", SAMPLE))
    assert [row.address for row in view.recent] == [
        f"{DEFAULT_PROJECT}/invocation/id-3",
        f"{DEFAULT_PROJECT}/invocation/id-2",
        f"{DEFAULT_PROJECT}/invocation/id-1",
    ]


def test_the_project_is_configurable_and_reaches_the_address(tmp_path):
    view = build(make_db(tmp_path / "a.db", SAMPLE), project="acme/thing")
    assert view.recent[0].address.startswith("acme/thing/invocation/")


def test_the_addresses_parse_with_the_grammar(tmp_path):
    from qmcp.addresses import parse

    view = build(make_db(tmp_path / "a.db", SAMPLE))
    assert all(parse(row.address) is not None for row in view.recent)
    assert all(parse(row.address).kind == "invocation" for row in view.recent)


# --- the counts --------------------------------------------------------------


def test_the_totals_are_counted(tmp_path):
    view = build(make_db(tmp_path / "a.db", SAMPLE))
    assert view.total == 3
    assert view.by_tool == {"planner": 1, "executor": 1, "reviewer": 1}


def test_failures_are_counted_separately_from_successes(tmp_path):
    """A dashboard whose headline number hides failures is worse than none."""
    view = build(make_db(tmp_path / "a.db", SAMPLE))
    assert view.failures == 1


def test_pending_is_not_counted_as_a_failure(tmp_path):
    rows = [("id-1", "planner", InvocationStatus.PENDING, None,
             "2026-08-18T10:00:00", None)]
    assert build(make_db(tmp_path / "a.db", rows)).failures == 0


def test_the_most_recent_come_first(tmp_path):
    view = build(make_db(tmp_path / "a.db", SAMPLE))
    assert [row.tool_name for row in view.recent] == ["reviewer", "executor", "planner"]


def test_the_recent_list_is_bounded(tmp_path):
    rows = [(f"id-{i}", "echo", InvocationStatus.SUCCESS, 1,
             f"2026-08-18T10:00:{i % 60:02d}", None) for i in range(30)]
    view = build(make_db(tmp_path / "a.db", rows), recent=5)
    assert len(view.recent) == 5
    assert view.total == 30, "the bound is on what is listed, not what is counted"


# --- the degraded states this database has actually been in ------------------


def test_a_database_with_no_invocations_table_still_renders(tmp_path):
    """This database has been missing a table the code expected. A dashboard
    that raised then would be unavailable exactly when it explains why."""
    view = build(make_db(tmp_path / "a.db", with_tables=False))
    assert view.total is None, "nobody counted; zero would say somebody did"
    assert view.tables == 1
    assert "unknown rather than zero" in render(view)


def test_an_empty_dashboard_is_distinguishable_from_a_broken_one(tmp_path):
    """Not by the table count -- by whether the counts were taken at all.

    This test asserted the table count, and the count does not carry the
    distinction: a database of unrelated tables reports one like any other. The
    honest signal is that an idle harness has every table and a real zero, and
    an unreadable one has an `unknown` with a reason.
    """
    idle = build(make_db(tmp_path / "idle.db", []))
    broken = build(make_db(tmp_path / "broken.db", with_tables=False))

    assert idle.total == 0 and idle.missing == ()
    assert broken.total is None and "tool_invocations" in broken.missing

    assert "Nothing has been invoked" in render(idle)
    assert "unknown rather than zero" in render(broken)


def test_a_count_nobody_took_is_unknown_rather_than_zero(tmp_path):
    """The name of this test used to be `..._count_as_zero_rather_than_raising`.

    Not raising was right and was never the question. Returning zero was the
    error: it made a database missing its tables report a clean bill of health,
    and the consumer stored that zero as a fact about the harness.
    """
    view = build(make_db(tmp_path / "a.db", with_tables=False))
    assert (view.human_requests, view.human_responses) == (None, None)

    totals = to_dict(view)["totals"]
    for name in ("invocations", "failures", "human_requests", "human_responses"):
        assert isinstance(totals[name], dict), f"{name} was reported as a number"
        assert "unknown" in totals[name]
        assert "table" in totals[name]["unknown"], "an unknown must say why"


def test_a_failure_count_is_unknown_when_nothing_could_be_classified(tmp_path):
    """No invocations table means no clean bill of health either.

    Mutation: return 0 from `failures` when `total` is None and this passes --
    which is a dashboard reporting no failures without having looked.
    """
    assert build(make_db(tmp_path / "a.db", with_tables=False)).failures is None


def test_a_database_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(tmp_path / "absent.db")


# --- the rendering is separate from the reading ------------------------------


def test_the_renderer_runs_no_query(tmp_path):
    """A renderer that queried would be a second place the query lives, and the
    two drift the first time one is fixed. Asserted on the source."""
    import qmcp.dashboard as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    body = source.split("def render(")[1]
    assert "execute" not in body
    assert "connect" not in body


def test_the_view_states_that_it_holds_no_opinion(tmp_path):
    """Two views disagreeing is a delta, not a winner to pick. The page says so
    rather than leaving a reader to assume this one is authoritative."""
    text = render(build(make_db(tmp_path / "a.db", SAMPLE)))
    assert "no opinion" in text
    assert "delta" in text


def test_an_error_reaches_the_page(tmp_path):
    assert "it broke" in render(build(make_db(tmp_path / "a.db", SAMPLE)))


# --- the data form -----------------------------------------------------------


def test_the_json_form_carries_the_same_addresses(tmp_path):
    payload = to_dict(build(make_db(tmp_path / "a.db", SAMPLE)))
    assert payload["totals"]["invocations"] == 3
    assert payload["totals"]["failures"] == 1
    assert all("/invocation/" in row["address"] for row in payload["recent"])


def test_the_json_form_declares_its_schema(tmp_path):
    payload = to_dict(build(make_db(tmp_path / "a.db", SAMPLE)))
    assert payload["schema"] == SCHEMA == 2, (
        "schema 2 is where a count nobody took became `unknown` rather than 0"
    )


def test_a_healthy_payload_carries_plain_numbers(tmp_path):
    """The unknown convention must not leak into the ordinary case."""
    totals = to_dict(build(make_db(tmp_path / "a.db", SAMPLE)))["totals"]
    assert all(isinstance(value, int) for value in totals.values())


# --- the shared contract, which the consumer is held to as well --------------


def test_the_emitter_conforms_to_the_shared_payload_vectors():
    """Every case the control panel is also held to.

    TWO IMPLEMENTATIONS, ONE SET OF CASES -- the same trade this project makes
    for the address grammar. The vectors were not hand-written: each payload in
    them came out of this emitter reading a database built from these models,
    so a case that stops being producible stops being in the file.

    This is what replaced a hand-written fixture on each side of the seam that
    agreed with each other and with nothing else.
    """
    if not VECTORS.is_file():
        pytest.fail(
            f"{VECTORS.relative_to(ROOT).as_posix()} is absent. The payload "
            f"contract ships through the governance submodule; without it this "
            f"emitter is verified against its own tests only, which is the "
            f"weaker claim that made the seam wrong."
        )
    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert document["cases"], "an empty vector file verifies nothing"

    assert document["schema"] == SCHEMA, (
        f"the vectors are schema {document['schema']}, this emitter is {SCHEMA}"
    )

    vocabulary = document["status_vocabulary"]["values"]
    assert vocabulary == [status.name for status in InvocationStatus], (
        "the contract's status vocabulary is not this enum's. Both sides read "
        "the stored form, which is the enum NAME rather than its value."
    )

    for case in document["cases"]:
        payload = case["payload"]
        assert payload["schema"] == SCHEMA, case["name"]
        assert set(payload) >= {"schema", "project", "totals", "recent"}, case["name"]
        for row in payload["recent"]:
            assert row["status"] in vocabulary, (
                f"{case['name']}: status {row['status']!r} is outside the "
                f"vocabulary this emitter can produce"
            )


# --- the cap, which used to be silent ------------------------------------------


def _queue_of(tmp_path, count: int):
    """A database with `count` outstanding questions in it."""
    import sqlite3

    database = tmp_path / "queue.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE human_requests (id TEXT PRIMARY KEY, request_type TEXT, "
        "prompt TEXT, options TEXT, status TEXT, created_at TEXT)")
    connection.executemany(
        "INSERT INTO human_requests VALUES (?, 'approval', ?, '[]', 'pending', ?)",
        [(f"ask-{n:03d}", f"question {n}", f"2026-08-25T00:{n:02d}:00")
         for n in range(count)])
    connection.commit()
    connection.close()
    return database


def test_a_truncated_queue_says_how_many_it_holds(tmp_path):
    """THE ONE THAT MATTERS.

    A queue of fifteen arrived as ten and nothing said so, so a person acting
    on `dossier`'s Outstanding list was acting on a work list that had quietly
    dropped the five most recently asked. Found by queueing two governed runs
    against a live harness and finding neither in the payload.

    Mutation, quoted as it printed: `"queue_total": view.queue_total` replaced
    with `len(view.waiting)`.

        AssertionError: a truncated queue reported its own length as the total
        assert 50 == (50 + 10)
         +  where 50 = dashboard.QUEUE
    """
    view = dashboard.build(_queue_of(tmp_path, dashboard.QUEUE + 10))
    payload = dashboard.to_dict(view)

    assert payload["queue_shown"] == dashboard.QUEUE
    assert payload["queue_total"] == dashboard.QUEUE + 10, (
        "a truncated queue reported its own length as the total")
    assert payload["queue_shown"] < payload["queue_total"]


def test_an_untruncated_queue_reports_the_same_number_twice(tmp_path):
    """Equal is the healthy case, and it must be sayable rather than implied."""
    payload = dashboard.to_dict(dashboard.build(_queue_of(tmp_path, 3)))

    assert payload["queue_shown"] == payload["queue_total"] == 3


def test_an_absent_queue_is_zero_of_zero_rather_than_a_dropped_list(tmp_path):
    """No table and a full table showing part of itself are different answers."""
    import sqlite3

    database = tmp_path / "bare.db"
    sqlite3.connect(database).close()
    payload = dashboard.to_dict(dashboard.build(database))

    assert payload["queue_shown"] == 0
    assert payload["queue_total"] == 0


def test_the_queue_cap_is_not_the_invocation_window():
    """A work list and a log window are different things.

    Mutation: setting `QUEUE = RECENT` fails here, which is the assertion that
    stops the two silently becoming one constant again.
    """
    assert dashboard.QUEUE != dashboard.RECENT
    assert dashboard.QUEUE > dashboard.RECENT
