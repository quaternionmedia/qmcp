"""qmcp's dashboard: what it shows, and what it refuses to pretend.

Every test builds its own database. The dashboard reads a file directly rather
than the HTTP API, so nothing here needs a server -- which is the same property
that lets the real one explain why the server is down.

THE TESTS WORTH READING ARE THE DEGRADED ONES. A dashboard is easy to write so
that it works on healthy data and raises on everything else, and the states it
has to survive here are real: this database has been missing a table the code
expected, and has been empty. An empty dashboard and a broken one look identical
unless something distinguishes them.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qmcp.dashboard import DEFAULT_PROJECT, build, render, to_dict


def make_db(path: Path, rows: list[tuple] | None = None,
            with_tables: bool = True) -> Path:
    connection = sqlite3.connect(str(path))
    if with_tables:
        connection.execute(
            "CREATE TABLE tool_invocations (id TEXT PRIMARY KEY, tool_name TEXT, "
            "status TEXT, duration_ms INTEGER, created_at TEXT, error TEXT)"
        )
        connection.execute("CREATE TABLE human_requests (id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE human_responses (id TEXT PRIMARY KEY)")
        for row in rows or []:
            connection.execute(
                "INSERT INTO tool_invocations "
                "(id, tool_name, status, duration_ms, created_at, error) "
                "VALUES (?,?,?,?,?,?)", row)
    else:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    return path


SAMPLE = [
    ("id-1", "planner", "SUCCESS", 5, "2026-08-18T10:00:00", None),
    ("id-2", "executor", "SUCCESS", 7, "2026-08-18T10:00:01", None),
    ("id-3", "reviewer", "ERROR", 2, "2026-08-18T10:00:02", "it broke"),
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
    rows = [("id-1", "planner", "PENDING", None, "2026-08-18T10:00:00", None)]
    assert build(make_db(tmp_path / "a.db", rows)).failures == 0


def test_the_most_recent_come_first(tmp_path):
    view = build(make_db(tmp_path / "a.db", SAMPLE))
    assert [row.tool_name for row in view.recent] == ["reviewer", "executor", "planner"]


def test_the_recent_list_is_bounded(tmp_path):
    rows = [(f"id-{i}", "echo", "SUCCESS", 1, f"2026-08-18T10:00:{i:02d}", None)
            for i in range(30)]
    view = build(make_db(tmp_path / "a.db", rows), recent=5)
    assert len(view.recent) == 5
    assert view.total == 30, "the bound is on what is listed, not what is counted"


# --- the degraded states this database has actually been in ------------------


def test_a_database_with_no_invocations_table_still_renders(tmp_path):
    """This database has been missing a table the code expected. A dashboard
    that raised then would be unavailable exactly when it explains why."""
    view = build(make_db(tmp_path / "a.db", with_tables=False))
    assert view.total == 0
    assert view.tables == 1
    assert "Nothing has been invoked" in render(view)


def test_an_empty_dashboard_is_distinguishable_from_a_broken_one(tmp_path):
    """The table count is what tells them apart, and it is on the page."""
    text = render(build(make_db(tmp_path / "a.db", [])))
    assert "tables        3" in text
    assert "Nothing has been invoked" in text


def test_missing_human_loop_tables_count_as_zero_rather_than_raising(tmp_path):
    view = build(make_db(tmp_path / "a.db", with_tables=False))
    assert (view.human_requests, view.human_responses) == (0, 0)


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
    assert to_dict(build(make_db(tmp_path / "a.db", SAMPLE)))["schema"] == 1
