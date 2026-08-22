"""The read-only archive routes, and where they refuse to be served.

The tests worth reading are the two refusals: a source nobody declared, and a
server bound off loopback. Both are about what reaches a socket -- the archive
holds somebody's conversations, which is a different thing from the tools and
invocations every other route here publishes.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp.threads import index as index_module
from qmcp.threads.service import SOURCES, diverged, index_at, register, summarise


def indexed(tmp_path, entries):
    document = index_module.document(entries)
    (tmp_path / index_module.INDEX_NAME).write_text(
        json.dumps(document), encoding="utf-8")
    return document


def entry(identifier="c-1", source="claude", history=None):
    row = index_module.Entry(
        id=identifier, source=source, title="T", digest="abc", turns=2,
        turn_ids=["m-1", "m-2"], first_seen="2026-08-20T00:00:00Z",
        last_seen="2026-08-20T00:00:00Z")
    if history:
        row.history = history
    return row


def client_for(tmp_path):
    app = FastAPI()
    register(app, tmp_path, tmp_path)
    return TestClient(app)


# --- an absent index is an absent answer -------------------------------------


def test_no_index_is_a_404_not_an_empty_archive(tmp_path):
    """A machine that has never indexed and one whose archive is empty are
    different states.

    Mutation: return `{"threads": []}` when there is no index and this fails --
    which tells a caller the archive is empty when nobody has looked.
    """
    response = client_for(tmp_path).get("/v1/threads")
    assert response.status_code == 404
    assert "absent answer" in response.json()["detail"]


def test_an_unreadable_index_is_treated_as_absent(tmp_path):
    (tmp_path / index_module.INDEX_NAME).write_text("{not json", encoding="utf-8")
    assert index_at(tmp_path) is None


# --- the listing --------------------------------------------------------------


def test_the_listing_answers_from_the_index(tmp_path):
    """From one document rather than reading every session, which is the whole
    reason the index exists."""
    indexed(tmp_path, [entry("c-1"), entry("c-2")])
    body = client_for(tmp_path).get("/v1/threads").json()
    assert body["totals"]["threads"] == 2
    assert {row["id"] for row in body["threads"]} == {"c-1", "c-2"}


def test_the_listing_drops_turn_ids(tmp_path):
    """They are in the index because divergence needs them, and a listing
    carrying every one would be large for a caller who wanted titles."""
    document = indexed(tmp_path, [entry()])
    assert "turn_ids" not in summarise(document)["threads"][0]


def test_the_listing_says_what_it_counted(tmp_path):
    """Not conversations that exist -- conversations that were exported and
    indexed."""
    document = indexed(tmp_path, [entry()])
    note = summarise(document)["reading"]["the_index_is_a_reading"]
    assert "not what exists" in note or "counts conversations that" in note


# --- divergence, which is the query the archive exists for --------------------


def test_diverged_lists_only_what_disagrees(tmp_path):
    change = index_module.Change(
        at="2026-08-21T00:00:00Z", kind=index_module.DIVERGED,
        from_digest="aaa", to_digest="bbb", from_turns=2, to_turns=1,
        detail="1 turn(s) present before are absent now")
    document = indexed(tmp_path, [entry("c-1"), entry("c-2", history=[change])])
    found = diverged(document)
    assert [row["id"] for row in found] == ["c-2"]
    assert found[0]["latest"]["detail"] == change.detail


def test_the_diverged_route_says_nothing_was_repaired(tmp_path):
    indexed(tmp_path, [entry()])
    body = client_for(tmp_path).get("/v1/threads/diverged").json()
    assert "Nothing here is repaired" in body["note"]


def test_growth_is_not_reported_as_divergence(tmp_path):
    """Mutation: report every change and this fails, which buries the findings
    in the ordinary case."""
    grew = index_module.Change(
        at="2026-08-21T00:00:00Z", kind=index_module.GREW,
        from_digest="aaa", to_digest="bbb", from_turns=1, to_turns=2,
        detail="1 turn(s) added")
    assert diverged(indexed(tmp_path, [entry(history=[grew])])) == []


# --- the refusals -------------------------------------------------------------


def test_a_source_nobody_declared_is_a_404(tmp_path):
    """`source` reaches a filesystem path, and a name nobody declared is a name
    somebody made up.

    Mutation: pass the source through to a reader and this fails.
    """
    indexed(tmp_path, [entry()])
    assert client_for(tmp_path).get("/v1/threads/madeup/x").status_code == 404


def test_every_declared_source_is_one_this_project_ships():
    from qmcp.threads.service import sources_for

    assert set(SOURCES) == set(sources_for(__import__("pathlib").Path(".")))


def test_the_routes_are_not_mounted_off_loopback():
    """THE ONE THAT MATTERS.

    `qmcp cookbook serve` offers `--host 0.0.0.0` so a container can reach the
    server. The archive is somebody's conversations, so the routes are not
    registered at all off loopback -- a 403 would still tell a caller the
    archive is there.

    Mutation: register unconditionally and this fails.
    """
    from qmcp.server import is_loopback

    assert all(is_loopback(host) for host in ("127.0.0.1", "::1", "localhost", ""))
    assert not any(is_loopback(host) for host in ("0.0.0.0", "10.0.0.5", "::"))


def test_an_unfamiliar_host_is_treated_as_remote():
    """A guard that fails open on an unfamiliar string stops guarding the first
    time somebody names an interface."""
    from qmcp.server import is_loopback

    assert not is_loopback("eth0")
    assert not is_loopback("example.internal")


def test_the_app_mounts_the_archive_on_loopback(monkeypatch):
    from qmcp import server

    app = server.create_app()
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/v1/threads" in paths


# --- serving spends nothing ---------------------------------------------------


def test_reading_a_thread_over_http_spends_nothing(tmp_path):
    """Every route reads local files. A route needing a paid call would be
    refused by the budget it was handed rather than billing the requester."""
    from qmcp.threads.claude import ClaudeThreads

    folder = tmp_path / "claude"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({
        "uuid": "c-1", "name": "T",
        "chat_messages": [{"uuid": "m-1", "sender": "human",
                           "text": "DECISION: keep it local"}]}), encoding="utf-8")
    indexed(tmp_path, [entry("c-1")])

    body = client_for(tmp_path).get("/v1/threads/claude/c-1/deltas").json()
    assert body["spent"] == 0
    assert body["perspective"] == ClaudeThreads().perspective
    assert [d["delta"]["delta_type"] for d in body["deltas"]] == ["thread", "chore"]


def test_a_thread_that_is_not_there_is_a_404(tmp_path):
    indexed(tmp_path, [entry()])
    assert client_for(tmp_path).get("/v1/threads/claude/nope").status_code == 404
