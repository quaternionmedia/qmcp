"""A thread's delta identity, published rather than re-derived downstream.

The listing had every fact about a conversation except the one that makes it a
delta: its address. A consumer wanting to show threads on a delta board had two
options -- ask for it, or reimplement `thread-{id}` and the project it hangs
under. This is the first option existing.
"""

from __future__ import annotations

from qmcp.threads.base import Thread, thread_name, to_thread_delta
from qmcp.threads.service import as_delta_row, source_classes, summarise


def test_the_address_matches_the_one_the_delta_actually_carries():
    """THE ONE THAT MATTERS.

    Two ways of naming the same thing, checked against each other. If they ever
    disagree, a board shows an address nothing can be found at.

    Mutation: change the prefix in `thread_name` and this fails -- which is the
    point, because without it only the payload would move and the listing would
    keep publishing the old address.
    """
    thread = Thread(id="abc123", title="A conversation")
    claude = source_classes()["claude"]

    delta = to_thread_delta(thread, project=claude.project,
                            perspective=claude.perspective)
    published = as_delta_row("claude", thread.id)

    addresses = [link["target_name"] for link in delta["links"]
                 if link["link_type"] == "address"]
    assert addresses == [published["address"]]
    assert published["address"].endswith(thread_name(thread))


def test_each_source_keeps_its_own_perspective():
    """Two assistants discussing one piece of work produce two perspectives on
    one strand, not one perspective twice. Collapsing them here would make a
    board show two rows that look like duplicates."""
    seen = {name: as_delta_row(name, "x")["perspective"]
            for name in source_classes()}
    assert len(set(seen.values())) == len(seen), seen
    assert seen["claude"] != seen["chatgpt"]


def test_an_unknown_source_gets_no_address_rather_than_a_guessed_one():
    """A made-up address is worse than an absent one: it looks like something a
    reader can go and find.

    Mutation: fall back to a default project and this fails.
    """
    assert as_delta_row("no-such-assistant", "x") == {
        "address": None, "perspective": None}


def test_deriving_an_address_reads_nothing_from_disk(tmp_path, monkeypatch):
    """It is called once per row in a listing. A version that touched the store
    would make a page of four hundred threads four hundred reads, and would
    fail entirely when the store moved.

    Mutation: build a real source instance in `as_delta_row` and this fails --
    `ClaudeCodeThreads` resolves a session root at construction.
    """
    def refuse(*args, **kwargs):
        raise AssertionError("the store was read to derive an address")

    monkeypatch.setattr("pathlib.Path.glob", refuse)
    monkeypatch.setattr("pathlib.Path.iterdir", refuse)
    assert as_delta_row("claude", "abc")["address"]


def test_the_listing_carries_the_address_for_every_row():
    """Mutation: drop the spread from `summarise` and this fails."""
    document = {
        "schema": 2,
        "generated_at": "2026-08-20T00:00:00Z",
        "totals": {"threads": 2},
        "threads": [
            {"source": "claude", "id": "one", "title": "One", "turns": 3,
             "digest": "d", "first_seen": "", "last_seen": "", "history": []},
            {"source": "claude-code", "id": "two", "title": "Two", "turns": 9,
             "digest": "e", "first_seen": "", "last_seen": "", "history": []},
        ],
    }
    rows = summarise(document)["threads"]
    assert [row["address"] for row in rows] == [
        "quaternionmedia/qmcp/delta/thread-one",
        "quaternionmedia/qmcp/delta/thread-two",
    ]
    assert rows[0]["perspective"] == "claude/thread"
    assert rows[1]["perspective"] == "claude-code/session"


def test_the_listing_still_carries_what_it_carried_before():
    """Adding a field must not quietly drop one. The panel reads these."""
    document = {
        "schema": 2, "generated_at": "x", "totals": {},
        "threads": [{"source": "claude", "id": "one", "title": "One",
                     "turns": 3, "digest": "d", "first_seen": "a",
                     "last_seen": "b", "history": []}],
    }
    row = summarise(document)["threads"][0]
    for field in ("source", "id", "title", "turns", "digest",
                  "first_seen", "last_seen", "diverged", "changes"):
        assert field in row, f"{field} was dropped"
