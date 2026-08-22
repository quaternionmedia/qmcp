"""The thread index, and an archive that never overwrites.

The tests worth reading are the classifications. Telling a conversation that
grew from one whose export disagrees with an earlier record of itself is the
whole point, and the failure that matters is one direction only: a divergence
reported as growth is a finding nobody sees.
"""

from __future__ import annotations

import json

from qmcp.threads.base import Thread, Turn
from qmcp.threads.index import (
    DIVERGED,
    GREW,
    Entry,
    build,
    classify,
    digest_of,
    document,
    drift,
    entry_for,
    load,
    merge,
    render,
)

AT = "2026-08-20T00:00:00Z"
LATER = "2026-08-21T00:00:00Z"


def thread(identifier="c-1", turns=(("m-1", "one"),)):
    return Thread(id=identifier, title="T",
                  turns=tuple(Turn(id=i, role="assistant", text=t)
                              for i, t in turns))


def entry(identifier="c-1", turns=(("m-1", "one"),), at=AT):
    return entry_for(thread(identifier, turns), "claude", at)


# --- the digest is over what a thread says -----------------------------------


def test_the_digest_covers_the_text_not_only_the_ids():
    """A turn edited in place must change the digest, or nothing downstream can
    tell. Mutation: digest the ids alone and this fails."""
    assert digest_of(thread(turns=(("m-1", "one"),))) != \
        digest_of(thread(turns=(("m-1", "two"),)))


def test_the_digest_is_over_content_and_not_the_file():
    """An exporter that reformats its JSON must not register as every
    conversation diverging at once -- a finding nobody could act on, which
    teaches its reader to ignore the real ones."""
    assert digest_of(thread()) == digest_of(thread())


def test_order_is_part_of_the_digest():
    a = thread(turns=(("m-1", "one"), ("m-2", "two")))
    b = thread(turns=(("m-2", "two"), ("m-1", "one")))
    assert digest_of(a) != digest_of(b)


# --- growth, and the three ways it is not growth -----------------------------


def test_turns_appended_is_growth():
    before = entry(turns=(("m-1", "one"),))
    after = entry(turns=(("m-1", "one"), ("m-2", "two")))
    kind, detail = classify(before, after)
    assert kind == GREW
    assert "1 turn(s) added" in detail


def test_a_turn_edited_in_place_is_divergence_not_growth():
    """THE ONE THAT WAS WRONG.

    The first `classify` checked only that the old ids were a prefix of the
    new. Identical ids with different text satisfied that, so an edited
    conversation reported as having grown by zero turns -- a divergence
    classified as the ordinary case, which is the single direction this must
    not fail in.

    Mutation: drop the `len(new) > len(old)` guard and this fails.
    """
    before = entry(turns=(("m-1", "the original"),))
    after = entry(turns=(("m-1", "rewritten"),))
    kind, detail = classify(before, after)
    assert kind == DIVERGED
    assert "edited after" in detail


def test_a_turn_that_vanished_is_divergence():
    before = entry(turns=(("m-1", "one"), ("m-2", "two")))
    after = entry(turns=(("m-1", "one"),))
    kind, detail = classify(before, after)
    assert kind == DIVERGED
    assert "absent now" in detail


def test_reordering_is_divergence_and_not_growth():
    """An export that kept every old turn but reordered them has not grown, it
    has rewritten. Calling that growth hides the interesting half."""
    before = entry(turns=(("m-1", "one"), ("m-2", "two")))
    after = entry(turns=(("m-2", "two"), ("m-1", "one"), ("m-3", "three")))
    assert classify(before, after)[0] == DIVERGED


# --- the archive keeps what the cache no longer has ---------------------------


def test_a_first_index_is_all_new():
    entries, changed = merge({}, [entry("c-1"), entry("c-2")], at=AT)
    assert len(entries) == len(changed) == 2


def test_an_unchanged_thread_is_not_reported_as_changed():
    previous = {e.key: e for e in [entry("c-1")]}
    entries, changed = merge(previous, [entry("c-1")], at=LATER)
    assert changed == []
    assert entries[0].last_seen == LATER, "seen again, so last_seen moves"


def test_first_seen_survives_every_later_index():
    """When a thread was first known is a fact the files cannot tell you, and
    it is the reason the archive layer exists."""
    previous = {e.key: e for e in [entry("c-1", at=AT)]}
    entries, _ = merge(previous, [entry("c-1", (("m-1", "one"), ("m-2", "two")))],
                       at=LATER)
    assert entries[0].first_seen == AT
    assert entries[0].last_seen == LATER


def test_the_change_is_recorded_rather_than_the_old_version_replaced():
    previous = {e.key: e for e in [entry("c-1", (("m-1", "old"),))]}
    entries, _ = merge(previous, [entry("c-1", (("m-1", "new"),))], at=LATER)
    history = entries[0].history
    assert len(history) == 1
    assert history[0].kind == DIVERGED
    assert history[0].from_digest and history[0].from_digest != entries[0].digest


def test_history_accumulates_across_indexes():
    state = {}
    for turns in ((("m-1", "a"),), (("m-1", "b"),), (("m-1", "c"),)):
        entries, _ = merge(state, [entry("c-1", turns)], at=LATER)
        state = {e.key: e for e in entries}
    assert len(state["claude/c-1"].history) == 2


def test_a_thread_the_cache_no_longer_holds_keeps_its_entry():
    """An export somebody deleted or replaced with a narrower one has not
    un-happened. Removing the row would make the index agree with the cache by
    forgetting.

    Mutation: build the merge from the current entries alone and this fails.
    """
    previous = {e.key: e for e in [entry("c-1"), entry("c-gone")]}
    entries, _ = merge(previous, [entry("c-1")], at=LATER)
    assert {e.id for e in entries} == {"c-1", "c-gone"}


def test_two_sources_may_use_the_same_conversation_id():
    """The source is part of the identity rather than assumed unique."""
    claude = entry_for(thread("shared"), "claude", AT)
    chatgpt = entry_for(thread("shared"), "chatgpt", AT)
    entries, _ = merge({}, [claude, chatgpt], at=AT)
    assert len(entries) == 2


# --- the document -------------------------------------------------------------


def test_the_document_counts_what_is_cached_and_says_so():
    doc = document([entry("c-1")], at=AT)
    assert doc["totals"]["threads"] == 1
    assert any("how many are cached" in line for line in doc["reading"]["do_not"])


def test_the_document_refuses_to_be_committed_in_writing():
    doc = document([], at=AT)
    assert any("commit this file" in line for line in doc["reading"]["do_not"])


def test_unreadable_files_are_counted_separately_from_threads():
    doc = document([entry("c-1")], unreadable={"claude": ["broken.json"]}, at=AT)
    assert doc["totals"]["threads"] == 1
    assert doc["totals"]["unreadable"] == 1
    assert "not counted above" in render(doc)


def test_a_divergence_reaches_the_rendered_page():
    previous = {e.key: e for e in [entry("c-1", (("m-1", "old"),))]}
    entries, changed = merge(previous, [entry("c-1", (("m-1", "new"),))], at=LATER)
    text = render(document(entries, at=LATER), changed)
    assert "changed in a way an export should not" in text
    assert "Nothing here is repaired" in text


def test_an_index_round_trips_through_a_file(tmp_path):
    previous = {e.key: e for e in [entry("c-1", (("m-1", "old"),))]}
    entries, _ = merge(previous, [entry("c-1", (("m-1", "new"),))], at=LATER)
    path = tmp_path / "index.json"
    path.write_text(json.dumps(document(entries, at=LATER)), encoding="utf-8")

    loaded = load(path)
    assert loaded["claude/c-1"].first_seen == AT
    assert loaded["claude/c-1"].history[0].kind == DIVERGED


def test_an_absent_index_is_not_an_error(tmp_path):
    assert load(tmp_path / "nothing.json") == {}


def test_an_unreadable_index_is_not_an_error(tmp_path):
    """A truncated write must not stop the next index from being built."""
    path = tmp_path / "index.json"
    path.write_text("{not json", encoding="utf-8")
    assert load(path) == {}


# --- drift, which compares only what the files can answer ---------------------


def test_drift_is_silent_when_the_index_matches_the_files():
    entries = [entry("c-1")]
    assert drift(document(entries, at=AT), entries) == []


def test_drift_reports_a_digest_that_moved():
    doc = document([entry("c-1", (("m-1", "old"),))], at=AT)
    problems = drift(doc, [entry("c-1", (("m-1", "new"),))])
    assert problems and "the files say" in problems[0]


def test_drift_reports_a_thread_the_index_never_saw():
    problems = drift(document([], at=AT), [entry("c-new")])
    assert problems and "not in the index" in problems[0]


def test_an_archived_thread_absent_from_the_cache_is_not_drift():
    """The archive keeps what the cache no longer holds, deliberately. Calling
    that drift would report the design as a defect on every run."""
    assert drift(document([entry("c-gone")], at=AT), []) == []


# --- building never spends ----------------------------------------------------


def test_building_the_index_spends_nothing(tmp_path):
    """Indexing must never be the thing that bills somebody.

    A source needing a paid call to be indexed refuses rather than quietly
    spending, because `build` hands it a budget of zero.
    """
    from qmcp.spend import Budget, Refused
    from qmcp.threads.base import Decision, Survey, ThreadSource

    class Metered(ThreadSource):
        name = "metered"
        perspective = "metered/thread"

        def survey(self):
            return Survey(source=self.name, available=1, would_need=1)

        def fetch(self, ids, budget: Budget):
            budget.spend()
            return [thread()]

        def decisions(self, thread_, budget):
            return []

    try:
        build([Metered()], at=AT)
    except Refused as refused:
        assert "0 authorised" in str(refused)
    else:
        raise AssertionError("a metered source was indexed without refusing")
