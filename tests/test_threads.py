"""The thread-source contract, exercised by a source that actually implements it.

The two shipped sources are stubs, so a suite that only tested them would be
testing `NotImplementedError`. `FakeThreads` below is a real implementation over
a dict, which is what lets the contract be exercised: the budget being spent,
the perspective being carried, a decision citing its turns.

The tests worth reading are the refusals. Every one of them is a convenience
this could have offered, and each is a place it would start lying.
"""

from __future__ import annotations

import json

import pytest

from qmcp.spend import Budget, Refused
from qmcp.threads import Decision, Survey, Thread, ThreadSource, Turn, to_delta
from qmcp.threads.base import NOTICED, THREAD_LINK, TURN_LINK
from qmcp.threads.chatgpt import ChatGPTThreads
from qmcp.threads.claude import ClaudeThreads

PROJECT = "quaternionmedia/qm"


def turn(identifier: str, text: str = "", role: str = "assistant") -> Turn:
    return Turn(id=identifier, role=role, at="2026-08-20T09:00:00Z", text=text)


def thread(identifier: str = "t-1", turns=(), partial: bool = False) -> Thread:
    return Thread(id=identifier, title="A thread", url=f"https://x/{identifier}",
                  turns=tuple(turns), _partial=partial)


class FakeThreads(ThreadSource):
    """A source over a dict. Free to survey, costs one call per thread fetched."""

    name = "fake"
    perspective = "fake/thread"

    def __init__(self, threads=None, decisions_per_thread=1, **kw):
        super().__init__(**kw)
        # `threads or {...}` would fall back on an empty dict, because empty is
        # falsy -- the same coercion that turned an unknown count into a zero in
        # the control panel earlier. An empty source is a real state to test.
        self._threads = {

            "t-1": thread("t-1", [turn("m-1", "we should split the gate"),
                                  turn("m-2", "agreed")]),
            "t-2": thread("t-2", [turn("m-3", "leave it")]),
        } if threads is None else threads
        self._per_thread = decisions_per_thread
        self.calls = 0

    def survey(self) -> Survey:
        return Survey(source=self.name, available=len(self._threads),
                      would_need=len(self._threads))

    def fetch(self, ids, budget):
        pulled = []
        for identifier in ids:
            budget.spend()          # before the call, never after
            self.calls += 1
            pulled.append(self._threads[identifier])
        return pulled

    def decisions(self, thread_, budget):
        return [
            Decision(name=f"{thread_.id}-decision-{index}",
                     title=f"Something {thread_.id} settled",
                     summary="a summary",
                     from_turns=(thread_.turns[0].id,))
            for index in range(self._per_thread)
        ]


# --- the free pass ----------------------------------------------------------


def test_survey_takes_no_budget_at_all():
    """The enforcement is the signature. There is nothing for it to spend
    against, so a source cannot spend in it by accident -- only by reaching
    around the contract, which is a different kind of mistake.

    Mutation: give `survey` a budget parameter and this fails.
    """
    import inspect

    parameters = inspect.signature(ThreadSource.survey).parameters
    assert list(parameters) == ["self"], "survey must have nothing to spend"


def test_an_empty_cache_establishes_a_real_zero(tmp_path):
    """These read a local export, so they always establish something.

    An absent export directory is zero *cached* threads, which is known. It is
    not a claim about how many conversations exist -- that is what an API
    source would answer, and it would answer differently.
    """
    surveyed = ClaudeThreads(root=tmp_path, project=PROJECT).survey()
    assert surveyed.available == 0
    assert surveyed.established is True


def test_a_zero_from_a_cache_says_it_looked(tmp_path):
    """`available=0` is only honest if somebody looked. A cache read did.

    The distinction this replaces is still live for the API source: one that
    could not list without spending must say `unknown` with that reason rather
    than reporting none.
    """
    surveyed = ClaudeThreads(root=tmp_path).survey()
    assert surveyed.available == 0
    assert "not a failure to look" in (surveyed.note or "")


def test_a_real_survey_reports_numbers_and_is_established():
    surveyed = FakeThreads(project=PROJECT).survey()
    assert (surveyed.available, surveyed.would_need) == (2, 2)
    assert surveyed.established is True


def test_a_genuine_zero_is_kept_as_zero():
    """There really is such a thing as an empty source, and it is not unknown."""
    surveyed = FakeThreads(threads={}, project=PROJECT).survey()
    assert surveyed.available == 0
    assert surveyed.established is True


def test_describe_shows_the_cost_of_the_paid_work_as_none(tmp_path):
    """Reading a file costs no calls, so the paid work really is zero, and a
    person deciding whether to authorise anything is told so plainly."""
    text = ClaudeThreads(root=tmp_path, project=PROJECT).describe()
    assert "paid calls" in text
    assert "0" in text


# --- the passes that spend --------------------------------------------------


def test_fetching_spends_the_budget():
    source = FakeThreads(project=PROJECT)
    budget = Budget(authorised=2, service="fake")
    source.fetch(["t-1", "t-2"], budget)
    assert (budget.made, source.calls) == (2, 2)


def test_a_zero_budget_pulls_nothing():
    """The default. A command nobody gave a number to spends nothing.

    Mutation: default `Budget.authorised` to anything above zero and this
    fails.
    """
    source = FakeThreads(project=PROJECT)
    with pytest.raises(Refused):
        source.fetch(["t-1"], Budget())
    assert source.calls == 0


def test_the_budget_stops_the_pull_partway_and_the_call_is_not_made():
    """Checked before the call. A refusal after the money is gone is a report."""
    source = FakeThreads(project=PROJECT)
    budget = Budget(authorised=1)
    with pytest.raises(Refused):
        source.fetch(["t-1", "t-2"], budget)
    assert source.calls == 1, "the second call was refused, not made and undone"


def test_a_partial_thread_says_so_rather_than_leaving_it_to_be_counted():
    """A consumer counting turns on an excerpt reports the size of the excerpt
    and calls it the conversation."""
    assert thread(turns=[turn("m-1")], partial=True).partial is True
    assert thread(turns=[turn("m-1")]).partial is False


# --- the perspective --------------------------------------------------------


def test_every_payload_names_the_perspective_it_speaks_from():
    source = FakeThreads(project=PROJECT)
    payloads = source.deltas(thread("t-1", [turn("m-1")]), Budget())
    assert payloads
    assert all(p["perspective"] == "fake/thread" for p in payloads)


def test_a_payload_without_a_perspective_is_refused_not_defaulted():
    """A default would be a silent claim about level.

    Mutation: default it to the source name and this fails, which is a receiver
    being told a level nobody stated.
    """
    with pytest.raises(ValueError) as raised:
        to_delta(Decision("n", "T", from_turns=("m-1",)), thread(),
                 project=PROJECT, perspective="")
    assert "perspective" in str(raised.value)


def test_two_assistants_speak_from_different_perspectives():
    """They are not duplicates of each other. Two perspectives on one strand,
    and `same-as` is how somebody says so after reading both."""
    assert ClaudeThreads().perspective != ChatGPTThreads().perspective


# --- what a decision must carry ---------------------------------------------


def test_a_decision_citing_no_turn_is_refused():
    """A claim a person cannot check against anything.

    Mutation: allow an empty `from_turns` and a source can put unfalsifiable
    rows on a board, which is worse there than an absent one.
    """
    with pytest.raises(ValueError) as raised:
        to_delta(Decision("n", "T"), thread(), project=PROJECT,
                 perspective="fake/thread")
    assert "cites no turn" in str(raised.value)


def test_a_delta_points_back_at_the_thread_and_the_turns():
    payload = to_delta(
        Decision("split-the-gate", "Split the gate", from_turns=("m-1", "m-2")),
        thread("t-1", [turn("m-1"), turn("m-2")]),
        project=PROJECT, perspective="fake/thread")
    kinds = [link["link_type"] for link in payload["links"]]
    assert kinds.count(TURN_LINK) == 2
    assert THREAD_LINK in kinds
    assert any(link["link_type"] == "address"
               and link["target_name"].endswith("/delta/split-the-gate")
               for link in payload["links"])


def test_an_extracted_decision_never_opens_past_noticed():
    """A source recognising a decision has noticed something. It has not
    established that anybody acted, and `planning` would assert they had.

    Mutation: raise the phase and this fails.
    """
    payload = to_delta(Decision("n", "T", from_turns=("m-1",)), thread(),
                       project=PROJECT, perspective="fake/thread")
    assert payload["delta"]["phase"] == NOTICED == "brainstorm"


def test_the_payload_carries_the_consumers_own_column_names():
    """`ProjectDelta(**payload["delta"])` and nothing translates in between.
    `perspective` sits beside the row rather than inside it, because it
    describes the claim rather than the work."""
    payload = to_delta(Decision("n", "T", from_turns=("m-1",)), thread(),
                       project=PROJECT, perspective="fake/thread")
    assert set(payload["delta"]) == {"name", "title", "description", "phase",
                                     "delta_type", "priority"}
    assert "perspective" not in payload["delta"]


def test_a_thread_produces_far_fewer_deltas_than_turns():
    """The filter, which is the point. A thread is mostly steps.

    Mirroring turns would bury the decisions; this asserts the shape refuses to.
    """
    many = thread("t-1", [turn(f"m-{i}") for i in range(40)])
    source = FakeThreads(threads={"t-1": many}, project=PROJECT)
    assert len(source.deltas(many, Budget())) < len(many.turns)


# --- the stubs are honest ---------------------------------------------------


def test_a_source_without_a_parse_refuses_rather_than_returning_nothing(tmp_path):
    """Returning `[]` would read as "there were none".

    The shipped sources both parse, so this holds the base class to it: a
    `LocalCacheSource` that forgot `parse` reports every file as unreadable and
    names them, rather than an empty directory.
    """
    from qmcp.threads.cache import LocalCacheSource

    class Forgetful(LocalCacheSource):
        name = "forgetful"
        folder = "forgetful"
        perspective = "forgetful/thread"

    _write(tmp_path, "forgetful", {"anything": True})
    source = Forgetful(root=tmp_path, project=PROJECT)
    with pytest.raises(NotImplementedError):
        source.fetch([], Budget())


def test_no_module_here_calls_a_paid_service_yet():
    """Asserted on the source. When one does, it goes through `qmcp.spend`, and
    this test is what says the day it changed."""
    from pathlib import Path

    import qmcp.threads as package

    for path in Path(package.__file__).parent.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for client in ("import anthropic", "import openai", "import httpx",
                       "import requests", "urllib.request"):
            assert client not in text, f"{path.name} became a caller: {client}"


# --- the local cache, which is what both sources actually are -----------------


def _write(root, folder, document, name="a.json"):
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        document if isinstance(document, str) else json.dumps(document),
        encoding="utf-8")
    return path


def _claude_export(identifier="c-1", messages=None):
    return {"uuid": identifier, "name": "A thread",
            "chat_messages": messages if messages is not None else
            [{"uuid": "m-1", "sender": "human", "text": "hello"}]}


def test_a_cache_read_costs_nothing_and_says_the_paid_work_is_zero(tmp_path):
    """A genuine zero, not the sentinel. There is no paid work to do here.

    Mutation: return `unknown(...)` for `would_need` and this fails -- which
    would tell a person the cost could not be established when it is known and
    is none.
    """
    _write(tmp_path, "claude", _claude_export())
    surveyed = ClaudeThreads(root=tmp_path).survey()
    assert surveyed.would_need == 0
    assert surveyed.established is True


def test_fetching_from_a_cache_spends_nothing_even_at_zero_budget(tmp_path):
    """The whole point of local-first: the path runs with no credential, no
    network and no bill."""
    _write(tmp_path, "claude", _claude_export())
    budget = Budget()
    threads = ClaudeThreads(root=tmp_path).fetch([], budget)
    assert len(threads) == 1
    assert budget.made == 0


def test_an_absent_export_is_zero_threads_and_not_a_failure(tmp_path):
    surveyed = ClaudeThreads(root=tmp_path / "nothing-here").survey()
    assert surveyed.available == 0
    assert "not a failure to look" in (surveyed.note or "")


def test_a_file_that_will_not_parse_is_named_and_not_counted(tmp_path):
    """THE ONE THIS MODULE'S DOCSTRING WARNS ABOUT, AND THE FIRST VERSION DID.

    `survey` counted files, so a directory of two with one malformed reported
    two and fetched one. It reads them now.

    Mutation: count `files()` instead of reading and this fails.
    """
    _write(tmp_path, "claude", _claude_export(), name="good.json")
    _write(tmp_path, "claude", "{not json", name="broken.json")
    surveyed = ClaudeThreads(root=tmp_path).survey()
    assert surveyed.available == 1, "the unreadable file is not a thread"
    assert "broken.json" in (surveyed.note or ""), "and it is named"


def test_a_malformed_file_does_not_cost_the_others(tmp_path):
    _write(tmp_path, "claude", "{not json", name="broken.json")
    _write(tmp_path, "claude", _claude_export("c-1"), name="a.json")
    _write(tmp_path, "claude", _claude_export("c-2"), name="b.json")
    source = ClaudeThreads(root=tmp_path)
    assert len(source.fetch([], Budget())) == 2
    assert [u.path for u in source.unreadable] == ["broken.json"]


def test_an_export_missing_its_id_is_unreadable_rather_than_invented(tmp_path):
    _write(tmp_path, "claude", {"name": "no id", "chat_messages": []})
    source = ClaudeThreads(root=tmp_path)
    assert source.fetch([], Budget()) == []
    assert "uuid" in source.unreadable[0].why


def test_both_content_shapes_are_read(tmp_path):
    """Newer exports carry typed blocks, older ones a flat string. A block that
    is not text is skipped rather than stringified."""
    _write(tmp_path, "claude", _claude_export(messages=[
        {"uuid": "m-1", "sender": "human", "text": "flat"},
        {"uuid": "m-2", "sender": "assistant",
         "content": [{"type": "text", "text": "blocks"}]},
        {"uuid": "m-3", "sender": "assistant",
         "content": [{"type": "image", "source": "x"}]},
    ]))
    turns = ClaudeThreads(root=tmp_path).fetch([], Budget())[0].turns
    assert [t.text for t in turns] == ["flat", "blocks", ""]


def test_a_chatgpt_tree_is_flattened_in_timestamp_order(tmp_path):
    """It is stored as a tree and read as a sequence, which is an inaccuracy the
    module states rather than hides."""
    _write(tmp_path, "chatgpt", {
        "conversation_id": "g-1", "title": "T", "mapping": {
            "n2": {"message": {"id": "m-2", "create_time": 2.0,
                               "author": {"role": "assistant"},
                               "content": {"parts": ["second"]}}},
            "n1": {"message": {"id": "m-1", "create_time": 1.0,
                               "author": {"role": "user"},
                               "content": {"parts": ["first"]}}},
        }})
    turns = ChatGPTThreads(root=tmp_path).fetch([], Budget())[0].turns
    assert [t.id for t in turns] == ["m-1", "m-2"]


# --- a thread is itself a delta ----------------------------------------------


def test_the_thread_is_emitted_as_a_delta_of_its_own_type(tmp_path):
    _write(tmp_path, "claude", _claude_export(messages=[
        {"uuid": "m-1", "sender": "assistant", "text": "DECISION: do the thing"}]))
    source = ClaudeThreads(root=tmp_path)
    payloads = source.deltas(source.fetch([], Budget())[0], Budget())
    assert payloads[0]["delta"]["delta_type"] == "thread"
    assert payloads[0]["delta"]["name"] == "thread-c-1"


def test_a_thread_that_settled_nothing_is_still_emitted(tmp_path):
    """A conversation that reached no conclusion is one somebody had, and it is
    the interesting one to be able to find.

    Mutation: emit only decisions and this fails.
    """
    _write(tmp_path, "claude", _claude_export())
    source = ClaudeThreads(root=tmp_path)
    payloads = source.deltas(source.fetch([], Budget())[0], Budget())
    assert len(payloads) == 1
    assert payloads[0]["delta"]["delta_type"] == "thread"


def test_each_decision_is_part_of_its_thread_and_the_relation_is_stated(tmp_path):
    """Stated, not derived. A consumer must not infer containment from two rows
    sharing a prefix -- `DRAFT-deltas-compose.md` 5."""
    _write(tmp_path, "claude", _claude_export(messages=[
        {"uuid": "m-1", "sender": "assistant",
         "text": "DECISION: first\nDECIDED: second"}]))
    source = ClaudeThreads(root=tmp_path)
    relations = source.relations(source.fetch([], Budget())[0], Budget())
    assert len(relations) == 2
    assert all(r["relation"] == "part-of" for r in relations)
    assert all(r["target"].endswith("/delta/thread-c-1") for r in relations)


def test_a_decision_name_carries_its_thread(tmp_path):
    """Two conversations may settle the same thing in the same words, and one
    row for both would be a claim nobody made."""
    for identifier, name in (("c-1", "a.json"), ("c-2", "b.json")):
        _write(tmp_path, "claude", _claude_export(identifier, messages=[
            {"uuid": "m-1", "sender": "assistant",
             "text": "DECISION: same words"}]), name=name)
    source = ClaudeThreads(root=tmp_path)
    names = {
        payload["delta"]["name"]
        for pulled in source.fetch([], Budget())
        for payload in source.deltas(pulled, Budget())
        if payload["delta"]["delta_type"] != "thread"
    }
    assert len(names) == 2


def test_extraction_finds_only_what_was_marked(tmp_path):
    """It recognises nothing on its own, and the docstring says so. A free
    heuristic dressed as comprehension would put confident rows on a board and
    a reader would have no way to tell which kind they were."""
    _write(tmp_path, "claude", _claude_export(messages=[
        {"uuid": "m-1", "sender": "assistant",
         "text": "We should probably split the gate, it would be cleaner."}]))
    source = ClaudeThreads(root=tmp_path)
    assert source.decisions(source.fetch([], Budget())[0], Budget()) == []


def test_the_deltas_belong_to_the_harness_that_pulled_them(tmp_path):
    """Decided rather than defaulted: a thread belongs to no repository, so
    somebody chose one."""
    assert ClaudeThreads().project == "quaternionmedia/qmcp"
    assert ChatGPTThreads().project == "quaternionmedia/qmcp"
