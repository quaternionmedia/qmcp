"""The thread-source contract, exercised by a source that actually implements it.

The two shipped sources are stubs, so a suite that only tested them would be
testing `NotImplementedError`. `FakeThreads` below is a real implementation over
a dict, which is what lets the contract be exercised: the budget being spent,
the perspective being carried, a decision citing its turns.

The tests worth reading are the refusals. Every one of them is a convenience
this could have offered, and each is a place it would start lying.
"""

from __future__ import annotations

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


def test_a_stub_survey_establishes_nothing_and_says_why():
    surveyed = ClaudeThreads(project=PROJECT).survey()
    assert isinstance(surveyed.available, dict)
    assert "stub" in surveyed.available["unknown"]
    assert surveyed.established is False


def test_a_stub_does_not_report_zero_threads():
    """`available=0` claims there are none. Nobody looked.

    Mutation: return 0 from the stub's survey and this fails -- which is the
    substitution the harness payload refuses one seam over, arriving here with
    money attached.
    """
    for source in (ClaudeThreads(), ChatGPTThreads()):
        assert source.survey().available != 0


def test_a_real_survey_reports_numbers_and_is_established():
    surveyed = FakeThreads(project=PROJECT).survey()
    assert (surveyed.available, surveyed.would_need) == (2, 2)
    assert surveyed.established is True


def test_a_genuine_zero_is_kept_as_zero():
    """There really is such a thing as an empty source, and it is not unknown."""
    surveyed = FakeThreads(threads={}, project=PROJECT).survey()
    assert surveyed.available == 0
    assert surveyed.established is True


def test_describe_says_nothing_was_spent_when_nothing_was_established():
    text = ClaudeThreads(project=PROJECT).describe()
    assert "unknown" in text
    assert "an unknown is not a zero" in text


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


def test_a_stub_refuses_rather_than_returning_nothing():
    """Returning `[]` would read as "there were none".

    Mutation: return an empty list from a stub's `fetch` and this fails, which
    is a source reporting an empty result for work it never did.
    """
    for source in (ClaudeThreads(project=PROJECT), ChatGPTThreads(project=PROJECT)):
        with pytest.raises(NotImplementedError):
            source.fetch(["anything"], Budget(authorised=1))
        with pytest.raises(NotImplementedError):
            source.decisions(thread(), Budget(authorised=1))


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
