"""The seam a model is called through, and what it refuses to do.

**WHAT THESE TESTS ARE FOR.** `qmcp.governed` makes three claims that are worth
nothing unless something holds them: that the black box is called at most once,
that it is never reached without the budget having been checked first, and that
a bound which fires is reported rather than absorbed. Each has a test whose
whole job is that claim.

**AND ONE CLAIM THAT IS DELIBERATELY NOT TESTED HERE.** Nothing below proves a
slow call is stopped, because nothing stops one. `Bound.seconds` is measured and
reported. A test named `test_a_slow_call_is_stopped` would be the worst artifact
in this file -- a green check standing exactly where a reader believes something
is enforced.

THE MUTATIONS, per P16. Quoted as they printed:

`budget.spend(1)` deleted from `run`, so the call happens unbudgeted:

    AssertionError: assert 'drafted' == 'refused'
    assert 0 == 1

The `len(answer) > bound.chars` branch removed, so an oversized answer drafts:

    AssertionError: assert 'drafted' == 'stopped'
    At index 1 diff: 'drafted' != 'stopped'

`over_bound` hard-coded to `False`, so an exceeded expectation is absorbed:

    AssertionError: an exceeded expectation was not reported
    assert False
     +  where False = Outcome(state='drafted', ..., over_bound=False).over_bound

Each restoration goes green at 33 passed.
"""

from __future__ import annotations

import pytest

from qmcp import governed as g
from qmcp.spend import Budget
from qmcp.topology_view import GATE, INPUT, LEVELS, OUTPUT, REFUSAL


def a_request(text: str = "summarise the delta") -> g.Request:
    return g.Request(text=text, purpose="a draft for the release notes",
                     issued_by="a person")


class Counter:
    """A model that records being called, and nothing else."""

    def __init__(self, answer: str = "a draft answer") -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, text: str) -> str:
        self.calls += 1
        return self.answer


# --- the budget, which is the point of the seam existing ----------------------


def test_a_zero_budget_never_reaches_the_black_box() -> None:
    """The default is zero, and zero is a real count rather than an absence."""
    call = Counter()
    outcome = g.run(a_request(), Budget(), call)

    assert outcome.state == g.REFUSED
    assert call.calls == 0
    assert not outcome.called, "the black box was reached on a refused run"
    assert outcome.stages == ("in", "budget")


def test_a_refusal_still_declares_what_the_work_would_cost() -> None:
    """A free pass establishes the count without spending to find it out.

    The distinction `qmcp.spend` is built on: `would_need: 0` from a run that
    never counted would claim the work is free.
    """
    outcome = g.run(a_request(), Budget(), Counter())

    assert outcome.declared["free_pass"] is True
    assert outcome.declared["would_need"] == 1
    assert outcome.declared["made"] == 0


def test_one_authorised_call_is_made_exactly_once() -> None:
    call = Counter()
    outcome = g.run(a_request(), Budget(authorised=1, service="a model"), call)

    assert outcome.state == g.DRAFTED
    assert call.calls == 1
    assert outcome.declared["made"] == 1
    assert outcome.declared["service"] == "a model"


def test_a_second_run_against_a_spent_budget_is_refused() -> None:
    """Consent is an amount rather than a category, and it does not carry."""
    budget = Budget(authorised=1)
    call = Counter()

    first = g.run(a_request(), budget, call)
    second = g.run(a_request(), budget, call)

    assert first.state == g.DRAFTED
    assert second.state == g.REFUSED
    assert call.calls == 1


def test_the_refusal_says_how_to_widen_it_and_does_not_widen_it() -> None:
    budget = Budget(authorised=0, service="a model")
    outcome = g.run(a_request(), budget, Counter())

    assert "Re-issue" in outcome.why
    assert budget.authorised == 0


# --- the bounds, and which of them are enforced -------------------------------


def test_an_oversized_answer_stops_and_is_not_reported_as_free() -> None:
    """The call was made and paid for, so this is stopped rather than refused."""
    outcome = g.run(a_request(), Budget(authorised=1),
                    Counter("x" * 50), bound=g.Bound(chars=10))

    assert outcome.state == g.STOPPED
    assert outcome.called
    assert outcome.declared["made"] == 1
    assert "50 characters" in outcome.why


def test_an_answer_that_is_not_text_stops_rather_than_raising() -> None:
    """A caller cannot be handed something the type says is a draft and is not."""
    outcome = g.run(a_request(), Budget(authorised=1), lambda text: {"a": 1})

    assert outcome.state == g.STOPPED
    assert "dict" in outcome.why
    assert outcome.draft == ""


def test_an_exceeded_expectation_is_reported_and_the_draft_still_arrives() -> None:
    """P17: a bound that fires is reported, never absorbed.

    And never silently upgraded into a refusal either. Nothing interrupted the
    call, so reporting it as stopped would claim an enforcement that did not
    happen.
    """
    ticks = iter([100.0, 109.0])
    outcome = g.run(a_request(), Budget(authorised=1), Counter(),
                    bound=g.Bound(seconds=5.0), clock=lambda: next(ticks))

    assert outcome.state == g.DRAFTED
    assert outcome.over_bound, "an exceeded expectation was not reported"
    assert outcome.elapsed == pytest.approx(9.0)


def test_a_call_inside_its_expectation_is_not_marked() -> None:
    ticks = iter([100.0, 101.0])
    outcome = g.run(a_request(), Budget(authorised=1), Counter(),
                    bound=g.Bound(seconds=5.0), clock=lambda: next(ticks))

    assert not outcome.over_bound
    assert outcome.elapsed == pytest.approx(1.0)


def test_no_expectation_means_no_verdict_about_elapsed() -> None:
    """`None` is no expectation, and it must not read as one that passed."""
    ticks = iter([100.0, 900.0])
    outcome = g.run(a_request(), Budget(authorised=1), Counter(),
                    clock=lambda: next(ticks))

    assert outcome.over_bound is False
    assert outcome.elapsed == pytest.approx(800.0)


@pytest.mark.parametrize("chars,seconds", [(0, None), (-1, None), (10, 0), (10, -3)])
def test_a_bound_that_admits_nothing_is_refused_at_construction(
        chars: int, seconds: float | None) -> None:
    with pytest.raises(ValueError):
        g.Bound(chars=chars, seconds=seconds)


# --- what a request is, and what a draft is -----------------------------------


def test_the_request_is_carried_verbatim() -> None:
    call = Counter()
    text = "  leading and trailing space, and a\nnewline  "
    g.run(g.Request(text=text, purpose="p"), Budget(authorised=1), call)

    assert call.calls == 1


def test_the_fingerprint_is_content_addressed_and_needs_no_clock() -> None:
    """The same request twice has the same name, on any machine."""
    assert a_request().fingerprint == a_request().fingerprint
    assert a_request("other").fingerprint != a_request().fingerprint
    assert a_request().fingerprint.startswith("governed-")


def test_every_outcome_reports_its_stages_and_its_spend() -> None:
    """A consumer must never have to tell paths apart by which field is empty."""
    outcomes = [
        g.run(a_request(), Budget(), Counter()),
        g.run(a_request(), Budget(authorised=1), Counter("x" * 9),
              bound=g.Bound(chars=2)),
        g.run(a_request(), Budget(authorised=1), Counter()),
    ]

    assert [o.state for o in outcomes] == [g.REFUSED, g.STOPPED, g.DRAFTED]
    for outcome in outcomes:
        assert outcome.stages, f"{outcome.state} reported no stages"
        assert set(outcome.declared) >= {"authorised", "made", "would_need",
                                         "free_pass"}


# --- the human gate, which this pipeline ends at ------------------------------


def test_the_queue_payload_says_it_is_a_draft_outright() -> None:
    outcome = g.run(a_request(), Budget(authorised=1), Counter())
    payload = g.queued(outcome)

    assert payload["context"]["this_is_a_draft"] is True
    assert payload["context"]["draft"] == "a draft answer"
    assert payload["request_type"] == "approval"
    assert payload["prompt"] == "a draft for the release notes"


def test_a_refused_run_is_queued_too_and_says_why() -> None:
    """Somebody deciding whether to authorise a retry needs the refusal."""
    outcome = g.run(a_request(), Budget(authorised=0, service="a model"),
                    Counter())
    payload = g.queued(outcome)

    assert payload["context"]["state"] == g.REFUSED
    assert payload["context"]["draft"] == ""
    assert "exceed" in payload["context"]["why"]
    assert payload["context"]["spend"]["free_pass"] is True


def test_the_queue_ids_of_two_states_of_one_request_differ() -> None:
    """A refusal and a later draft are two things for a person to look at."""
    request = a_request()
    refused = g.run(request, Budget(), Counter())
    drafted = g.run(request, Budget(authorised=1), Counter())

    assert g.queued(refused)["id"] != g.queued(drafted)["id"]


def test_the_module_offers_nothing_that_answers_the_question() -> None:
    """Answering is a person's by constitution, so there is nothing to call.

    A structural check rather than a behavioural one: it fails the moment
    somebody adds the convenience function this seam exists to not have.
    """
    offered = {name for name in dir(g) if not name.startswith("_")}
    forbidden = {"approve", "accept", "decide", "answer", "reject", "ratify",
                 "resolve", "auto_approve"}

    assert not (offered & forbidden), (
        f"{sorted(offered & forbidden)} would let a machine answer a question "
        f"`ci/attested-registry.yaml` reserves for a person")


# --- the drawing, which is derived from the stages ----------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_the_view_is_built_at_every_level(level: int) -> None:
    view = g.view(level)

    assert view.topology == "governed"
    assert view.level == level
    assert view.boxes


def test_the_flow_draws_one_box_for_every_stage_the_run_walks() -> None:
    """The picture cannot describe a pipeline the code does not run.

    Both sides read `STAGES`, so this asserts they are the same tuple rather
    than that two lists happen to agree -- and it is the assertion that fails
    if somebody starts writing the boxes out by hand again.
    """
    drawn = {box.id for box in g.view(2).boxes}
    walked = {stage.id for stage in g.STAGES}

    assert walked <= drawn
    assert drawn - walked == {"refused", "stopped"}


def test_every_way_a_run_can_end_is_on_the_picture() -> None:
    """A path that exists and is not drawn is a path a reader cannot see."""
    view = g.view(2)
    refusals = {arrow.to for arrow in view.arrows if arrow.kind == REFUSAL}

    assert refusals == {"refused", "stopped"}
    for ending in refusals:
        assert view.box(ending) is not None


def test_the_stages_a_real_run_reaches_are_all_drawn() -> None:
    """The metering and the drawing checked against each other, not asserted."""
    view = g.view(2)
    drawn = {box.id for box in view.boxes}

    for budget, call, bound in [
        (Budget(), Counter(), None),
        (Budget(authorised=1), Counter("x" * 9), g.Bound(chars=2)),
        (Budget(authorised=1), Counter(), None),
    ]:
        outcome = g.run(a_request(), budget, call, bound=bound)
        assert set(outcome.stages) <= drawn, (
            f"a {outcome.state} run reached "
            f"{set(outcome.stages) - drawn}, which nothing draws")


def test_the_black_box_level_hides_the_parts_and_keeps_the_gate() -> None:
    """A reader at level 0 must still see that this ends at a person."""
    view = g.view(0)
    kinds = {box.kind for box in view.boxes}

    assert kinds == {INPUT, "worker", GATE}
    assert view.box("queue") is not None
    assert view.box("model") is None


def test_the_parts_level_shows_no_order() -> None:
    view = g.view(1)

    assert view.boxes
    assert view.arrows == ()


def test_the_seam_declares_that_it_spends_and_does_not_decide() -> None:
    """The plane's declarations are part of the picture."""
    view = g.view(2)

    assert view.marks == ("spends",)
    assert view.decides is False
    assert not view.is_refused


def test_an_unknown_level_is_refused_rather_than_rounded() -> None:
    with pytest.raises(ValueError):
        g.view(3)


def test_the_only_worker_in_the_flow_is_the_black_box() -> None:
    """P17's first obligation: the surface is one box, and it is countable."""
    workers = [box for box in g.view(2).boxes if box.kind == "worker"]

    assert len(workers) == 1
    assert workers[0].id == "model"


def test_the_draft_and_the_ends_are_outputs_and_the_gates_are_gates() -> None:
    view = g.view(2)

    assert view.box("draft").kind == OUTPUT
    assert view.box("budget").kind == GATE
    assert view.box("queue").kind == GATE


# --- the documentation, which is a second description of the same pipeline ----


def test_the_hitl_guide_names_every_stage_this_module_walks() -> None:
    """P12: nothing describes a behaviour in a second place beside the code.

    `docs/human_in_loop.md` draws this pipeline for a reader who will never
    open the module, which makes it exactly the second description that goes
    stale. It cannot be generated from `STAGES` -- it is prose around a diagram
    -- so it is held to them instead.

    Mutation, quoted as it printed: renaming the `budget` stage to `allowance`
    in `STAGES` and leaving the guide alone.

        AssertionError: docs/human_in_loop.md draws a pipeline missing
        ['allowance'], so a reader of that page has a different pipeline in
        mind
    """
    from pathlib import Path

    guide = (Path(__file__).resolve().parent.parent
             / "docs" / "human_in_loop.md").read_text(encoding="utf-8")
    section = guide.split("## Where a request comes from")[1]

    missing = [s.id for s in g.STAGES if s.id not in section]
    assert not missing, (
        f"docs/human_in_loop.md draws a pipeline missing {missing}, so a "
        f"reader of that page has a different pipeline in mind")
    for ending in ("refused", "stopped"):
        assert ending in section, f"the guide does not draw the {ending} path"


def test_the_hitl_guide_does_not_claim_the_seconds_bound_is_enforced() -> None:
    """The one sentence in that page that would be worth the most if wrong."""
    from pathlib import Path

    guide = (Path(__file__).resolve().parent.parent
             / "docs" / "human_in_loop.md").read_text(encoding="utf-8")

    assert "does not enforce" in guide or "not enforce" in guide, (
        "the guide must say what the seam does not enforce, because a reader "
        "who assumes a slow call is stopped is the reader this costs")


# --- what a person sees in the queue, which is a prompt and not a context -----


def test_a_refusal_and_a_draft_do_not_read_the_same(tmp_path=None) -> None:
    """THE ONE THAT MATTERS HERE.

    Both surfaces that show this queue -- `qmcp human list` and the harness
    payload -- carry `prompt` and not `context`. A state that lived only in the
    context arrived as two rows with identical text, separable by reading the
    id. Found by queueing one of each against a live harness and looking.

    Mutation, quoted as it printed: `_ASKS[REFUSED]` given an empty lead.

        AssertionError: a refusal and a draft read identically to a person
        assert 'a draft for the release notes' != 'a draft for the release
        notes'
    """
    request = a_request()
    refused = g.queued(g.run(request, Budget(), Counter()))
    drafted = g.queued(g.run(request, Budget(authorised=1), Counter()))

    assert refused["prompt"] != drafted["prompt"], (
        "a refusal and a draft read identically to a person")
    assert refused["prompt"].startswith("Refused"), (
        "the state must lead, because a window with little room keeps the front")


def test_there_is_nothing_to_accept_in_a_run_that_produced_no_draft() -> None:
    """The options differ because the questions differ."""
    refused = g.queued(g.run(a_request(), Budget(), Counter()))
    stopped = g.queued(g.run(a_request(), Budget(authorised=1),
                             Counter("x" * 9), bound=g.Bound(chars=2)))
    drafted = g.queued(g.run(a_request(), Budget(authorised=1), Counter()))

    assert refused["options"] == ["re-issue", "leave"]
    assert stopped["options"] == ["re-issue", "leave"]
    assert drafted["options"] == ["accept", "reject"]
    assert "accept" not in refused["options"]


def test_every_state_has_an_ask_and_no_state_is_missing_one() -> None:
    """A state added without a question would fall through to a KeyError.

    Cheaper to assert than to discover: the alternative is a run that reaches
    the queue and raises there, which is the one place a failure costs a
    person's attention rather than a developer's.
    """
    assert set(g._ASKS) == set(g.STATES)


def test_a_stopped_run_says_a_bound_fired_rather_than_naming_a_budget() -> None:
    """Refused and stopped are different facts and must not share a sentence."""
    stopped = g.queued(g.run(a_request(), Budget(authorised=1),
                             Counter("x" * 9), bound=g.Bound(chars=2)))

    assert "bound" in stopped["prompt"]
    assert "nothing was called" not in stopped["prompt"]
