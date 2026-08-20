"""What a command may spend, and the distinction that makes zero meaningful.

`governance/qm/records/DRAFT-no-unattended-spending.md` is the decision. The
tests worth reading are the ones asserting a refusal, and the pair that keeps
`0` from collapsing into `unknown` -- which is the same substitution the harness
payload refuses one seam over.
"""

from __future__ import annotations

import pytest

from qmcp.spend import FREE, Budget, Refused, declare, render, unknown


# --- zero is the default, and it is a real count -----------------------------


def test_a_command_nobody_gave_a_number_to_may_spend_nothing():
    """Not an absence of a budget. A budget of zero."""
    assert Budget().authorised == FREE == 0
    assert Budget().free is True


def test_a_zero_budget_refuses_the_first_call():
    with pytest.raises(Refused) as raised:
        Budget().spend()
    assert "0 authorised" in str(raised.value)


def test_a_refusal_says_how_to_proceed_and_that_nothing_widens_itself():
    """Clause 5: consent does not carry forward, so a refusal points at
    re-issuing rather than at a flag that would remember."""
    with pytest.raises(Refused) as raised:
        Budget(authorised=2, service="a model").spend(3)
    message = str(raised.value)
    assert "Re-issue" in message
    assert "widens a budget" in message
    assert "a model" in message


def test_the_budget_is_checked_before_the_call_not_after():
    """A refusal that arrives after the money is gone is a report.

    Mutation: increment `made` before the comparison and this fails, because
    the third call would be recorded and then refused.
    """
    budget = Budget(authorised=2)
    budget.spend()
    budget.spend()
    with pytest.raises(Refused):
        budget.spend()
    assert budget.made == 2, "the refused call was not recorded as made"


def test_a_negative_budget_is_not_a_smaller_one():
    with pytest.raises(ValueError):
        Budget(authorised=-1)


def test_spending_counts_down_and_stops_at_zero():
    budget = Budget(authorised=3)
    budget.spend(2)
    assert budget.remaining == 1
    with pytest.raises(Refused):
        budget.spend(2)
    assert budget.remaining == 1


# --- zero is never a sentinel for unknown ------------------------------------


def test_a_requirement_nobody_established_is_unknown_and_not_zero():
    """THE LOAD-BEARING PAIR, HALF ONE.

    Reporting `would_need: 0` when nobody could count claims the work is free.
    That is the substitution this organisation refuses at the harness seam, and
    it would arrive here as a much more expensive version of the same mistake.
    """
    declared = declare(Budget(), would_need=None)
    assert isinstance(declared["would_need"], dict)
    assert "unknown" in declared["would_need"]
    assert declared["would_need"] != 0


def test_a_genuine_requirement_of_zero_is_kept_as_zero():
    """THE LOAD-BEARING PAIR, HALF TWO.

    There really is such a thing as no paid work to do, and it must not be
    reported as unknown either. Mutation: treat a falsy `would_need` as unknown
    and this fails.
    """
    declared = declare(Budget(), would_need=0)
    assert declared["would_need"] == 0
    assert "There is no paid work to do" in render(declared)


def test_the_two_are_told_apart_in_what_a_person_reads():
    nothing_to_do = render(declare(Budget(), would_need=0))
    could_not_count = render(declare(Budget(), would_need=unknown("no index yet")))
    assert "no paid work" in nothing_to_do
    assert "unknown" in could_not_count
    assert "not a count of zero" in could_not_count


def test_a_negative_requirement_is_not_a_count():
    with pytest.raises(ValueError):
        declare(Budget(), would_need=-1)


# --- the signal that travels downstream --------------------------------------


def test_a_free_pass_says_so_rather_than_leaving_it_to_be_derived():
    """`made == 0` is also true of an authorised run that found nothing to do.

    Mutation: drop `free_pass` and let a consumer infer it from `made`, and a
    complete run with no work becomes indistinguishable from a partial one.
    """
    partial = declare(Budget(authorised=0), would_need=12)
    complete = declare(Budget(authorised=5), would_need=0)
    assert partial["made"] == complete["made"] == 0
    assert partial["free_pass"] is True
    assert complete["free_pass"] is False


def test_the_declaration_carries_what_a_consumer_needs():
    declared = declare(Budget(authorised=4, service="a model"), would_need=4)
    assert set(declared) == {"authorised", "made", "would_need", "service",
                             "free_pass"}


def test_a_free_pass_tells_a_person_the_number_to_re_issue_against():
    """The whole point of the zero-budget pass: it establishes the count so a
    person can consent to an amount rather than to a category."""
    text = render(declare(Budget(), would_need=37))
    assert "37" in text
    assert "Re-issue" in text
    assert "does not" in text and "carry forward" in text


def test_an_authorised_run_reports_what_it_used():
    budget = Budget(authorised=3, service="a model")
    budget.spend(2)
    text = render(declare(budget, would_need=0))
    assert "2 of 3" in text
    assert "not remembered" in text


# --- what this module is -----------------------------------------------------


def test_this_module_calls_nothing(tmp_path):
    """It is what a paid module passes through, not one itself.

    Asserted on the source, the way dossier asserts its renderer runs no query:
    a module that acquired a client would stop being the thing a reader can
    check quickly.
    """
    from pathlib import Path

    import qmcp.spend as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    for forbidden in ("import requests", "import httpx", "import anthropic",
                      "import openai", "urllib.request"):
        assert forbidden not in source, f"{forbidden} would make this a caller"
