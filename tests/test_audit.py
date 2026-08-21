"""Answers that two runs over the same rows agree about.

THE TESTS WORTH READING ARE THE FIRST THREE. Determinism is not a property you
add; it is three specific ways an answer stops being reproducible -- a clock, a
default, and a silent omission -- and each one has a test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from qmcp.audit import (
    OPEN_PHASES,
    UNKNOWN,
    Window,
    compare,
    in_flight,
    model_of,
    models_run,
)

UTC = timezone.utc


def invocation(when, model=None, status="SUCCESS", tool="planner"):
    params = {"goal": "x"}
    if model:
        params["model"] = model
    return {"tool_name": tool, "input_params": params,
            "status": status, "created_at": when}


# --- the three ways an answer stops being reproducible ------------------------


def test_a_window_must_be_stated_in_utc():
    """THE CLOCK. "Today" is a fact about a timezone, not about the world. A
    naive datetime would take its meaning from wherever it was read.

    Mutation: accept naive datetimes and this fails.
    """
    with pytest.raises(ValueError):
        Window(datetime(2026, 8, 21), datetime(2026, 8, 22))


def test_a_day_window_is_named_by_the_caller_and_reads_no_clock():
    """THE DEFAULT. A window filled in from the wall clock returns different
    rows on two runs, so `Window.day` takes the day.

    Mutation: default `on` to `date.today()` and this stops being assertable --
    which is the point.
    """
    window = Window.day(date(2026, 8, 21))
    assert window.start == datetime(2026, 8, 21, tzinfo=UTC)
    assert window.end == datetime(2026, 8, 22, tzinfo=UTC)
    assert Window.day(date(2026, 8, 21)) == window, "two calls, one answer"


def test_an_invocation_with_no_model_is_counted_as_unknown():
    """THE SILENT OMISSION. Dropping it would report two models where fifty-five
    invocations ran, and the missing fifty-three would look like nothing.

    Measured on the real database: 55 of 55 recorded no model.

    Mutation: skip rows with no model and this fails.
    """
    rows = [invocation(datetime(2026, 8, 21, 10, tzinfo=UTC)),
            invocation(datetime(2026, 8, 21, 11, tzinfo=UTC), model="qwen")]
    report = models_run(rows, Window.day(date(2026, 8, 21)))

    assert report.rows_read == 2
    assert report.unrecorded == 1
    assert {r.model for r in report.runs} == {UNKNOWN, "qwen"}
    assert "unknown" in report.summary()


# --- windows ------------------------------------------------------------------


def test_the_window_is_half_open_so_consecutive_days_neither_overlap_nor_drop():
    """A closed interval double-counts a row landing exactly on midnight --
    rare, real, and invisible until two windows disagree by one.

    Mutation: make the end inclusive and this fails.
    """
    midnight = datetime(2026, 8, 22, 0, 0, tzinfo=UTC)
    first = Window.day(date(2026, 8, 21))
    second = Window.day(date(2026, 8, 22))

    assert first.holds(midnight) is False
    assert second.holds(midnight) is True


def test_a_window_that_ends_before_it_starts_is_refused():
    with pytest.raises(ValueError):
        Window(datetime(2026, 8, 22, tzinfo=UTC), datetime(2026, 8, 21, tzinfo=UTC))


def test_rows_outside_the_window_are_not_read():
    rows = [invocation(datetime(2026, 8, 20, 23, tzinfo=UTC)),
            invocation(datetime(2026, 8, 21, 1, tzinfo=UTC))]
    assert models_run(rows, Window.day(date(2026, 8, 21))).rows_read == 1


def test_a_naive_stored_timestamp_is_read_as_utc():
    """The column is written in UTC. Reading it as local shifts every row by
    the offset, which on this machine is four hours -- enough to move an
    invocation into the wrong day."""
    rows = [invocation(datetime(2026, 8, 21, 2, 0))]
    assert models_run(rows, Window.day(date(2026, 8, 21))).rows_read == 1


def test_the_summary_names_the_window_before_any_figure():
    """A count without a window is not an answer to anything."""
    summary = models_run([], Window.day(date(2026, 8, 21))).summary()
    assert summary.startswith("2026-08-21T00:00Z")


# --- where a model comes from -------------------------------------------------


def test_nothing_is_inferred_from_a_tool_name():
    """`planner` is a tool, not a model. Mapping one to the other would be this
    module inventing the fact it was asked to report.

    Mutation: fall back to the tool name and this fails.
    """
    assert model_of({"goal": "x"}) == UNKNOWN


def test_a_model_is_read_from_any_of_the_recognised_keys():
    assert model_of({"model": "qwen2.5-coder:7b"}) == "qwen2.5-coder:7b"
    assert model_of({"model_name": "a"}) == "a"
    assert model_of({"model_id": "b"}) == "b"


def test_params_stored_as_json_text_are_read():
    """The column is JSON in the database and a dict in memory. Both arrive."""
    assert model_of('{"model": "qwen"}') == "qwen"
    assert model_of("not json at all") == UNKNOWN
    assert model_of(None) == UNKNOWN


def test_a_blank_model_is_unknown_rather_than_a_model_called_nothing():
    assert model_of({"model": "   "}) == UNKNOWN


def test_failures_are_counted_against_the_model_that_ran():
    rows = [invocation(datetime(2026, 8, 21, 1, tzinfo=UTC), "m", "SUCCESS"),
            invocation(datetime(2026, 8, 21, 2, tzinfo=UTC), "m", "FAILED")]
    run = models_run(rows, Window.day(date(2026, 8, 21))).runs[0]
    assert run.invocations == 2 and run.failures == 1


def test_first_and_last_bracket_the_window_of_actual_use():
    rows = [invocation(datetime(2026, 8, 21, h, tzinfo=UTC), "m")
            for h in (9, 3, 17)]
    run = models_run(rows, Window.day(date(2026, 8, 21))).runs[0]
    assert run.first.hour == 3 and run.last.hour == 17


# --- what is in flight --------------------------------------------------------


def test_only_open_phases_are_in_flight():
    rows = [{"name": "a", "phase": "review"},
            {"name": "b", "phase": "complete"},
            {"name": "c", "phase": "abandoned"}]
    found = in_flight(rows, datetime(2026, 8, 21, tzinfo=UTC))
    assert [i.delta for i in found.items] == ["a"]


def test_the_open_phases_are_listed_rather_than_derived_from_not_complete():
    """A phase added later should be a decision somebody makes here, not a
    silent reclassification of something into flight."""
    assert "complete" not in OPEN_PHASES
    assert "abandoned" not in OPEN_PHASES
    assert "review" in OPEN_PHASES


def test_a_phase_is_matched_whatever_its_case():
    """The panel stores `REVIEW`; the corpus writes `review`."""
    found = in_flight([{"name": "a", "phase": "REVIEW"}],
                      datetime(2026, 8, 21, tzinfo=UTC))
    assert len(found.items) == 1


def test_a_delta_with_no_harness_recorded_is_unattributed_not_dropped():
    """Measured on the real database: 51 of 59 in flight had no harness. A
    report showing only the 8 would describe a tenth of the work.

    Mutation: filter to rows with a harness and this fails.
    """
    rows = [{"name": "a", "phase": "review", "harness": "reviewer"},
            {"name": "b", "phase": "review"}]
    found = in_flight(rows, datetime(2026, 8, 21, tzinfo=UTC))

    assert len(found.items) == 2
    assert [i.delta for i in found.unattributed] == ["b"]
    assert "no harness recorded" in found.summary()


def test_the_time_of_the_reading_is_passed_rather_than_read():
    """An answer that depends on when it was asked cannot be checked against
    itself."""
    at = datetime(2026, 8, 21, 12, tzinfo=UTC)
    assert in_flight([], at).at == at


# --- the bridge ---------------------------------------------------------------


def test_agreement_across_the_pair_is_an_empty_list():
    assert compare({"threads": 237}, {"threads": 237}) == []


def test_a_disagreement_names_both_sides_and_picks_no_winner():
    """Deciding which side is right needs somebody who can look at both --
    exactly the split that had the panel reporting an absent archive while the
    harness served 203 threads.

    Mutation: return the harness's figure and this fails.
    """
    found = compare({"threads": 237}, {"threads": 203})
    assert len(found) == 1
    assert found[0].harness_says == 237
    assert found[0].panel_says == 203


def test_a_figure_only_one_side_reports_is_a_disagreement():
    """A bridge that drops a field is a bridge that lost it. Comparing only
    the shared keys would report that as agreement.

    Mutation: intersect the keys instead of uniting them and this fails.
    """
    found = compare({"threads": 1, "deltas": 5}, {"threads": 1})
    assert [d.figure for d in found] == ["deltas"]
    assert found[0].panel_says == "not reported"


def test_comparing_beyond_what_a_route_carries_is_a_category_error():
    """The rule above -- a one-sided figure is a loss -- is right for a figure
    the route promised and wrong for one it never did.

    The first real bridge run compared the harness's invocation count against
    the threads route and reported a disagreement: correct by the rule, wrong
    about the world. The remedy is the caller's, so this pins the shape of it
    rather than changing `compare`.
    """
    carried = ("threads", "diverged")
    harness = {"threads": 237, "diverged": 0, "invocations": 55}
    panel = {"threads": 237, "diverged": 0}

    assert compare(harness, panel), "unscoped, the extra figure disagrees"
    scoped = compare({k: v for k, v in harness.items() if k in carried},
                     {k: v for k, v in panel.items() if k in carried})
    assert scoped == [], "scoped to the route, the two sides agree"


def test_recording_a_model_makes_it_findable_afterwards():
    """The remedy for `unknown`, end to end: record it, then read it back.

    Mutation: write to a key `model_of` does not read and this fails.
    """
    from qmcp.audit import record_model

    assert model_of(record_model({"goal": "x"}, "qwen2.5-coder:7b")) == "qwen2.5-coder:7b"


def test_recording_nothing_leaves_the_key_absent_rather_than_writing_unknown():
    """An absent key and a recorded "unknown" read the same today. Only the
    absent one stays true if somebody later learns which model it was.

    Mutation: write the string "unknown" and this fails.
    """
    from qmcp.audit import record_model

    assert record_model({"goal": "x"}, None) == {"goal": "x"}
    assert record_model({"goal": "x"}, "  ") == {"goal": "x"}


def test_recording_does_not_mutate_what_it_was_given():
    """A recorder that edited the caller's dict would change an invocation's
    parameters as a side effect of describing them."""
    from qmcp.audit import record_model

    original = {"goal": "x"}
    record_model(original, "m")
    assert original == {"goal": "x"}
