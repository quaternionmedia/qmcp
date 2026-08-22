"""qmcp pointed at its own repository, and what it declines to conclude.

Nothing here runs a real subprocess. `run_check` is the one function that does,
and it is exercised by running the command for real rather than by asserting on
a mocked `subprocess.run` -- a mock of the thing under test would only prove the
mock. What these cover is the reasoning around it: which findings become units
of work, what phase they open at, and what identity they carry across runs.
"""

from __future__ import annotations

from qmcp.selfcheck import (
    BRAINSTORM,
    PLANNING,
    Finding,
    ask,
    checks,
    delta_name,
    last_line,
    phase_of,
    render,
    to_delta,
)

PROJECT = "quaternionmedia/qmcp"


def finding(check: str = "tag-claims", ok: bool = False, detail: str = "1 of 1 failed") -> Finding:
    return Finding(
        check=check, what="whether that run supports a release claim", ok=ok,
        detail=detail, invocation_id="abc-123",
        address=f"{PROJECT}/invocation/abc-123", duration_ms=140,
    )


# --- which findings are work ------------------------------------------------


def test_a_failing_check_becomes_a_delta():
    payload = to_delta(finding(), PROJECT)
    assert payload["delta"]["name"] == "tag-claims-does-not-pass"
    assert payload["project"] == PROJECT


def test_the_delta_carries_dossiers_own_column_names():
    """The consumer writes `ProjectDelta(**payload["delta"])` and translates
    nothing. A key that is not a column there would break that."""
    assert set(to_delta(finding(), PROJECT)["delta"]) == {
        "name", "title", "description", "phase", "delta_type", "priority"}


def test_the_same_failing_check_is_the_same_delta_across_runs():
    """Mutation: put the invocation id in the name and this fails.

    A queue that grows by one row per run is a queue nobody reads -- the same
    reasoning `DRAFT-a-disagreement-is-a-delta.md` 2 gives for identifying a
    divergence by what disagrees rather than by when it was noticed.
    """
    first = to_delta(finding(), PROJECT)["delta"]["name"]
    second = to_delta(
        Finding(check="tag-claims", what="x", ok=False, detail="y",
                invocation_id="a-different-run", address="a/b/invocation/z",
                duration_ms=1), PROJECT)["delta"]["name"]
    assert first == second


def test_the_link_points_back_at_the_invocation_that_found_it():
    links = {link["link_type"]: link["target_name"]
             for link in to_delta(finding(), PROJECT)["links"]}
    assert links["invocation"] == "abc-123"
    assert links["address"] == f"{PROJECT}/delta/tag-claims-does-not-pass"


# --- the phase comes from facts ---------------------------------------------


def test_a_finding_nobody_has_been_asked_about_opens_at_brainstorm():
    """Noticing is not deciding.

    Mutation: return PLANNING unconditionally and this fails -- which is a
    detector asserting that somebody had looked.
    """
    assert phase_of(finding(), answered=False) == BRAINSTORM
    assert to_delta(finding(), PROJECT)["delta"]["phase"] == BRAINSTORM


def test_a_finding_a_human_answered_moves_to_planning():
    assert phase_of(finding(), answered=True) == PLANNING
    assert to_delta(finding(), PROJECT, answered=True)["delta"]["phase"] == PLANNING


def test_nothing_here_reaches_implementation_or_complete():
    """No amount of re-running establishes that anybody acted on a finding.

    `complete` is the value that would flatter, and it is unreachable from
    here -- the same refusal `qmcp/cookbook/delta.py` makes for a step that
    asked for a review and did not get one.
    """
    reachable = {phase_of(finding(), answered=answered) for answered in (True, False)}
    assert reachable == {BRAINSTORM, PLANNING}


# --- what is not work -------------------------------------------------------


def test_a_passing_check_produces_no_delta():
    """A green gate is not a thing anybody has to do.

    Asserted on the delta names rather than on the prose. The first version of
    this test asserted the phrase "unit of work" was absent, and the all-green
    message contains it -- so it was testing the sentence, and would have gone
    red on a rewording that changed no behaviour.
    """
    green = [finding(check="suite", ok=True, detail="413 passed"),
             finding(check="walkthrough", ok=True, detail="1 passed")]
    report = render(green, PROJECT)
    for entry in green:
        assert delta_name(entry) not in report
    assert "No unit of work follows from a green gate" in report


def test_a_report_with_a_failure_names_the_delta_it_produced():
    report = render([finding(check="suite", ok=True, detail="413 passed"), finding()],
                    PROJECT)
    assert "tag-claims-does-not-pass" in report
    assert "1 check(s) did not pass" in report


# --- the question a failure raises ------------------------------------------


def test_a_failure_raises_one_question_with_the_options_named():
    request = ask(finding(), PROJECT)
    assert request.options == ["fix", "accept", "defer"]
    assert "tag-claims" in request.prompt
    assert request.context["delta"] == "tag-claims-does-not-pass"


def test_the_question_has_the_same_id_every_run():
    """Asking again each run would bury the one nobody had answered yet."""
    assert ask(finding(), PROJECT).id == ask(finding(), PROJECT).id


# --- the checks themselves --------------------------------------------------


def test_the_tag_gate_reads_what_the_suite_captured(tmp_path):
    """The ordering is the point of that gate: it judges a captured run rather
    than running one, so it cannot be satisfied by a run nobody kept."""
    suite, tag_claims, _ = checks(tmp_path)
    assert suite.capture is not None
    assert suite.capture in tag_claims.argv
    assert list(checks(tmp_path)).index(suite) < list(checks(tmp_path)).index(tag_claims)


def test_a_check_with_no_output_still_reports_something():
    assert last_line("") == "(no output)"
    assert last_line("first\n\nlast\n") == "last"
