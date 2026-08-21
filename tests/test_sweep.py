"""Dispatching a sweep: a worker per shape, and nothing chosen in advance.

The test worth reading is the last one in the first section: a share whose shape
has no worker is reported and queued, never dropped. A sweep that quietly
omitted fifteen of twenty-four repositories would look finished.
"""

from __future__ import annotations

import pytest

from qmcp.sweep import (
    DONE,
    FAILED,
    HUMAN,
    JUDGEMENT,
    MECHANICAL,
    NEEDS_WORKER,
    REFUSED,
    UNKNOWN,
    Outcome,
    branch_for,
    mechanical_worker,
    run,
)


def share(project="org/a", shape=MECHANICAL, declared=">=0.100.0", why=""):
    return {"project": project, "shape": shape, "declared": declared, "why": why}


# --- the topology comes from the work -----------------------------------------


def test_the_mix_of_workers_follows_the_shares(monkeypatch):
    """THE ARCHITECTURE, ASSERTED.

    Nothing chooses "one agent per repository" or "one for all". Nine
    mechanical shares and six judgement ones produce nine parser runs and six
    questions, and the same dispatcher produces a different mix tomorrow
    without changing.

    Mutation: hard-code a single worker and this fails.
    """
    shares = ([share(f"org/m{i}") for i in range(9)]
              + [share(f"org/j{i}", shape=JUDGEMENT) for i in range(6)])

    done = run(shares, "0.116.0")
    assert len(done.ready) == 9
    assert len(done.waiting) == 6
    assert len(done.outcomes) == 15, "every share got an outcome"


def test_a_shape_with_no_worker_is_queued_and_named():
    """THE ONE THAT MATTERS.

    Dropping it would leave the sweep looking finished with two thirds of the
    repositories untouched.

    Mutation: skip unregistered shapes and this fails.
    """
    done = run([share("org/x", shape=UNKNOWN)], "1.0.0")
    assert len(done.outcomes) == 1
    assert done.outcomes[0].state == NEEDS_WORKER
    assert "unknown" in done.outcomes[0].detail


def test_a_different_worker_changes_what_runs_and_not_the_dispatcher():
    """Registering a model for `judgement` is a deployment decision. The
    dispatcher does not know or care which it got."""
    def stands_in(share, to_version):
        return Outcome(share["project"], DONE, "a worker looked at it")

    done = run([share("org/j", shape=JUDGEMENT)], "1.0.0",
               workers={JUDGEMENT: stands_in})
    assert done.ready and done.ready[0].detail == "a worker looked at it"


# --- the mechanical worker ----------------------------------------------------


def test_it_prepares_the_edit_and_does_not_apply_it():
    """The sweep is approved whole. A worker that wrote to a repository before
    anybody saw the batch would commit to the part before the whole was
    decided.

    Mutation: have the worker write a file and this stops being assertable --
    which is the point of returning the edit instead.
    """
    outcome = mechanical_worker(share(declared=">=0.100.0"), "0.116.0")
    assert outcome.state == DONE
    assert outcome.edit == ">=0.116.0"
    assert "->" in outcome.detail


def test_it_keeps_the_operator_it_found():
    assert mechanical_worker(share(declared="~=0.95"), "1.0.0").edit == "~=1.0.0"


def test_it_refuses_two_constraints_rather_than_flattening_them():
    """A ceiling somebody put there on purpose."""
    outcome = mechanical_worker(share(declared="<1.0.0,>=0.92.0"), "0.116.0")
    assert outcome.state == REFUSED


def test_it_refuses_what_it_cannot_parse():
    assert mechanical_worker(share(declared="whatever"), "1.0.0").state == REFUSED
    assert mechanical_worker(share(declared=None), "1.0.0").state == REFUSED


# --- what is never dispatched -------------------------------------------------


def test_a_human_share_is_refused_on_purpose():
    """Some acts change meaning when a machine does them. A dispatcher that
    could do them would make the registry a description of what it chose not
    to do.

    Mutation: register a worker that completes `human` shares and this fails.
    """
    done = run([share("org/h", shape="human")], "1.0.0")
    assert done.outcomes[0].state == REFUSED
    assert "constitution" in done.outcomes[0].detail


# --- one bad share does not take the sweep ------------------------------------


def test_a_worker_that_throws_loses_one_share_and_not_the_rest():
    """The other twenty-three are still worth preparing.

    Mutation: let the exception propagate and this fails.
    """
    def explodes(share, to_version):
        raise RuntimeError("the manifest was a directory")

    done = run([share("org/a"), share("org/bad", shape=JUDGEMENT), share("org/c")],
               "1.0.0", workers={MECHANICAL: mechanical_worker,
                                 JUDGEMENT: explodes})
    states = {o.project: o.state for o in done.outcomes}
    assert states["org/bad"] == FAILED
    assert states["org/a"] == DONE and states["org/c"] == DONE


def test_a_failure_names_what_went_wrong():
    def explodes(share, to_version):
        raise ValueError("no such file")

    done = run([share("org/bad")], "1.0.0", workers={MECHANICAL: explodes})
    assert "ValueError" in done.outcomes[0].detail
    assert "no such file" in done.outcomes[0].detail


# --- what a person is handed --------------------------------------------------


def test_the_batch_is_what_is_ready_and_the_queue_is_what_is_not():
    done = run([share("org/a"), share("org/b"),
                share("org/j", shape=JUDGEMENT)], "1.0.0")
    assert sorted(o.project for o in done.ready) == ["org/a", "org/b"]
    assert [o.project for o in done.waiting] == ["org/j"]


def test_the_summary_counts_every_state():
    done = run([share("org/a"), share("org/j", shape=JUDGEMENT),
                share("org/h", shape="human")], "1.0.0")
    summary = done.summary()
    assert "3 share(s)" in summary
    for word in (DONE, NEEDS_WORKER, REFUSED):
        assert word in summary


# --- the branch ---------------------------------------------------------------


def test_every_repository_uses_the_same_branch_name():
    """One sweep, one branch name, so a person checking twenty-four
    repositories is checking one thing."""
    first = branch_for("fastapi", "0.116.0")
    assert first == branch_for("fastapi", "0.116.0")
    assert first.startswith("evolve/"), "a sweep is org-level work"
    assert "fastapi" in first and "0.116.0" in first


# --- the vocabulary is shared across the seam ---------------------------------


def test_the_shapes_match_the_ones_the_panel_states():
    """Copied across the seam rather than imported, because the two
    repositories do not depend on each other -- so something has to check that
    the copies still agree.

    Skipped where the panel is not beside this clone, since that is a state of
    the machine rather than a disagreement.
    """
    import pathlib
    import re as _re

    here = pathlib.Path(__file__).resolve()
    panel = None
    for parent in here.parents:
        candidate = parent / "dossier" / "src" / "dossier" / "sweep.py"
        if candidate.is_file():
            panel = candidate
            break
    if panel is None:
        pytest.skip("dossier is not beside this clone")

    text = panel.read_text(encoding="utf-8")
    for name, value in (("MECHANICAL", MECHANICAL), ("JUDGEMENT", JUDGEMENT),
                        ("HUMAN", HUMAN),
                        ("UNKNOWN", UNKNOWN)):
        found = _re.search(rf'^{name} = "([^"]+)"', text, _re.MULTILINE)
        assert found, f"{name} is not declared in the panel's sweep module"
        assert found.group(1) == value, (
            f"{name}: panel says {found.group(1)!r}, harness says {value!r}")
