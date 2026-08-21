"""Which projects a thread is about, and the line between measuring and claiming.

The test worth reading is the first one: evidence and claim move independently.
Changing the rule must change what is claimed and leave every measurement
exactly as it was, because that is what makes the claim arguable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qmcp.threads.base import Thread, Turn, thread_name
from qmcp.threads.consolidate import (
    CROSSES,
    DEFAULT_MIN_TURNS,
    PART_OF,
    Reading,
    about,
    consolidate,
    mentions,
    relations_for,
    roster,
)

NAMES = {"dossier": "quaternionmedia/dossier", "qmcp": "quaternionmedia/qmcp",
         "rad": "quaternionmedia/rad", "alfred": "quaternionmedia/alfred"}


def thread(*texts, title=None, id="abc"):
    return Thread(id=id, title=title,
                  turns=tuple(Turn(id=f"t{i}", role="human", text=t)
                              for i, t in enumerate(texts)))


# --- measuring is not claiming ------------------------------------------------


def test_the_rule_changes_the_claim_and_not_the_evidence():
    """THE ONE THAT MATTERS.

    A reading carries both. If the rule could alter what was measured, nobody
    could check the claim against anything -- the evidence would already agree
    with whatever was concluded.

    Mutation: filter `evidence` down to the chosen projects and this fails.
    """
    conversation = thread("dossier here", "and dossier again", "rad once")

    strict = about(conversation, NAMES, min_turns=2)
    loose = about(conversation, NAMES, min_turns=1)

    assert strict.projects == ("dossier",)
    assert set(loose.projects) == {"dossier", "rad"}
    assert strict.evidence == loose.evidence, "the measurement moved with the rule"


def test_a_reading_says_which_rule_produced_it():
    """A verdict without its rule is one a person cannot argue with."""
    found = about(thread("dossier", "dossier"), NAMES, min_turns=2)
    assert "2 turns" in found.rule


def test_mentions_records_which_turns_so_somebody_can_look():
    """A count nobody can check is a number to be believed rather than read."""
    found = mentions(thread("about dossier", "unrelated", "dossier again"), NAMES)
    by_name = {m.project: m for m in found}
    assert by_name["dossier"].turns == ("t0", "t2")
    assert by_name["dossier"].total == 2
    assert by_name["dossier"].turn_count == 2


# --- what counts as being about something -------------------------------------


def test_the_title_is_stronger_evidence_than_a_passing_mention():
    """A title is a person's own summary of what the conversation was about, so
    one occurrence there outweighs the turn threshold.

    Mutation: drop the `in_title` branch and this fails.
    """
    found = about(thread("we touched it once: dossier",
                         title="Dossier rewrite"), NAMES)
    assert found.projects == ("dossier",)


def test_one_passing_mention_is_not_enough_on_its_own():
    """The difference between a conversation that referenced a repository and
    one that was about it."""
    found = about(thread("this reminds me of dossier", "anyway, lunch"), NAMES)
    assert found.is_unknown
    assert found.evidence, "the mention was still measured"


def test_a_thread_about_nothing_is_unknown_rather_than_placed():
    """Most conversations in a personal archive are not about this
    organisation. Guessing a home for them fills sixteen boards with noise, and
    the guess is indistinguishable from a finding.

    Mutation: fall back to a default project and this fails.
    """
    found = about(thread("what temperature for a roast chicken"), NAMES)
    assert found.is_unknown
    assert found.projects == ()
    assert found.relation is None


def test_unknown_is_not_the_same_as_no_evidence():
    """Two different facts: nothing was named, versus nothing named enough."""
    nothing = about(thread("entirely unrelated"), NAMES)
    almost = about(thread("dossier came up once"), NAMES)
    assert nothing.is_unknown and almost.is_unknown
    assert nothing.evidence == ()
    assert almost.evidence != ()


# --- names are words, not substrings ------------------------------------------


def test_a_name_inside_a_longer_word_is_not_a_mention():
    """`rad` lives inside `gradient`, `radius` and `radical`. A substring match
    reports a conversation about colour as a conversation about the menu
    library.

    Mutation: drop the word boundaries and this fails.
    """
    found = about(thread("a radical gradient with a wide radius",
                         "more radians"), NAMES)
    assert found.is_unknown, found.evidence


def test_a_name_is_matched_whatever_its_case():
    found = mentions(thread("Dossier", "DOSSIER"), NAMES)
    assert found and found[0].project == "dossier"
    assert found[0].turn_count == 2


# --- how a thread joins what it is about --------------------------------------


def test_one_project_is_part_of_it():
    found = about(thread("dossier", "dossier again"), NAMES)
    assert found.relation == PART_OF


def test_several_projects_cross_rather_than_belonging_to_each():
    """`part-of` twice would claim the conversation belongs wholly to each of
    two repositories, which is the shape that made a hierarchy the wrong model.

    Mutation: return `part-of` for the multi-project case and this fails.
    """
    found = about(thread("dossier and qmcp", "dossier and qmcp again"), NAMES)
    assert len(found.projects) == 2
    assert found.relation == CROSSES


def test_the_relations_name_addresses_on_both_sides():
    """A relation joins addresses rather than rows, so one side may name a
    delta that does not exist yet."""
    conversation = thread("dossier", "dossier", id="xyz")
    found = about(conversation, NAMES)
    payloads = relations_for(conversation, found, project_of=NAMES)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["source"].endswith(thread_name(conversation))
    assert payload["target"].startswith("quaternionmedia/dossier/delta/")
    assert payload["relation"] == PART_OF


def test_a_relation_carries_the_rule_that_produced_it():
    """Somebody who disagrees with a relation should not have to guess where it
    came from."""
    conversation = thread("qmcp", "qmcp")
    payload = relations_for(conversation, about(conversation, NAMES),
                            project_of=NAMES)[0]
    assert "consolidate" in payload["stated_by"]
    assert "turns" in payload["stated_by"]


def test_an_unknown_thread_produces_no_relations():
    conversation = thread("nothing to do with any of it")
    assert relations_for(conversation, about(conversation, NAMES),
                         project_of=NAMES) == []


# --- the set of them ----------------------------------------------------------


def test_the_summary_reports_what_was_not_placed_first():
    """A consolidator that printed only its hits would read as though it had
    placed everything.

    Mutation: omit the unknown count and this fails.
    """
    found = consolidate([
        thread("dossier", "dossier", id="a"),
        thread("lunch", id="b"),
        thread("nothing", id="c"),
    ], NAMES)
    assert len(found.unknown) == 2
    assert "about none of them" in found.summary()
    assert "3 thread(s) read" in found.summary()


def test_the_overlap_is_reported_per_project():
    found = consolidate([
        thread("dossier", "dossier", id="a"),
        thread("dossier and qmcp", "dossier and qmcp", id="b"),
    ], NAMES)
    overlap = found.by_project()
    assert sorted(overlap["dossier"]) == ["a", "b"]
    assert overlap["qmcp"] == ["b"]


def test_crossing_threads_are_counted_separately():
    found = consolidate([thread("dossier and rad", "dossier and rad", id="x")],
                        NAMES)
    assert len(found.crossing) == 1


# --- the roster comes from the corpus -----------------------------------------


def test_the_roster_is_read_from_the_embedded_corpus():
    """Asking the control panel would make the harness depend on the panel for
    a list the corpus publishes.

    Skipped rather than failed where the submodule is not checked out, because
    that is a state of the clone.
    """
    corpus = Path("governance/qm")
    if not (corpus / "ci" / "workspace.yaml").is_file():
        pytest.skip("governance/qm is not checked out in this clone")

    found = roster(corpus)
    assert "dossier" in found
    assert found["dossier"] == "quaternionmedia/dossier"
    assert len(found) > 5


def test_the_default_threshold_is_a_stated_number():
    """A threshold buried in a comparison is a decision nobody can find."""
    assert DEFAULT_MIN_TURNS == 2


def test_a_reading_is_frozen():
    with pytest.raises(Exception):
        Reading(thread="a").thread = "b"


# --- an inventory is not a crossing -------------------------------------------


def test_a_thread_naming_most_of_the_roster_is_a_survey_not_a_crossing():
    """THE ONE THE REAL ARCHIVE FOUND.

    `crosses` means both must happen, they interact at one point, and neither
    contains the other. A status sweep names every repository in most of its
    turns and measures identically -- on the real archive one thread came out
    as crossing eleven projects of thirteen, which it plainly did not do.

    Reported as its own kind rather than dropped: a workspace sweep is a real
    thing somebody did.

    Mutation: remove the survey branch and this fails.
    """
    from qmcp.threads.consolidate import MIN_SURVEY_PROJECTS

    wide = {f"repo{i}": f"quaternionmedia/repo{i}" for i in range(8)}
    body = " ".join(wide)
    found = about(thread(body, body), wide)

    assert len(found.projects) >= MIN_SURVEY_PROJECTS
    assert found.surveys_the_roster is True
    assert found.relation is None, "a survey was given a relation"
    assert "survey of the workspace" in found.rule


def test_a_genuine_crossing_of_two_is_still_a_crossing():
    """The guard must not swallow the case it was protecting.

    Mutation: drop the `MIN_SURVEY_PROJECTS` floor and this fails, because two
    of four is half the roster.
    """
    found = about(thread("dossier and qmcp", "dossier and qmcp"), NAMES)
    assert found.surveys_the_roster is False
    assert found.relation == CROSSES


def test_a_survey_is_counted_apart_from_crossings():
    wide = {f"repo{i}": f"quaternionmedia/repo{i}" for i in range(8)}
    body = " ".join(wide)
    found = consolidate([
        thread(body, body, id="sweep"),
        thread("repo0 and repo1", "repo0 and repo1", id="real"),
    ], wide)

    assert [r.thread for r in found.surveys] == ["sweep"]
    assert [r.thread for r in found.crossing] == ["real"]
    assert "survey the workspace" in found.summary()


def test_a_survey_produces_no_relations():
    """Every relation in this vocabulary says something specific about two
    pieces of work. "Was mentioned in a list of everything" is not one."""
    wide = {f"repo{i}": f"quaternionmedia/repo{i}" for i in range(8)}
    body = " ".join(wide)
    conversation = thread(body, body, id="sweep")
    found = about(conversation, wide)
    assert relations_for(conversation, found, project_of=wide) == []
