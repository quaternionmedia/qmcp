"""Which project an instruction is for, and the line between a match and a guess.

Every test here was seen red first. The mutation for each is in its docstring,
applied to `qmcp.instructions.resolve` and restored.
"""

from __future__ import annotations

from qmcp.db.models import InstructionStatus
from qmcp.instructions import (
    RULE_NO_ROSTER,
    RULE_NONE,
    RULE_ONE,
    RULE_SEVERAL,
    RULE_STATED,
    resolve,
    roster_names,
)

NAMES = ("qmcp", "rad", "vox", "dossier")


def test_one_project_named_resolves_to_it():
    """Mutation: return `Resolution(None, named, RULE_ONE)` for one match -- red."""
    found = resolve("Deploy qmcp to the pi", NAMES)

    assert found.project == "qmcp"
    assert found.candidates == ("qmcp",)
    assert found.rule == RULE_ONE
    assert found.status is InstructionStatus.RECORDED


def test_no_project_named_is_unresolved_with_no_candidates():
    """Mutation: fall through to `named[0]` when nothing matched -- IndexError."""
    found = resolve("Rotate the logs", NAMES)

    assert found.project is None
    assert found.candidates == ()
    assert found.rule == RULE_NONE
    assert found.status is InstructionStatus.UNRESOLVED


def test_several_projects_named_is_unresolved_with_them_as_candidates():
    """Mutation: return `named[0]` as the project when several match -- red on
    `project is None`. A conversation naming two repositories is about both,
    and picking the first would be the guess this refuses to make."""
    found = resolve("Move the vectors from vox into qmcp", NAMES)

    assert found.project is None
    assert found.candidates == ("qmcp", "vox")
    assert found.rule == RULE_SEVERAL


def test_a_substring_inside_another_word_is_not_a_match():
    """`rad` lives inside `gradient`. Mutation: match with `name.lower() in
    text.lower()` instead of `_pattern` -- red, project `rad`."""
    found = resolve("Fix the gradient on the radial menu", NAMES)

    assert found.project is None
    assert found.candidates == ()


def test_casing_is_ignored():
    """A transcript capitalises as it pleases. Mutation: drop IGNORECASE from
    `_pattern` -- red."""
    assert resolve("Restart QMCP", NAMES).project == "qmcp"
    assert resolve("restart Dossier please", NAMES).project == "dossier"


def test_a_name_followed_by_punctuation_matches():
    """Whisper ends sentences. Mutation: require a space or end after the
    name -- red on the full stop and the comma."""
    assert resolve("Deploy qmcp.", NAMES).project == "qmcp"
    assert resolve("In vox, add a pause parameter", NAMES).project == "vox"
    assert resolve("What about dossier?", NAMES).project == "dossier"


def test_the_recorded_name_is_the_rosters_spelling():
    """The project is the roster's name, not the transcript's casing of it."""
    assert resolve("DEPLOY QMCP", NAMES).project == "qmcp"


def test_a_stated_project_skips_the_matching():
    """Mutation: resolve the text anyway and prefer the match -- red on `rule`
    and on a text naming a different project than the one stated."""
    found = resolve("Deploy qmcp to the pi", NAMES, project="dossier")

    assert found.project == "dossier"
    assert found.candidates == ()
    assert found.rule == RULE_STATED
    assert found.status is InstructionStatus.RECORDED


def test_no_roster_is_its_own_answer():
    """Nothing to match against is not the same as nothing named."""
    found = resolve("Deploy qmcp", ())

    assert found.project is None
    assert found.rule == RULE_NO_ROSTER


def test_a_hyphenated_name_is_whole():
    """`rad` inside `rad-godot` is not `rad`, which is what the hyphen in
    `_pattern`'s boundary is for."""
    assert resolve("Pin rad-godot to the vectors", NAMES).project is None


def test_the_roster_comes_from_the_checkouts_own_corpus(tmp_path):
    """Read beside the package, not from the working directory; and empty,
    not an error, where the submodule is absent."""
    assert roster_names(tmp_path) == ()
    names = roster_names()
    if names:
        assert "qmcp" in names
