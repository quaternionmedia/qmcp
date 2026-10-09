"""Phrases a person adds to the spoken commands, kept in a file of their own.

THE TEST WORTH READING IS THE CONFLICT ONE. A phrase that already means
something -- another command, an answer, a check, a project's name -- is refused
rather than given a second meaning, because the loop matches the first meaning
it finds and the person would hear the wrong thing happen.

Every test reads the empty phrase file `tests/conftest.py` gives it.
"""

from __future__ import annotations

import json

import pytest

from qmcp.instructions import converse
from qmcp.instructions.converse import NOTHING_RAN, STOPPING
from qmcp.integrations.voice import vocabulary


def _saved(name: str, phrase: str) -> vocabulary.PhraseEdit:
    edit = vocabulary.prepare_add(name, phrase)
    vocabulary.apply_edit(edit)
    return edit


def _journal() -> dict:
    return json.loads(vocabulary.overlay_path().read_text(encoding="utf-8"))


# --- preparing and saving ----------------------------------------------------------


def test_an_add_is_saved_only_when_applied_and_read_from_the_file():
    """Mutation: skip the `os.replace` in `apply_edit` and the saved phrase is
    never read back."""
    edit = vocabulary.prepare_add("stop", "Farewell, now!")
    assert edit.phrase == "farewell now"
    assert not vocabulary.overlay_path().exists()

    vocabulary.apply_edit(edit)
    assert vocabulary.overrides("conversation.stop") == ("farewell now",)
    assert _journal()["history"][0]["action"] == "add"
    assert "farewell now" not in vocabulary.phrases("conversation.stop"), (
        "the package's own phrases stay as declared")


# The controls, the iteration commands and the diagnostics, by the names a
# person says. Spelled here rather than read from `EDITABLE`, so a command
# dropped from it fails its case instead of losing it.
SPOKEN = ("stop", "repeat", "done", "more", "try again", "same in", "never mind",
          "what heard", "how did it go", "whats waiting", "test voice", "which projects",
          "help")


@pytest.mark.parametrize("name", SPOKEN)
def test_every_editable_command_takes_a_phrase_by_its_spoken_name(name):
    """Mutation: drop a key from `EDITABLE` and its case fails."""
    _saved(name, "zebra crossing")
    added = vocabulary.overrides(vocabulary.TARGETS[name])
    assert added in (("zebra crossing",), ("zebra crossing {project}",))


def test_the_spoken_names_are_every_editable_command():
    assert set(vocabulary.TARGETS) == set(SPOKEN)


@pytest.mark.parametrize("key", ["answer.yes", "answer.no", "answer.agree",
                                 "projects.qmcp.checks", "vocabulary.add"])
def test_what_answers_a_consent_and_what_is_not_a_command_are_not_editable(key):
    """Consent is heard through `answer.*`; a phrase of one's own there would
    change what agreeing sounds like.

    Mutation: add `answer.yes` to `EDITABLE` and its case fails.
    """
    with pytest.raises(ValueError, match="not editable"):
        vocabulary.prepare_add(key, "okey dokey")


@pytest.mark.parametrize("phrase, means", [
    ("stop", "conversation.stop"),
    ("go ahead", "answer.yes"),
    ("hold off", "answer.no"),
    ("run the tests", "check"),
    ("qmcp", "project qmcp"),
    ("same in qmcp", "iteration.same_in"),
    ("add vocabulary phrase", "vocabulary.add"),
])
def test_a_phrase_that_already_means_something_is_refused(phrase, means):
    """THE ONE THAT MATTERS.

    Mutation: return an empty map from `_meanings` and every case fails.
    """
    with pytest.raises(ValueError, match=means):
        vocabulary.prepare_add("repeat", phrase)


@pytest.mark.parametrize("phrase", [
    "", "!!!", "this phrase has nine separate words which exceeds the limit"])
def test_an_empty_or_long_phrase_is_refused(phrase):
    with pytest.raises(ValueError, match="at least one word|at most"):
        vocabulary.prepare_add("repeat", phrase)


def test_a_phrase_added_to_one_command_is_refused_for_another():
    """A person's own phrases mean something too.

    Mutation: skip `state["entries"]` in `_meanings` and this fails.
    """
    _saved("repeat", "farewell")
    with pytest.raises(ValueError, match="already means conversation.repeat"):
        vocabulary.prepare_add("stop", "farewell")


def test_the_same_phrase_twice_on_one_command_says_so():
    _saved("done", "thats enough")
    with pytest.raises(ValueError, match="already a phrase for done"):
        vocabulary.prepare_add("done", "That's enough.")


def test_same_in_takes_a_lead_and_the_project_follows():
    """"same in" phrases end in a project's name, so an added one is stored
    with the slot and heard with any project after it.

    Mutation: drop the slot in `_candidate` and `same_in` finds nothing.
    """
    edit = _saved("same in", "redo in")
    assert edit.phrase == "redo in {project}"
    assert vocabulary.shown(edit.phrase) == "redo in a project"
    assert converse.same_in("redo in joe") == "joe"


# --- removing, undoing, and what cannot be changed ---------------------------------


def test_the_package_phrases_cannot_be_removed():
    with pytest.raises(ValueError, match="package vocabulary is fixed"):
        vocabulary.prepare_remove("stop", "goodbye")


def test_a_remove_and_its_undo_are_journalled():
    _saved("done", "thats enough")
    vocabulary.apply_edit(vocabulary.prepare_remove("done", "that's enough"))
    assert vocabulary.overrides("conversation.done") == ()

    undo = vocabulary.prepare_undo()
    assert undo is not None and undo.action == "undo"
    vocabulary.apply_edit(undo)
    history = _journal()["history"]
    assert vocabulary.overrides("conversation.done") == ("thats enough",)
    assert [event["action"] for event in history] == ["add", "remove", "undo"]
    assert history[-1]["target_event"] == history[-2]["id"]


def test_undo_with_nothing_to_undo_is_none():
    assert vocabulary.prepare_undo() is None


def test_an_edit_does_not_overwrite_a_change_made_while_it_waited():
    waiting = vocabulary.prepare_add("stop", "farewell")
    _saved("stop", "later")
    with pytest.raises(RuntimeError, match="changed while"):
        vocabulary.apply_edit(waiting)
    assert vocabulary.overrides("conversation.stop") == ("later",)


def test_an_undo_does_not_overwrite_a_file_changed_outside_the_journal():
    """An undo is prepared from the journal, so only the file's phrases as
    they stand show a change made by hand.

    Mutation: drop the `before` comparison in `apply_edit` and the hand-made
    phrase is lost.
    """
    _saved("stop", "farewell")
    waiting = vocabulary.prepare_undo()
    state = _journal()
    state["entries"]["conversation.stop"] = ["farewell", "by hand"]
    vocabulary.overlay_path().write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed while"):
        vocabulary.apply_edit(waiting)
    assert vocabulary.overrides("conversation.stop") == ("farewell", "by hand")


def test_an_edit_checks_its_meaning_again_before_saving():
    waiting = vocabulary.prepare_add("stop", "farewell")
    _saved("repeat", "farewell")
    with pytest.raises(ValueError, match="already means conversation.repeat"):
        vocabulary.apply_edit(waiting)
    assert vocabulary.overrides("conversation.stop") == ()


def test_a_malformed_phrase_file_is_refused_by_name():
    path = vocabulary.overlay_path()
    path.write_text('{"schema": 1, "entries": {"answer.yes": ["sure thing"]}, "history": []}',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="malformed overlay entry 'answer.yes'"):
        vocabulary.overrides("conversation.stop")


def test_entries_show_what_a_person_added_beside_what_is_declared():
    _saved("help", "remind me")
    """`qmcp vocabulary` and joe's page read `phrases`, so an added phrase is
    in it as well as under `added`.

    Mutation: leave `phrases` as declared and the first assertion fails."""
    shown = {item["key"]: item for item in vocabulary.entries()}
    assert shown["diagnostic.help"]["phrases"][-1] == "remind me"
    assert shown["diagnostic.help"]["added"] == ["remind me"]
    assert "added" not in shown["answer.yes"]


def test_an_added_repeat_phrase_repeats_a_question_anywhere():
    """`asks_repeat` serves every question, not only the conversation's.

    Mutation: read `REPEAT` alone in `asks_repeat` and this fails.
    """
    from qmcp.integrations.voice.adapter import asks_repeat

    _saved("repeat", "pardon me")
    assert asks_repeat("Pardon me?")


def test_an_added_never_mind_phrase_drops_a_read_back():
    """Mutation: read `NEVER_MIND` alone in `dialog._drop_if_asked` and this
    fails."""
    from qmcp.instructions.dialog import _drop_if_asked
    from qmcp.integrations.voice.adapter import Abandoned

    _saved("never mind", "drop the lot")
    with pytest.raises(Abandoned):
        _drop_if_asked("Drop the lot.")


def test_no_test_reads_the_real_phrase_file(tmp_path):
    """`tests/conftest.py` gives every test a file of its own.

    Mutation: drop that fixture and this fails on any workstation.
    """
    assert vocabulary.overlay_path().parent == tmp_path


# --- by voice ----------------------------------------------------------------------


def _talk(*takes):
    from tests.test_spoken_iteration import _talk as talk
    return talk(*takes)


def test_an_approved_edit_is_heard_on_the_next_utterance():
    """Mutation: read the declared phrases alone at the stop check and the
    conversation never stops on the new phrase."""
    ended, tts, client, acted = _talk(
        "add vocabulary phrase farewell now to stop", "approve", "farewell now")

    assert ended.reason == "told to stop"
    assert acted == [] and client.created == []
    assert "Add 'farewell now' to stop. Approve or hold?" in tts.spoken
    assert "Saved. Added 'farewell now' to stop." in tts.spoken
    assert tts.spoken[-1] == STOPPING


def test_an_added_diagnostic_phrase_is_answered():
    _, tts, client, acted = _talk(
        "add vocabulary phrase status please to how did it go", "approve",
        "status please", "stop")

    assert acted == [] and client.created == []
    assert NOTHING_RAN in tts.spoken


def test_a_refused_phrase_is_said_and_nothing_waits():
    _, tts, _, _ = _talk("add vocabulary phrase go ahead to stop", "stop")

    assert any(line.startswith("That cannot be changed:") and "answer.yes" in line
               for line in tts.spoken)
    assert tts.spoken[-1] == STOPPING
    assert not vocabulary.overlay_path().exists()


def test_hold_leaves_the_vocabulary_unchanged():
    _, tts, _, _ = _talk("add vocabulary phrase farewell to stop", "hold", "stop")

    assert not vocabulary.overlay_path().exists()
    assert "Held. The vocabulary is unchanged." in tts.spoken


def test_approve_and_hold_together_is_asked_again():
    """Mutation: let `match_option` decide alone and "approve hold" saves."""
    _, tts, _, _ = _talk("add vocabulary phrase farewell to stop", "approve hold", "hold",
                         "stop")

    assert not vocabulary.overlay_path().exists()
    assert "Say approve to save the change, or hold to leave it." in tts.spoken


def test_a_no_beside_approve_holds():
    _talk("add vocabulary phrase farewell to stop", "dont approve", "stop")

    assert not vocabulary.overlay_path().exists()


def test_an_undo_by_voice_reverses_the_last_change():
    _saved("more", "carry on then")
    _, tts, _, _ = _talk("undo vocabulary change", "approve", "stop")

    assert vocabulary.overrides("conversation.more") == ()
    assert "Saved. Undid the last change, to 'carry on then' in more." in tts.spoken
