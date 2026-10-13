"""`qmcp.instructions.spoken`: what is said back about an instruction, and how
the panel is told.

Pure functions over a row as the server serves it, and one function that
speaks; the synthesizer and the engine are stood in for.
"""

from __future__ import annotations

import pytest

from qmcp.instructions.spoken import (
    SPOKEN_CHARS,
    first_sentence,
    say,
    starting,
    summarise,
)


def _row(status, **fields):
    return {"id": "row-1", "text": "Add a health check to qmcp.", "project": "qmcp",
            "status": status, "outcome_text": None, "exit_code": None, **fields}


# --- first_sentence -------------------------------------------------------------------


def test_the_first_sentence_is_taken_and_the_rest_is_flagged():
    """Mutation: return the whole text -- red, two sentences come back."""
    assert first_sentence("Added the route. Two files changed.") == ("Added the route.", True)
    assert first_sentence("Added the route.") == ("Added the route.", False)


def test_a_long_sentence_is_cut_at_a_word_boundary_within_the_limit():
    """Mutation: cut at `chars` exactly -- red, a word is broken."""
    text = "word " * 60
    sentence, cut = first_sentence(text, chars=23)

    assert cut and sentence == "word word word word"
    assert len(first_sentence("x " * 200)[0]) <= SPOKEN_CHARS


def test_markdown_marks_and_line_breaks_are_not_read_aloud():
    """A synthesizer reads `**` and `#` aloud. Mutation: drop the marks
    pattern -- red."""
    assert first_sentence("## Summary\n\n**Added** the `health` route.\nMore.") == (
        "Summary Added the health route.", True)


def test_nothing_says_nothing():
    assert first_sentence("") == ("", False)
    assert first_sentence("  \n ") == ("", False)
    assert first_sentence(None) == ("", False)


# --- summarise ------------------------------------------------------------------------


def test_a_done_run_says_where_and_the_outcomes_first_sentence():
    """Only the first sentence; the rest is on the record and is not
    announced. Mutation: say the whole outcome -- red."""
    row = _row("done", outcome_text="Added a health route. Two files changed.", exit_code=0)

    assert summarise(row) == "Done in qmcp. Added a health route."


def test_an_outcome_is_said_without_its_syntax():
    """A path is said by its last name, before the sentence is cut, so the
    budget is spent on words. Mutation: summarise the raw outcome -- red."""
    row = _row("done", outcome_text="Edited qmcp/instructions/act.py (2 lines). Tests pass.")

    assert summarise(row) == "Done in qmcp. Edited act.py 2 lines."


def test_a_done_run_closes_an_unpunctuated_outcome():
    """Two sentences said together are still two. Mutation: drop `_closed`
    -- red, "Done in qmcp. route added" runs on."""
    assert summarise(_row("done", outcome_text="route added")) == "Done in qmcp. route added."


def test_a_done_run_that_reported_nothing_says_so():
    assert summarise(_row("done", outcome_text="")) == "Done in qmcp. Nothing reported."


def test_a_failed_run_says_where_and_what_it_reported():
    """The exit code is on the record, not in the sentence. Mutation: say
    `done` for every finished run -- red."""
    row = _row("failed", outcome_text="Tests failed in test_api.py", exit_code=2)

    assert summarise(row) == "Failed in qmcp. Tests failed in test api.py."
    assert summarise(_row("failed")) == "Failed in qmcp."


@pytest.mark.parametrize(("status", "said"), [
    ("refused", "Held. Nothing ran."),
    ("unanswered", "No answer. Nothing ran."),
    ("asking", "Waiting for consent in qmcp."),
    ("consented", "Running in qmcp."),
    ("acting", "Running in qmcp."),
    ("recorded", "Recorded for qmcp."),
])
def test_every_status_is_said_as_what_it_is(status, said):
    """Mutation: fall through to the generic sentence for any one of these --
    red on that row."""
    assert summarise(_row(status)) == said


def test_an_unresolved_row_says_it_has_no_project():
    assert summarise(_row("unresolved", project=None)) == "Recorded, no project."


def test_a_refusal_before_the_ask_says_nothing_ran_and_not_why():
    """The reason names flags and paths and is written for a terminal.
    Mutation: ignore `why` -- red, the row reads as merely recorded."""
    why = "no checkout for 'qmcp' in the thread archive; pass --cwd <path to the project's clone>."

    said = summarise(_row("recorded"), why=why)

    assert said == "Nothing ran."
    assert "--cwd" not in said


def test_a_status_given_as_an_enum_is_read_by_its_value():
    from qmcp.db.models import InstructionStatus

    assert summarise(_row(InstructionStatus.REFUSED)).startswith("Held.")


def test_an_unknown_status_is_said_rather_than_guessed():
    assert summarise(_row("archived")) == "Instruction in qmcp: archived."


def test_the_running_line_names_the_project_when_there_is_one():
    assert starting("qmcp") == "Running in qmcp."
    assert starting(None) == "Running."


# --- say -------------------------------------------------------------------------------


class _TTS:
    def __init__(self):
        self.spoken = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)
        return "spoken.wav"


class _Engine:
    def __init__(self, fails=False):
        self.announced = []
        self.fails = fails

    def announce(self, state, text="", reason=None):
        self.announced.append((state, text))
        if self.fails:
            raise RuntimeError("the engine went away")
        return True


def test_the_panel_is_told_it_is_being_said_and_then_that_the_turn_is_over():
    """`recorded` would show on the panel as an answer accepted. Mutation:
    end on `recorded` -- red."""
    tts, engine = _TTS(), _Engine()

    say("Done in qmcp.", tts, engine)

    assert tts.spoken == ["Done in qmcp."]
    assert engine.announced == [("speaking", "Done in qmcp."), ("idle", "Done in qmcp.")]


def test_an_announcement_that_fails_costs_the_sentence_nothing():
    """Mutation: let the exception out of `announce` -- red, nothing is said."""
    tts = _TTS()

    say("Held.", tts, _Engine(fails=True))

    assert tts.spoken == ["Held."]


def test_a_sentence_is_said_without_its_syntax():
    """Mutation: speak `text` unwrapped -- red, "(s)" reaches the synthesizer."""
    tts = _TTS()

    say("Two run(s) in qmcp.", tts)

    assert tts.spoken == ["Two run in qmcp."]


def test_without_an_engine_the_sentence_is_still_said():
    tts = _TTS()

    say("Held.", tts)
    say("Held.", tts, object())

    assert tts.spoken == ["Held.", "Held."]
