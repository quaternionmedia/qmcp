"""What is said aloud is a few plain words and nothing a synthesizer reads as syntax.

`speakable` cleans a sentence -- "(s)", markdown, brackets, a path's folders, a
link -- and every dialog speaks through it, so a sentence written for a
terminal or a record is never read with its punctuation. Counts are words, the
grammar is a short question, and an accepted answer is acknowledged with the
word a listener expects.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qmcp.instructions.converse import Conversation
from qmcp.instructions.dialog import InstructionDialog
from qmcp.integrations.voice.adapter import (
    Speakable,
    VoiceApprovalLoop,
    counted,
    say_options,
    speakable,
    speakably,
)


class _STT:
    def __init__(self, *takes):
        self.takes = list(takes)

    def listen(self, duration=5.0, *, pause_ms=None, hint=None):
        return (self.takes.pop(0) if self.takes else ""), "take.wav"


class _TTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)
        return out_path


class _Requests:
    def __init__(self, said, options=("approve", "hold")):
        self.request = SimpleNamespace(prompt=said, options=list(options), context={"spoken": said})

    def get_human_request(self, request_id):
        return self.request, None

    def submit_human_response(self, **kw):
        return SimpleNamespace(response=kw["response"])


# --- speakable ------------------------------------------------------------------------


def test_a_plural_mark_is_not_read():
    """Heard as "run s". Mutation: drop the plural-mark rule -- red."""
    assert speakable("budget 1 run(s), carrying 2 instruction(s)") == "budget 1 run, carrying 2 instruction"


@pytest.mark.parametrize(("written", "said"), [
    (r"clone C:\work\clones\qmcp, ready", "clone qmcp, ready"),
    ("in /home/me/clones/vox now", "in vox now"),
    ("edited qmcp/instructions/act.py.", "edited act.py."),
    ("edited docs/quickstart.md", "edited quickstart.md"),
])
def test_a_path_is_said_by_its_last_name(written, said):
    """Mutation: drop the path rule -- red on every row."""
    assert speakable(written) == said


def test_a_fraction_and_an_either_or_are_not_paths():
    """A single slash between words or numbers is not a path. Mutation: drop
    the fraction rule -- red, "3 3"; take any one slash as a path -- red,
    "Tests pass: 3"."""
    assert speakable("Tests pass: 3/3, and/or more.") == "Tests pass: 3 of 3, and or more."


def test_a_link_is_said_as_a_link_and_a_markdown_link_by_its_words():
    """Mutation: drop the link rule -- red, the address is spelled."""
    assert speakable("See [the notes](https://x.y/z) and https://github.com/a/b.") == (
        "See the notes and a link")


def test_markdown_marks_and_brackets_are_not_read():
    """Mutation: drop the symbols rule -- red."""
    assert speakable("**Added** `read_file` [twice] (it works) # done") == (
        "Added read file twice it works done")


def test_ordinary_punctuation_and_numbers_are_kept():
    assert speakable("Version 1.2.3 is out, e.g. it works. 50% done?") == (
        "Version 1.2.3 is out, e.g. it works. 50% done?")
    assert speakable("") == "" and speakable(None) == ""


# --- counts and grammar ---------------------------------------------------------------


def test_a_count_is_said_in_words_with_its_plural():
    assert (counted(1, "run"), counted(2, "run"), counted(0, "run")) == ("one run", "two runs", "zero runs")
    assert counted(12, "run") == "12 runs"


def test_the_grammar_is_a_short_question():
    """Mutation: drop the capital -- red, "Ship it? approve or hold?"."""
    assert say_options(["approve", "hold"]) == "Approve or hold?"
    assert say_options(["red", "green", "blue"]) == "Red, green, or blue?"
    assert say_options(["agree"]) == "Agree?"


# --- every dialog speaks through it ---------------------------------------------------


def test_an_approval_is_asked_without_its_syntax():
    """Mutation: keep the bare synthesizer in `VoiceApprovalLoop` -- red,
    "(s)" reaches it."""
    tts = _TTS()

    VoiceApprovalLoop(_STT("approve"), tts, client=_Requests("Run 2 run(s) in qmcp.")).run_once("r")

    assert tts.spoken[0] == "Run 2 run in qmcp. Approve or hold?"


@pytest.mark.parametrize(("answer", "said"), [
    ("approve", "Approved."), ("hold", "Holding."), ("red", "Noted."),
])
def test_an_accepted_answer_is_acknowledged_with_a_word(answer, said):
    """Not "Recorded: approve". Mutation: say the answer back -- red."""
    tts = _TTS()
    options = ("approve", "hold") if answer != "red" else ("red", "green")

    VoiceApprovalLoop(_STT(answer), tts, client=_Requests("Ship it?", options)).run_once("r")

    assert tts.spoken[-1] == said


def test_an_instruction_is_read_back_without_its_syntax():
    """Mutation: keep the bare synthesizer in `InstructionDialog` -- red."""

    class Client:
        def create_instruction(self, text, source, project, heard):
            return {"id": "i", "text": text, "project": "qmcp", "status": "recorded"}

    tts = _TTS()
    InstructionDialog(stt=_STT("Fix qmcp/instructions/act.py (now).", "agree"), tts=tts,
                      client=Client(), names=["qmcp"]).run_once()

    assert tts.spoken[1] == "I heard: Fix act.py now. Agree or again?"


def test_the_conversation_speaks_through_speakable_and_wraps_once():
    """Mutation: keep the bare synthesizer in `Conversation` -- red."""
    tts = _TTS()
    conversation = Conversation(_STT(), tts, client=None, runtime=None, names=["qmcp"])

    assert isinstance(conversation.tts, Speakable) and conversation.tts.tts is tts
    assert speakably(conversation.tts) is conversation.tts and speakably(None) is None


def test_the_wrapper_passes_everything_else_through():
    tts = _TTS()
    wrapped = speakably(tts)

    assert wrapped.speak("Done (s).", "out.wav") == "out.wav"
    assert wrapped.spoken is tts.spoken == ["Done."]
