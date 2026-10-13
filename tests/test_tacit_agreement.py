"""An instruction heard confidently is agreed to tacitly; any interruption cancels that.

The engine says how sure it is of each take (`last_confidence`); at or above
the dialog's `tacit_above`, the read-back says what was heard, asks nothing,
and listens briefly: silence agrees. A word or a key in that moment cancels
the tacit agreement, and what interrupted decides.
"""

from __future__ import annotations

import pytest

from qmcp.instructions.converse import Conversation
from qmcp.instructions.dialog import CONFIRM, PROMPT, InstructionDialog


class _STT:
    """Takes as (text, confidence); records each listen's duration and hint."""

    def __init__(self, *takes):
        self.takes = [t if isinstance(t, tuple) else (t, None) for t in takes]
        self.listens: list[tuple[float, object]] = []
        self.last_confidence = None

    def listen(self, duration=5.0, *, pause_ms=None, hint=None):
        self.listens.append((duration, hint))
        text, self.last_confidence = self.takes.pop(0) if self.takes else ("", None)
        return text, "take.wav"

    def announce(self, state, text="", reason=None, options=None):
        pass


class _TTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)


class _Client:
    def __init__(self):
        self.recorded: list[str] = []

    def create_instruction(self, text, source, project, heard):
        self.recorded.append(text)
        return {"id": "i", "text": text, "project": "qmcp", "status": "recorded"}


def _dialog(*takes, tacit_above=0.7, max_retries=2):
    stt, tts, client = _STT(*takes), _TTS(), _Client()
    dialog = InstructionDialog(stt=stt, tts=tts, client=client, names=["qmcp"],
                               tacit_above=tacit_above, tacit_seconds=2.5, max_retries=max_retries)
    return dialog, stt, tts, client


def _readbacks(tts):
    return [t for t in tts.spoken if t.startswith("I heard: ")]


def test_a_confident_instruction_is_read_back_without_a_question_and_silence_agrees():
    """Mutation: drop the tacit offer -- red, the read-back asks."""
    dialog, stt, tts, client = _dialog(("Deploy qmcp.", 0.9), "")

    dialog.run_once()

    assert client.recorded == ["Deploy qmcp."]
    assert _readbacks(tts) == ["I heard: Deploy qmcp."]
    assert stt.listens[1] == (2.5, CONFIRM)  # a short window, hinted for an interruption


def test_a_transcript_already_taken_is_agreed_to_tacitly_on_its_own_confidence():
    dialog, _, tts, client = _dialog("")

    dialog.run_once(heard="Deploy qmcp.", confidence=0.95)

    assert client.recorded == ["Deploy qmcp."] and _readbacks(tts) == ["I heard: Deploy qmcp."]


def test_interrupting_with_agree_records_it():
    dialog, _, _, client = _dialog(("Deploy qmcp.", 0.9), "agree")

    dialog.run_once()

    assert client.recorded == ["Deploy qmcp."]


def test_interrupting_with_again_takes_the_instruction_again():
    """Mutation: let silence-or-anything agree -- red, the misheard one is kept."""
    dialog, _, tts, client = _dialog(("Deploy box.", 0.9), "again", ("Deploy vox.", 0.5), "agree")

    dialog.run_once()

    assert client.recorded == ["Deploy vox."]
    assert PROMPT in tts.spoken
    assert _readbacks(tts)[-1] == "I heard: Deploy vox. Agree or again?"  # the retake asks


def test_interrupting_with_anything_else_cancels_the_tacit_agreement_and_asks():
    """Mutation: take any interruption as agreement -- red."""
    dialog, _, tts, client = _dialog(("Deploy qmcp.", 0.9), "wait a moment", "agree")

    dialog.run_once()

    assert client.recorded == ["Deploy qmcp."]
    assert _readbacks(tts) == ["I heard: Deploy qmcp.", "I heard: Deploy qmcp. Agree or again?"]


def test_a_key_asking_to_repeat_cancels_it_too():
    dialog, _, tts, _ = _dialog(("Deploy qmcp.", 0.9), "repeat", "agree")

    dialog.run_once()

    assert _readbacks(tts)[1].endswith("Agree or again?")


@pytest.mark.parametrize("confidence, tacit_above", [(0.6, 0.7), (None, 0.7), (1.0, None)])
def test_below_the_threshold_or_unweighed_or_off_the_read_back_asks(confidence, tacit_above):
    """Mutation: offer tacitly whatever the confidence -- red."""
    dialog, _, tts, client = _dialog(("Deploy qmcp.", confidence), "agree", tacit_above=tacit_above)

    dialog.run_once()

    assert _readbacks(tts) == ["I heard: Deploy qmcp. Agree or again?"]
    assert client.recorded == ["Deploy qmcp."]


def test_the_earlier_word_is_still_taken_as_agree_and_never_said():
    dialog, _, tts, client = _dialog(("Deploy qmcp.", 0.5), "record")

    dialog.run_once()

    assert client.recorded == ["Deploy qmcp."]
    assert not any("record" in t.lower() for t in tts.spoken if t.startswith("I heard"))


def test_the_conversation_hands_each_take_s_confidence_to_the_read_back(monkeypatch):
    """Mutation: drop the confidence on the way to the dialog -- red, the
    conversation always asks."""
    from qmcp.integrations.voice.adapter import UnclearResponse

    class Queue:
        def list_human_requests(self, **kw):
            return []

    seen = []

    def run_once(self, heard=None, confidence=None):
        seen.append((heard, confidence, self.tacit_above))
        raise UnclearResponse("stood in for: the turn ends here")

    monkeypatch.setattr(InstructionDialog, "run_once", run_once)
    stt, tts = _STT(("Deploy qmcp.", 0.9), "stop listening"), _TTS()
    Conversation(stt, tts, Queue(), runtime=None, names=["qmcp"], idle_limit=2,
                 tacit_above=0.7).run()

    assert seen == [("Deploy qmcp.", 0.9, 0.7)]
