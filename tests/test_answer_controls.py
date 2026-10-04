"""Answering a closed question with fewer misses, and without speaking.

Each closed question tells the engine the words its answer is expected to be
(`hint`) and announces them as `options`, so a display can offer each as a key
or a button; "repeat" -- said, or sent by a key -- asks the question again
without spending a retry; and a word the transcriber returned over and over is
read once. A backend written before `hint` or `options` existed keeps working.
"""

from __future__ import annotations

import pytest

from qmcp.instructions import converse
from qmcp.instructions.converse import ANYTHING_ELSE, OKAY, READY, Conversation, plain
from qmcp.instructions.dialog import CONFIRM, InstructionDialog
from qmcp.integrations.voice.adapter import (
    MAX_REPEATS,
    UnclearResponse,
    VoiceApprovalLoop,
    announce_to,
    asks_repeat,
    listen_for,
    plain_words,
)


class _STT:
    """A backend of today's shape: it takes `hint` and announces `options`."""

    def __init__(self, *takes):
        self.takes = list(takes)
        self.hints: list = []
        self.announced: list[dict] = []

    def listen(self, duration=5.0, *, pause_ms=None, hint=None):
        self.hints.append(hint)
        return (self.takes.pop(0) if self.takes else ""), "take.wav"

    def announce(self, state, text="", reason=None, options=None):
        self.announced.append({"state": state, "text": text, "reason": reason, "options": options})
        return True


class _OldSTT:
    """A backend written before `hint` and `options` existed."""

    def __init__(self, *takes):
        self.takes = list(takes)
        self.listens = 0
        self.announced: list[tuple] = []

    def listen(self, duration=5.0, *, pause_ms=None):
        self.listens += 1
        return (self.takes.pop(0) if self.takes else ""), "take.wav"

    def announce(self, state, text="", reason=None):
        self.announced.append((state, text, reason))


class _TTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)


# --- reading a short answer -----------------------------------------------------


def test_a_word_returned_over_and_over_is_read_once():
    """Seen on a clipped "no": "No. No. No. No. No." Mutation: drop the
    collapse from `plain` -- red, it is not a no to "Anything else?"."""
    assert plain_words("No. No. No.") == "no" and plain("No. No. No. No. No.") == "no"
    assert plain("no thank you") == "no thank you"


@pytest.mark.parametrize("said", ["Repeat.", "repeat that", "Say that again?", "What?", "Pardon"])
def test_asking_to_hear_it_again_is_recognised(said):
    assert asks_repeat(said)


@pytest.mark.parametrize("said", ["again", "repeat the deploy", "approve", ""])
def test_an_answer_is_not_a_request_to_repeat(said):
    """"again" is the read-back's own option."""
    assert not asks_repeat(said)


# --- the seam to the backend ---------------------------------------------------------


def test_the_hint_reaches_a_backend_that_takes_it_and_is_dropped_for_one_that_does_not():
    """Mutation: drop the fallback -- red, an old backend fails the question."""
    new, old = _STT("approve"), _OldSTT("approve")

    assert listen_for(new, 2.0, hint=("approve", "hold"))[0] == "approve"
    assert new.hints == [["approve", "hold"]]
    assert listen_for(old, 2.0, hint=("approve", "hold"))[0] == "approve"
    assert old.listens == 1


def test_a_backend_s_own_type_error_is_not_mistaken_for_an_old_shape():
    """A second call would record a second take. Mutation: retry on any
    TypeError -- red, the backend is asked twice."""

    class Broken:
        calls = 0

        def listen(self, duration=5.0, *, pause_ms=None, hint=None):
            Broken.calls += 1
            raise TypeError("unsupported operand")

    with pytest.raises(TypeError, match="unsupported operand"):
        listen_for(Broken(), 2.0, hint=("approve",))
    assert Broken.calls == 1


def test_options_reach_a_backend_that_takes_them_and_an_old_one_still_hears_the_state():
    new, old = _STT(), _OldSTT()

    announce_to(new, "speaking", "Say approve or hold.", options=("approve", "hold"))
    announce_to(old, "speaking", "Say approve or hold.", reason="nomatch", options=("approve", "hold"))

    assert new.announced == [{"state": "speaking", "text": "Say approve or hold.", "reason": None,
                              "options": ["approve", "hold"]}]
    assert old.announced == [("speaking", "Say approve or hold.", "nomatch")]


# --- the approval dialog -------------------------------------------------------------


def test_an_approval_hints_and_offers_its_options():
    stt, tts = _STT("approve"), _TTS()

    assert VoiceApprovalLoop(stt, tts)._ask("Ship it?", ["approve", "hold"]) == "approve"

    assert stt.hints == [["approve", "hold"]]
    assert stt.announced[0]["options"] == ["approve", "hold"]


def test_repeat_says_the_question_again_and_spends_no_retry():
    """No retries at all, and still answered after a repeat. Mutation: count
    a repeat as an attempt -- red, UnclearResponse."""
    stt, tts = _STT("repeat", "hold"), _TTS()

    assert VoiceApprovalLoop(stt, tts, max_retries=0)._ask("Ship it?", ["approve", "hold"]) == "hold"
    assert tts.spoken == ["Ship it? Say approve or hold."] * 2
    assert stt.announced[1]["reason"] == "repeat"


def test_repeats_are_bounded():
    """A room that says "what" for ever is not a question asked for ever."""
    stt = _STT(*["what"] * (MAX_REPEATS + 5))

    with pytest.raises(UnclearResponse):
        VoiceApprovalLoop(stt, _TTS(), max_retries=0)._ask("Ship it?", ["approve", "hold"])
    assert len(stt.hints) == MAX_REPEATS + 1


# --- the instruction dialog ----------------------------------------------------------


class _Client:
    def create_instruction(self, text, source, project, heard):
        return {"id": "i", "text": text, "project": project, "status": "recorded"}


def test_the_read_back_hints_record_or_again_and_repeats_on_request():
    """Mutation: drop the hint on the confirmation -- red."""
    stt, tts = _STT("repeat", "record"), _TTS()
    dialog = InstructionDialog(stt=stt, tts=tts, client=_Client(), names=["qmcp"], max_retries=0)

    row = dialog.run_once(heard="Deploy qmcp.")

    assert row["text"] == "Deploy qmcp."
    readbacks = [t for t in tts.spoken if t.startswith("I heard: ")]
    assert len(readbacks) == 2
    assert stt.hints == [CONFIRM, CONFIRM]
    assert [a["options"] for a in stt.announced if a["state"] == "speaking"][:2] == [CONFIRM, CONFIRM]


def test_a_project_choice_hints_and_offers_its_candidates():
    stt, tts = _STT("vox"), _TTS()
    dialog = InstructionDialog(stt=stt, tts=tts, client=_Client(), names=["qmcp", "vox"])

    assert dialog._ask_choice("Which project?", ["qmcp", "vox"]) == "vox"
    assert stt.hints == [["qmcp", "vox"]]
    assert stt.announced[0]["options"] == ["qmcp", "vox"]


# --- the standing conversation -------------------------------------------------------


class _Queue:
    def list_human_requests(self, **kw):
        return []


def _talk(*takes, taken=None):
    stt, tts = _STT(*takes), _TTS()
    conversation = Conversation(stt, tts, _Queue(), runtime=None, names=["qmcp"], idle_limit=2)
    taken = [] if taken is None else taken
    conversation._take = lambda heard: taken.append(heard) or converse.Turn("id", heard, "done", "Done.")
    conversation.run()
    return stt, tts


def test_a_prompt_leaves_the_turn_open_so_the_engine_can_cue_the_person():
    """joe cues a person to speak when the last state is `speaking`. Mutation:
    say the prompt with `say`, which closes the turn with `idle` -- red."""
    stt, _ = _talk("stop listening")

    assert stt.announced[0] == {"state": "speaking", "text": READY, "reason": None, "options": None}
    assert stt.announced[1]["state"] != "idle"


def test_anything_else_offers_yes_or_no_and_hints_the_take_that_answers_it():
    stt, _ = _talk("Deploy qmcp.", "no", "stop listening")

    asked = [a for a in stt.announced if a["text"] == ANYTHING_ELSE]
    assert asked and asked[0]["options"] == ["yes", "no"]
    # Takes: the instruction (no hint), the answer to "Anything else?", the stop.
    assert stt.hints[0] is None and stt.hints[1] == ["yes", "no"]


def test_an_answer_by_key_needs_no_wake_word():
    """A key's answer comes back with no recording behind it. Mutation: apply
    the wake word to it -- red, the key does nothing in a noisy room."""

    class Keyed(_STT):
        def listen(self, duration=5.0, *, pause_ms=None, hint=None):
            text, _ = super().listen(duration, pause_ms=pause_ms, hint=hint)
            return text, ("" if text in ("yes", "stop listening") else "take.wav")

    stt, tts = Keyed("Deploy qmcp.", "qmcp deploy qmcp", "yes", "stop listening"), _TTS()
    conversation = Conversation(stt, tts, _Queue(), runtime=None, names=["qmcp"], idle_limit=2,
                                wake="qmcp")
    taken = []
    conversation._take = lambda heard: taken.append(heard) or converse.Turn("id", heard, "done", "Done.")
    ended = conversation.run()

    assert taken == ["deploy qmcp"]  # the spoken one without its word was ignored
    assert ended.reason == "told to stop"  # by key, with no wake word


def test_repeat_in_the_conversation_asks_the_last_question_again():
    """Mutation: drop the repeat from the loop -- red, "repeat" is taken as an
    instruction."""
    taken: list[str] = []
    stt, tts = _talk("Deploy qmcp.", "repeat", "No. No. No.", "stop listening", taken=taken)

    assert taken == ["Deploy qmcp."]
    assert tts.spoken.count(ANYTHING_ELSE) == 2
    assert OKAY in tts.spoken  # the repeated "no" was a no


def test_a_request_s_spoken_form_is_what_is_said_and_its_prompt_otherwise():
    """Mutation: drop `context.spoken` -- red, the half-minute path is read out."""
    from types import SimpleNamespace

    class Client:
        def __init__(self, context):
            self.request = SimpleNamespace(prompt="Act on C:/a/very/long/path.",
                                           options=["approve", "hold"], context=context)

        def get_human_request(self, request_id):
            return self.request, None

        def submit_human_response(self, **kw):
            return SimpleNamespace(response=kw["response"])

    for context, said in (({"spoken": "Act on qmcp."}, "Act on qmcp. Say approve or hold."),
                          (None, "Act on C:/a/very/long/path. Say approve or hold.")):
        tts = _TTS()
        VoiceApprovalLoop(_STT("approve"), tts, client=Client(context)).run_once("r")
        assert tts.spoken[0] == said
