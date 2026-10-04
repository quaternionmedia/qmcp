"""A closed question can be answered before it ends, by key or by voice.

Each question is said through `ask_over`: the engine is watched first, with
the parameters of the listen that follows, so an answer said over the
question is heard from its first word, and the voice is told to stop as soon
as the engine says the person interrupted. A backend or a voice without
either says the question whole.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qmcp.instructions.converse import Conversation
from qmcp.instructions.dialog import CONFIRM, InstructionDialog
from qmcp.integrations.voice.adapter import VoiceApprovalLoop, ask_over


class _STT:
    """An engine that can be watched and asked whether it was interrupted."""

    def __init__(self, *takes, confidence=None):
        self.takes = list(takes)
        self.events: list = []
        self.last_confidence = confidence

    def watch(self, duration=5.0, *, pause_ms=None, hint=None):
        self.events.append(("watch", duration, pause_ms, hint))
        return True

    def interrupted(self):
        return False

    def listen(self, duration=5.0, *, pause_ms=None, hint=None):
        self.events.append(("listen", duration, pause_ms, hint))
        return (self.takes.pop(0) if self.takes else ""), "take.wav"


class _PlainSTT:
    """An engine written before watching existed."""

    def __init__(self, *takes):
        self.takes = list(takes)

    def listen(self, duration=5.0, *, pause_ms=None, hint=None):
        return (self.takes.pop(0) if self.takes else ""), "take.wav"


class _TTS:
    """A voice that can be cut short, recording what it was told."""

    def __init__(self, stt=None):
        self.stt = stt
        self.spoken: list[str] = []
        self.untils: list = []

    def speak(self, text, out_path=None, until=None):
        if self.stt is not None:
            self.stt.events.append(("speak", text))
        self.spoken.append(text)
        self.untils.append(until)


class _OldTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)


class _Requests:
    def __init__(self, said="Run in qmcp: deploy. A script, one run.", options=("approve", "hold")):
        self.request = SimpleNamespace(prompt=said, options=list(options), context={"spoken": said})

    def get_human_request(self, request_id):
        return self.request, None

    def submit_human_response(self, **kw):
        return SimpleNamespace(response=kw["response"])


def _kinds(stt):
    return [event[0] for event in stt.events]


# --- the approval -------------------------------------------------------------


def test_a_consent_is_watched_with_the_listen_s_parameters_before_it_is_said():
    """Mutation: say the question without `ask_over` -- red, nothing is watched."""
    stt = _STT("approve")
    tts = _TTS(stt)
    loop = VoiceApprovalLoop(stt, tts, client=_Requests(), listen_duration=6.0)

    loop.run_once("r")

    assert _kinds(stt)[:3] == ["watch", "speak", "listen"]
    assert stt.events[0] == ("watch", 6.0, None, ["approve", "hold"])
    assert stt.events[2] == ("listen", 6.0, None, ["approve", "hold"])


def test_the_voice_is_told_to_stop_when_the_engine_says_it_was_interrupted():
    """Mutation: speak without `until` -- red, a key or a word over the
    question cannot stop it."""
    stt = _STT("approve")
    tts = _TTS(stt)

    VoiceApprovalLoop(stt, tts, client=_Requests()).run_once("r")

    assert tts.untils[0] == stt.interrupted


def test_a_repeat_and_a_re_ask_are_watched_too():
    stt = _STT("repeat", "banana", "hold")
    tts = _TTS(stt)

    VoiceApprovalLoop(stt, tts, client=_Requests(), max_retries=2).run_once("r")

    kinds = _kinds(stt)
    assert kinds.count("watch") == kinds.count("speak") - 1 == 3  # the acknowledgement is no question
    assert all(kinds[i + 1] == "speak" for i, kind in enumerate(kinds) if kind == "watch")


def test_a_voice_that_cannot_be_cut_short_says_the_question_whole():
    """Mutation: no fallback for a voice without `until` -- red, TypeError."""
    tts = _OldTTS()

    answer = VoiceApprovalLoop(_STT("hold"), tts, client=_Requests()).run_once("r")

    assert answer.response == "hold" and tts.spoken[0].endswith("Approve or hold?")


def test_a_backend_that_cannot_be_watched_still_asks():
    tts = _TTS()

    answer = VoiceApprovalLoop(_PlainSTT("approve"), tts, client=_Requests()).run_once("r")

    assert answer.response == "approve" and tts.untils[0] is None


def test_a_watch_that_fails_costs_the_question_nothing():
    class Failing(_STT):
        def watch(self, *args, **kwargs):
            raise ConnectionError("the engine went away")

    stt = Failing("approve")
    tts = _TTS(stt)

    assert VoiceApprovalLoop(stt, tts, client=_Requests()).run_once("r").response == "approve"
    assert tts.spoken[0].startswith("Run in qmcp")


def test_a_type_error_from_inside_the_voice_is_not_taken_for_an_old_voice():
    """Mutation: retry on any TypeError -- red, a broken voice speaks twice."""

    class Broken(_TTS):
        def speak(self, text, out_path=None, until=None):
            self.spoken.append(text)
            raise TypeError("cannot synthesize")

    tts = Broken()
    with pytest.raises(TypeError, match="cannot synthesize"):
        ask_over(_STT(), tts, "Approve or hold?", duration=5.0)
    assert tts.spoken == ["Approve or hold?"]


# --- the dialog and the conversation -------------------------------------------


class _Instructions:
    def create_instruction(self, text, source, project, heard):
        return {"id": "i", "text": text, "project": "qmcp", "status": "recorded"}


def test_the_prompt_and_the_read_back_are_watched_for_the_takes_that_follow_them():
    """Mutation: say the read-back without `_say_over` -- red."""
    stt = _STT("Deploy qmcp.", "agree")
    dialog = InstructionDialog(stt=stt, tts=_TTS(stt), client=_Instructions(), names=["qmcp"],
                               listen_duration=20.0, pause_ms=1500, answer_duration=4.0,
                               vocabulary=["qmcp", "vox"])

    dialog.run_once()

    watches = [e for e in stt.events if e[0] == "watch"]
    assert watches[0] == ("watch", 20.0, 1500, ["qmcp", "vox"])  # the instruction's own take
    assert watches[1] == ("watch", 4.0, None, CONFIRM)  # the read-back's


def test_the_tacit_offer_is_watched_for_its_brief_window():
    stt = _STT("", confidence=0.9)
    dialog = InstructionDialog(stt=stt, tts=_TTS(stt), client=_Instructions(), names=["qmcp"],
                               tacit_above=0.7, tacit_seconds=2.5)

    dialog.run_once(heard="Deploy qmcp.", confidence=0.9)

    assert stt.events[0] == ("watch", 2.5, None, CONFIRM)


def test_a_choice_of_project_is_watched_with_its_options():
    stt = _STT("vox")
    dialog = InstructionDialog(stt=stt, tts=_TTS(stt), client=_Instructions(), names=["qmcp", "vox"],
                               answer_duration=4.0)

    assert dialog._ask_choice("Which project?", ["qmcp", "vox"]) == "vox"
    assert stt.events[0] == ("watch", 4.0, None, ["qmcp", "vox"])


def test_anything_else_is_watched_with_the_conversation_s_own_hint():
    """Mutation: say the conversation's question without `ask_over` -- red."""
    stt = _STT()
    conversation = Conversation(stt, _TTS(stt), client=None, runtime=None, names=["qmcp"],
                                listen_duration=12.0, pause_ms=1200)
    conversation._vocabulary = ["qmcp", "vox"]

    conversation.ask("Anything else?", ("yes", "no"))

    assert stt.events[0] == ("watch", 12.0, 1200, ["yes", "no", "qmcp", "vox"])


# --- over the wire ----------------------------------------------------------------


def test_over_the_wire_the_watch_carries_the_engine_s_spelling_and_the_voice_stops(tmp_path):
    """Against vox's stand-in engine on joe's contract: the watch arrives with
    the listen's parameters, and a voice polling `interrupted` is told to stop
    when the engine says so."""
    from vox.adapters import JOE
    from vox.engine import EngineState, serve
    from vox.stt import HttpSTT

    state = EngineState(microphone="m", heard="approve", capture_dir=tmp_path, audio_dirs=[tmp_path],
                        contract=JOE, interrupted=True)
    stopped: list[bool] = []

    class Polling(_TTS):
        def speak(self, text, out_path=None, until=None):
            stopped.append(bool(until and until()))

    with serve(state) as (base_url, state), HttpSTT(base_url, contract=JOE) as stt:
        answer = VoiceApprovalLoop(stt, Polling(), client=_Requests(), listen_duration=5.0).run_once("r")

    assert answer.response == "approve"
    assert state.watches[0] == {"duration": "5.0", "hint": "approve, hold"}
    assert stopped[0] is True
