"""`qmcp.instructions.converse`: the standing conversation's turns, its waiting
questions, where it finds a clone, and how it waits for the servers.

The loop is driven by a stand-in for the speech engine that returns scripted
takes; a turn is stood in for where the test is about the loop rather than the
instruction, and the whole session runs for real in `tests/test_cookbook_instruct.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qmcp.instructions import converse
from qmcp.instructions.converse import (
    ANYTHING_ELSE,
    FAILED,
    OKAY,
    QUESTION_WAITING,
    READY,
    STOPPING,
    WAITING,
    Conversation,
    plain,
    wait_for,
)
from qmcp.instructions.dialog import PROMPT


class _STT:
    def __init__(self, *takes):
        self.takes = list(takes)
        self.announced: list[tuple[str, str]] = []

    def listen(self, duration=5.0, *, pause_ms=None):
        return (self.takes.pop(0) if self.takes else ""), "take.wav"

    def announce(self, state, text="", reason=None):
        self.announced.append((state, text))
        return True


class _TTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text, out_path=None):
        self.spoken.append(text)
        return "out.wav"


class _Queue:
    def __init__(self, *pending):
        self.pending = list(pending)

    def list_human_requests(self, **kw):
        return list(self.pending)


def _talk(*takes, queue=None, wake=None, idle_limit=2, turn=None):
    stt, tts = _STT(*takes), _TTS()
    conversation = Conversation(stt, tts, queue or _Queue(), runtime=None, names=["qmcp"],
                                wake=wake, idle_limit=idle_limit)
    taken: list[str] = []

    def take(heard):
        taken.append(heard)
        if turn:
            turn(heard)
        return converse.Turn("id", heard, "done", "Done.")

    conversation._take = take
    return conversation, conversation.run(), stt, tts, taken


def test_words_are_compared_without_case_or_punctuation():
    assert plain("Stop listening!") == "stop listening"
    assert plain("That's all.") == "thats all"
    assert plain("  No,  thanks ") == "no thanks"


def test_it_says_it_is_ready_and_ends_when_told():
    """Mutation: drop `stop` from the loop -- red, the conversation idles out
    instead."""
    _, ended, _, tts, taken = _talk("stop listening")

    assert tts.spoken == [READY, STOPPING]
    assert ended.reason == "told to stop" and taken == []


def test_an_instruction_is_taken_and_anything_else_asked():
    _, ended, _, tts, taken = _talk("Read the README.", "goodbye")

    assert taken == ["Read the README."] and len(ended.turns) == 1
    assert tts.spoken == [READY, ANYTHING_ELSE, STOPPING]


def test_no_goes_back_to_waiting_and_yes_asks_for_the_instruction():
    """Mutation: treat "no" as an instruction -- red, it is read back."""
    _, _, _, tts, taken = _talk("No.", "Yes.", "Read the README.", "stop")

    assert tts.spoken == [READY, OKAY, PROMPT, ANYTHING_ELSE, STOPPING]
    assert taken == ["Read the README."]


def test_silence_is_waited_through_and_announced_once():
    """Mutation: say the waiting line aloud on every silent take -- red, the
    room hears it again and again."""
    conversation, ended, stt, tts, _ = _talk("", "", idle_limit=2)

    assert tts.spoken == [READY]
    assert stt.announced.count(("idle", WAITING)) == 1
    assert ended.reason == "2 silent takes in a row"


def test_without_its_wake_word_an_utterance_is_ignored():
    """Mutation: drop the wake check -- red, the room's talk is read back."""
    _, _, _, tts, taken = _talk("Somebody else talking.", "Computer, read the README.",
                                "Computer.", "computer stop", wake="computer")

    assert taken == ["read the README."]
    assert tts.spoken == [READY, ANYTHING_ELSE, PROMPT, STOPPING]


def test_a_turn_that_fails_is_said_and_the_conversation_goes_on():
    """Mutation: let the exception out -- red, the standing conversation ends."""
    def broken(heard):
        raise RuntimeError("the queue went away")

    _, ended, _, tts, _ = _talk("Read the README.", "stop", turn=broken)

    assert tts.spoken == [READY, FAILED, STOPPING] and ended.reason == "told to stop"


def _request(identifier):
    return SimpleNamespace(id=identifier)


def test_a_waiting_question_is_asked_once_and_its_own_consents_are_not(monkeypatch):
    """Mutation: drop the `asked` set -- red, the question is asked on every
    turn; drop the `OWN` filter -- red, a consent is asked out of its act."""
    asked: list[str] = []

    class Loop:
        def __init__(self, **kw):
            pass

        def run_once(self, request_id):
            asked.append(request_id)
            return SimpleNamespace(response="approve")

    monkeypatch.setattr("qmcp.integrations.voice.adapter.VoiceApprovalLoop", Loop)
    queue = _Queue(_request("agent-question"), _request("instruction-abc"))

    _, ended, _, tts, _ = _talk("", "stop", queue=queue, idle_limit=5)

    assert asked == ["agent-question"] and ended.answered == ["agent-question"]
    assert tts.spoken.count(QUESTION_WAITING) == 1


def test_an_unclear_answer_leaves_the_question_for_anyone(monkeypatch):
    from qmcp.integrations.voice.adapter import UnclearResponse

    class Loop:
        def __init__(self, **kw):
            pass

        def run_once(self, request_id):
            raise UnclearResponse("banana")

    monkeypatch.setattr("qmcp.integrations.voice.adapter.VoiceApprovalLoop", Loop)

    _, ended, _, _, _ = _talk("stop", queue=_Queue(_request("agent-question")))

    assert ended.answered == []


def test_a_queue_that_cannot_be_read_costs_the_conversation_nothing():
    class Down:
        def list_human_requests(self, **kw):
            raise ConnectionError("refused")

    _, ended, _, tts, _ = _talk("stop", queue=Down())

    assert tts.spoken == [READY, STOPPING]


# --- the clone ---------------------------------------------------------------------------


def test_a_first_project_runs_in_the_clone_named_for_it_beside_the_others(tmp_path, monkeypatch):
    """Mutation: return the sibling even when the record knows the clone -- red,
    the remembered clone would be overruled."""
    # Imported before the patch, as a running server has it: `act` binds
    # `last_clone` when first imported, and a first import inside the patch
    # would keep the stand-in for every later test.
    import qmcp.instructions.act  # noqa: F401

    (tmp_path / "qmcp").mkdir()
    remembered: list = [None]
    monkeypatch.setattr("qmcp.instructions.continuity.last_clone",
                        lambda rows, project, before: remembered[0])
    conversation = Conversation(_STT(), _TTS(), _Queue(), None, ["qmcp"], rows=object(),
                                clones=tmp_path)

    assert conversation._clone_for("qmcp", "id") == tmp_path / "qmcp"
    assert conversation._clone_for("quaternionmedia/qmcp", "id") == tmp_path / "qmcp"
    assert conversation._clone_for("vox", "id") is None
    assert conversation._clone_for(None, "id") is None
    remembered[0] = tmp_path / "elsewhere"
    assert conversation._clone_for("qmcp", "id") is None


def test_the_clones_are_looked_for_beside_this_checkout_by_default():
    from pathlib import Path

    import qmcp

    assert converse.sibling_clones() == Path(qmcp.__file__).resolve().parents[2]


# --- waiting for the servers -------------------------------------------------------------


class _Client:
    def __init__(self, up):
        self.up = list(up)

    def health(self):
        if not self.up.pop(0):
            raise ConnectionError("not yet")
        return {"status": "ok"}


class _Engine:
    def __init__(self, up):
        self.up = list(up)

    def reachable(self):
        return self.up.pop(0)


def test_it_waits_for_both_servers_in_either_order_and_says_so_once():
    """Mutation: return when either answers -- red, the engine is not up."""
    lines: list[str] = []
    waits: list[float] = []

    up = wait_for(_Client([False, True, True]), _Engine([False, False, True]),
                  echo=lines.append, sleep=waits.append)

    assert up is True and len(waits) == 2
    assert lines == ["waiting for qmcp and the speech engine"]


def test_a_limit_ends_the_wait():
    lines: list[str] = []

    up = wait_for(_Client([True] * 5), _Engine([False] * 5), echo=lines.append,
                  sleep=lambda s: None, limit=2)

    assert up is False and lines == ["waiting for the speech engine"]


# --- the dialog, entered with a transcript ------------------------------------------------


def test_the_dialog_starts_at_the_read_back_when_it_is_handed_what_was_heard():
    """Mutation: speak the prompt even when handed a transcript -- red, the
    person is asked a question they just answered."""
    from qmcp.instructions.dialog import InstructionDialog

    class Client:
        def create_instruction(self, text, source, project, heard):
            return {"id": "r1", "text": text, "project": "qmcp"}

    stt, tts = _STT("agree"), _TTS()
    row = InstructionDialog(stt, tts, Client(), names=["qmcp"]).run_once(heard="Deploy qmcp.")

    assert row["text"] == "Deploy qmcp."
    assert tts.spoken[0] == "I heard: Deploy qmcp. Say agree or again."
    assert PROMPT not in tts.spoken


@pytest.mark.parametrize("words", ["stop", "stop listening", "Goodbye.", "good bye"])
def test_every_stop_phrase_stops(words):
    _, ended, _, _, _ = _talk(words)

    assert ended.reason == "told to stop"
