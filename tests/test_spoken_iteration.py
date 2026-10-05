"""What the standing conversation does itself between instructions.

"try again" and "same in <project>" take the last instruction again, recorded
anew and asked for consent; "never mind" drops what is being asked; and the
diagnostics say how the loop stands -- what it heard, how the last instruction
went, what is waiting, that it can hear, which projects it knows, and what can
be said. The act itself is stood in for: these are about the loop.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from qmcp.instructions import converse
from qmcp.instructions.converse import (
    ANYTHING_ELSE,
    HEARING_YOU,
    HELP_SAID,
    NO_SUCH_PROJECT,
    NOTHING_HEARD,
    NOTHING_RAN,
    NOTHING_TO_TRY,
    OKAY,
    READY,
    STOPPING,
    Conversation,
    same_in,
)
from qmcp.integrations.voice.adapter import Abandoned, parse_yes_no


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


class _Client:
    def __init__(self, waiting=0):
        self.created: list[dict] = []
        self.waiting = waiting

    def list_human_requests(self, **kw):
        return [SimpleNamespace(id=f"agent-{i}") for i in range(self.waiting)]

    def get_human_request(self, request_id):
        # Answered elsewhere already: asked before a take, it is not asked aloud.
        return SimpleNamespace(id=request_id), SimpleNamespace(response="approve")

    def list_instructions(self, limit=50):
        return []

    def get_instruction(self, instruction_id):
        return {"id": instruction_id, "project": "qmcp"}

    def create_instruction(self, text, source, project, heard):
        row = {"id": f"again-{len(self.created)}", "text": text, "project": project}
        self.created.append(row)
        return row


def _talk(*takes, client=None, names=("qmcp", "joe", "vox")):
    stt, tts, client = _STT(*takes), _TTS(), client or _Client()
    conversation = Conversation(stt, tts, client, runtime=None, names=list(names), idle_limit=2)
    acted: list[dict] = []

    def take(heard):
        return converse.Turn("first", heard, "done", "Done in qmcp. It reads well.")

    def act_on(row):
        acted.append(row)
        return converse.Turn(row["id"], row["text"], "done", f"Done in {row['project']}.")

    conversation._take = take
    conversation._act_on = act_on
    ended = conversation.run()
    return ended, tts, client, acted


# --- the last instruction again ------------------------------------------------------


def test_try_again_records_the_last_instruction_anew_and_acts_on_it():
    """Mutation: drop try-again from the loop -- red, "try again" is taken as
    an instruction of its own."""
    ended, tts, client, acted = _talk("Read the README in qmcp.", "try again", "stop")

    assert client.created == [{"id": "again-0", "text": "Read the README in qmcp.", "project": "qmcp"}]
    assert acted == client.created and len(ended.turns) == 2
    assert tts.spoken == [READY, ANYTHING_ELSE, ANYTHING_ELSE, STOPPING]


def test_same_in_another_project_records_it_for_that_project():
    """Mutation: never read the project out of "same in" -- red."""
    _, _, client, acted = _talk("Read the README in qmcp.", "Same in vox.", "stop")

    assert [row["project"] for row in acted] == ["vox"]


def test_same_in_a_project_nobody_knows_records_nothing():
    _, tts, client, _ = _talk("Read the README in qmcp.", "same in nowhere", "stop")

    assert client.created == [] and NO_SUCH_PROJECT in tts.spoken


def test_try_again_before_anything_was_taken_says_so():
    _, tts, client, _ = _talk("try again", "stop")

    assert client.created == [] and tts.spoken == [READY, NOTHING_TO_TRY, STOPPING]


def test_the_project_words_are_read_out_of_every_same_in_phrase():
    assert same_in("same in rad godot") == "rad godot"
    assert same_in("do that in vox") == "vox"
    assert same_in("same in") is None and same_in("read the readme") is None


# --- never mind ---------------------------------------------------------------------


def test_never_mind_between_instructions_goes_back_to_waiting():
    _, tts, _, acted = _talk("never mind", "stop")

    assert tts.spoken == [READY, OKAY, STOPPING] and acted == []


def test_never_mind_at_the_read_back_drops_the_instruction_unrecorded():
    """Mutation: drop the never-mind check from the read-back -- red, the
    instruction is taken again instead."""
    from qmcp.instructions.dialog import InstructionDialog

    client = _Client()
    dialog = InstructionDialog(stt=_STT("never mind"), tts=_TTS(), client=client, names=["qmcp"])

    with pytest.raises(Abandoned):
        dialog.run_once(heard="Deploy qmcp.")
    assert client.created == []


def test_never_mind_to_a_consent_is_a_hold():
    assert parse_yes_no("Never mind.") is False


# --- how the loop stands --------------------------------------------------------------


@pytest.mark.parametrize(("takes", "said"), [
    (("what did you hear",), NOTHING_HEARD),
    (("Read the README in qmcp.", "what did you hear"), "I heard: Read the README in qmcp."),
    (("how did that go",), NOTHING_RAN),
    (("Read the README in qmcp.", "how did it go"), "Done in qmcp. It reads well."),
    (("can you hear me",), HEARING_YOU),
    (("which projects",), "I know qmcp, joe and vox."),
    (("help",), HELP_SAID),
])
def test_each_diagnostic_says_how_the_loop_stands(takes, said):
    """Mutation: drop the diagnostics from the loop -- red, each is taken as an
    instruction."""
    _, tts, client, acted = _talk(*takes, "stop")

    assert tts.spoken[-2] == said and client.created == []


@pytest.mark.parametrize(("waiting", "said"), [
    (0, "Nothing is waiting."), (1, "One question is waiting."), (2, "Two questions are waiting."),
])
def test_whats_waiting_counts_the_questions(waiting, said):
    _, tts, _, _ = _talk("whats waiting", "stop", client=_Client(waiting=waiting))

    assert said in tts.spoken


def test_a_long_roster_is_named_six_and_counted():
    _, tts, _, _ = _talk("which projects", "stop", names=[f"p{i}" for i in range(8)])

    assert "I know p0, p1, p2, p3, p4 and p5, and 2 more." in tts.spoken


def test_the_conversations_own_consents_are_not_counted_as_waiting():
    """Mutation: count every pending request -- red."""

    class Own(_Client):
        def list_human_requests(self, **kw):
            return [SimpleNamespace(id="agent-1"), SimpleNamespace(id="instruction-abc")]

    _, tts, _, _ = _talk("whats waiting", "stop", client=Own())

    assert "One question is waiting." in tts.spoken
