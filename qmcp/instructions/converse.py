"""One standing conversation, so the loop is spoken from end to end.

    uv run qmcp serve --converse --runtime local    # the server, and the conversation beside it
    uv run qmcp converse --runtime local            # the conversation, against a running server

**ONCE THE SERVERS ARE UP, NOTHING IS TYPED.** With the speech engine running
and `qmcp serve --converse`, this says it is ready and asks what should be
done. An instruction is spoken, read back and recorded -- tacitly when the
engine heard it confidently, otherwise on `agree`
(`qmcp.instructions.dialog`); consent is asked aloud and given with `approve`
(`qmcp.instructions.act`); the runtime carries it out in the project's clone;
the outcome is said back (`qmcp.instructions.spoken`); and it asks whether
there is anything else. No id is read off a screen, no flag is typed, no
button is pressed.

**WHAT IS WAITING IS ASKED FIRST.** Before each instruction is taken, questions
an agent has put on the human queue are asked aloud, oldest first, through the
same loop `qmcp human voice` runs -- closed choices by their options, open
questions read back. A question nobody answers stays pending for anyone, and
is not asked again in this conversation.

**NOTHING RUNS WITHOUT A SPOKEN APPROVE.** The instruction is agreed to once it
is read back -- with `agree`, or, heard confidently, by a moment's uninterrupted
silence -- and consent is given with `approve`, after it says what will run
where. Every
runtime is asked, the local model included. Continuity comes from qmcp, not
the model: each run is handed the project's earlier instructions and what
they found, from this server's record.

**THE CLONE IS FOUND WITHOUT A PATH.** A project acted on before runs in the
clone its last act ran in, from the record. A project acted on for the first
time runs in a clone named for it beside this checkout -- the workspace keeps
its repositories as siblings -- or in `--clones`. With neither, the act refuses
before it asks, and the refusal is said.

**SILENCE IS WAITED THROUGH, AND IT ENDS WHEN TOLD.** Nobody speaking is not an
error: the microphone stays open, take after take, until somebody does. "No"
or "that's all" after "Anything else?" goes back to waiting; "yes" asks for the
instruction; "stop listening" or "goodbye" ends the conversation. A turn that
fails is said to have failed and the conversation goes on.

WHAT THIS CANNOT DO. Tell speech meant for it from speech in the room. Every
utterance is read back before it is recorded, and nothing runs without
`approve`; `wake` requires a word to begin an instruction, for a room where
people talk. Nor can it interrupt itself: the microphone is closed while it
speaks, so a person waits for it to finish.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qmcp.instructions.dialog import LISTEN_DURATION, PAUSE_MS, PROMPT, InstructionDialog
from qmcp.integrations.voice.adapter import REPEAT

READY = "Ready. What should be done?"
ANYTHING_ELSE = "Anything else?"
WAITING = "Listening. Say an instruction whenever you are ready."
OKAY = "Okay. I am listening."
STOPPING = "Stopping. Start the server again to talk."
QUESTION_WAITING = "A question is waiting."
FAILED = "That turn failed, and nothing ran. Say the instruction again."
NOT_RECORDED = "Nothing was recorded."

# What ends the conversation, what goes back to waiting, and what asks for an
# instruction, matched on the whole utterance with punctuation and case gone.
STOP = ("stop", "stop listening", "goodbye", "good bye", "stop the conversation", "stop talking")
DONE = ("no", "nope", "nothing", "no thanks", "no thank you", "thats all", "that is all",
        "nothing else", "not now")
MORE = ("yes", "yeah", "yep", "sure", "yes please", "please")
# The answers "Anything else?" offers as keys and buttons, and hints to the engine.
MORE_OPTIONS = ("yes", "no")

# Request ids this conversation puts on the queue itself, asked inside the act
# that created them rather than as a waiting question.
OWN = ("instruction-",)


def plain(text: str) -> str:
    """An utterance as words alone: lower case, no punctuation, single spaces,
    and a word said over and over read once ("No. No. No." is "no")."""
    words = re.sub(r"[^\w\s]", "", (text or "").lower().replace("'", "")).split()
    if len(words) > 1 and len(set(words)) == 1:
        words = words[:1]
    return " ".join(words)


def sibling_clones() -> Path:
    """Where the workspace keeps repositories: beside this checkout."""
    return Path(__file__).resolve().parents[3]


@dataclass
class Turn:
    """One instruction taken in the conversation, and how it ended."""

    instruction_id: str | None
    text: str
    status: str
    summary: str
    carried: tuple[str, ...] = ()


@dataclass
class Ended:
    """What a conversation did, for a caller that ran it to an end."""

    turns: list[Turn] = field(default_factory=list)
    answered: list[str] = field(default_factory=list)
    """Waiting questions answered by voice, by request id."""
    reason: str = ""


class Conversation:
    """Takes instructions by voice, acts on each with consent, and says what happened."""

    def __init__(self, stt: Any, tts: Any, client: Any, runtime: Any, names: Iterable[str], *,
                 rows: Any = None, clones: Path | None = None, wake: str | None = None,
                 listen_duration: float = LISTEN_DURATION, pause_ms: int = PAUSE_MS,
                 answer_duration: float = 5.0, max_retries: int = 2,
                 idle_limit: int | None = None, poll_interval: float = 1.0,
                 echo: Callable[[str], None] | None = None,
                 tacit_above: float | None = None) -> None:
        self.stt = stt
        self.tts = tts
        self.client = client
        self.runtime = runtime
        self.names = tuple(names)
        self.rows = rows
        self.clones = Path(clones) if clones is not None else sibling_clones()
        self.wake = plain(wake) if wake else None
        self.listen_duration = listen_duration
        self.pause_ms = pause_ms
        self.answer_duration = answer_duration
        self.max_retries = max_retries
        # Consecutive silent takes before the conversation ends on its own. None
        # -- the default -- waits for ever, which is what a standing conversation
        # does; a check sets it so a script that runs out of words ends.
        self.idle_limit = idle_limit
        self.poll_interval = poll_interval
        self.echo = echo or (lambda line: None)
        # The engine confidence at or above which an instruction's read-back
        # asks nothing and silence agrees (`InstructionDialog`); None always asks.
        self.tacit_above = tacit_above
        self._confidence: float | None = None
        self._asked: set[str] = set()
        # The question the next take answers, and the options it offered, so
        # "repeat" can say it again and the engine can be hinted.
        self._question: tuple[str, tuple[str, ...]] = (READY, ())

    # --- speaking ---------------------------------------------------------------

    def say(self, text: str) -> None:
        from qmcp.instructions.spoken import say

        self.echo(f"said: {text}")
        say(text, self.tts, self.stt)

    def ask(self, text: str, options: tuple[str, ...] = ()) -> None:
        """Say a question and leave the turn to the person: announced as
        `speaking`, with its options, and not closed with `idle`, so the
        engine knows the next take answers it -- joe cues the person to
        speak -- and the page can offer the options."""
        from qmcp.instructions.spoken import announce

        self._question = (text, tuple(options))
        self.echo(f"said: {text}")
        announce(self.stt, "speaking", text, options=list(options) or None)
        self.tts.speak(text)

    def _announce(self, state: str, text: str) -> None:
        from qmcp.instructions.spoken import announce

        announce(self.stt, state, text)

    # --- the conversation -------------------------------------------------------

    def run(self) -> Ended:
        """Talk until told to stop, or until `idle_limit` silent takes in a row."""
        from qmcp.integrations.voice.adapter import listen_for

        ended = Ended()
        self.ask(READY)
        idle = 0
        while True:
            self._ask_waiting(ended)
            heard, recording = listen_for(self.stt, self.listen_duration, pause_ms=self.pause_ms,
                                          hint=self._question[1])
            # An answer with no recording behind it came from a key or a button
            # on the engine's page: it was meant for this conversation.
            keyed = not recording
            self._confidence = getattr(self.stt, "last_confidence", None)
            words = plain(heard)
            if not words:
                idle += 1
                if idle == 1:
                    self._announce("idle", WAITING)
                if self.idle_limit is not None and idle >= self.idle_limit:
                    ended.reason = f"{idle} silent takes in a row"
                    return ended
                continue
            idle = 0
            self.echo(f"heard: {heard.strip()}")
            if self.wake and not keyed:
                if not words.startswith(self.wake):
                    continue
                heard = _after(heard, self.wake)
                words = plain(heard)
                if not words:
                    self.say(PROMPT)
                    continue
            if words in STOP:
                self.say(STOPPING)
                ended.reason = "told to stop"
                return ended
            if words in REPEAT:
                self.ask(*self._question)
                continue
            if words in DONE:
                self.ask(OKAY)
                continue
            if words in MORE:
                self.ask(PROMPT)
                continue
            try:
                ended.turns.append(self._take(heard))
            except Exception as exc:  # noqa: BLE001 -- a standing conversation outlives a turn
                self.echo(f"turn failed: {type(exc).__name__}: {exc}")
                self.ask(FAILED)
                continue
            self.ask(ANYTHING_ELSE, MORE_OPTIONS)

    def _take(self, heard: str) -> Turn:
        """One instruction: read back and recorded, then acted on with consent, then said."""
        from qmcp.instructions.act import RULE_SIBLING, act
        from qmcp.instructions.spoken import starting, summarise
        from qmcp.integrations.voice.adapter import UnclearResponse
        from qmcp.spend import Budget

        dialog = InstructionDialog(stt=self.stt, tts=self.tts, client=self.client,
                                   names=self.names, max_retries=self.max_retries,
                                   listen_duration=self.listen_duration,
                                   pause_ms=self.pause_ms, answer_duration=self.answer_duration,
                                   tacit_above=self.tacit_above)
        try:
            row = dialog.run_once(heard=heard, confidence=self._confidence)
        except UnclearResponse:
            self.say(NOT_RECORDED)
            return Turn(None, heard.strip(), "not recorded", NOT_RECORDED)
        project = row.get("project")
        self.echo(f"recorded: {row['id']} for {project or 'no project'}")

        def event(state: str, text: str) -> None:
            if state == "acting":
                self.say(starting(project))
            elif state == "output":
                self.echo(f"read: {text}")

        done = act(row["id"], self.runtime, Budget(authorised=1), client=self.client,
                   rows=self.rows, cwd=self._clone_for(project, row["id"]),
                   cwd_rule=RULE_SIBLING, stt=self.stt, tts=self.tts, on_event=event,
                   poll_interval=self.poll_interval)
        recorded = self.client.get_instruction(row["id"])
        summary = summarise(recorded, why=done.why)
        if done.why and not done.ran:
            self.echo(f"why: {done.why}")
        self.say(summary)
        return Turn(row["id"], row["text"], recorded["status"], summary, done.carried)

    def _clone_for(self, project: str | None, instruction_id: str) -> Path | None:
        """None when the record knows the project's clone -- the act reads it
        there -- else the sibling clone named for the project, if one exists."""
        from qmcp.instructions import act as act_module
        from qmcp.instructions.continuity import last_clone

        if not project:
            return None
        rows = self.rows or act_module.configured_rows()
        self.rows = rows
        if last_clone(rows, project, before=instruction_id) is not None:
            return None
        candidate = self.clones / project.rsplit("/", 1)[-1]
        return candidate if candidate.is_dir() else None

    def _ask_waiting(self, ended: Ended) -> None:
        """Ask aloud what agents have put on the human queue, oldest first, once each."""
        from qmcp.integrations.voice.adapter import UnclearResponse, VoiceApprovalLoop

        try:
            pending = self.client.list_human_requests(status_filter="pending", limit=20,
                                                      oldest_first=True)
        except Exception:  # noqa: BLE001 -- an unreachable queue costs this turn nothing
            return
        for request in pending:
            if request.id in self._asked or request.id.startswith(OWN):
                continue
            self._asked.add(request.id)
            self.say(QUESTION_WAITING)
            try:
                answer = VoiceApprovalLoop(stt=self.stt, tts=self.tts,
                                           client=self.client).run_once(request.id)
            except UnclearResponse:
                continue
            ended.answered.append(request.id)
            self.echo(f"answered: {request.id} -> {answer.response}")


def wait_for(client: Any, stt: Any, echo: Callable[[str], None] = print,
             sleep: Callable[[float], None] | None = None, every: float = 2.0,
             limit: int | None = None) -> bool:
    """Wait until this server and the speech engine both answer, in either order.

    The conversation is started with the server, and the engine may come up
    before it or long after; neither is a reason to fail. Says once what it is
    waiting for. `limit` bounds the waits for a test; None waits for ever.
    Returns True once both answer, False when `limit` ran out first.
    """
    import time

    sleep = sleep or time.sleep
    said = False
    waits = 0
    while True:
        try:
            client.health()
            server = True
        except Exception:  # noqa: BLE001 -- not up yet is the expected answer
            server = False
        engine = bool(stt.reachable())
        if server and engine:
            return True
        if not said:
            missing = [name for name, up in (("qmcp", server), ("the speech engine", engine))
                       if not up]
            echo(f"waiting for {' and '.join(missing)}")
            said = True
        if limit is not None and waits >= limit:
            return False
        waits += 1
        sleep(every)


def _after(heard: str, wake: str) -> str:
    """The utterance with its wake word, and the punctuation after it, taken off."""
    words = heard.strip().split()
    taken = len(wake.split())
    return " ".join(words[taken:]).lstrip(",.!? ")


__all__ = [
    "ANYTHING_ELSE",
    "DONE",
    "MORE",
    "READY",
    "STOP",
    "STOPPING",
    "Conversation",
    "Ended",
    "Turn",
    "plain",
    "sibling_clones",
    "wait_for",
]
