"""The spoken instruction: ask, listen long, read back, resolve, record.

**AN OPEN QUESTION, THEN A CLOSED ONE.** "What should be done?" has no grammar
for its answer, so the transcript is the answer and the speaker is the only
check on it: it is read back as a closed choice between `record` and `again`,
parsed with the helpers the approval dialog uses (`parse_yes_no`,
`match_option`, `say_options`), so a yes records and a no asks again. The
re-asks say which of two things went wrong, `noinput` or `nomatch`, and the two
turns this dialog adds are announced as `confirm` (the read-back) and `again`
(the speaker asked for another take).

**AN OPEN QUESTION ON THE HUMAN QUEUE AND AN INSTRUCTION ARE ONE DIALOG.** A
request without options, answered by voice, is asked and confirmed in exactly
this shape; the two differ in where the confirmed text goes and in how long the
microphone stays open. Written here beside the inbox, on the shared helpers and
with the same grammar and announcement reasons, so that when both are on one
branch they converge on one implementation rather than two that drift.

**THE TAKE IS LONG, AND THE ANSWER IS SHORT.** A closed-choice answer is a word,
and the engine's own endpointing serves it. An instruction is a sentence with
pauses mid-thought, so the first listen asks the engine for a longer pause
before it ends the take and a longer cap -- `pause_ms` and `listen_duration`,
both options on `qmcp instruct --voice` -- and the confirmation that follows
asks for neither. The protocol's `pause_ms` keyword exists for this one call.

**THE PROJECT IS CONFIRMED, NOT TRUSTED.** Whether a project name transcribes
reliably is unmeasured. A text naming one project resolves; one naming several
is asked back as a closed choice over the candidates by name; one naming none
is asked for the project once, open. An answer that still matches nothing is
recorded `unresolved` rather than dropped -- the person spoke it and confirmed
it, and the inbox is where it is kept until somebody says which project.

**THE ROW SAYS HOW THE PROJECT WAS SETTLED.** A text that named one project is
sent without a project, so the server reads the same text against the same
roster and the row carries the name it matched and the rule, as a typed one
does. A project the person chose from the candidates or spoke when asked is
stated to the server, and the row says `stated` with `source: voice` and the
answer among the transcripts in `heard`: the dialog is the caller that stated
it, and the evidence for its choice is kept beside it.
"""

from __future__ import annotations

from typing import Any, Iterable

from qmcp.instructions import resolve
from qmcp.integrations.voice.adapter import (
    MAX_REPEATS,
    SpeechToText,
    TextToSpeech,
    UnclearResponse,
    announce_to,
    asks_repeat,
    listen_for,
    match_option,
    parse_yes_no,
    say_options,
)

PROMPT = "What should be done?"
WHICH_PROJECT = "Which project?"
CONFIRM = ["record", "again"]

# Seconds an instruction may take, and how long a pause ends it. A sentence
# with a thought in the middle of it; the engine's own default pause is tuned
# for a one-word answer and ended a take mid-sentence.
LISTEN_DURATION = 30.0
PAUSE_MS = 1500


def _said(text: str, chars: int | None = None) -> str:
    """What was heard, to be said back inside a sentence. A transcript carries
    its own closing stop, which the sentence around it would double."""
    return text.strip()[:chars].rstrip(".!?")


class InstructionDialog:
    """Takes one instruction by voice and records it. Nothing runs."""

    def __init__(self, stt: SpeechToText, tts: TextToSpeech, client: Any,
                 names: Iterable[str], max_retries: int = 2,
                 listen_duration: float = LISTEN_DURATION, pause_ms: int = PAUSE_MS,
                 answer_duration: float = 5.0):
        self.stt = stt
        self.tts = tts
        self.client = client
        self.names = tuple(names)
        self.max_retries = max_retries
        self.listen_duration = listen_duration
        self.pause_ms = pause_ms
        # For the confirmation and the project: a word, at the engine's own pause.
        self.answer_duration = answer_duration
        # Every transcript taken, in order; recorded with the row.
        self.heard: list[str] = []

    def run_once(self, heard: str | None = None) -> dict[str, Any]:
        """Ask, confirm, resolve, record. Returns the row as the server recorded it.

        `heard` is a transcript already taken -- the standing conversation
        listens before it knows an instruction is coming -- and starts the
        dialog at the read-back rather than at the prompt.

        Raises `UnclearResponse` when the instruction itself was never
        confirmed within the budget; a project that cannot be settled is
        recorded `unresolved` rather than raised, because by then the person
        has said what they want done.
        """
        self.heard = []
        text = self._ask_instruction(first=heard)
        found = resolve(text, self.names)
        # None when the text itself named the project: the server reads it
        # again and the row carries that match, not a statement.
        project = None
        if found.project is None and found.candidates:
            project = self._ask_choice(WHICH_PROJECT, list(found.candidates))
        elif found.project is None:
            project = self._ask_project_once()
        row = self.client.create_instruction(text, source="voice", project=project,
                                             heard=list(self.heard))
        said = (f"Recorded for {row['project']}." if row.get("project")
                else "Recorded. The project is unresolved.")
        self._announce("recorded", said)
        self.tts.speak(said)
        return row

    def _listen(self, *, long: bool, hint: list[str] | None = None) -> str:
        """One take. The instruction gets the long cap and the long pause; a
        word in answer gets neither, so the engine's own default serves it."""
        if long:
            heard, _ = self.stt.listen(duration=self.listen_duration, pause_ms=self.pause_ms)
        else:
            heard, _ = listen_for(self.stt, self.answer_duration, hint=hint)
        self.heard.append(heard)
        return heard

    def _ask_instruction(self, first: str | None = None) -> str:
        """Speak the prompt, listen long, read the transcript back, return it once confirmed.

        One budget covers the whole exchange, as the approval dialog's open
        question does: every turn the speaker has to be asked a second time
        costs a retry -- nothing heard for the instruction (noinput), nothing
        usable for the confirmation (noinput or nomatch), or `again` -- and
        the turns that move the dialog forward cost nothing.
        """
        grammar = say_options(CONFIRM)
        heard = ""
        answer: str | None = None
        repeats = 0
        if first is not None and first.strip():
            self.heard.append(first)
            heard, answer = first, first.strip()
            readback = f"I heard: {_said(answer)}. {grammar}"
            self._announce("speaking", readback, reason="confirm", options=CONFIRM)
            self.tts.speak(readback)
        else:
            self._announce("speaking", PROMPT)
            self.tts.speak(PROMPT)
        for reasks in range(self.max_retries + 1):
            while True:
                if answer is None:
                    heard = self._listen(long=True)
                    if not heard.strip():
                        reask, reason = f"I didn't hear anything. {PROMPT}", "noinput"
                        break
                    if asks_repeat(heard) and repeats < MAX_REPEATS:
                        repeats += 1
                        self._announce("speaking", PROMPT, reason="repeat")
                        self.tts.speak(PROMPT)
                        continue
                    answer = heard.strip()
                    readback = f"I heard: {_said(answer)}. {grammar}"
                    self._announce("speaking", readback, reason="confirm", options=CONFIRM)
                    self.tts.speak(readback)
                    continue
                heard = self._listen(long=False, hint=CONFIRM)
                # Asked to say it again: the read-back again, and no retry spent.
                if asks_repeat(heard) and repeats < MAX_REPEATS:
                    repeats += 1
                    self._announce("speaking", readback, reason="repeat", options=CONFIRM)
                    self.tts.speak(readback)
                    continue
                decision = parse_yes_no(heard)
                named = match_option(heard, CONFIRM)
                # The decision is consulted before the option named, as the
                # approval dialog does: "don't record" names record and is a no.
                if decision is True or (decision is None and named == "record"):
                    return answer
                if decision is False or named == "again":
                    reask, reason, answer = PROMPT, "again", None
                elif not heard.strip():
                    reask, reason = f"I didn't hear anything. {grammar}", "noinput"
                else:
                    reask, reason = f"I heard: {_said(heard, 80)}. {grammar}", "nomatch"
                break
            if reasks < self.max_retries:
                # A re-ask of the read-back offers its options; a re-ask for
                # the instruction itself has none to offer.
                self._announce("speaking", reask, reason=reason,
                               options=CONFIRM if answer is not None else None)
                self.tts.speak(reask)
        self._announce("gave_up", heard.strip())
        raise UnclearResponse(
            f"No usable instruction after {self.max_retries + 1} attempts; last heard {heard!r}"
        )

    def _ask_choice(self, prompt: str, options: list[str]) -> str | None:
        """A closed choice by name only, or None once the budget is spent.

        The same grammar and re-asks as an approval, without the yes/no
        reading: "yes" to "Which project? Say qmcp or vox." chooses nothing,
        and a project is never picked by position.
        """
        grammar = say_options(options)
        question = f"{prompt} {grammar}"
        self._announce("speaking", question, options=options)
        self.tts.speak(question)
        attempt = repeats = 0
        while attempt <= self.max_retries:
            heard = self._listen(long=False, hint=options)
            if asks_repeat(heard) and repeats < MAX_REPEATS:
                repeats += 1
                self._announce("speaking", question, reason="repeat", options=options)
                self.tts.speak(question)
                continue
            named = match_option(heard, options)
            if named is not None:
                return named
            if attempt < self.max_retries:
                if not heard.strip():
                    reask, reason = f"I didn't hear anything. {grammar}", "noinput"
                else:
                    reask, reason = f"I heard: {_said(heard, 80)}. {grammar}", "nomatch"
                self._announce("speaking", reask, reason=reason, options=options)
                self.tts.speak(reask)
            attempt += 1
        return None

    def _ask_project_once(self) -> str | None:
        """Ask for the project with no grammar to offer, once.

        The text named nothing on the roster, so there are no candidates to
        say. The answer is read against the whole roster the same way the
        text was; one match is the project, anything else leaves it
        unresolved for a person to settle.
        """
        self._announce("speaking", WHICH_PROJECT)
        self.tts.speak(WHICH_PROJECT)
        heard = self._listen(long=False)
        return resolve(heard, self.names).project

    def _announce(self, state: str, text: str = "", reason: str | None = None,
                  options: list[str] | None = None) -> None:
        """Tell whatever shows the exchange what the dialog is doing, as the
        approval dialog does, a question carrying its options; a backend
        without `announce` costs nothing."""
        announce_to(self.stt, state, text, reason, options)

