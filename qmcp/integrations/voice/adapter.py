"""The voice-approval loop: speak a prompt, listen, parse the answer, submit.

A request carrying `options` is a closed choice and the options are its
grammar. A request carrying none is an open question: the transcript itself is
the answer, read back once and recorded when the speaker says so.

Structurally typed against vox's `SpeechToText`/`TextToSpeech` shape rather
than importing vox at module load — this module only needs objects with a
`.listen()`/`.speak()` method, so qmcp does not require the `voice` extra
just to import it.
"""

from __future__ import annotations

import re
import time
from typing import Protocol

from qmcp.client import HumanRequestExpiredError, MCPClient
from qmcp.client.mcp_client import HumanResponse

_YES_WORDS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "affirmative",
    "correct",
    "approve",
    "approved",
    "confirm",
    "confirmed",
}
_NO_WORDS = {
    "no",
    "nope",
    "nah",
    "negative",
    "reject",
    "rejected",
    "deny",
    "denied",
    "stop",
    "cancel",
}
_YES_PHRASES = ("go ahead", "do it", "sounds good")
_NO_PHRASES = ("do not", "dont", "hold off", "not now")
_UNCLEAR_PHRASES = ("not sure", "not certain", "dont know")

# The grammar of an open question's read-back, in the order it is spoken.
# `_ask_open` parses the confirmation with `parse_yes_no` and `match_option`:
# a yes or `record` records the answer, a no or `again` asks the question
# again. Neither word is in the yes/no vocabulary.
_CONFIRM = ["record", "again"]

# Asks for the question again rather than answering it, matched on the whole
# utterance -- what a key on joe's page sends as well. "again" alone is not
# one: it is the read-back's own option.
REPEAT = ("repeat", "repeat that", "say that again", "come again", "pardon", "what",
          "what was that", "sorry")
# Repeats granted per question before a request to repeat is read as no answer.
MAX_REPEATS = 3


def plain_words(text: str) -> str:
    """An utterance as words alone -- lower case, no punctuation -- with a word
    said over and over read once: a transcriber that returns "No. No. No."
    has heard "no"."""
    words = re.sub(r"[^\w\s]", " ", (text or "").lower().replace("'", "")).split()
    if len(words) > 1 and len(set(words)) == 1:
        words = words[:1]
    return " ".join(words)


def asks_repeat(text: str) -> bool:
    """Whether an utterance asks for the question again."""
    return plain_words(text) in REPEAT


def listen_for(stt, duration: float, *, pause_ms: int | None = None, hint=None):
    """`stt.listen`, told the words a short answer is expected to be.

    vox's backends take `hint` and an engine biases its transcription toward
    it; a backend written against the shape before `hint` existed refuses the
    keyword, and is asked again without it rather than failing the question.
    """
    kwargs: dict = {"duration": duration}
    if pause_ms is not None:
        kwargs["pause_ms"] = pause_ms
    if hint:
        kwargs["hint"] = list(hint)
    try:
        return stt.listen(**kwargs)
    except TypeError as exc:
        if "hint" not in kwargs or "hint" not in str(exc):
            raise
        del kwargs["hint"]
        return stt.listen(**kwargs)


def announce_to(stt, state: str, text: str = "", reason: str | None = None,
                options=None) -> None:
    """Post one dialog state to the backend's `announce`, a question carrying
    its options so a display can offer them as keys and buttons. A backend
    without `announce`, one that fails, or one written before options existed
    costs the dialog nothing."""
    post = getattr(stt, "announce", None)
    if post is None:
        return
    extra = {"reason": reason} if reason else {}
    try:
        if options:
            try:
                post(state, text, options=list(options), **extra)
                return
            except TypeError as exc:
                if "options" not in str(exc):
                    raise
        post(state, text, **extra)
    except Exception:
        pass


def parse_yes_no(text: str) -> bool | None:
    """Parse a spoken response as approve (True), reject (False), or unclear (None).

    Whisper transcripts carry punctuation and casing a strict match would
    miss ("Yes.", "No!", "Yeah, go ahead."), so this normalizes first. Negatives
    are checked before positives so a phrase like "no, don't" isn't read as
    a stray match on an unrelated word.
    """
    normalized = re.sub(r"[^\w\s]", "", text.strip().lower())
    words = set(normalized.split())

    if any(phrase in normalized for phrase in _UNCLEAR_PHRASES):
        return None
    if words & _NO_WORDS or any(phrase in normalized for phrase in _NO_PHRASES):
        return False
    if words & _YES_WORDS or any(phrase in normalized for phrase in _YES_PHRASES):
        return True
    return None


def choose_option(decision: bool, options: list[str]) -> str:
    """Pick the option a yes or a no means, by reading the options themselves.

    Position is not meaning. A request carrying `["reject", "approve"]`
    would have a spoken "yes" recorded as `reject` if the first option were
    taken to be the approving one, and nothing in the request says which
    position is which.

    Where no option is recognisable — `["ship it", "wait"]` — position is
    the only information available, and the conventional ordering puts the
    affirmative first.
    """
    wanted = _YES_WORDS if decision else _NO_WORDS
    for option in options:
        if set(re.sub(r"[^\w\s]", "", option.lower()).split()) & wanted:
            return option
    return options[0] if decision else options[-1]


def _words(text: str) -> set[str]:
    """The words of a transcript or an option, lower-cased, punctuation gone.

    A hyphen is a word break and not punctuation: an option `rad-godot` is
    the two words a transcript carries when somebody says it, since no
    transcript writes the hyphen.
    """
    return set(re.sub(r"[^\w\s]", "", text.lower().replace("-", " ")).split())


def match_option(text: str, options: list[str]) -> str | None:
    """The request's own option a transcript names, or None.

    The options are the grammar, as a VoiceXML field's are: an answer that
    says one of them is that answer, whether or not it is a yes/no word.
    An option is named when every word of it appears in the transcript.
    Naming none is no match; naming several is no match unless one of them
    says every word the others do, when it is the one meant -- `rad godot`
    names `rad` too, and the longer option is what was said.
    """
    words = _words(text)
    named = [option for option in options
             if (option_words := _words(option)) and option_words <= words]
    if len(named) == 1:
        return named[0]
    covering = [option for option in named
                if all(_words(other) <= _words(option) for other in named)]
    return covering[0] if len(covering) == 1 else None


def _said(text: str, chars: int | None = None) -> str:
    """What was heard, to be said back inside a sentence. A transcript carries
    its own closing stop, which the sentence around it would double."""
    return text.strip()[:chars].rstrip(".!?")


def say_options(options: list[str]) -> str:
    """The spoken grammar: "Say approve or hold." / "Say red, green, or blue."."""
    if len(options) == 1:
        return f"Say {options[0]}."
    if len(options) == 2:
        return f"Say {options[0]} or {options[1]}."
    return f"Say {', '.join(options[:-1])}, or {options[-1]}."


class SpeechToText(Protocol):
    def listen(self, duration: float = 5.0, *, pause_ms: int | None = None) -> tuple[str, str]:
        """Up to `duration` seconds, transcribed; `pause_ms` is how long a pause
        ends the take early, for a backend that stops when the speaker does.

        The closed-choice dialog never passes it, so a backend's own default
        serves a one-word answer. An instruction has pauses mid-thought and
        asks for a longer one (`qmcp.instructions.dialog`)."""
        ...


class TextToSpeech(Protocol):
    def speak(self, text: str, out_path: str | None = None) -> str: ...


class UnclearResponse(Exception):
    """Raised when no spoken answer was usable within the retry budget."""


class VoiceApprovalLoop:
    """Answers pending qmcp human-approval requests by voice.

    `run_once` handles one request end-to-end, including a retry sub-loop
    when the spoken answer is ambiguous. `run_forever` keeps polling for the
    next pending request after each one — the chat-loop continuation this
    class exists to demonstrate: answering one request does not end the
    session, it goes back to listening for the next.
    """

    def __init__(
        self,
        stt: SpeechToText,
        tts: TextToSpeech,
        client: MCPClient | None = None,
        max_retries: int = 2,
        listen_duration: float = 5.0,
    ):
        self.stt = stt
        self.tts = tts
        self.client = client or MCPClient()
        self.max_retries = max_retries
        self.listen_duration = listen_duration
        # Requests `run_forever` asked and got no usable answer to, oldest first.
        self.unanswered: list[str] = []

    def _ask(self, prompt: str, options: list[str]) -> str:
        """Speak the prompt and its options, listen, and return the option chosen.

        A closed-choice dialog in VoiceXML's shape: the prompt says the
        grammar, which is the request's own options; a yes or no maps onto
        them by meaning and an answer naming one is taken directly; and a
        re-ask says which of two things went wrong — nothing heard
        (noinput) or something heard and unusable (nomatch), echoing what
        was heard so the speaker can hear the mishearing. Unclear after the
        retry budget raises, and nothing is guessed.
        """
        grammar = say_options(options)
        question = f"{prompt} {grammar}"
        self._announce("speaking", question, options=options)
        self.tts.speak(question)
        heard = ""
        attempt = repeats = 0
        while attempt <= self.max_retries:
            heard, _ = listen_for(self.stt, self.listen_duration, hint=options)
            # Asked to say it again: said again, and no retry spent.
            if asks_repeat(heard) and repeats < MAX_REPEATS:
                repeats += 1
                self._announce("speaking", question, reason="repeat", options=options)
                self.tts.speak(question)
                continue
            decision = parse_yes_no(heard)
            if decision is not None:
                return choose_option(decision, options)
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
        self._announce("gave_up", heard.strip())
        raise UnclearResponse(
            f"No usable answer after {self.max_retries + 1} attempts; last heard {heard!r}"
        )

    def _ask_open(self, prompt: str) -> str:
        """Speak the prompt, listen, read the transcript back, and return it once confirmed.

        An open question has no grammar for the answer, so the transcript is
        the answer and the speaker is the only check on it: it is read back
        as a closed choice between `record` and `again`, parsed with the same
        helpers as any closed choice, so a yes records and a no re-asks.

        One budget covers the whole dialog. Every turn the speaker has to be
        asked for a second time costs a retry -- nothing heard for the answer
        (noinput), nothing usable for the confirmation (noinput or nomatch),
        or `again` -- and the turns that move the dialog forward cost nothing.
        Exhausting it raises, and nothing is guessed or recorded.
        """
        grammar = say_options(_CONFIRM)
        self._announce("speaking", prompt)
        self.tts.speak(prompt)
        heard = ""
        answer: str | None = None
        for reasks in range(self.max_retries + 1):
            while True:
                heard, _ = self.stt.listen(duration=self.listen_duration)
                if answer is None:
                    if not heard.strip():
                        reask, reason = f"I didn't hear anything. {prompt}", "noinput"
                        break
                    answer = heard.strip()
                    readback = f"I heard: {_said(answer)}. {grammar}"
                    self._announce("speaking", readback, reason="confirm")
                    self.tts.speak(readback)
                    continue
                decision = parse_yes_no(heard)
                named = match_option(heard, _CONFIRM)
                # The decision is consulted before the option named, as `_ask`
                # does: "don't record" names record and is a no.
                if decision is True or (decision is None and named == "record"):
                    return answer
                if decision is False or named == "again":
                    reask, reason, answer = prompt, "again", None
                elif not heard.strip():
                    reask, reason = f"I didn't hear anything. {grammar}", "noinput"
                else:
                    reask, reason = f"I heard: {_said(heard, 80)}. {grammar}", "nomatch"
                break
            if reasks < self.max_retries:
                self._announce("speaking", reask, reason=reason)
                self.tts.speak(reask)
        self._announce("gave_up", heard.strip())
        raise UnclearResponse(
            f"No usable answer after {self.max_retries + 1} attempts; last heard {heard!r}"
        )

    def run_once(self, request_id: str) -> HumanResponse:
        """Answer one pending request by voice.

        If the request already has a response, returns it without asking
        again. Raises `UnclearResponse` if no spoken answer was usable.
        """
        request, existing = self.client.get_human_request(request_id)
        if existing is not None:
            return existing

        if request.options:
            answer = self._ask(request.prompt, request.options)
        else:
            answer = self._ask_open(request.prompt)

        response = self.client.submit_human_response(
            request_id=request_id, response=answer, responded_by="vox"
        )
        self._announce("recorded", answer)
        self.tts.speak(f"Recorded: {answer}")
        return response

    def _announce(self, state: str, text: str = "", reason: str | None = None,
                  options: list[str] | None = None) -> None:
        """Tell whatever shows the exchange what the dialog is doing.

        The dialog's own states -- `speaking` (with the reason on a re-ask,
        and the question's options), `recorded`, `gave_up` -- go to the STT
        backend's `announce` (`announce_to`).
        """
        announce_to(self.stt, state, text, reason, options)

    def run_forever(self, poll_interval: float = 2.0, max_iterations: int | None = None) -> int:
        """Keep answering pending requests as they appear.

        Discovers work via `list_human_requests(status_filter="pending")`,
        which does not expire anything as a side effect (unlike polling
        `get_human_request` directly — see qmcp's AGENTS.md on that hazard).

        `max_iterations` bounds the loop for tests/demos; omit it to run
        until interrupted.

        **A request is asked once per run.** One that gets no usable answer
        stays pending -- nothing is guessed -- and this loop moves on rather
        than asking it again. Re-asking at once repeated the same question to
        an empty room every few seconds for as long as the loop ran, and a
        person who was not listening came back to a queue the loop had been
        talking at. The request remains answerable by `qmcp human respond`,
        by `qmcp human voice <id>`, or by the next run; `self.unanswered`
        names them.

        Returns the number of requests answered.
        """
        answered = 0
        iterations = 0
        self.unanswered = []
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            # One more than the requests already passed over: at most that
            # many of the oldest can be ones this run has asked.
            pending = self.client.list_human_requests(
                status_filter="pending",
                limit=min(len(self.unanswered) + 1, 500),
                oldest_first=True,
            )
            fresh = [r for r in pending if r.id not in self.unanswered]
            if not fresh:
                if max_iterations is None:
                    time.sleep(poll_interval)
                    continue
                break

            try:
                self.run_once(fresh[0].id)
                answered += 1
            except UnclearResponse:
                self.unanswered.append(fresh[0].id)
            except HumanRequestExpiredError:
                pass

        return answered
