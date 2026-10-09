"""The voice-approval loop: speak a prompt, listen, parse yes/no, submit.

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
from qmcp.integrations.voice import vocabulary

# A yes or a no: a word anywhere in the answer, or a phrase inside it
# (`vocabulary.toml`, `answer.*`).
_YES_WORDS = set(vocabulary.words("answer.yes"))
_NO_WORDS = set(vocabulary.words("answer.no"))
_YES_PHRASES = vocabulary.phrases("answer.yes")
_NO_PHRASES = vocabulary.phrases("answer.no")
_UNCLEAR_PHRASES = vocabulary.phrases("answer.unclear")

# Asks for the question again rather than answering it, matched on the whole
# utterance -- what a key on joe's page sends as well. "again" alone is not
# one: it is the read-back's own option.
REPEAT = vocabulary.phrases("conversation.repeat")
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
    """Whether an utterance asks for the question again, by a declared phrase
    or one a person added."""
    return plain_words(text) in (*REPEAT, *vocabulary.overrides("conversation.repeat"))


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


def ask_over(stt, tts, text: str, *, duration: float, pause_ms: int | None = None,
             hint=None) -> None:
    """Say a question so it can be answered before it ends.

    The engine is watched first (`stt.watch`), with the parameters of the
    listen that follows, so an answer said over the question is heard from
    its first word; and the voice stops as soon as the engine says the person
    interrupted (`stt.interrupted`) -- answered by key, held the talk key, or
    spoke over it. A backend without either, or a voice that cannot be cut
    short, says the question whole, as before. The listen is the caller's.
    """
    watch = getattr(stt, "watch", None)
    if callable(watch):
        try:
            watch(duration, pause_ms=pause_ms, hint=list(hint) if hint else None)
        except Exception:  # noqa: BLE001 -- a question that cannot be watched is still asked
            pass
    interrupted = getattr(stt, "interrupted", None)
    if callable(interrupted):
        try:
            tts.speak(text, until=interrupted)
            return
        except TypeError as exc:
            if "until" not in str(exc):
                raise
    tts.speak(text)


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


def say_options(options: list[str]) -> str:
    """The spoken grammar, in as few words as it takes: "Approve or hold?" /
    "Red, green, or blue?"."""
    if len(options) == 1:
        said = f"{options[0]}?"
    elif len(options) == 2:
        said = f"{options[0]} or {options[1]}?"
    else:
        said = f"{', '.join(options[:-1])}, or {options[-1]}?"
    return said[:1].upper() + said[1:]


NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def counted(n: int, word: str) -> str:
    """"one run", "two runs": a count as a listener hears it, never "1 run(s)"."""
    said = NUMBERS[n] if 0 <= n < len(NUMBERS) else str(n)
    return f"{said} {word}" if n == 1 else f"{said} {word}s"


# What a synthesizer would read aloud as syntax: "(s)", markdown, brackets,
# a path's folders, a link. Each is a rule a test names.
_PLURAL_MARK = re.compile(r"\(s\)")
_LINK = re.compile(r"\bhttps?://\S+")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_FRACTION = re.compile(r"\b(\d+)/(\d+)\b")
# A drive or rooted path, anything with two separators, or one separator
# before a file's extension; "and/or" and "3/3" are not paths.
_PATH = re.compile(r"[A-Za-z]:(?:[\\/][^\s\\/]+)+"
                   r"|(?<!\S)[\\/][^\s\\/]+(?:[\\/][^\s\\/]+)*"
                   r"|(?:[^\s\\/]+[\\/]){2,}[^\s\\/]+"
                   r"|[^\s\\/]+[\\/][^\s\\/]+\.[A-Za-z]\w{0,4}\b")
_SYMBOLS = re.compile(r"[*`#_~<>\[\]{}()|\\/]+")


def speakable(text: str) -> str:
    """`text` as a synthesizer should say it: words and ordinary punctuation.

    A "(s)" is dropped, a markdown link keeps its words, a link becomes "a
    link", "3/3" is "3 of 3", a path keeps its last name
    ("C:/work/clones/qmcp" is "qmcp"), and brackets, parentheses, slashes
    and markdown marks go, so a sentence is never read with its syntax.
    Whitespace is collapsed.
    """
    said = _PLURAL_MARK.sub("", text or "")
    said = _MARKDOWN_LINK.sub(r"\1", said)
    said = _LINK.sub("a link", said)
    said = _FRACTION.sub(r"\1 of \2", said)
    said = _PATH.sub(lambda m: re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1], said)
    said = _SYMBOLS.sub(" ", said)
    said = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(said.split()))
    return said


class Speakable:
    """A synthesizer that says every sentence `speakable`, whoever wrote it."""

    def __init__(self, tts):
        self.tts = tts

    def speak(self, text: str, *args, **kwargs):
        return self.tts.speak(speakable(text), *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tts, name)


def speakably(tts):
    """`tts`, wrapped once in `Speakable`."""
    return tts if isinstance(tts, Speakable) or tts is None else Speakable(tts)


# How an accepted answer is acknowledged: the word, as a listener expects it.
ACKNOWLEDGED = {"approve": "Approved.", "hold": "Holding.", "reject": "Rejected.",
                "yes": "Yes.", "no": "No."}


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
    """Raised when a spoken answer never parsed as yes/no within the retry budget."""


class Abandoned(UnclearResponse):
    """Raised when the person said to drop what was being asked ("never mind"):
    nothing is recorded, as for an answer that never parsed."""


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
        self.tts = speakably(tts)
        self.client = client or MCPClient()
        self.max_retries = max_retries
        self.listen_duration = listen_duration
        # Requests `run_forever` asked and got no usable answer to, oldest first.
        self.unanswered: list[str] = []

    def _say(self, text: str, options: list[str]) -> None:
        """Say a question the person may answer before it ends (`ask_over`)."""
        ask_over(self.stt, self.tts, text, duration=self.listen_duration, hint=options)

    def _ask(self, prompt: str, options: list[str]) -> str:
        """Speak the prompt and its options, listen, and return the option chosen.

        A closed-choice dialog in VoiceXML's shape: the prompt says the
        grammar, which is the request's own options; a yes or no maps onto
        them by meaning and an answer naming one is taken directly; and a
        re-ask says which of two things went wrong — nothing heard
        (noinput) or something heard and unusable (nomatch), echoing what
        was heard so the speaker can hear the mishearing. Unclear after the
        retry budget raises, and nothing is guessed. Every question can be
        answered before it ends, by key or by voice (`ask_over`).
        """
        grammar = say_options(options)
        question = f"{prompt} {grammar}"
        self._announce("speaking", question, options=options)
        self._say(question, options)
        heard = ""
        attempt = repeats = 0
        while attempt <= self.max_retries:
            heard, _ = listen_for(self.stt, self.listen_duration, hint=options)
            # Asked to say it again: said again, and no retry spent.
            if asks_repeat(heard) and repeats < MAX_REPEATS:
                repeats += 1
                self._announce("speaking", question, reason="repeat", options=options)
                self._say(question, options)
                continue
            decision = parse_yes_no(heard)
            if decision is not None:
                return choose_option(decision, options)
            named = match_option(heard, options)
            if named is not None:
                return named
            if attempt < self.max_retries:
                if not heard.strip():
                    reask, reason = f"Didn't catch that. {grammar}", "noinput"
                else:
                    reask, reason = f"Heard {heard.strip()[:80].rstrip('.!?')}. {grammar}", "nomatch"
                self._announce("speaking", reask, reason=reason, options=options)
                self._say(reask, options)
            attempt += 1
        self._announce("gave_up", heard.strip())
        raise UnclearResponse(
            f"No usable answer after {self.max_retries + 1} attempts; last heard {heard!r}"
        )

    def run_once(self, request_id: str) -> HumanResponse:
        """Answer one pending request by voice.

        If the request already has a response, returns it without asking
        again. Raises `UnclearResponse` if the spoken answer never parses.
        """
        request, existing = self.client.get_human_request(request_id)
        if existing is not None:
            return existing

        options = request.options or ["approve", "reject"]
        # A request may carry a shorter form for the ear (`context.spoken`):
        # what a page shows in full can be too long to listen to.
        spoken = (request.context or {}).get("spoken")
        answer = self._ask(spoken if isinstance(spoken, str) and spoken.strip() else request.prompt,
                           options)

        response = self.client.submit_human_response(
            request_id=request_id, response=answer, responded_by="vox"
        )
        self._announce("recorded", answer)
        self.tts.speak(ACKNOWLEDGED.get(answer, "Noted."))
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
