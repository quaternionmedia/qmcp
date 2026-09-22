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


class SpeechToText(Protocol):
    def listen(self, duration: float = 5.0) -> tuple[str, str]: ...


class TextToSpeech(Protocol):
    def speak(self, text: str, out_path: str | None = None) -> str: ...


class UnclearResponse(Exception):
    """Raised when a spoken answer never parsed as yes/no within the retry budget."""


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

    def _ask(self, prompt: str) -> bool:
        """Speak `prompt`, listen, parse — re-asking on an unclear answer."""
        self.tts.speak(prompt)
        for attempt in range(self.max_retries + 1):
            transcript, _ = self.stt.listen(duration=self.listen_duration)
            decision = parse_yes_no(transcript)
            if decision is not None:
                return decision
            if attempt < self.max_retries:
                self.tts.speak("Sorry, I didn't catch that. Yes or no?")
        raise UnclearResponse(f"No clear yes/no after {self.max_retries + 1} attempts")

    def run_once(self, request_id: str) -> HumanResponse:
        """Answer one pending request by voice.

        If the request already has a response, returns it without asking
        again. Raises `UnclearResponse` if the spoken answer never parses.
        """
        request, existing = self.client.get_human_request(request_id)
        if existing is not None:
            return existing

        options = request.options or ["approve", "reject"]
        decision = self._ask(request.prompt)
        answer = options[0] if decision else (options[1] if len(options) > 1 else options[0])

        response = self.client.submit_human_response(
            request_id=request_id, response=answer, responded_by="vox"
        )
        self.tts.speak(f"Recorded: {answer}")
        return response

    def run_forever(self, poll_interval: float = 2.0, max_iterations: int | None = None) -> int:
        """Keep answering pending requests as they appear.

        Discovers work via `list_human_requests(status_filter="pending")`,
        which does not expire anything as a side effect (unlike polling
        `get_human_request` directly — see qmcp's AGENTS.md on that hazard).

        `max_iterations` bounds the loop for tests/demos; omit it to run
        until interrupted.

        Returns the number of requests answered.
        """
        answered = 0
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            pending = self.client.list_human_requests(status_filter="pending", limit=1)
            if not pending:
                if max_iterations is None:
                    time.sleep(poll_interval)
                    continue
                break

            try:
                self.run_once(pending[0].id)
                answered += 1
            except (HumanRequestExpiredError, UnclearResponse):
                pass

        return answered
