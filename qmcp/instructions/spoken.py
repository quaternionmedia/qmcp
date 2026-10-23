"""The result of an instruction, said back: what was asked, where, and how it ended.

    uv run qmcp instructions say <id> [--speak]

`summarise` says whether the instruction ran, in which project, and the
outcome's opening sentence, cut at a word boundary; the whole outcome stays on
the row. Every sentence is a function of the row, so every caller says one row
the same way. Markdown markers are dropped before speech.

`docs/integrations/voice.md`, *The result, spoken*, describes what is said and how it is announced.

What this cannot do: tell whether the outcome is true; it repeats the
runtime's first sentence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# How much of an outcome is said. About two breaths: long enough for one
# sentence of what happened, short enough that the loop stays faster than reading.
SPOKEN_CHARS = 160

# Characters a synthesizer would read aloud and a reader would not: the
# emphasis, heading and code marks an agent's markdown carries, and the
# language a code fence names ("```bash"), which would be read as a word.
_FENCE = re.compile(r"```[\w+.-]*")
_MARKS = re.compile(r"[*`#]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
# The number that opens a list item: the period after "1." ends no sentence,
# or a list is said as its lead-in and "1.". A sentence that does end on a
# bare number runs on into the next, to the cut.
_LIST_NUMBER = re.compile(r"(?:^|\s)\d{1,2}\.$")


def first_sentence(text: str, chars: int = SPOKEN_CHARS) -> tuple[str, bool]:
    """The opening sentence of `text`, and whether anything after it was left out.

    Whitespace is collapsed and markdown marks dropped. A sentence longer than
    `chars` is cut at the last word boundary inside it. The flag is True when
    the text went on past what is returned, by a cut or by further sentences.
    """
    flat = " ".join(_MARKS.sub("", _FENCE.sub("", text or "")).split())
    if not flat:
        return "", False
    end = next((found for found in _SENTENCE_END.finditer(flat)
                if not _LIST_NUMBER.search(flat[:found.start()])), None)
    sentence = flat[:end.start()] if end else flat
    cut = len(sentence) < len(flat)
    if len(sentence) > chars:
        sentence = sentence[:chars].rsplit(" ", 1)[0].rstrip(",;:-")
        cut = True
    return sentence, cut


def _closed(sentence: str) -> str:
    """A sentence that ends as one, so two run together are still two."""
    return sentence if sentence[-1:] in ".!?" else sentence + "."


def summarise(row: Mapping[str, Any], why: str = "") -> str:
    """A few sentences a synthesizer can say about one instruction row.

    `row` is the instruction as the server serves it. `why`, when an act was
    refused before it asked, makes a row still `recorded` or `unresolved` say
    that nothing was asked or run.
    """
    from qmcp.integrations.voice.adapter import speakable

    project = row.get("project")
    where = f"in {project}" if project else "with no project"
    status = row.get("status")
    status = getattr(status, "value", status)
    outcome, _ = first_sentence(speakable(row.get("outcome_text") or ""))

    if status == "done":
        return f"Done {where}. {_closed(outcome)}" if outcome else f"Done {where}. Nothing reported."
    if status == "failed":
        return f"Failed {where}. {_closed(outcome)}" if outcome else f"Failed {where}."
    if status == "refused":
        return "Held. Nothing ran."
    if status == "unanswered":
        return "No answer. Nothing ran."
    if status == "asking":
        return f"Waiting for consent {where}."
    if status in ("consented", "acting"):
        return f"Running {where}."
    if why:
        return "Nothing ran."
    if status == "unresolved":
        return "Recorded, no project."
    if status == "recorded":
        return f"Recorded for {project}."
    return f"Instruction {where}: {status}."


def starting(project: str | None) -> str:
    """What is said between the approval and the run, so the wait is not silence."""
    return f"Running in {project}." if project else "Running."


def announce(stt: Any, state: str, text: str, options: Any = None) -> None:
    """Post one state to the engine's conversation route, if it has one.

    The same contract as the approval loop's (`announce_to`): a backend
    without `announce`, or one that fails, is dropped rather than stopping
    the sentence.
    """
    from qmcp.integrations.voice.adapter import announce_to

    announce_to(stt, state, text, options=options)


def say(text: str, tts: Any, stt: Any = None) -> None:
    """Say `text`, with the panel told it is being said and then that the turn is over."""
    from qmcp.integrations.voice.adapter import speakably

    announce(stt, "speaking", text)
    speakably(tts).speak(text)
    announce(stt, "idle", text)


__all__ = [
    "SPOKEN_CHARS",
    "announce",
    "first_sentence",
    "say",
    "starting",
    "summarise",
]
