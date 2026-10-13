"""The result of an instruction, said back: what was asked, where, and how it ended.

    uv run qmcp instructions say <id> [--speak]

**THE RECORD IS THE RESULT; THE SENTENCE POINTS AT IT.** An agent's outcome
can run to pages, and nobody listens to pages. `summarise` says whether the
instruction ran, in which project, and the outcome's opening sentence, cut at
a word boundary; when it had to cut, it says the rest is on the record, which
is where the whole text stays (`qmcp instructions show <id>`). A summary that
read the whole outcome aloud would make the spoken loop slower than reading
the terminal, which is the one thing it exists not to be.

**IT SAYS WHAT THE ROW SAYS, AND NOTHING THE ROW DOES NOT.** Every sentence is
a function of the row's status, project, text, outcome and exit code, so the
same row is said the same way by `instructions act --voice`, by
`instructions say`, and by the offline check. A refusal before anything was
asked leaves the row as it was, so the caller passes `why` to have the summary
say nothing ran -- not the reason itself, which names flags and paths and is
written for a terminal.

**THE PANEL IS TOLD IN STATES IT ALREADY HAS.** The speech engine accepts a
fixed set of posted states and refuses any other. The summary is announced as
`speaking` while it is said, and the turn then ends `idle` carrying it, so the
panel shows it as the last thing said. It is never announced as `recorded`,
which the panel shows as a person's answer accepted. An engine without the
route, or an announcement that fails, costs the sentence nothing.

WHAT THIS CANNOT DO. Tell whether the outcome is true. The runtime reported
it; this repeats its first sentence. Markdown markers are dropped because a
synthesizer reads them aloud, and nothing else about the text is changed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

# How much of an outcome is said. About two breaths: long enough for one
# sentence of what happened, short enough that the loop stays faster than reading.
SPOKEN_CHARS = 160

REST = "The rest is on the record."

# Characters a synthesizer would read aloud and a reader would not: the
# emphasis, heading and code marks an agent's markdown carries.
_MARKS = re.compile(r"[*`#]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def first_sentence(text: str, chars: int = SPOKEN_CHARS) -> tuple[str, bool]:
    """The opening sentence of `text`, and whether anything after it was left out.

    Whitespace is collapsed and markdown marks dropped. A sentence longer than
    `chars` is cut at the last word boundary inside it. The flag is True when
    the text went on past what is returned, by a cut or by further sentences.
    """
    flat = " ".join(_MARKS.sub("", text or "").split())
    if not flat:
        return "", False
    end = _SENTENCE_END.search(flat)
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
    project = row.get("project")
    where = f"in {project}" if project else "with no project"
    status = row.get("status")
    status = getattr(status, "value", status)
    asked, _ = first_sentence(row.get("text") or "")
    outcome, cut = first_sentence(row.get("outcome_text") or "")
    rest = f" {REST}" if cut else ""

    if status == "done":
        if not outcome:
            return f"Done {where}, and the run reported nothing."
        return f"Done {where}. {_closed(outcome)}{rest}"
    if status == "failed":
        code = row.get("exit_code")
        head = f"The run {where} failed" + (f", exit {code}." if code is not None else ".")
        return f"{head} {_closed(outcome)}{rest}" if outcome else head
    if status == "refused":
        return f"Held. Nothing ran for: {_closed(asked)}"
    if status == "unanswered":
        return f"Nobody answered in time, so nothing ran for: {_closed(asked)}"
    if status == "asking":
        return f"Waiting for consent to act {where}."
    if status in ("consented", "acting"):
        return f"Running {where}."
    if why:
        return f"Nothing was asked and nothing ran {where}."
    if status == "unresolved":
        return "Recorded with no project. Nothing has run."
    if status == "recorded":
        return f"Recorded for {project}. Nothing has run."
    return f"The instruction {where} is {status}."


def starting(project: str | None) -> str:
    """What is said between the approval and the run, so the wait is not silence."""
    return f"Approved. Running in {project}." if project else "Approved. Running."


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
    announce(stt, "speaking", text)
    tts.speak(text)
    announce(stt, "idle", text)


__all__ = [
    "REST",
    "SPOKEN_CHARS",
    "announce",
    "first_sentence",
    "say",
    "starting",
    "summarise",
]
