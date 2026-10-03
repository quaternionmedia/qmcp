"""Where was I: a project's latest work, read from the archive, in sentences.

    uv run qmcp threads recall <project>            # printed
    uv run qmcp threads recall <project> --speak    # said, through the synthesizer
    uv run qmcp threads recall <project> --json     # as data
    GET  /v1/threads/recall/<project>               # the same data, on loopback

**CONTINUITY COMES FROM THE RECORD, NOT FROM WHICHEVER CONVERSATION IS OPEN.**
A person with several sessions running, in several repositories, asks "where
was I in qmcp" and the honest answer lives in none of the windows: it is in the
session store, which already knows which branch each session was on, which
checkout it ran in, and which pull requests it opened. This reads that store
and says the latest of it. It is the first thing a spoken instruction loop
needs -- an answer about a project that is the same after every process has
restarted, because nothing about it was held in a process.

**READ-ONLY, AND NOTHING IS SPENT.** Every source is read with a budget of
nothing, no agent runs, and no network is reached. A recall that could spend
would be a question whose cost depended on who asked it.

**WHICH SESSION IS "THE LAST" IS A CLAIM, AND THE RULE TRAVELS WITH IT.** Two
rules compose here and neither is this module's own. Which threads are *about*
the project is `consolidate.about`, with its evidence and its stated rule -- a
second matcher here would be a second opinion about the same conversations,
and the two would disagree the day one was tuned. A thread that surveys the
workspace is not about any one project in it, so it is passed over, as
`consolidate` already passes it over for relations. Among the threads about the
project, the one whose last turn is latest is chosen; a thread with no
timestamp at all cannot be latest and sorts last.

**NOTHING CHOSEN IS AN ANSWER, NOT AN ERROR.** A project nobody has talked
about yet, or whose sessions all surveyed the roster rather than working in it,
gets a `Recall` with no thread and a sentence saying so. An exception here would
read, at a speaker, as the archive being broken rather than as the archive
being empty of this.

**THE CHECKOUT IS REPORTED TWICE, AND THE SECOND TIME IS MEASURED.** A session
names its working directory, and that directory may have been a worktree
somebody has since removed. `cwd` is what the session said; `cwd_exists` is
whether `Path.is_dir()` says so now. A spoken "in such-and-such a folder" that
sent somebody to a directory that is gone is the kind of confident wrong answer
this archive exists to avoid, so the two are never collapsed into one field.

WHAT THIS CANNOT SEE. Whether the session's work landed. A pull request named
here was opened; the archive does not know if it merged, and the sentence says
"opened" rather than anything stronger. Whether a thread is about a project at
all is `consolidate`'s reading, and its known weakness -- a long session that
names a repository twice in passing -- is inherited rather than papered over.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qmcp.spend import Budget
from qmcp.threads.base import Thread
from qmcp.threads.consolidate import about

# How many of the thread's final turns are carried, and how long each may be
# once flattened for speech. A synthesizer reading two hundred characters takes
# about as long as a person is willing to listen to a status, which is a
# judgement and not a measurement; the full turns stay in the archive.
LAST_TURNS = 3
SPEECH_CHARS = 200

# The embedded corpus, whose roster names the projects. Resolved from this file
# rather than from the working directory, because the server that answers the
# route was started from wherever somebody started it.
EMBEDDED_CORPUS = Path(__file__).resolve().parents[2] / "governance" / "qm"


@dataclass(frozen=True)
class Said:
    """One turn, flattened for a speaker. `text` is an excerpt and says so."""

    role: str
    at: str | None
    text: str
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "at": self.at, "text": self.text,
                "truncated": self.truncated}


@dataclass(frozen=True)
class Recall:
    """What the archive says a project's latest session was.

    Every field describing the chosen thread is `None` or empty when nothing was
    chosen, and `chosen` says which case this is. `considered` and `read` are
    both carried because they answer different questions: how many threads
    were about this project, and how many were looked at to find them.
    """

    project: str
    rule: str
    read: int = 0
    considered: int = 0
    as_of: str | None = None

    source: str | None = None
    thread: str | None = None
    title: str | None = None
    started: str | None = None
    last_activity: str | None = None
    branches: tuple[str, ...] = ()
    cwd: str | None = None
    cwd_exists: bool | None = None
    pulls: tuple[tuple[str, int], ...] = ()
    last_turns: tuple[Said, ...] = field(default_factory=tuple)

    @property
    def chosen(self) -> bool:
        return self.thread is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "chosen": self.chosen,
            "source": self.source,
            "thread": self.thread,
            "title": self.title,
            "started": self.started,
            "last_activity": self.last_activity,
            "branches": list(self.branches),
            "cwd": self.cwd,
            "cwd_exists": self.cwd_exists,
            "pulls": [{"repository": repository, "number": number}
                      for repository, number in self.pulls],
            "last_turns": [said.as_dict() for said in self.last_turns],
            "considered": self.considered,
            "read": self.read,
            "rule": self.rule,
            "as_of": self.as_of,
            "spoken": self.spoken(),
        }

    def spoken(self) -> str:
        """A few short sentences a synthesizer can say.

        Everything a listener could act on is in the first two sentences: the
        branch, the checkout and whether it is still there, the pull requests.
        The excerpt comes after, because it is the longest part and the least
        decisive. The count closes it so a listener knows how much was behind
        the answer.
        """
        if not self.chosen:
            looked = (f"{self.read} thread{'s' if self.read != 1 else ''} "
                      f"{'were' if self.read != 1 else 'was'} read")
            return (f"Nothing in the archive is about {self.project}. {looked}; "
                    f"the rule was: {self.rule}.")

        parts: list[str] = []
        opening = f"In {self.project}, the last session"
        if self.title:
            opening += f", titled {self.title},"
        age = _age(self.last_activity, self.as_of)
        opening += f" was {age}" if age else " was"
        if self.branches:
            branches = ", ".join(self.branches)
            opening += (f" on branch {branches}" if len(self.branches) == 1
                        else f" on branches {branches}")
        if self.cwd:
            opening += f" in {self.cwd}"
            if self.cwd_exists is False:
                opening += ", which is no longer on disk"
        parts.append(opening + ".")

        if self.pulls:
            named = ", ".join(f"pull request {number} in {repository}"
                              for repository, number in self.pulls)
            parts.append(f"It opened {named}.")
        else:
            parts.append("It opened no pull request.")

        if self.last_turns:
            last = self.last_turns[-1]
            # An excerpt that ends mid-sentence runs straight into the count
            # when spoken, so it is closed here if the turn did not close it.
            quoted = last.text if last.text.endswith((".", "!", "?")) else last.text + "."
            parts.append(f"Its last turn said: {quoted}")

        counted = (f"{self.considered} session{'s' if self.considered != 1 else ''} "
                   f"about {self.project} {'were' if self.considered != 1 else 'was'} "
                   f"read, of {self.read} in all.")
        parts.append(counted)
        return " ".join(parts)


def names_for(project: str, corpus: Path | None = None) -> dict[str, str]:
    """The roster `about` is read against, with this project in it.

    The whole roster rather than the one name, because `about`'s survey rule
    needs to know how many repositories a thread named out of how many exist --
    a thread listing the workspace is not about the project it happened to
    include. Without a roster the project is the only name, and the survey rule
    cannot fire; that is a weaker reading and the returned rule says what it
    was read against.
    """
    from qmcp.threads.consolidate import roster

    where = corpus if corpus is not None else EMBEDDED_CORPUS
    names: dict[str, str] = {}
    if (where / "ci" / "workspace.yaml").is_file():
        names = roster(where)
    names.setdefault(project, project)
    return names


def recall(project: str, sources: Iterable[Any], names: Iterable[str],
           now: datetime | None = None, *, turns: int = LAST_TURNS,
           chars: int = SPEECH_CHARS) -> Recall:
    """The latest session about `project`, from every source given.

    Reads each source with a budget of nothing. A source that keeps a
    `context` per thread -- the session store does -- contributes branches,
    checkout and pull requests; a source that does not contributes the thread
    alone, and the fields stay empty rather than guessed.
    """
    known = list(names)
    moment = now or datetime.now(UTC)
    read = 0
    candidates: list[tuple[datetime | None, str, Thread, Any]] = []
    rule = ""

    for source in sources:
        for thread in source.fetch([], Budget()):
            read += 1
            reading = about(thread, known)
            rule = rule or reading.rule
            # `relation` rather than `projects`: a survey of the workspace has
            # projects and no relation, and `consolidate` already declines to
            # relate it to any of them. The same reading applies here.
            if project not in reading.projects or reading.relation is None:
                continue
            candidates.append((_last_activity(thread), thread.id, thread, source))

    if not rule:
        rule = about(Thread(id="-"), known).rule
    rule += "; a thread surveying the workspace is passed over"

    if not candidates:
        return Recall(project=project, rule=rule, read=read, considered=0,
                      as_of=_stamp(moment))

    # Latest last turn first. `None` sorts after every real time, so a thread
    # with no timestamps is chosen only when nothing dated is about the project.
    candidates.sort(key=lambda c: (c[0] is None,
                                   -(c[0].timestamp()) if c[0] else 0,
                                   c[1]))
    when, _, thread, source = candidates[0]
    context = getattr(source, "context", {}).get(thread.id, {})
    cwds = list(context.get("cwds") or [])
    cwd = cwds[-1] if cwds else None

    return Recall(
        project=project,
        rule=rule,
        read=read,
        considered=len(candidates),
        as_of=_stamp(moment),
        source=getattr(source, "name", None) or type(source).__name__,
        thread=thread.id,
        title=thread.title,
        started=thread.started_at,
        last_activity=_stamp(when) if when else None,
        branches=tuple(sorted(context.get("branches") or [])),
        cwd=cwd,
        # Measured now, not remembered. The session said where it ran; whether
        # that directory is still there is a fact about this machine today.
        cwd_exists=Path(cwd).is_dir() if cwd else None,
        pulls=tuple(sorted(context.get("pulls") or [])),
        last_turns=tuple(_said(turn, chars) for turn in thread.turns[-turns:]),
    )


# --- helpers ------------------------------------------------------------------


def _when(value: str | None) -> datetime | None:
    """An ISO 8601 timestamp as an aware datetime, or None when it is not one.

    A naive time is read as UTC rather than refused: the session store writes
    a trailing `Z`, and a source that wrote none has still said *when*.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _last_activity(thread: Thread) -> datetime | None:
    """When the thread was last spoken in: the latest turn, else when it began."""
    times = [t for t in (_when(turn.at) for turn in thread.turns) if t is not None]
    if times:
        return max(times)
    return _when(thread.started_at)


def _age(then: str | None, now: str | None) -> str | None:
    """How long ago, in the units a person would use aloud."""
    start, end = _when(then), _when(now)
    if start is None or end is None:
        return None
    seconds = max(0, int((end - start).total_seconds()))
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        count = seconds // size
        if count:
            return f"{count} {unit}{'s' if count != 1 else ''} ago"
    return "moments ago"


_MARKUP = re.compile(r"[`*_#>\[\]|]")
_SPACE = re.compile(r"\s+")


def for_speech(text: str, chars: int = SPEECH_CHARS) -> tuple[str, bool]:
    """A turn's text as one line a synthesizer can read, and whether it was cut.

    Markup characters go, because a speaker reading "backtick" and "hash" is
    reading the formatting rather than the words. The cut lands on a word
    boundary and is marked, so a listener hears that there was more.
    """
    flat = _SPACE.sub(" ", _MARKUP.sub("", text or "")).strip()
    if len(flat) <= chars:
        return flat, False
    # One character past the limit, so a word that ends exactly at it is kept
    # whole rather than dropped as if it had been cut.
    cut = flat[:chars + 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "...", True


def _said(turn: Any, chars: int) -> Said:
    text, truncated = for_speech(turn.text, chars)
    return Said(role=turn.role, at=turn.at, text=text, truncated=truncated)
