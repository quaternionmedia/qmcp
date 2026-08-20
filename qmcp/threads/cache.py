"""Threads read from a local export, which costs nothing.

**NOTHING HERE SPENDS.** A cache read is a file read, so `survey` establishes
real numbers and `fetch` reports `would_need: 0` -- a genuine zero, meaning
there is no paid work to do, and not the sentinel that would mean nobody
counted. `qmcp/spend.py` is where that distinction is kept.

WHY LOCAL FIRST AND NOT ONLY. An export is a snapshot: it is as current as the
day somebody downloaded it, and it says so rather than implying otherwise. That
is a real limitation and it buys three things worth having before an API exists
-- the whole path can be exercised with no credential, no bill and no network,
the extraction can be judged against real conversations, and when the API
arrives it is a second source behind the same contract rather than a rewrite.

WHAT AN EXPORT LOOKS LIKE IS NOT THIS MODULE'S BUSINESS. Each assistant ships
its own shape and each changes it without asking. `parse` is the subclass's, and
this holds only the part that is the same everywhere: find the files, read them,
skip what cannot be read *and say which*, and never present a partial directory
as a whole one.

**A FILE THAT WILL NOT PARSE IS REPORTED, NOT SKIPPED QUIETLY.** A source that
silently dropped three of forty threads would report thirty-seven and look
correct. The count of unreadable files travels with the survey.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qmcp.spend import Budget
from qmcp.threads.base import Decision, Survey, Thread, ThreadSource

# Where an export is looked for when nobody says. Under the user's home rather
# than in a repository, because a conversation export is not source and must not
# end up committed by somebody running from the wrong directory.
DEFAULT_ROOT = Path.home() / ".qmcp" / "threads"


@dataclass(frozen=True)
class Unreadable:
    """A file in the cache that could not be read as a thread."""

    path: str
    why: str


class LocalCacheSource(ThreadSource):
    """A source over a directory of exported conversations.

    A subclass supplies `folder` and `parse`. Everything else -- counting,
    reading, refusing to pretend -- is here, so two assistants differ only where
    they actually differ.
    """

    #: Subdirectory under the cache root, so two assistants' exports do not mix.
    folder: str = ""

    #: What a reader must know about how current this is. An export is a
    #: snapshot; a live store is not, and a source reading one should not
    #: inherit a sentence about the other.
    caveat: str = ("An export is a snapshot: as current as the day it was "
                   "downloaded.")

    def __init__(self, root: Path | None = None, **kw) -> None:
        super().__init__(**kw)
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.unreadable: list[Unreadable] = []

    @property
    def directory(self) -> Path:
        return self.root / self.folder if self.folder else self.root

    def files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob("*.json"))

    def parse(self, document: Any, path: Path) -> Thread:
        """This assistant's export shape, as a `Thread`.

        Raises `ValueError` on anything it cannot read. The caller turns that
        into an `Unreadable` and keeps going, so one malformed file does not
        cost the other thirty-nine.
        """
        raise NotImplementedError

    # --- the contract -------------------------------------------------------

    def survey(self) -> Survey:
        """How many threads are here, established by reading them.

        Spends nothing, and there is nothing to spend. `would_need` is 0 and
        that is a fact rather than a placeholder: reading a local file costs no
        calls, so the paid work really is none.
        """
        if not self.directory.is_dir():
            return Survey(
                source=self.name,
                available=0,
                would_need=0,
                note=(f"No export at {self.directory}. That is zero threads "
                      f"found, not a failure to look -- the directory is "
                      f"simply not there."),
            )

        # READ, DO NOT COUNT. Counting files reports how many are there, and
        # the question is how many are threads. A directory of forty with three
        # malformed would have surveyed as forty and fetched as thirty-seven --
        # this module's own docstring warns about that and the first version of
        # this method did it.
        #
        # Reading them is affordable precisely because nothing here spends: the
        # cost of being right is disk, and the cost of being wrong is a board
        # quietly short three conversations.
        threads = self.fetch([], Budget())
        note = f"{len(threads)} thread(s) in {self.directory}. {self.caveat}"
        if self.unreadable:
            note += (f" {len(self.unreadable)} file(s) could not be read and "
                     f"are not counted: "
                     f"{', '.join(u.path for u in self.unreadable)}.")
        return Survey(source=self.name, available=len(threads), would_need=0,
                      note=note)

    def fetch(self, ids: list[str], budget: Budget) -> list[Thread]:
        """Read the named threads, or every one when `ids` is empty.

        Takes a budget it never spends. The parameter stays because the
        contract has it and because the API source will spend against the same
        one -- a signature that changed between sources would make the caller
        know which it was talking to.
        """
        wanted = set(ids)
        threads: list[Thread] = []
        self.unreadable = []

        for path in self.files():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.unreadable.append(Unreadable(path.name, str(exc)))
                continue
            try:
                thread = self.parse(document, path)
            except (ValueError, KeyError, TypeError) as exc:
                self.unreadable.append(Unreadable(path.name, str(exc)))
                continue
            if wanted and thread.id not in wanted:
                continue
            threads.append(thread)
        return threads

    def decisions(self, thread: Thread, budget: Budget) -> list[Decision]:
        """What this thread marked as settled.

        **THIS RECOGNISES NOTHING ON ITS OWN.** It finds decisions a
        conversation *marked*, by an agreed opening on a line. Reading an
        unmarked conversation and working out what it settled needs a model,
        and that is the API path -- which will spend, and will do it against
        the budget this method already takes.

        Saying that plainly is the point. A free heuristic dressed up as
        comprehension would put confident rows on a board and the person
        reading them would have no way to know which kind they were.
        """
        found: list[Decision] = []
        for turn in thread.turns:
            for line in turn.text.splitlines():
                stripped = line.strip()
                for marker in ("DECISION:", "DECIDED:"):
                    if stripped.upper().startswith(marker):
                        title = stripped[len(marker):].strip()
                        if not title:
                            continue
                        found.append(Decision(
                            name=slug(title, thread.id, len(found)),
                            title=title,
                            summary=turn.text.strip()[:400],
                            from_turns=(turn.id,),
                        ))
                        break
        return found

def slug(title: str, thread_id: str, index: int) -> str:
    """A delta name from a decision's title.

    The thread id is in it because two conversations may settle the same thing
    in the same words, and one row for both would be a claim nobody made. The
    index disambiguates within a thread for the same reason.
    """
    keep = [c.lower() if c.isalnum() else "-" for c in title[:48]]
    body = "".join(keep).strip("-")
    while "--" in body:
        body = body.replace("--", "-")
    return f"{thread_id}-{body or 'decision'}-{index}"
