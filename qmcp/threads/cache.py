"""Threads read from a local export, which costs nothing.

Nothing here spends. A cache read is a file read, so `survey` establishes real
numbers and `would_need` is `0`: there is no paid work to do.

An export is a snapshot, as current as the day it was downloaded, and the
survey's note says so.

`parse` is the subclass's: each assistant's export shape is its own. This
class finds the files, reads them, and records every file it cannot read as
an `Unreadable`, whose count and names travel with the survey.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qmcp.spend import Budget
from qmcp.threads.base import Decision, Survey, Thread, ThreadSource

# Where an export is looked for by default: under the user's home rather than
# in a repository, so a conversation export is never committed as source.
DEFAULT_ROOT = Path.home() / ".qmcp" / "threads"


@dataclass(frozen=True)
class Unreadable:
    """A file in the cache that could not be read as a thread."""

    path: str
    why: str


class LocalCacheSource(ThreadSource):
    """A source over a directory of exported conversations.

    A subclass supplies `folder` and `parse`. Counting, reading and reporting
    unreadable files are here.
    """

    #: Subdirectory under the cache root, so two assistants' exports do not mix.
    folder: str = ""

    #: How current the source is, for the survey's note. A source over a live
    #: store overrides it.
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

        Raises `ValueError` on anything it cannot read. The caller records an
        `Unreadable` and reads the remaining files.
        """
        raise NotImplementedError

    # --- the contract -------------------------------------------------------

    def survey(self) -> Survey:
        """How many threads are here, established by reading them.

        Spends nothing. `would_need` is `0` because reading a local file costs
        no calls.
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

        # Every file is parsed rather than counted, so the survey reports how
        # many are threads and agrees with what `fetch` returns.
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

        Takes a budget it never spends, so every source has one signature.
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

        Finds only decisions a conversation marked, with a line opening
        `DECISION:` or `DECIDED:`. It does not read an unmarked conversation
        for what it settled; that needs a model, which would spend against
        the budget this method takes.
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

    The thread id and the index within the thread are in it, so the same
    words settled in two places make two rows.
    """
    keep = [c.lower() if c.isalnum() else "-" for c in title[:48]]
    body = "".join(keep).strip("-")
    while "--" in body:
        body = body.replace("--", "-")
    return f"{thread_id}-{body or 'decision'}-{index}"
