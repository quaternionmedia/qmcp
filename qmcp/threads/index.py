"""An index over the local thread cache, and an archive that never overwrites.

    qmcp threads index            # what is cached, and what changed
    qmcp threads index --write    # write it
    qmcp threads index --check    # re-derive and report drift

The index answers from one file what `survey` answers by reading every file.
The files stay the source: `--check` re-derives from them and reports where
the two differ.

    cache layer     id, title, turn count, digest. A pure function of the files
                    on disk.
    archive layer   what earlier versions of a thread were, and how each
                    changed. A function of every index before this one.

The archive never overwrites. A later export of a conversation is a later
state of it, classified as one of:

    grew        every earlier turn is still there, in order, and there are
                more. The ordinary case.
    diverged    an earlier turn is gone or changed: a tool changing its format,
                somebody editing history, or an id being reused.

Neither is repaired: the prior digest and the change are kept
(`governance/qm/records/DRAFT-deltas-compose.md` 7 keeps a tangle the same
way).

What this cannot see: whether an export is complete. The index counts what is
cached, not what exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from qmcp.threads.base import Thread

SCHEMA = 1

# The index sits with what it indexes, outside any repository: it describes one
# machine's cache, with local paths and timestamps.
INDEX_NAME = "index.json"

# Long, because the cache changes only when somebody exports.
STALENESS_BUDGET_HOURS = 720

GREW = "grew"
DIVERGED = "diverged"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest_of(thread: Thread) -> str:
    """A content digest over what a thread *says*, not over its file.

    Turn ids and text, in order, and not the file bytes, so an exporter that
    reformats its JSON does not register as every conversation diverging.
    """
    body = hashlib.sha256()
    for turn in thread.turns:
        body.update(turn.id.encode("utf-8"))
        body.update(b"\x00")
        body.update(turn.text.encode("utf-8"))
        body.update(b"\x1e")
    return body.hexdigest()[:16]


@dataclass
class Change:
    """One transition between two versions of a thread."""

    at: str
    kind: str                 # GREW | DIVERGED
    from_digest: str
    to_digest: str
    from_turns: int
    to_turns: int
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at, "kind": self.kind,
            "from_digest": self.from_digest, "to_digest": self.to_digest,
            "from_turns": self.from_turns, "to_turns": self.to_turns,
            "detail": self.detail,
        }


@dataclass
class Entry:
    """One thread, as the index holds it."""

    id: str
    source: str
    title: str | None = None
    started_at: str | None = None
    url: str | None = None
    digest: str = ""
    turns: int = 0
    turn_ids: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    history: list[Change] = field(default_factory=list)

    @property
    def key(self) -> str:
        """What identifies a thread across sources.

        Two assistants may use the same conversation id for different
        conversations, so the source is part of the identity.
        """
        return f"{self.source}/{self.id}"

    @property
    def diverged(self) -> bool:
        return any(change.kind == DIVERGED for change in self.history)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source": self.source, "title": self.title,
            "started_at": self.started_at, "url": self.url,
            "digest": self.digest, "turns": self.turns,
            "turn_ids": list(self.turn_ids),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "history": [change.as_dict() for change in self.history],
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Entry":
        return cls(
            id=row["id"], source=row["source"], title=row.get("title"),
            started_at=row.get("started_at"), url=row.get("url"),
            digest=row.get("digest", ""), turns=row.get("turns", 0),
            turn_ids=list(row.get("turn_ids") or []),
            first_seen=row.get("first_seen", ""), last_seen=row.get("last_seen", ""),
            history=[Change(**change) for change in row.get("history") or []],
        )


def entry_for(thread: Thread, source: str, at: str) -> Entry:
    """A fresh entry from a thread as it is on disk right now."""
    return Entry(
        id=thread.id, source=source, title=thread.title,
        started_at=thread.started_at, url=thread.url,
        digest=digest_of(thread), turns=len(thread.turns),
        turn_ids=[turn.id for turn in thread.turns],
        first_seen=at, last_seen=at,
    )


def classify(previous: Entry, current: Entry) -> tuple[str, str]:
    """How this thread changed, and why that reading.

    Growth is the old turn ids being a proper prefix of the new ones. A
    prefix rather than a subset: an export that kept every old turn but
    reordered them has diverged.
    """
    old, new = previous.turn_ids, current.turn_ids

    # Growth needs the old ids as a prefix and something after them. The same
    # ids with a different digest is a turn edited after the fact: diverged.
    if len(new) > len(old) and new[:len(old)] == old:
        return GREW, f"{len(new) - len(old)} turn(s) added"

    if old == new:
        return DIVERGED, ("the turns are the same and what they say is not: a "
                          "turn was edited after it was first exported")

    if len(new) < len(old):
        return DIVERGED, (f"{len(old) - len(new)} turn(s) present before are "
                          f"absent now")
    for index, (before, after) in enumerate(zip(old, new)):
        if before != after:
            return DIVERGED, (f"turn {index + 1} was {before!r} and is now "
                              f"{after!r}")
    return DIVERGED, "the turns are the same and the text is not"


def merge(previous: dict[str, Entry], current: list[Entry],
          at: str | None = None) -> tuple[list[Entry], list[Entry]]:
    """Fold what is on disk into what was known. Returns (entries, changed).

    Nothing is overwritten and nothing is dropped: a thread the cache no
    longer holds keeps its entry.
    """
    stamped = at or now()
    merged: dict[str, Entry] = dict(previous)
    changed: list[Entry] = []

    for entry in current:
        before = merged.get(entry.key)
        if before is None:
            merged[entry.key] = entry
            changed.append(entry)
            continue

        entry.first_seen = before.first_seen or entry.first_seen
        entry.history = list(before.history)
        if before.digest == entry.digest:
            # Seen again, unchanged. `last_seen` moves; nothing else does.
            entry.last_seen = stamped
            merged[entry.key] = entry
            continue

        kind, detail = classify(before, entry)
        entry.history.append(Change(
            at=stamped, kind=kind,
            from_digest=before.digest, to_digest=entry.digest,
            from_turns=before.turns, to_turns=entry.turns, detail=detail))
        entry.last_seen = stamped
        merged[entry.key] = entry
        changed.append(entry)

    return sorted(merged.values(), key=lambda e: e.key), changed


def build(sources: Iterable[Any], at: str | None = None) -> list[Entry]:
    """Read every source's cache and describe what is there.

    The sources are read through `fetch`, because a digest needs the threads
    themselves, not `survey`'s count.
    """
    from qmcp.spend import Budget

    stamped = at or now()
    entries: list[Entry] = []
    for source in sources:
        # A budget of zero: indexing never spends, so a source that would need
        # a paid call refuses.
        for thread in source.fetch([], Budget()):
            entries.append(entry_for(thread, source.name, stamped))
    return entries


def document(entries: list[Entry], unreadable: dict[str, list[str]] | None = None,
             at: str | None = None) -> dict[str, Any]:
    """The index as data."""
    stamped = at or now()
    return {
        "schema": SCHEMA,
        "generated_at": stamped,
        "reading": {
            "refresh": "uv run qmcp threads index --write",
            "verify": "uv run qmcp threads index --check",
            "staleness_budget_hours": STALENESS_BUDGET_HOURS,
            "layers": {
                "cache": ("id, title, turns, digest. A pure function of the "
                          "files on disk."),
                "archive": ("first_seen, last_seen, history. A function of "
                            "every index before this one, which the files "
                            "cannot tell you."),
            },
            "do_not": [
                "read a count here as how many conversations exist -- it is "
                "how many are cached, and an export is a snapshot",
                "delete a diverged entry to make this tidy: the prior digest "
                "is the only record that it changed",
                "commit this file. It describes one machine's cache",
            ],
        },
        "totals": {
            "threads": len(entries),
            "diverged": sum(1 for entry in entries if entry.diverged),
            "unreadable": sum(len(paths) for paths in (unreadable or {}).values()),
        },
        "unreadable": unreadable or {},
        "threads": [entry.as_dict() for entry in entries],
    }


def load(path: Path) -> dict[str, Entry]:
    """What the index knew, keyed for merging. An absent index is not an error."""
    if not path.is_file():
        return {}
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = [Entry.from_dict(row) for row in found.get("threads") or []]
    return {entry.key: entry for entry in entries}


def render(doc: dict[str, Any], changed: list[Entry] | None = None) -> str:
    """The index, for a person."""
    totals = doc["totals"]
    out = [
        f"threads indexed at {doc['generated_at']}",
        "",
        f"  cached        {totals['threads']}",
        f"  diverged      {totals['diverged']}",
    ]
    if totals["unreadable"]:
        out.append(f"  unreadable    {totals['unreadable']}   <- not counted above")
    out.append("")

    diverged = [row for row in doc["threads"]
                if any(c["kind"] == DIVERGED for c in row["history"])]
    if diverged:
        out.append("changed in a way an export should not:")
        for row in diverged:
            last = [c for c in row["history"] if c["kind"] == DIVERGED][-1]
            out.append(f"  {row['source']}/{row['id']}  {last['detail']}")
            out.append(f"      {last['from_digest']} -> {last['to_digest']}"
                       f"   ({last['at']})")
        out += [
            "",
            "  An export is a record. One that disagrees with an earlier record",
            "  of itself is a tool changing its format, somebody editing",
            "  history, or an id being reused. Nothing here is repaired: the",
            "  prior digest is the only evidence it changed.",
            "",
        ]

    if changed:
        out.append(f"{len(changed)} thread(s) new or changed this run.")
    elif changed is not None:
        out.append("Nothing changed since the last index.")
    return "\n".join(out)


def drift(doc: dict[str, Any], entries: list[Entry]) -> list[str]:
    """Where the committed index and the files disagree, for `--check`.

    Only the cache layer is compared; the archive layer cannot be re-derived
    from the files.
    """
    known = {row["source"] + "/" + row["id"]: row for row in doc["threads"]}
    problems: list[str] = []

    for entry in entries:
        row = known.get(entry.key)
        if row is None:
            problems.append(f"{entry.key}: on disk and not in the index")
            continue
        if row["digest"] != entry.digest:
            problems.append(
                f"{entry.key}: index says {row['digest']}, the files say "
                f"{entry.digest}")
        if row["turns"] != entry.turns:
            problems.append(
                f"{entry.key}: index says {row['turns']} turn(s), the files "
                f"say {entry.turns}")

    on_disk = {entry.key for entry in entries}
    for key in known:
        if key not in on_disk:
            # Not drift: the archive keeps what the cache no longer holds.
            continue
    return problems
