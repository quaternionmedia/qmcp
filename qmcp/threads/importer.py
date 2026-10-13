"""Unpacking an official data export into the local cache.

    qmcp threads import <export.zip>
    qmcp threads import <conversations.json> --source claude

A model API does not expose the conversation history of `claude.ai` or
`chatgpt.com`. The route to it is each service's data export, requested by the
account holder in the web interface.

An export is a ZIP holding, among other files, one `conversations.json`: an
array of conversations. The cache reads a file per conversation, so this
splits the array into `<root>/<source>/<id>.json`.

Which service wrote a conversation is detected by structure rather than by
filename. Every conversation that cannot be read is counted and named, and a
run that recognised nothing says so.

Nothing here spends and nothing reaches the network: it reads a file the
operator downloaded, and a test asserts the module imports no HTTP client.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# The member of the archive that holds the conversations, in both services'
# exports; the search is by suffix so a nested layout still resolves.
CONVERSATIONS = "conversations.json"


@dataclass
class Imported:
    """What one import run did, in the terms a person would check it in."""

    source: str = ""
    written: list[str] = field(default_factory=list)
    identical: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.written) + len(self.identical) + len(self.replaced)


def detect(conversation: dict) -> str | None:
    """Which service this conversation came from, by shape rather than filename.

    `mapping` is ChatGPT's node tree; `chat_messages` is Claude's list.
    Anything carrying neither returns None.
    """
    if not isinstance(conversation, dict):
        return None
    if isinstance(conversation.get("mapping"), dict):
        return "chatgpt"
    if isinstance(conversation.get("chat_messages"), list):
        return "claude"
    return None


def identity(conversation: dict, source: str, index: int) -> str:
    """A filename for one conversation.

    Falls back to the position in the array when the export carries no id.
    A positional name shifts when an earlier conversation is deleted from a
    later export, which the index reads as divergence; `positional` lists the
    names that needed it.
    """
    for key in ("uuid", "conversation_id", "id"):
        found = conversation.get(key)
        if isinstance(found, str) and found:
            return safe(found)
    return f"{source}-position-{index:05d}"


def safe(identifier: str) -> str:
    """A filename that cannot escape the cache directory.

    Every character other than a letter, digit, `-` or `_` becomes `-`, so an
    id containing a separator cannot write outside the folder.
    """
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in identifier]
    return "".join(keep).strip("-")[:120] or "unnamed"


def conversations_in(path: Path) -> tuple[list[Any], str]:
    """Every conversation in an export, and where they were read from.

    Accepts the downloaded ZIP or an unpacked `conversations.json`.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist()
                       if name.endswith(CONVERSATIONS)]
            if not members:
                raise ValueError(
                    f"{path.name} holds no {CONVERSATIONS}. It carries: "
                    f"{', '.join(sorted(archive.namelist())[:8])}"
                    f"{' ...' if len(archive.namelist()) > 8 else ''}"
                )
            member = min(members, key=len)   # the outermost, if nested
            with archive.open(member) as handle:
                return json.loads(handle.read().decode("utf-8")), member
        # unreachable, kept for the reader
    return json.loads(path.read_text(encoding="utf-8")), path.name


def unpack(export: Path, root: Path, source: str | None = None,
           dry_run: bool = False) -> Imported:
    """Split an export into one file per conversation under `root/<source>`.

    Overwrites a conversation already cached and reports, in `replaced`,
    each one that changed. Re-importing a later export is the ordinary use;
    `qmcp threads index` records how each changed conversation changed.
    """
    found, where = conversations_in(export)
    if not isinstance(found, list):
        raise ValueError(f"{where} is not a list of conversations")

    report = Imported(source=source or "")
    for index, conversation in enumerate(found):
        detected = detect(conversation) if isinstance(conversation, dict) else None
        service = source or detected
        if service is None:
            report.unreadable.append((
                f"conversation {index}",
                "carries neither `mapping` nor `chat_messages`, so which "
                "service wrote it could not be established",
            ))
            continue
        if source and detected and detected != source:
            report.unreadable.append((
                f"conversation {index}",
                f"looks like a {detected} conversation and --source says "
                f"{source}. Refusing rather than filing it under the wrong one",
            ))
            continue

        report.source = report.source or service
        name = identity(conversation, service, index)
        target = root / service / f"{name}.json"
        body = json.dumps(conversation, indent=2, ensure_ascii=False) + "\n"

        if target.is_file():
            if target.read_text(encoding="utf-8") == body:
                report.identical.append(name)
                continue
            report.replaced.append(name)
        else:
            report.written.append(name)

        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8", newline="\n")

    return report


def render(report: Imported, root: Path, dry_run: bool = False) -> str:
    """What the import did, for a person about to index it."""
    if report.total == 0 and not report.unreadable:
        return ("The export held no conversations. That is what the file "
                "contained, not a failure to read it.")

    verb = "would write" if dry_run else "wrote"
    out = [
        f"  {verb} {len(report.written)} new conversation(s)",
        f"  {len(report.identical)} already cached and unchanged",
        f"  {len(report.replaced)} replaced with a different version",
    ]
    if report.unreadable:
        out.append(f"  {len(report.unreadable)} not recognised, and not imported:")
        for name, why in report.unreadable[:5]:
            out.append(f"      {name}: {why}")
        if len(report.unreadable) > 5:
            out.append(f"      ... and {len(report.unreadable) - 5} more")

    out.append("")
    if report.replaced:
        out += [
            "  A replaced conversation is one that changed since the last "
            "export.",
            "  Whether it grew or disagrees with the earlier record is what",
            "  `uv run qmcp threads index` establishes, and it keeps both.",
            "",
        ]
    out.append(f"  Cache: {root / report.source if report.source else root}")
    if dry_run:
        out.append("  Nothing was written. Drop --dry-run to keep it.")
    return "\n".join(out)


def positional(report: Imported) -> Iterable[str]:
    """Names that had to fall back to a position in the array."""
    return (name for name in report.written + report.replaced
            if "-position-" in name)
