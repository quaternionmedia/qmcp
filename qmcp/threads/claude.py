"""Claude threads, read from a local export. Costs nothing and calls nothing.

    ~/.qmcp/threads/claude/*.json      one conversation per file

**VERIFIED ONCE, AGAINST ONE REAL EXPORT.** This was written from the documented
format with no export to hand, and was checked on 2026-08-20 against an actual
one: 94 conversations, every one detected as Claude, and **no conversation
carrying messages parsed to zero turns**. Four came through empty and were empty
in the export too -- untitled, with no `chat_messages` at all.

That is one export from one account on one date. It is much stronger than the
guess it replaces and it is not a guarantee: an exporter that changes its shape
will be found by a conversation that will not parse, which is reported by name
rather than skipped. The failure being guarded against is a board quietly
showing thirty-seven of forty.

WHAT IS DELIBERATELY ABSENT. The API. It is a second source behind the same
contract, and when it arrives it spends against the `Budget` the contract
already passes. This one never spends, so `survey` reports `would_need: 0` --
a real zero, meaning there is no paid work to do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmcp.threads.base import Thread, Turn
from qmcp.threads.cache import LocalCacheSource


class ClaudeThreads(LocalCacheSource):
    """Conversations exported from Claude."""

    name = "claude"
    folder = "claude"

    # A claim about level: this source speaks about whole conversations and
    # what they settled, never about turns.
    perspective = "claude/thread"

    # Decided rather than defaulted: thread deltas belong to the harness that
    # pulled them. `plans/qmpm-standardisations.md` 1 still has the open
    # question of whether a delta may span owners; until it is settled, this is
    # a home somebody chose.
    project = "quaternionmedia/qmcp"

    def parse(self, document: Any, path: Path) -> Thread:
        if not isinstance(document, dict):
            raise ValueError("a conversation export is an object")

        identifier = _first(document, "uuid", "id", "conversation_id")
        if not identifier:
            raise ValueError("no uuid/id on this conversation")

        messages = _first(document, "chat_messages", "messages") or []
        if not isinstance(messages, list):
            raise ValueError("chat_messages is not a list")

        turns = tuple(
            Turn(
                id=str(_first(message, "uuid", "id") or f"{identifier}-{index}"),
                role=str(_first(message, "sender", "role") or "unknown"),
                at=_maybe_str(_first(message, "created_at", "timestamp")),
                text=_text_of(message),
            )
            for index, message in enumerate(messages)
            if isinstance(message, dict)
        )

        return Thread(
            id=str(identifier),
            title=_maybe_str(_first(document, "name", "title")),
            started_at=_maybe_str(_first(document, "created_at")),
            url=f"https://claude.ai/chat/{identifier}",
            turns=turns,
        )


def _first(document: dict, *keys: str) -> Any:
    for key in keys:
        if key in document and document[key] is not None:
            return document[key]
    return None


def _maybe_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _text_of(message: dict) -> str:
    """The message's text, from whichever shape this export uses.

    Newer exports carry a `content` list of typed blocks; older ones a flat
    `text`. Both are read, and a block whose type is not text is skipped rather
    than stringified -- an image rendered as its repr would become searchable
    prose that says nothing.
    """
    text = message.get("text")
    if isinstance(text, str) and text:
        return text

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        return "\n".join(part for part in parts if part)
    return ""
