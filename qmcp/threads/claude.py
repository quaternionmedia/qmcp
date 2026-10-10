"""Claude threads, read from a local export. Costs nothing and calls nothing.

    ~/.qmcp/threads/claude/*.json      one conversation per file

The format is the documented one, checked against one real export; that is
not a guarantee. A conversation that will not parse is reported by name.

This source never spends, so `survey` reports `would_need: 0`. An API source
would be a second source behind the same contract, spending against the
`Budget` the contract passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmcp.threads.base import Thread, Turn
from qmcp import identity
from qmcp.threads.cache import LocalCacheSource


class ClaudeThreads(LocalCacheSource):
    """Conversations exported from Claude."""

    name = "claude"
    folder = "claude"

    # A claim about level: this source speaks about whole conversations and
    # what they settled, never about turns.
    perspective = "claude/thread"

    # Thread deltas belong to the harness that pulled them, derived from the
    # checkout so a fork's deltas belong to the fork; see `qmcp.identity`.
    # Whether a delta may span owners is open in
    # `plans/qmpm-standardisations.md` 1.
    project = identity.this_project()

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

    Reads a flat `text` or a `content` list of typed blocks. A block whose type
    is not text, such as an image, is skipped rather than rendered as its repr.
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
