"""ChatGPT threads, read from a local export. Costs nothing and calls nothing.

    ~/.qmcp/threads/chatgpt/*.json     one conversation per file

The format is the documented one, checked against one real export. It stores
messages as a `mapping` of nodes with parent pointers: a tree, of which the
conversation a person read is one path.

This flattens the tree: every node with text is taken, in timestamp order. A
conversation with regenerated or branched replies therefore yields turns that
were alternatives to each other rather than one sequence.

The perspective is `chatgpt/thread`, distinct from the Claude source's, so two
assistants discussing the same work produce two sets of deltas. `same-as`
states that they are one strand, and neither address is retired
(`governance/qm/records/DRAFT-deltas-compose.md` 4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmcp.threads.base import Thread, Turn
from qmcp import identity
from qmcp.threads.cache import LocalCacheSource


class ChatGPTThreads(LocalCacheSource):
    """Conversations exported from ChatGPT."""

    name = "chatgpt"
    folder = "chatgpt"
    perspective = "chatgpt/thread"
    # Derived from the checkout, so a fork's deltas belong to the fork; see
    # `qmcp.identity`.
    project = identity.this_project()

    def parse(self, document: Any, path: Path) -> Thread:
        if not isinstance(document, dict):
            raise ValueError("a conversation export is an object")

        identifier = (document.get("conversation_id") or document.get("id")
                      or path.stem)
        mapping = document.get("mapping")
        if mapping is None:
            raise ValueError("no mapping on this conversation")
        if not isinstance(mapping, dict):
            raise ValueError("mapping is not an object")

        collected = []
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            text = _text_of(message)
            if not text:
                continue
            collected.append((
                message.get("create_time") or 0.0,
                Turn(
                    id=str(message.get("id") or node_id),
                    role=str((message.get("author") or {}).get("role")
                             or "unknown"),
                    at=_maybe_str(message.get("create_time")),
                    text=text,
                ),
            ))

        collected.sort(key=lambda pair: pair[0])
        return Thread(
            id=str(identifier),
            title=_maybe_str(document.get("title")),
            started_at=_maybe_str(document.get("create_time")),
            url=f"https://chatgpt.com/c/{identifier}",
            turns=tuple(turn for _, turn in collected),
        )


def _maybe_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _text_of(message: dict) -> str:
    """The message's text, from the `content.parts` shape.

    A part that is not a string, such as an attachment, is skipped rather than
    rendered as its repr.
    """
    content = message.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if isinstance(parts, list):
        return "\n".join(part for part in parts if isinstance(part, str) and part)
    if isinstance(parts, str):
        return parts
    return ""
