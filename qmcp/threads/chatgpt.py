"""ChatGPT threads, read from a local export. Costs nothing and calls nothing.

    ~/.qmcp/threads/chatgpt/*.json     one conversation per file

**VERIFIED ONCE, AGAINST ONE REAL EXPORT.** Written from the documented format
with none to hand, and checked on 2026-08-21 against an actual one: 34
conversations, all 34 parsed, and **no conversation carrying messages that
parsed to zero turns**. 354 turns, 116 from the person and 238 from the
assistant, which is the ratio a real archive has and a broken reading does not.

That is one export from one account on one date, and this format has more room
to be wrong than the Claude one: it stores messages as a `mapping` of nodes with
parent pointers rather than a list -- a tree, of which the conversation you
actually read is one path.

**THIS FLATTENS THE TREE AND SAYS SO.** Every node with text is taken, in
timestamp order. A conversation with regenerated or branched replies therefore
yields turns that were alternatives to each other rather than a sequence
somebody read. That is a real inaccuracy; it is preferred to silently choosing
one path, because choosing would look correct and this looks like what it is.
The `same-as` relation exists for the day somebody wants to say two branches
were one strand.

ONE DIFFERENCE FROM THE CLAUDE SOURCE, AND IT IS NOT COSMETIC. Its perspective
is its own — `chatgpt/thread`. Two assistants discussing the same work produce
two sets of deltas, and neither is the other's duplicate: they are two
perspectives on one strand. `same-as` is how somebody says they are the same
strand once they have read both, and
`governance/qm/records/DRAFT-deltas-compose.md` 4 is why neither address is
retired when they do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qmcp.threads.base import Thread, Turn
from qmcp.threads.cache import LocalCacheSource


class ChatGPTThreads(LocalCacheSource):
    """Conversations exported from ChatGPT."""

    name = "chatgpt"
    folder = "chatgpt"
    perspective = "chatgpt/thread"
    project = "quaternionmedia/qmcp"

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

    A part that is not a string is skipped rather than stringified. Attachments
    arrive as objects, and one rendered as its repr becomes prose that reads
    like content and is not.
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
