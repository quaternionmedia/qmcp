"""Read-only access to the thread archive, for something that is not this process.

    GET /v1/threads                       what is indexed
    GET /v1/threads/diverged              exports disagreeing with an earlier record
    GET /v1/threads/{source}/{id}         one thread
    GET /v1/threads/{source}/{id}/deltas  what it settled

**READ-ONLY, AND LOCAL.** Nothing here writes to the archive or to the stores it
reads. The archive holds somebody's conversations, so this binds where the rest
of this server binds -- loopback -- and `handbook/async-contract.md` 4 is the
standing rule about that.

WHY A SERVICE AT ALL, WHEN `qmcp.threads` IS IMPORTABLE. Because importing it
means being this process. A dashboard, a second tool, or a person with `curl`
should not have to become qmcp to read what qmcp archived, and the alternative
-- everyone parsing the session format themselves -- is how two readers come to
disagree about what a thread was.

**IT SERVES THE INDEX, NOT THE FILES.** `GET /v1/threads` answers from one
document rather than reading every session, which is the whole reason the index
exists. An absent index is an absent answer and says so; it is not an empty one.

WHAT IT DOES NOT DO. Spend. Every route here reads local files, and a route that
needed a paid call would be refused by the budget it was handed rather than
quietly billing whoever made the request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmcp.spend import Budget
from qmcp.threads import index as index_module

# The sources a request may name. Anything else is a 404 rather than an attempt,
# because `source` reaches a filesystem path and a name nobody declared is a
# name somebody made up.
SOURCES = ("claude", "chatgpt", "claude-code")


def sources_for(root: Path, sessions: Path | None = None) -> dict[str, Any]:
    """Every source, each pointed at the store it reads.

    Two roots, not one: the web exports live in a cache this project unpacks
    into, and Claude Code sessions live in a store somebody else owns.
    """
    from qmcp.threads.chatgpt import ChatGPTThreads
    from qmcp.threads.claude import ClaudeThreads
    from qmcp.threads.claudecode import SESSION_ROOT, ClaudeCodeThreads

    return {
        "claude": ClaudeThreads(root=root),
        "chatgpt": ChatGPTThreads(root=root),
        "claude-code": ClaudeCodeThreads(root=sessions or SESSION_ROOT),
    }


def index_at(root: Path) -> dict[str, Any] | None:
    """The written index, or None when there is not one.

    None rather than an empty document. A machine that has never indexed and
    one whose archive is empty are different states, and a route that returned
    `{"threads": []}` for both would tell a caller the archive is empty when
    nobody has looked.
    """
    path = root / index_module.INDEX_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def summarise(document: dict[str, Any]) -> dict[str, Any]:
    """The index as a listing: totals, and a row per thread without its turns.

    Turn ids are dropped here. They are in the index because divergence needs
    them, and a listing that carried every one would be large for a caller who
    wanted titles.
    """
    rows = []
    for row in document.get("threads") or []:
        rows.append({
            "source": row["source"],
            "id": row["id"],
            "title": row.get("title"),
            "turns": row.get("turns", 0),
            "digest": row.get("digest"),
            "first_seen": row.get("first_seen"),
            "last_seen": row.get("last_seen"),
            "diverged": any(change["kind"] == index_module.DIVERGED
                            for change in row.get("history") or []),
            "changes": len(row.get("history") or []),
        })
    return {
        "schema": document.get("schema"),
        "generated_at": document.get("generated_at"),
        "totals": document.get("totals", {}),
        "threads": rows,
        "reading": {
            "the_index_is_a_reading": (
                "These figures are as of `generated_at`, from a cache that is "
                "itself a snapshot. Nothing here counts conversations that "
                "exist; it counts conversations that were exported and "
                "indexed."
            ),
            "refresh": "uv run qmcp threads index --write",
        },
    }


def diverged(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Threads whose export disagrees with an earlier record of itself.

    The interesting query, and the reason the archive keeps history rather than
    the latest version. Each row carries the change that made it interesting.
    """
    found = []
    for row in document.get("threads") or []:
        changes = [change for change in row.get("history") or []
                   if change["kind"] == index_module.DIVERGED]
        if not changes:
            continue
        found.append({
            "source": row["source"], "id": row["id"], "title": row.get("title"),
            "changes": changes,
            "latest": changes[-1],
        })
    return found


def one(root: Path, source: str, identifier: str,
        sessions: Path | None = None) -> dict[str, Any] | None:
    """One thread, read from its store rather than from the index.

    From the store because the index holds what a thread *is*, not what it
    said. A caller asking for a thread wants the turns, and those were never in
    the index.
    """
    if source not in SOURCES:
        return None
    reader = sources_for(root, sessions)[source]
    for thread in reader.fetch([identifier], Budget()):
        if thread.id != identifier:
            continue
        return {
            "source": source,
            "id": thread.id,
            "title": thread.title,
            "started_at": thread.started_at,
            "url": thread.url,
            "partial": thread.partial,
            "turns": [
                {"id": turn.id, "role": turn.role, "at": turn.at,
                 "text": turn.text}
                for turn in thread.turns
            ],
        }
    return None


def deltas_of(root: Path, source: str, identifier: str,
              sessions: Path | None = None) -> dict[str, Any] | None:
    """What a thread settled, as payloads and relations.

    The same payloads `dossier deltas ingest` reads. A caller wanting to put a
    conversation on a board does not have to know how one is turned into rows.
    """
    if source not in SOURCES:
        return None
    reader = sources_for(root, sessions)[source]
    for thread in reader.fetch([identifier], Budget()):
        if thread.id != identifier:
            continue
        budget = Budget()
        return {
            "source": source,
            "id": thread.id,
            "perspective": reader.perspective,
            "deltas": reader.deltas(thread, budget),
            "relations": reader.relations(thread, budget),
            "spent": budget.made,
        }
    return None


def register(app: Any, root: Path, sessions: Path | None = None) -> None:
    """Attach the read-only routes to a FastAPI app.

    Takes the app rather than creating one, so the archive is served by the
    same process that already serves tools and the human queue -- one thing to
    start, one port, one place bound to loopback.
    """
    from fastapi import HTTPException

    @app.get("/v1/threads")
    async def list_threads() -> dict[str, Any]:
        document = index_at(root)
        if document is None:
            raise HTTPException(
                status_code=404,
                detail=("no index. `uv run qmcp threads index --write` builds "
                        "one. An absent index is an absent answer rather than "
                        "an empty archive."),
            )
        return summarise(document)

    @app.get("/v1/threads/diverged")
    async def list_diverged() -> dict[str, Any]:
        document = index_at(root)
        if document is None:
            raise HTTPException(status_code=404, detail="no index")
        found = diverged(document)
        return {
            "generated_at": document.get("generated_at"),
            "diverged": found,
            "note": ("An export is a record. One that disagrees with an "
                     "earlier record of itself is a tool changing its format, "
                     "somebody editing history, or an id being reused. Nothing "
                     "here is repaired."),
        }

    @app.get("/v1/threads/{source}/{identifier}")
    async def get_thread(source: str, identifier: str) -> dict[str, Any]:
        found = one(root, source, identifier, sessions)
        if found is None:
            raise HTTPException(status_code=404,
                                detail=f"no {source} thread {identifier!r}")
        return found

    @app.get("/v1/threads/{source}/{identifier}/deltas")
    async def get_deltas(source: str, identifier: str) -> dict[str, Any]:
        found = deltas_of(root, source, identifier, sessions)
        if found is None:
            raise HTTPException(status_code=404,
                                detail=f"no {source} thread {identifier!r}")
        return found
