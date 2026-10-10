"""The thread archive over HTTP, for a caller that is not this process.

    GET /v1/threads                       what is indexed
    GET /v1/threads/diverged              exports disagreeing with an earlier record
    GET /v1/threads/{source}/{id}         one thread
    GET /v1/threads/{source}/{id}/deltas  what it settled

    POST /v1/threads/reindex              rebuild the index, import nothing
    POST /v1/threads/import               unpack an export, then re-index

The routes bind where the rest of this server binds, loopback
(`handbook/async-contract.md` 4). The two `POST` routes write only into the
cache and its index: the harness writes, and the caller is a caller.

`GET /v1/threads` answers from the index, not from the files. An absent index
is a 404 that says so, not an empty listing.

Nothing here spends: every route reads local files, and every source is handed
an empty budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmcp.spend import Budget
from qmcp.threads import index as index_module

# The sources a request may name. Anything else is a 404, because `source`
# reaches a filesystem path.
SOURCES = ("claude", "chatgpt", "claude-code")


def source_classes() -> dict[str, Any]:
    """Every source class, with no store behind it.

    A thread's project and perspective are class attributes, so a caller
    asking only those needs no store; `sources_for` is for reading.
    """
    from qmcp.threads.chatgpt import ChatGPTThreads
    from qmcp.threads.claude import ClaudeThreads
    from qmcp.threads.claudecode import ClaudeCodeThreads

    return {"claude": ClaudeThreads, "chatgpt": ChatGPTThreads,
            "claude-code": ClaudeCodeThreads}


def as_delta_row(source: str, identifier: str) -> dict[str, Any]:
    """One thread's delta identity: its address and the level it speaks at.

    Built by `to_thread_delta`, so a consumer reads the address rather than
    reimplementing the naming rule.

    An unknown source gets `None` for every field.
    """
    from qmcp.threads.base import to_thread_delta

    found = source_classes().get(source)
    if found is None:
        return {"address": None, "perspective": None,
                "phase": None, "delta_type": None}

    # `to_thread_delta` decides the name, address, phase and type.
    built = to_thread_delta(_Named(identifier), project=found.project,
                            perspective=found.perspective)
    address = next(
        (link["target_name"] for link in built["links"]
         if link["link_type"] == "address"), None)
    return {
        "address": address,
        "perspective": built["perspective"],
        "phase": built["delta"]["phase"],
        "delta_type": built["delta"]["delta_type"],
    }


class _Named:
    """Just the fields `to_thread_delta` reads off a thread.

    A real `Thread` would mean reading the store. `title` and `url` are None:
    the listing carries the title from the index.
    """

    __slots__ = ("id", "title", "url")

    def __init__(self, identifier: str) -> None:
        self.id = identifier
        self.title = None
        self.url = None


def sources_for(root: Path, sessions: Path | None = None) -> dict[str, Any]:
    """Every source, each pointed at the store it reads.

    Two roots: the web exports live in the cache this project unpacks into,
    and Claude Code sessions live in a store another tool owns.
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

    None rather than an empty document: never indexed and an empty archive
    are different states.
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

    Turn ids, which the index keeps for classifying divergence, are dropped.
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
            # A thread is a delta, so each row carries its address, from class
            # attributes and the naming rule with nothing read from the store.
            **as_delta_row(row["source"], row["id"]),
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

    Each row carries its divergences and the latest of them.
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

    From the store, because the index does not hold the turns' text.
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

    The same payloads `dossier deltas ingest` reads.
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


def _reindex(root: Path, sessions: Path | None) -> dict[str, Any]:
    """Rebuild the index over every source, keeping what earlier ones knew.

    Spends nothing: `index.build` hands each source a budget of zero, so a
    source that would need a paid call refuses rather than billing whoever
    asked for an import.
    """
    document = index_at(root) or {}
    previous = {
        f"{row['source']}/{row['id']}": index_module.Entry.from_dict(row)
        for row in document.get("threads") or []
    }
    entries = index_module.build(sources_for(root, sessions).values())
    merged, changed = index_module.merge(previous, entries)
    written = index_module.document(merged)
    (root / index_module.INDEX_NAME).write_text(
        json.dumps(written, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "threads": written["totals"]["threads"],
        "diverged": written["totals"]["diverged"],
        "changed": len(changed),
        "generated_at": written["generated_at"],
    }


def register(app: Any, root: Path, sessions: Path | None = None) -> None:
    """Attach the read-only routes to a FastAPI app.

    Takes the app rather than creating one, so the archive is served by the
    process that serves tools and the human queue, on one loopback port.
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

    @app.post("/v1/threads/reindex")
    async def reindex_only(body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Rebuild the index from what is already in the store. Imports nothing.

        A refresh that needs no export file. Spends nothing: `index.build`
        hands every source a budget of zero.
        """
        indexed = _reindex(root, sessions)
        return {
            "reindexed": True,
            "indexed": indexed,
            "reading": ("the index is rebuilt from the local store. Nothing "
                        "was fetched and nothing was paid for; a conversation "
                        "that is not on this machine is not here afterwards."),
        }

    @app.post("/v1/threads/import")
    async def import_export(body: dict[str, Any]) -> dict[str, Any]:
        """Unpack an export the operator points at, then re-index.

        The harness writes the archive; the caller names a file. The person's
        act is requesting the export from the service, and what follows may be
        automated
        (`governance/qm/records/DRAFT-acts-that-are-a-persons-by-constitution.md`
        clause 3).

        Reads a path on this machine, writes only into the cache, and spends
        nothing.
        """
        from qmcp.threads.importer import positional, unpack

        raw = (body or {}).get("path") or ""
        source = (body or {}).get("source") or None
        if not raw:
            raise HTTPException(status_code=400, detail="no path given")

        export = Path(raw).expanduser()
        if not export.exists():
            raise HTTPException(status_code=404, detail=f"{export} is not there")
        if export.is_dir():
            # An unpacked export is a directory holding conversations.json.
            candidate = export / "conversations.json"
            if not candidate.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=(f"{export} holds no conversations.json. Point at "
                            f"the export archive or the file itself."))
            export = candidate

        try:
            report = unpack(export, root, source=source)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Re-index in the same call, so the listing reflects the import.
        indexed = _reindex(root, sessions)

        return {
            "source": report.source,
            "written": len(report.written),
            "identical": len(report.identical),
            "replaced": len(report.replaced),
            "unreadable": [{"what": name, "why": why}
                           for name, why in report.unreadable],
            "positional": len(list(positional(report))),
            "indexed": indexed,
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
