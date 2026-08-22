"""Read-only access to the thread archive, for something that is not this process.

    GET /v1/threads                       what is indexed
    GET /v1/threads/diverged              exports disagreeing with an earlier record
    GET /v1/threads/{source}/{id}         one thread
    GET /v1/threads/{source}/{id}/deltas  what it settled

    POST /v1/threads/import               unpack an export, then re-index

**LOCAL, AND ONE WRITE.** Every route but the last only reads. The archive holds
somebody's conversations, so this binds where the rest of this server binds --
loopback -- and `handbook/async-contract.md` 4 is the standing rule.

**THE ONE WRITE IS THE HARNESS DOING ITS OWN JOB.** A control panel asking for
an import is not a control panel authoring the archive: the archive stays one
record with one author, and the caller is a caller. It writes only into the
cache, from a file the operator points at, and re-indexes in the same call --
an import that left the listing stale would look like it had done nothing.

The human act happened before any of this: requesting the export from the
service. `records/DRAFT-acts-that-are-a-persons-by-constitution.md` clause 3 is
explicit that everything after it may be automated, and should be.

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


def source_classes() -> dict[str, Any]:
    """Every source class, with no store behind it.

    `sources_for` needs a root because it is going to *read*. Whether a thread
    is a delta, what project it belongs to and what level it speaks at are
    class attributes, so a caller asking only those does not need a filesystem
    -- and a listing that had to construct readers to state an address would be
    a listing that fails when the store is not where it was last time.
    """
    from qmcp.threads.chatgpt import ChatGPTThreads
    from qmcp.threads.claude import ClaudeThreads
    from qmcp.threads.claudecode import ClaudeCodeThreads

    return {"claude": ClaudeThreads, "chatgpt": ChatGPTThreads,
            "claude-code": ClaudeCodeThreads}


def as_delta_row(source: str, identifier: str) -> dict[str, Any]:
    """One thread's delta identity: its address and the level it speaks at.

    **DERIVED HERE BECAUSE THE NAMING RULE LIVES HERE.** `base.thread_name` is
    what `to_thread_delta` uses to name the delta, and a consumer that wanted
    an address had two options: ask for it, or reimplement `thread-{id}` and
    the project it hangs under. The second is a second copy of a naming rule,
    and the failure mode of that is two systems that agree until somebody
    changes the prefix.

    An unknown source gets `None` for both rather than a guess. A made-up
    address is worse than an absent one: it looks like something a reader can
    go and find.
    """
    from qmcp.threads.base import to_thread_delta

    found = source_classes().get(source)
    if found is None:
        return {"address": None, "perspective": None,
                "phase": None, "delta_type": None}

    # Built rather than assembled from parts. `to_thread_delta` is what decides
    # the name, the address, the phase and the type; restating any of them here
    # would be a second copy of a decision this module does not own, and a
    # consumer would then read whichever copy was edited last.
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

    Constructing a real `Thread` would mean reading the store, which is the
    cost this whole path exists to avoid. `title` and `url` are absent here on
    purpose: the listing already carries the title from the index, and a title
    invented from the id would be a worse one than the real one sitting beside
    it.
    """

    __slots__ = ("id", "title", "url")

    def __init__(self, identifier: str) -> None:
        self.id = identifier
        self.title = None
        self.url = None


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
            # A thread is a delta, so a listing of threads carries the address
            # that delta has. Cheap: both come from class attributes and the
            # naming rule, with nothing read from the store.
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

    @app.post("/v1/threads/reindex")
    async def reindex_only(body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Rebuild the index from what is already in the store. Imports nothing.

        **THE REFRESH THAT IS NOT AN IMPORT.** `/v1/threads/import` re-indexes
        as well, which is right for an import and wrong as the only way to get
        a fresh reading: it makes "refresh the cache" require a path to an
        export somebody may not have any more, and it re-unpacks megabytes to
        answer a question about what is already unpacked.

        `_reindex` was already a function and already spent nothing --
        `index.build` hands every source a budget of zero, so a source that
        would need a paid call refuses rather than billing whoever pressed
        refresh. Only the route was missing.
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

        **THE HARNESS DOES THE WRITING, AND THAT IS THE POINT.** A control
        panel asking for an import is not a control panel authoring the
        archive: the archive stays one record with one author, and the panel is
        a caller. What it may not do is write rows itself.

        The human act happened before any of this: requesting the export from
        the service.
        `governance/qm/records/DRAFT-acts-that-are-a-persons-by-constitution.md`
        clause 3 is explicit that everything after it may be automated, and
        should be.

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
            # Accepting the folder is what an operator will try first.
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

        # Re-index in the same call. An import that left the listing stale
        # would look like it had done nothing, and the caller would have no way
        # to tell that from an export with nothing new in it.
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
