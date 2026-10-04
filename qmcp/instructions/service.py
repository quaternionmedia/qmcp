"""The inbox over HTTP, for a page and for the CLI.

    POST /v1/instructions              record one; 201 with the row
    GET  /v1/instructions?status=      what has been recorded, newest first
    GET  /v1/instructions/{id}         one row
    POST /v1/instructions/voice        take one by voice on this machine; 202 or 409
    GET  /v1/instructions/voice        whether that is running, and how the last ended

**LOOPBACK ONLY, LIKE THE VOICE ROUTES.** An instruction is a person's own
words about what should be done, and the voice route makes this machine speak
and listen. Neither belongs on a socket somebody elsewhere can reach, so
`qmcp.server` registers these only when it is bound to loopback and they do not
exist otherwise -- nothing to reach, rather than something that refuses.

**THE SPOKEN ROUTE STARTS A COMMAND.** `POST /v1/instructions/voice` runs
`qmcp instruct --voice` in a process of its own, through the tracker the
approval route uses, for the reasons `qmcp.integrations.voice.service` gives:
the synthesizer wants a main thread, the command already carries the preflight
and every message a person needs, and there is one microphone. Sharing the
tracker is what makes "one conversation at a time" true across both kinds.

**THE SERVER RESOLVES; THE RECORD IS THE SERVER'S.** A typed instruction and a
page's arrive as text and are read against the roster here, so every row's
`detail` was produced by one function on one list. The spoken dialog reads the
same roster before it records, because an ambiguous name has to be asked back
before anything is written; it states a project only when the person chose or
spoke one, and sends a text that named one project without it, so that row too
carries the match the server made and the rule that made it.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

# Module level, not inside `register`: the routes' annotations are strings
# under `from __future__ import annotations`, and FastAPI resolves them here.
from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from qmcp.db.models import Instruction, InstructionSource, InstructionStatus
from qmcp.instructions import resolve, roster_names
from qmcp.integrations.voice.service import VoiceRuns

Sessions = Callable[[], Any]
"""A callable returning an async context manager that yields a session,
committed on exit -- what `qmcp.db.get_session` is."""

Names = Callable[[], Iterable[str]]
"""Where the roster comes from, asked on every record so a changed workspace
is seen without a restart."""


class InstructionCreate(BaseModel):
    """What a caller sends to record an instruction."""

    text: str = Field(..., min_length=1, description="The instruction, in the person's words")
    source: InstructionSource = Field(
        default=InstructionSource.TYPED,
        description="How it arrived: voice, typed, or page")
    project: str | None = Field(
        default=None,
        description="The project it is for, stated outright; skips the matching."
                    " Blank is nothing stated.")
    heard: list[str] | None = Field(
        default=None,
        description="For a spoken instruction, every transcript the dialog took, in order")

    @field_validator("text")
    @classmethod
    def _said_something(cls, text: str) -> str:
        # Stripped before the length is read: whitespace alone is not an
        # instruction, and `min_length` alone would record it as one.
        if not text.strip():
            raise ValueError("an instruction has to say something")
        return text.strip()


def register(app: Any, runs: VoiceRuns, engine: str = "joe",
             engine_url: str | None = None, names: Names | None = None,
             sessions: Sessions | None = None) -> None:
    """Attach the inbox routes. `runs` is the machine's one conversation.

    `names` defaults to the checkout's own roster, read when a row is
    recorded; `sessions` defaults to the configured database. A walkthrough
    or a test hands in its own, because the configured one is somebody's queue.
    """
    from sqlmodel import select

    if sessions is None:
        from qmcp.db import get_session
        sessions = get_session

    def known() -> Iterable[str]:
        # Looked up on each call rather than bound as a default, so the roster
        # a route reads is the module's at that moment.
        return names() if names is not None else roster_names()

    # Registered before `/v1/instructions/{instruction_id}`: a path is matched
    # in registration order, and `voice` would otherwise be read as an id.
    @app.post("/v1/instructions/voice", status_code=202)
    async def instruct_by_voice(request: Request) -> dict[str, Any]:
        """Take one instruction by voice, here, as `qmcp instruct --voice`.

        202 once the dialog has started; `GET /v1/instructions/voice` says how
        it ends. 409 while any conversation runs on this machine, an approval
        included.
        """
        # The socket this request arrived on, not the Host header: a page
        # reaches this through a dev server's proxy, whose Host names the page.
        host, port = request.scope.get("server") or ("127.0.0.1", 3141)
        argv = [sys.executable, "-m", "qmcp", "instruct", "--voice",
                "--base-url", f"http://{host}:{port}", "--engine", engine]
        if engine_url:
            argv += ["--engine-url", engine_url]
        try:
            return runs.start(argv, kind="instruction")
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"A conversation is already running, for '{exc}'. One at a time:"
                       " there is one microphone.")

    @app.get("/v1/instructions/voice")
    async def voice_status() -> dict[str, Any]:
        """Whether a conversation is running on this machine, and how the last
        one ended: its kind, its exit code and the last lines it printed."""
        return runs.status()

    @app.post("/v1/instructions", status_code=201)
    async def record_instruction(body: InstructionCreate) -> Instruction:
        """Record one instruction. Nothing runs.

        The project is read from the text against the roster unless stated;
        the row says which names matched and by what rule, and is
        `unresolved` rather than guessed when none or several did.
        """
        found = resolve(body.text, known(), project=body.project)
        detail = found.detail()
        if body.heard is not None:
            detail["heard"] = body.heard
        # Naive UTC, as the rest of this database keeps time: SQLite stores no
        # zone, so a row read back carries none, and the row returned here
        # must read the same as the row read later.
        now = datetime.now(UTC).replace(tzinfo=None)
        row = Instruction(text=body.text, project=found.project, source=body.source,
                          status=found.status, created_at=now, updated_at=now,
                          detail=detail)
        async with sessions() as session:
            session.add(row)
        return row

    @app.get("/v1/instructions")
    async def list_instructions(
        status_filter: InstructionStatus | None = Query(
            default=None, alias="status", description="recorded or unresolved"),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        """What has been recorded, newest first."""
        async with sessions() as session:
            query = select(Instruction).order_by(Instruction.created_at.desc())
            if status_filter is not None:
                query = query.where(Instruction.status == status_filter)
            rows = (await session.execute(query.limit(limit))).scalars().all()
        return {"instructions": list(rows), "count": len(rows)}

    @app.get("/v1/instructions/{instruction_id}")
    async def get_instruction(instruction_id: str) -> Instruction:
        """One instruction, with its evidence."""
        async with sessions() as session:
            row = (await session.execute(
                select(Instruction).where(Instruction.id == instruction_id)
            )).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404,
                                detail=f"Instruction '{instruction_id}' not found")
        return row
