"""Answer a pending request by voice, started over HTTP.

`POST /v1/human/requests/{id}/voice` asks the request aloud on this machine and
takes the spoken answer, so a page -- joe's front end -- can start the
conversation without a terminal. `GET /v1/human/voice` says how the last one
went.

The conversation runs as `qmcp human voice <id>`, in a process of its own. On
Windows the synthesizer drives the sound card through COM, which wants a
process's main thread and not a server's worker, and the command already
carries the preflight, the dialog and every message a person needs when the
engine is down or the answer never came.

One conversation at a time: there is one microphone and one pair of speakers.
`VoiceRuns` is that fact as an object, and it is shared: the instruction inbox
(`qmcp.instructions.service`) starts `qmcp instruct --voice` through the same
tracker, so an approval being asked and an instruction being taken cannot
overlap, and either route's status names whichever is running. The routes are
registered only on loopback, like the thread archive, because a caller
elsewhere has no business making this machine speak and listen.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Module level, not inside `register`: the routes' annotations are strings
# under `from __future__ import annotations`, and FastAPI resolves them here.
from fastapi import HTTPException, Request

OUTPUT_LINES = 20


class VoiceRuns:
    """The one conversation this machine can hold, and how the last one ended."""

    def __init__(self, log_dir: Path | None = None, popen=subprocess.Popen):
        self._lock = threading.Lock()
        self._popen = popen
        self._log_dir = log_dir or Path(tempfile.gettempdir())
        self._process = None
        # What the conversation is: `approval` for a request on the human
        # queue, `instruction` for one being taken. The request id is set only
        # for the first, and stays in the status for the second so a reader of
        # the payload sees one shape.
        self._kind: str | None = None
        self._request_id: str | None = None
        self._log: Path | None = None
        self._started: float | None = None

    def running(self) -> str | None:
        """What is being asked right now -- the request id, or the kind when the
        conversation has no request -- or None."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self._request_id or self._kind
            return None

    def start(self, argv: list[str], kind: str = "approval",
              request_id: str | None = None) -> dict[str, Any]:
        """Start a conversation. Raises RuntimeError, naming what runs, if one is."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(self._request_id or self._kind or "")
            self._log = self._log_dir / f"qmcp-voice-{os.getpid()}.log"
            with open(self._log, "w", encoding="utf-8") as out:
                self._process = self._popen(
                    argv, stdout=out, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                )
            self._kind = kind
            self._request_id = request_id
            self._started = time.time()
            return {"kind": kind, "request_id": request_id, "running": True,
                    "started_at": self._when()}

    def status(self) -> dict[str, Any]:
        """Whether a conversation is running, and how the last one ended."""
        with self._lock:
            if self._process is None:
                return {"running": False, "kind": None, "request_id": None,
                        "exit_code": None, "output": [], "started_at": None}
            code = self._process.poll()
            return {
                "running": code is None,
                "kind": self._kind,
                "request_id": self._request_id,
                "exit_code": code,
                "output": self._tail(),
                "started_at": self._when(),
            }

    def _when(self) -> str | None:
        return (datetime.fromtimestamp(self._started, UTC).isoformat()
                if self._started else None)

    def _tail(self) -> list[str]:
        try:
            lines = self._log.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, AttributeError):
            return []
        return [line for line in lines if line.strip()][-OUTPUT_LINES:]


def register(app: Any, engine: str, engine_url: str | None,
             runs: VoiceRuns | None = None) -> VoiceRuns:
    """Attach the voice routes to a FastAPI app. Returns the run tracker."""
    from sqlmodel import select

    from qmcp.db import HumanRequest, get_session
    from qmcp.db.models import HumanRequestStatus

    runs = runs or VoiceRuns()

    @app.post("/v1/human/requests/{request_id}/voice", status_code=202)
    async def answer_by_voice(request_id: str, request: Request) -> dict[str, Any]:
        """Ask a pending request aloud here, and take the spoken answer.

        202 once the conversation has started; `GET /v1/human/voice` says how
        it ends. 404 for no such request, 409 for one that is not waiting (it
        has an answer or has expired) or while another conversation runs.
        """
        async with get_session() as session:
            result = await session.execute(
                select(HumanRequest).where(HumanRequest.id == request_id))
            human_request = result.scalar_one_or_none()
        if human_request is None:
            raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found")

        expires = human_request.expires_at
        if expires is not None and expires.tzinfo is not None:
            expires = expires.astimezone(UTC).replace(tzinfo=None)
        expired = expires is not None and expires <= datetime.now(UTC).replace(tzinfo=None)
        if human_request.status != HumanRequestStatus.PENDING or expired:
            raise HTTPException(
                status_code=409,
                detail=f"Request '{request_id}' is not waiting on anyone"
                       f" ({'expired' if expired else human_request.status.value}).")

        # The socket this request arrived on, not the Host header: a page
        # reaches this through a dev server's proxy, whose Host names the page.
        host, port = request.scope.get("server") or ("127.0.0.1", 3141)
        argv = [sys.executable, "-m", "qmcp", "human", "voice", request_id,
                "--base-url", f"http://{host}:{port}", "--engine", engine]
        if engine_url:
            argv += ["--engine-url", engine_url]
        try:
            return runs.start(argv, kind="approval", request_id=request_id)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"A conversation is already running, for '{exc}'. One at a time:"
                       " there is one microphone.")

    @app.get("/v1/human/voice")
    async def voice_status() -> dict[str, Any]:
        """Whether a conversation is running, and how the last one ended:
        its exit code and the last lines it printed."""
        return runs.status()

    return runs
