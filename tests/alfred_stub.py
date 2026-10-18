"""A stand-in alfred for the alfred tools' tests, served through `httpx.MockTransport`.

One account, tokens shaped as alfred's JWTs are, and canned answers per route,
so nothing needs a running alfred. `CaptionedAlfred` also holds projects,
their captions and their renders the way alfred's caption routes do, with a
transcription job and a render that each move one state per look. Not
collected: the name does not start with `test_`.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

import httpx

from qmcp.tools.alfred import AlfredClient, AlfredConfig

URL = "http://alfred.test"
EMAIL, PASSWORD = "editor@example.com", "correct horse"

Route = Callable[[httpx.Request], httpx.Response]


def jwt(lifetime: float = 3600, *, iat: float | None = None, n: int = 0) -> str:
    """A token shaped as alfred's: `iat` and `exp` in an unsigned payload."""
    iat = time.time() if iat is None else iat

    def part(data: dict) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    claims = {"sub": "user", "aud": ["fastapi-users:auth"], "iat": int(iat),
              "exp": int(iat + lifetime), "n": n}
    return f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part(claims)}.signature"


class FakeAlfred:
    """A stand-in alfred: logs one account in, accepts the tokens it issued
    (and any added to `valid`), and answers each route it is given. `prefix` is
    a path a proxy in front of it would strip."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.valid: set[str] = set()
        self.issued = 0
        self.lifetime = 3600
        self.open = False  # answers without a token, as alfred's dev bypass does
        self.refresh_works = True
        self.skew = 0.0  # seconds alfred's clock is ahead of this machine's
        self.prefix = ""
        self.routes: dict[tuple[str, str], Route] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def route(self, method: str, path: str, answer: Route | object) -> None:
        """Answer `method path` with `answer`: a callable taking the request, or
        a value served as JSON with 200."""
        if callable(answer):
            self.routes[(method, path)] = answer
        else:
            self.routes[(method, path)] = lambda request: httpx.Response(200, json=answer)

    def issue(self) -> str:
        self.issued += 1
        token = jwt(self.lifetime, iat=time.time() + self.skew, n=self.issued)
        self.valid.add(token)
        return token

    def bearer(self, request: httpx.Request) -> str | None:
        header = request.headers.get("authorization")
        return header.removeprefix("Bearer ") if header else None

    def path(self, request: httpx.Request) -> str:
        return request.url.path.removeprefix(self.prefix)

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if (r.method, self.path(r)) == (method, path)]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, self.path(request))
        if key == ("POST", "/auth/jwt/login"):
            form = parse_qs(request.content.decode())
            if form.get("username") == [EMAIL] and form.get("password") == [PASSWORD]:
                return httpx.Response(200, json={"access_token": self.issue(),
                                                 "token_type": "bearer"})
            return httpx.Response(400, json={"detail": "LOGIN_BAD_CREDENTIALS"})
        if key == ("GET", "/openapi.json"):  # served to anyone, as FastAPI serves it
            return httpx.Response(200, json={"info": {"title": "Alfred API"}})
        token = self.bearer(request)
        if not self.open and token not in self.valid:
            return httpx.Response(401, json={"detail": "Unauthorized"})
        if key == ("POST", "/auth/jwt/refresh"):
            if not self.refresh_works:
                return httpx.Response(500, json={"detail": "Internal Server Error"})
            return httpx.Response(200, json={"access_token": self.issue(),
                                             "token_type": "bearer"})
        answer = self.routes.get(key)
        if answer is None:
            return self.unrouted(request)
        return answer(request)

    def unrouted(self, request: httpx.Request) -> httpx.Response:
        """The answer to a route nobody gave: FastAPI's own 404."""
        return httpx.Response(404, json={"detail": "Not Found"})


def connect(fake: FakeAlfred, **config) -> AlfredClient:
    """A client for the stand-in, logging in as its account unless told otherwise."""
    config = {"email": EMAIL, "password": PASSWORD, **config}
    return AlfredClient(AlfredConfig(url=URL, **config), transport=fake.transport())


# --- captions ------------------------------------------------------------------

NO_ENGINE_DETAIL = "No speech engine is configured: set ALFRED_SPEECH_URL"
RUNNING_DETAIL = "A transcription of this project is already running"
NO_OUTPUT_DETAIL = "That render has no output file yet"
SIGNED_UNTIL = 1893456000  # the `exp` on every download link, 2030-01-01T00:00:00Z

_CAPTIONS = re.compile(r"/project/([^/]+)/captions(/transcribe)?")
_PROJECT = re.compile(r"/project/([^/]+)")
_RENDER = re.compile(r"/render/([^/]+)/(status|deliver)")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CaptionedAlfred(FakeAlfred):
    """The stand-in, holding projects with captions and their renders.

    A project carries `captions`, `caption_style` and `captions_job`, and
    answers the caption routes: `GET` and `PUT /project/{id}/captions`, and
    `POST .../captions/transcribe`, which answers 503 while `speech` says no
    engine is configured and 409 while a job runs. Each look at the captions
    moves a job one state -- queued, transcribing, then done with what the
    engine `heard`, or failed -- while `workers` is above 0. A render moves the
    same way, queued, rendering, rendered, on each look at its status, and is
    delivered by `POST /render/{id}/deliver`. `GET /health` reports `workers`
    and, unless it is None, `speech`. A route given with `route` answers first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.speech: dict[str, Any] | None = {"configured": True, "ok": True}
        self.workers = 1
        self.projects: dict[str, dict[str, Any]] = {}
        self.heard: dict[str, list[dict[str, Any]]] = {}
        self.failure: dict[str, str | None] = {}
        self.renders: dict[str, dict[str, Any]] = {}
        self.queued: list[dict[str, Any]] = []  # each POST /render body, in order

    def add_project(
        self,
        project_id: str,
        name: str,
        *,
        heard: list[dict[str, Any]] | None = None,
        captions: list[dict[str, Any]] | None = None,
        style: dict[str, Any] | None = None,
        render_settings: dict[str, Any] | None = None,
        failure: str | None = None,
    ) -> dict[str, Any]:
        """A project with a one-clip timeline. `heard` is what the speech engine
        will hear in it; `failure`, when given, is why its transcription fails."""
        project = {
            "_id": project_id,
            "name": name,
            "logic": {},
            "edl": {"clips": [{"type": "video", "name": "videos/talk.mp4", "duration": 9.0}],
                    "duration": 9.0},
            "render_settings": render_settings,
            "captions": list(captions or []),
            "caption_style": style,
            "captions_job": None,
        }
        self.projects[project_id] = project
        self.heard[project_id] = list(heard or [])
        self.failure[project_id] = failure
        return project

    # --- the routes ------------------------------------------------------------

    def unrouted(self, request: httpx.Request) -> httpx.Response:
        path = self.path(request)
        if (request.method, path) == ("GET", "/health"):
            return httpx.Response(200, json=self.health())
        if (request.method, path) == ("POST", "/render"):
            return self.queue(json.loads(request.content))
        if match := _CAPTIONS.fullmatch(path):
            project = self.projects.get(match[1])
            if project is None:
                return httpx.Response(404, json={"detail": "Item not found"})
            if match[2]:
                if request.method == "POST":
                    return self.transcribe(project)
            elif request.method == "GET":
                self.tick(project)
                return httpx.Response(200, json=self.held(project))
            elif request.method == "PUT":
                return self.store(project, json.loads(request.content))
        elif match := _PROJECT.fullmatch(path):
            project = self.projects.get(match[1])
            if request.method == "GET":
                if project is None:
                    return httpx.Response(404, json={"detail": "Item not found"})
                return httpx.Response(200, json=project)
        elif match := _RENDER.fullmatch(path):
            render = self.renders.get(match[1])
            if render is None:
                return httpx.Response(404, json={"detail": "Item not found"})
            if (request.method, match[2]) == ("GET", "status"):
                return httpx.Response(200, json=self.status(render))
            if (request.method, match[2]) == ("POST", "deliver"):
                return self.deliver(render, request.url.params.get("destination"))
        return super().unrouted(request)

    def health(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "api": "ok", "db": "ok",
            "renderer": {"ok": self.workers > 0, "workers": self.workers},
            "queue": {"pending": sum(r["state"] == "queued" for r in self.renders.values())},
        }
        if self.speech is not None:
            report["speech"] = self.speech
        return report

    @staticmethod
    def held(project: dict[str, Any]) -> dict[str, Any]:
        return {"captions": project["captions"], "style": project["caption_style"],
                "job": project["captions_job"]}

    def transcribe(self, project: dict[str, Any]) -> httpx.Response:
        if not (self.speech or {}).get("configured"):
            return httpx.Response(503, json={"detail": NO_ENGINE_DETAIL})
        job = project["captions_job"]
        if job and job["state"] in ("queued", "transcribing"):
            return httpx.Response(409, json={"detail": RUNNING_DETAIL})
        project["captions_job"] = {"state": "queued", "error": None, "engine": None,
                                   "segments": None, "updated_at": _now()}
        return httpx.Response(202, json={"job": project["captions_job"]})

    def tick(self, project: dict[str, Any]) -> None:
        """One step of the project's transcription, when a worker runs."""
        job = project["captions_job"]
        if not job or self.workers < 1 or job["state"] not in ("queued", "transcribing"):
            return
        job["updated_at"] = _now()
        if job["state"] == "queued":
            job["state"] = "transcribing"
            return
        failure = self.failure[project["_id"]]
        if failure:
            job.update(state="failed", error=failure)
            return
        heard = self.heard[project["_id"]]
        project["captions"] = [dict(caption) for caption in heard]
        job.update(state="done", engine="stub engine", segments=len(heard))

    def store(self, project: dict[str, Any], body: dict[str, Any]) -> httpx.Response:
        """`PUT /project/{id}/captions`: checked as alfred checks them, and only
        the captions and the style stored."""
        captions = body.get("captions", project["captions"])
        last = 0.0
        for n, caption in enumerate(captions, 1):
            start, end = caption.get("start"), caption.get("end")
            text = str(caption.get("text") or "").strip()
            if not isinstance(start, (int, float)) or start < 0:
                return httpx.Response(422, json={"detail": f"caption {n}: start must be 0 or more"})
            if not isinstance(end, (int, float)) or end <= start:
                return httpx.Response(422, json={"detail": f"caption {n}: end must be after start"})
            if not text:
                return httpx.Response(422, json={"detail": f"caption {n}: text is empty"})
            if start < last:
                return httpx.Response(422, json={"detail": f"caption {n}: starts before the one "
                                                           "ahead of it"})
            last = start
        project["captions"] = [dict(caption) for caption in captions]
        project["caption_style"] = body.get("style", project["caption_style"])
        return httpx.Response(200, json=self.held(project))

    def queue(self, body: dict[str, Any]) -> httpx.Response:
        self.queued.append(body)
        render_id = f"r{len(self.queued)}"
        self.renders[render_id] = {"id": render_id, "state": "queued", "progress": None,
                                   "destination": None}
        return httpx.Response(200, json=render_id)

    def status(self, render: dict[str, Any]) -> dict[str, Any]:
        """One step of the render, when a worker runs, then its status."""
        if self.workers > 0:
            step = {"queued": ("rendering", 50.0), "rendering": ("rendered", 100.0)}
            render["state"], render["progress"] = step.get(
                render["state"], (render["state"], render["progress"]))
        rendered = render["state"] == "rendered"
        link = f"/render/{render['id']}/download?exp={SIGNED_UNTIL}&sig=c2ln"
        return {"id": render["id"], "state": render["state"], "progress": render["progress"],
                "error": None, "download_url": link if rendered else None}

    def deliver(self, render: dict[str, Any], destination: str | None) -> httpx.Response:
        if render["state"] != "rendered":
            return httpx.Response(409, json={"detail": NO_OUTPUT_DETAIL})
        chosen = destination or render["destination"] or "local"
        if chosen != "local":
            return httpx.Response(502, json={
                "detail": f"unknown destination {chosen!r}; known: ['bucket', 'local']"})
        render["destination"] = chosen
        return httpx.Response(200, json={"destination": chosen,
                                         "location": f"/render/{render['id']}/download"})
