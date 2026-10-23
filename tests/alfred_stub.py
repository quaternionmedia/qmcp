"""A stand-in alfred for the alfred tools' tests, served through `httpx.MockTransport`.

One account, tokens shaped as alfred's JWTs are, and canned answers per route,
so nothing needs a running alfred. `CaptionedAlfred` also holds projects,
their captions and their renders the way alfred's caption routes do, with a
transcription job and a render that each move one state per look. What it
answers in alfred's own words is copied from alfred: `HEALTH` is alfred's
`GET /health`, kept in `tests/alfred_health.json`; `CAPTIONS` its
`GET /project/{id}/captions` for a project never captioned, kept in
`tests/alfred_captions.json`; and `REFUSALS` its refusals, each its status and
its body, `{detail, code}`, kept in `tests/alfred_refusals.json`. Not
collected: the name does not start with `test_`.
"""

from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
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


# --- alfred's own answers ------------------------------------------------------

# alfred's `GET /health` as alfred's code answers it with no audio engine set:
# its `Health` model, alfred/core/routes/health.py.
HEALTH: dict[str, Any] = json.loads(
    Path(__file__).with_name("alfred_health.json").read_text(encoding="utf-8"))

# alfred's `GET /project/{id}/captions` for a project never captioned, as its
# code answers it: no captions, no job, and a caption style with every field,
# each null, `enabled` false (alfred/core/routes/captions.py, `view_of`).
CAPTIONS: dict[str, Any] = json.loads(
    Path(__file__).with_name("alfred_captions.json").read_text(encoding="utf-8"))

# alfred's refusals as alfred's code answers them, each `{status, body}` by its
# `code`: captured from alfred's routes (alfred/core/errors.py lists the codes,
# its docs/api/errors.md says what each means). Where a refusal names what was
# asked -- how many captions, which destination -- the stand-in words it as
# alfred does (`too_many_captions`, `unknown_destination`), held to these.
REFUSALS: dict[str, dict[str, Any]] = json.loads(
    Path(__file__).with_name("alfred_refusals.json").read_text(encoding="utf-8"))


def said(code: str) -> str:
    """The sentence alfred refuses with, beside `code`."""
    return REFUSALS[code]["body"]["detail"]


# alfred's details, from alfred/core/routes/captions.py and render.py. The 503
# for no audio engine is the sentence the health report's `detail` carries:
# both are `problem()`'s, in alfred/core/utils/audio_engine.py. The 503 for a
# job the broker would not take carries what the broker raised, here a refused
# connection, and alfred records the job as failed with it.
NO_ENGINE_DETAIL: str = HEALTH["audio_engine"]["detail"]
BROKER_DOWN = "[Errno 111] Connection refused"
NOT_QUEUED_DETAIL = said("transcription_not_queued")
NOT_QUEUED_ERROR = f"could not be queued: {BROKER_DOWN}"
RUNNING_DETAIL = said("transcription_running")
NO_OUTPUT_DETAIL = said("render_not_finished")
# A job's error when the engine heard no speech (alfred/core/utils/transcription.py).
NO_SPEECH_ERROR = "the audio engine heard no speech in the timeline's sound"
# alfred's answers for an id (alfred/core/utils/bucket.py): one that is not an
# ObjectId is refused before anything is looked up, one nobody holds is 404.
INVALID_ID_DETAIL = said("invalid_id")
NOT_FOUND_DETAIL = said("not_found")

# --- captions ------------------------------------------------------------------

SIGNED_UNTIL = 1893456000  # the `exp` on every download link, 2030-01-01T00:00:00Z

_CAPTIONS = re.compile(r"/project/([^/]+)/captions(/transcribe)?")
_PROJECT = re.compile(r"/project/([^/]+)")
_RENDER = re.compile(r"/render/([^/]+)/(status|deliver)")
_OBJECT_ID = re.compile(r"[0-9a-fA-F]{24}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def render_id(n: int) -> str:
    """The id of the `n`th render the stand-in queues: an ObjectId's 24 hex
    digits, as alfred's `POST /render` answers."""
    return f"6a79{n:020d}"


# The most captions alfred keeps on a project (alfred/core/utils/captions.py).
MAX_CAPTIONS = 1000


def too_many_captions(count: int) -> str:
    """alfred's sentence for `count` captions, more than a project carries."""
    return f"{count} captions is more than the {MAX_CAPTIONS} a project may carry"


def unknown_destination(name: str) -> str:
    """alfred's sentence for a destination it has no such name for."""
    return f"unknown destination {name!r}; known: ['bucket', 'local']"


def project_style(stored: Any) -> dict[str, Any]:
    """A project's caption style as alfred answers it: every field of one,
    null where nothing is stored, and `enabled` false unless stored true or
    false (alfred/core/utils/captions.py, `project_style`)."""
    stored = stored if isinstance(stored, dict) else {}
    style = {key: stored.get(key) for key in CAPTIONS["style"]}
    if style["enabled"] is None:
        style["enabled"] = False
    return style


def style_to_store(sent: dict[str, Any] | None, stored: Any) -> dict[str, Any] | None:
    """The style alfred keeps when a project's captions are written, or None
    when the stored one stands (alfred/core/utils/captions.py,
    `style_to_store`): one sent, with `enabled` as sent, else as stored, else
    false; none sent, the stored one, given `enabled: false` if it says nothing."""
    stored = stored if isinstance(stored, dict) else {}
    if sent is not None:
        style = {key: sent.get(key) for key in CAPTIONS["style"]}
        if "enabled" not in sent:
            style["enabled"] = bool(stored.get("enabled"))
        return style
    if stored.get("enabled") is not None:
        return None
    return project_style(stored)


def body_errors(body: dict[str, Any]) -> list[dict[str, Any]]:
    """What alfred refuses in a `PUT /project/{id}/captions` body, as FastAPI
    answers its CaptionsUpdate (alfred/core/routes/captions.py) before the
    route looks at the id: captions it requires and checks, and any field
    besides the captions and the style."""
    if "captions" in body:
        errors = caption_errors(body["captions"])
    else:
        errors = [{"loc": ["body", "captions"], "msg": "field required",
                   "type": "value_error.missing"}]
    return errors + [{"loc": ["body", key], "msg": "extra fields not permitted",
                      "type": "value_error.extra"}
                     for key in body if key not in ("captions", "style")]


def caption_errors(captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What alfred refuses in a `PUT`'s captions, as FastAPI answers alfred's
    `Caption` checks (alfred/core/models/captions.py): a time before 0, text
    with nothing to read, and -- once both of those hold -- an end that is not
    after the start. alfred puts captions in order rather than refusing them."""
    errors = []
    for n, caption in enumerate(captions):
        where, times, sound = ["body", "captions", n], {}, True
        for key in ("start", "end"):
            if caption[key] < 0:
                errors.append({"loc": [*where, key], "msg": f"{key} must not be negative",
                               "type": "value_error"})
                sound = False
            times[key] = round(float(caption[key]), 3)
        if not " ".join(str(caption.get("text") or "").split()):
            errors.append({"loc": [*where, "text"], "msg": "a caption needs some text",
                           "type": "value_error"})
            sound = False
        if sound and times["end"] <= times["start"]:
            errors.append({"loc": [*where, "__root__"],
                           "msg": f"a caption must end after it starts "
                                  f"(start {times['start']}, end {times['end']})",
                           "type": "value_error"})
    return errors


class CaptionedAlfred(FakeAlfred):
    """The stand-in, holding projects with captions and their renders.

    A project carries `captions`, `caption_style` and `captions_job`, and
    answers the caption routes as alfred does, its style always whole
    (`CAPTIONS`'s shape): `GET` and `PUT /project/{id}/captions`, and
    `POST .../captions/transcribe`, which answers 503 while `audio_engine`
    says none is configured, 409 while a job runs, and 503 again, with the
    job failed, while `broker` is false. Each look at the captions moves a
    job one state -- queued, transcribing, then done with what the engine
    `heard`, or failed, as when it heard nothing, with the captions the
    project had kept -- while `workers` is above 0. A render moves the
    same way, queued, rendering, rendered, on each look at its status, and is
    delivered by `POST /render/{id}/deliver`. `GET /health` is alfred's report,
    `HEALTH`'s shape, with `workers`, the renders queued and `audio_engine`.
    Projects and renders are named as alfred names them, by ObjectId, and
    refused as alfred refuses them: a body first, then a malformed id, then an
    unknown one. Every refusal of alfred's own is one of `REFUSALS`, its
    `code` beside its `detail` -- or the detail alone while `codes` is false,
    as an alfred from before codes answers. A route given with `route`
    answers first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.audio_engine: dict[str, Any] = {"configured": True, "ok": True, "detail": None}
        self.broker = True  # whether alfred's broker takes a job
        self.codes = True  # whether alfred puts a `code` beside a refusal's detail
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
        """A project with a one-clip timeline, held as alfred stores one: no
        captions or style is null until one is given. `heard` is what the audio
        engine will hear in it; `failure`, when given, is why its transcription
        fails."""
        project = {
            "_id": project_id,
            "name": name,
            "logic": {},
            "edl": {"clips": [{"type": "video", "name": "videos/talk.mp4", "duration": 9.0}],
                    "duration": 9.0},
            "render_settings": render_settings,
            "captions": None if captions is None else list(captions),
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
        # As alfred refuses: a body FastAPI will not take, then an id that is
        # not an ObjectId, then one nobody holds; only then the route's own.
        if match := _CAPTIONS.fullmatch(path):
            transcribing = bool(match[2])
            if (request.method, transcribing) == ("PUT", False):
                body = json.loads(request.content)
                if errors := body_errors(body):
                    return httpx.Response(422, json={"detail": errors})
                return self._with(self.projects, match[1], lambda p: self.store(p, body))
            if (request.method, transcribing) == ("GET", False):
                return self._with(self.projects, match[1], self.look)
            if (request.method, transcribing) == ("POST", True):
                return self._with(self.projects, match[1], self.transcribe)
        elif (match := _PROJECT.fullmatch(path)) and request.method == "GET":
            return self._with(self.projects, match[1],
                              lambda project: httpx.Response(200, json=project))
        elif match := _RENDER.fullmatch(path):
            if (request.method, match[2]) == ("GET", "status"):
                return self._with(self.renders, match[1],
                                  lambda render: httpx.Response(200, json=self.status(render)))
            if (request.method, match[2]) == ("POST", "deliver"):
                destination = request.url.params.get("destination")
                return self._with(self.renders, match[1],
                                  lambda render: self.deliver(render, destination))
        return super().unrouted(request)

    def refuse(self, code: str, detail: str | None = None) -> httpx.Response:
        """alfred's refusal `code`, with its status and sentence (`REFUSALS`), or
        `detail` where alfred's sentence names what was asked; without the code
        while `codes` is false."""
        body = {**REFUSALS[code]["body"], **({} if detail is None else {"detail": detail})}
        if not self.codes:
            del body["code"]
        return httpx.Response(REFUSALS[code]["status"], json=body)

    def _with(self, held: dict[str, dict[str, Any]], item_id: str,
              answer: Callable[[dict[str, Any]], httpx.Response]) -> httpx.Response:
        """`answer` for what `held` keeps under `item_id`, or alfred's refusal:
        422 for an id that is not an ObjectId, before anything is looked up,
        and 404 for one nobody holds."""
        if not _OBJECT_ID.fullmatch(item_id):
            return self.refuse("invalid_id")
        if item_id not in held:
            return self.refuse("not_found")
        return answer(held[item_id])

    def without_an_audio_engine(self) -> None:
        """Answer as alfred does with no ALFRED_AUDIO_ENGINE_URL set."""
        self.audio_engine = dict(HEALTH["audio_engine"])

    def health(self) -> dict[str, Any]:
        return {
            "api": "ok", "db": "ok",
            "renderer": {"ok": self.workers > 0, "workers": self.workers},
            "queue": {"pending": sum(r["state"] == "queued" for r in self.renders.values())},
            "audio_engine": dict(self.audio_engine),
        }

    @staticmethod
    def held(project: dict[str, Any]) -> dict[str, Any]:
        """A project's captions, style and job as alfred answers them."""
        return {"captions": project["captions"] or [],
                "style": project_style(project["caption_style"]),
                "job": project["captions_job"]}

    def look(self, project: dict[str, Any]) -> httpx.Response:
        """`GET /project/{id}/captions`, which moves the job one step first."""
        self.tick(project)
        return httpx.Response(200, json=self.held(project))

    def transcribe(self, project: dict[str, Any]) -> httpx.Response:
        if not self.audio_engine.get("configured"):
            return self.refuse("audio_engine_missing")
        job = project["captions_job"]
        if job and job["state"] in ("queued", "transcribing"):
            return self.refuse("transcription_running")
        project["captions_job"] = {"state": "queued", "error": None, "engine": None,
                                   "segments": None, "updated_at": _now()}
        if not self.broker:
            project["captions_job"].update(state="failed", error=NOT_QUEUED_ERROR)
            return self.refuse("transcription_not_queued")
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
        heard = self.heard[project["_id"]]
        failure = self.failure[project["_id"]] or (None if heard else NO_SPEECH_ERROR)
        if failure:
            job.update(state="failed", error=failure)
            return
        project["captions"] = [dict(caption) for caption in heard]
        # As alfred's worker does: a style that says nothing of `enabled` is
        # stored saying false, so new captions are not drawn until someone says.
        if (style := style_to_store(None, project["caption_style"])) is not None:
            project["caption_style"] = style
        job.update(state="done", engine="stub engine", segments=len(heard))

    def store(self, project: dict[str, Any], body: dict[str, Any]) -> httpx.Response:
        """`PUT /project/{id}/captions` for a body already checked (`body_errors`)
        and a project found: counted as alfred counts them, stored in order of
        when they start, as alfred stores them, with the style alfred keeps
        (`style_to_store`), and nothing else stored."""
        captions = body["captions"]
        if len(captions) > MAX_CAPTIONS:
            return self.refuse("too_many_captions", too_many_captions(len(captions)))
        ordered = sorted(captions, key=lambda caption: (caption["start"], caption["end"]))
        project["captions"] = [dict(caption) for caption in ordered]
        if (style := style_to_store(body.get("style"), project["caption_style"])) is not None:
            project["caption_style"] = style
        return httpx.Response(200, json=self.held(project))

    def queue(self, body: dict[str, Any]) -> httpx.Response:
        self.queued.append(body)
        queued = render_id(len(self.queued))
        self.renders[queued] = {"id": queued, "state": "queued", "progress": None,
                                "destination": None}
        return httpx.Response(200, json=queued)

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
            return self.refuse("render_not_finished")
        chosen = destination or render["destination"] or "local"
        if chosen != "local":
            return self.refuse("destination_unknown", unknown_destination(chosen))
        render["destination"] = chosen
        return httpx.Response(200, json={"destination": chosen,
                                         "location": f"/render/{render['id']}/download"})
