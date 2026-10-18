"""Tools that drive alfred, the self-hosted video editor, over its HTTP API.

`alfred_projects`, `alfred_templates`, `alfred_render_enqueue`,
`alfred_render_status`, `alfred_render_deliver`, `alfred_health`, and the
caption tools `alfred_captions`, `alfred_captions_transcribe` and
`alfred_captions_set` each ask the alfred named by `QMCP_ALFRED_URL` and
return what it answered, reshaped for a caller that is not alfred's own
editor. `docs/integrations/alfred.md` is the reference: the settings, each
tool's input and output, and the errors.

Nothing from alfred or its render engine is imported; every call is HTTP. A
request alfred refuses raises `AlfredError` carrying alfred's own `detail`,
which the server records as the invocation's error. Between calls the tools
keep one thing: the client, with alfred's token, so a call does not log in
again while the token works.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from qmcp import config
from qmcp.tools.registry import tool_registry

# Seconds alfred has to answer one request when `QMCP_ALFRED_TIMEOUT` is unset
# (the settings' own default), and the most a connection may take to open.
DEFAULT_TIMEOUT: float = config.Settings.model_fields["alfred_timeout"].default
CONNECT_TIMEOUT = 3.0

# A token with fewer seconds than this left is renewed before it is sent.
RENEW_MARGIN = 300.0

# What a render is asked to be, beside its EDL. A project's saved
# `render_settings` supply any of these the caller leaves out.
RENDER_OPTIONS = ("width", "height", "fps", "quality")

NOT_CONFIGURED = (
    "alfred is not configured: set QMCP_ALFRED_URL to its API's address, with "
    "QMCP_ALFRED_EMAIL and QMCP_ALFRED_PASSWORD, or QMCP_ALFRED_TOKEN, for an "
    "alfred that asks for an account"
)

# How `alfred_captions_transcribe` opens the two refusals a transcription can
# meet: alfred's 503, when it has no speech engine, and its 409, while one of
# the project's transcriptions is running. `qmcp.cookbook.alfred_captions`
# tells the two apart by these, through the error the server records.
NO_SPEECH_ENGINE = "alfred has no speech engine"
TRANSCRIPTION_RUNNING = "a transcription is already running"


class AlfredError(RuntimeError):
    """alfred could not be asked, or refused what it was asked.

    `status` is alfred's HTTP status when it answered, and `detail` its `detail`.
    """

    def __init__(self, message: str, *, status: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


class AlfredNotConfiguredError(AlfredError):
    """The settings name no alfred, or half of an account."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlfredConfig:
    """Where alfred answers and who asks it. The secrets stay out of `repr`."""

    url: str
    email: str | None = None
    password: str | None = field(default=None, repr=False)
    token: str | None = field(default=None, repr=False)
    timeout: float = DEFAULT_TIMEOUT

    @property
    def can_log_in(self) -> bool:
        return bool(self.email and self.password)

    @classmethod
    def from_settings(cls, settings: Any = None) -> AlfredConfig:
        """The configuration qmcp's settings hold.

        Raises `AlfredNotConfiguredError` for no URL, or for an email without a
        password or the reverse. Fields are read with defaults, so a settings
        object without them is an unconfigured alfred.
        """
        settings = settings if settings is not None else config.get_settings()
        url = (getattr(settings, "alfred_url", None) or "").strip().rstrip("/")
        if not url:
            raise AlfredNotConfiguredError(NOT_CONFIGURED)
        email = (getattr(settings, "alfred_email", None) or "").strip() or None
        password = _reveal(getattr(settings, "alfred_password", None))
        if bool(email) != bool(password):
            missing = "QMCP_ALFRED_PASSWORD" if email else "QMCP_ALFRED_EMAIL"
            raise AlfredNotConfiguredError(
                f"alfred's account is half configured: {missing} is unset"
            )
        return cls(
            url=url,
            email=email,
            password=password,
            token=_reveal(getattr(settings, "alfred_token", None)),
            timeout=float(getattr(settings, "alfred_timeout", None) or DEFAULT_TIMEOUT),
        )


def _reveal(value: Any) -> str | None:
    """A secret's text, exactly, from a `SecretStr` or a plain string; blank is None."""
    if value is None:
        return None
    text = value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)
    return text if text.strip() else None


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _claims(token: str) -> dict[str, Any]:
    """A JWT's payload, read without verifying it; {} when it is not one."""
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _seconds_left(token: str, *, issued_now: bool) -> float | None:
    """Seconds `token` works for from now, or None when it does not say.

    A token alfred has just issued works for `exp - iat`, both on alfred's
    clock. A token issued earlier is measured against this machine's clock.
    """
    claims = _claims(token)
    exp, iat = claims.get("exp"), claims.get("iat")
    if not isinstance(exp, (int, float)):
        return None
    if issued_now and isinstance(iat, (int, float)):
        return float(exp - iat)
    return float(exp) - time.time()


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class AlfredClient:
    """One alfred, asked as one account.

    The token is kept and reused. One about to lapse is renewed with
    `POST /auth/jwt/refresh` before it is sent, and a request refused with 401
    is sent once more after a fresh login, when an account is configured.
    With neither an account nor a token, requests carry no token.
    """

    def __init__(
        self,
        cfg: AlfredConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = cfg
        self._clock = clock
        self._http = httpx.Client(
            base_url=cfg.url,
            timeout=httpx.Timeout(cfg.timeout, connect=min(CONNECT_TIMEOUT, cfg.timeout)),
            transport=transport,
        )
        self._lock = threading.Lock()
        self._token: str | None = None
        self._renew_at: float | None = None
        if cfg.token:
            self._keep(cfg.token, issued_now=False)

    def close(self) -> None:
        self._http.close()

    def url_for(self, path: str) -> str:
        """`path` on this alfred as an absolute URL; an absolute URL is returned as it is."""
        if urlsplit(path).scheme:
            return path
        return f"{self.config.url}/{path.lstrip('/')}"

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, body: Any) -> Any:
        return self.request("POST", path, json=body)

    def put(self, path: str, body: Any) -> Any:
        return self.request("PUT", path, json=body)

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        """alfred's JSON answer to one request; `AlfredError` for anything else."""
        token = self._authorization()
        response = self._send(method, path, token, **kwargs)
        if response.status_code == 401 and self.config.can_log_in:
            response = self._send(method, path, self._log_in_again(token), **kwargs)
        return self._answer(method, path, response)

    # --- the token ----------------------------------------------------------

    def _keep(self, token: str, *, issued_now: bool) -> None:
        left = _seconds_left(token, issued_now=issued_now)
        self._token = token
        self._renew_at = None if left is None else self._clock() + left - RENEW_MARGIN

    def _authorization(self) -> str | None:
        with self._lock:
            if self._token is None:
                if self.config.can_log_in:
                    self._log_in()
            elif self._renew_at is not None and self._clock() >= self._renew_at:
                self._renew()
            return self._token

    def _log_in_again(self, refused: str | None) -> str | None:
        with self._lock:
            if self._token == refused:
                self._log_in()
            return self._token

    def _log_in(self) -> None:
        response = self._send(
            "POST",
            "/auth/jwt/login",
            None,
            data={"username": self.config.email, "password": self.config.password},
        )
        if response.is_error:
            detail = _detail(response)
            raise AlfredError(
                f"alfred refused the login for {self.config.email} "
                f"({response.status_code}): {_text(detail)}",
                status=response.status_code,
                detail=detail,
            )
        self._keep(_access_token(response, "/auth/jwt/login"), issued_now=True)

    def _renew(self) -> None:
        response = self._send("POST", "/auth/jwt/refresh", self._token)
        if response.status_code == 200:
            try:
                self._keep(_access_token(response, "/auth/jwt/refresh"), issued_now=True)
                return
            except AlfredError:
                pass
        if self.config.can_log_in:
            self._log_in()
        else:
            # Sent as it is from here on; alfred's answer says whether it works.
            self._renew_at = None

    # --- one exchange -------------------------------------------------------

    def _send(self, method: str, path: str, token: str | None, **kwargs: Any) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"} if token else None
        try:
            return self._http.request(method, path, headers=headers, **kwargs)
        except httpx.TimeoutException as exc:
            raise AlfredError(
                f"alfred at {self.config.url} did not answer {method} {path} in time "
                f"({type(exc).__name__}, {self.config.timeout:g} s)"
            ) from exc
        except httpx.HTTPError as exc:
            raise AlfredError(
                f"alfred at {self.config.url} could not be reached for {method} {path}: {exc}"
            ) from exc

    def _answer(self, method: str, path: str, response: httpx.Response) -> Any:
        if response.is_error:
            detail = _detail(response)
            message = f"alfred refused {method} {path} ({response.status_code}): {_text(detail)}"
            if response.status_code == 401:
                message += f"; {self._unauthorized()}"
            elif response.status_code == 404 and detail == "Not Found":
                message += self._who_answers()
            raise AlfredError(message, status=response.status_code, detail=detail)
        try:
            return response.json()
        except ValueError:
            kind = response.headers.get("content-type", "a body")
            raise AlfredError(
                f"alfred answered {method} {path} with {kind}, not JSON{self._who_answers()}",
                status=response.status_code,
            ) from None

    def _unauthorized(self) -> str:
        if self.config.can_log_in:
            return f"it refused {self.config.email} after a fresh login"
        if self.config.token:
            return (
                "it refused QMCP_ALFRED_TOKEN, and the tokens alfred issues expire; set "
                "QMCP_ALFRED_EMAIL and QMCP_ALFRED_PASSWORD so qmcp can log in again"
            )
        return (
            "it asks for an account: set QMCP_ALFRED_EMAIL and QMCP_ALFRED_PASSWORD, "
            "or QMCP_ALFRED_TOKEN"
        )

    def _who_answers(self) -> str:
        """What the server at the configured address calls itself, for an error
        that suggests it may not be alfred; empty when it does not say."""
        try:
            title = self._http.get("/openapi.json").json()["info"]["title"]
        except Exception:  # noqa: BLE001 -- a note on an error already being raised
            return ""
        return f" (the server at {self.config.url} calls itself {title!r})"


def _access_token(response: httpx.Response, path: str) -> str:
    try:
        token = response.json().get("access_token")
    except (ValueError, AttributeError):
        token = None
    if not isinstance(token, str) or not token:
        raise AlfredError(f"alfred answered POST {path} without an access_token")
    return token


def _detail(response: httpx.Response) -> Any:
    """alfred's `detail` for a refusal, or the body when it carries none."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:300] or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        return body["detail"]
    return body


def _text(detail: Any) -> str:
    """A `detail` as one line: a string as it is, a validation error list as
    `where: what` pairs joined by semicolons."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail and all(isinstance(e, dict) for e in detail):
        parts = []
        for error in detail:
            where = ".".join(str(step) for step in error.get("loc") or ())
            what = error.get("msg", json.dumps(error, default=str))
            parts.append(f"{where}: {what}" if where else str(what))
        return "; ".join(parts)
    return json.dumps(detail, default=str)


_current: AlfredClient | None = None
_current_lock = threading.Lock()


def client() -> AlfredClient:
    """The client for the alfred qmcp's settings name, kept while they name the same one.

    Raises `AlfredNotConfiguredError` when they name none.
    """
    global _current
    cfg = AlfredConfig.from_settings()
    with _current_lock:
        if _current is None or _current.config != cfg:
            if _current is not None:
                _current.close()
            _current = AlfredClient(cfg)
        return _current


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def _expect(value: Any, kind: type, method: str, path: str) -> Any:
    if not isinstance(value, kind):
        raise AlfredError(
            f"alfred answered {method} {path} with {type(value).__name__}, "
            f"not the {kind.__name__} alfred serves there"
        )
    return value


def _path_id(value: Any) -> str:
    """An id as one path segment: `/` and every other reserved character escaped."""
    return quote(str(value), safe="")


def _required(params: dict[str, Any], key: str) -> Any:
    value = params.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required")
    return value


def captions_view(project_id: Any, held: dict[str, Any]) -> dict[str, Any]:
    """A project's captions as alfred holds them (`GET /project/{id}/captions`)."""
    return {
        "project_id": str(project_id),
        "captions": held.get("captions"),
        "style": held.get("style"),
        "job": held.get("job"),
    }


def project_captions(
    project: dict[str, Any], wanted: bool | None
) -> tuple[list[Any], dict[str, Any]] | None:
    """The captions and caption style a render of `project` carries, or None.

    Unset (`wanted` None), the project decides: its captions go when its
    caption style is enabled. `True` sends them with the style switched on,
    whatever the project saved; `False` sends none. They go as alfred stores
    them: alfred turns them into clips, not qmcp.
    """
    if wanted is False:
        return None
    captions = project.get("captions")
    captions = captions if isinstance(captions, list) else []
    style = project.get("caption_style")
    style = style if isinstance(style, dict) else None
    if wanted is None:
        if not captions or style is None or style.get("enabled") is not True:
            return None
        return captions, style
    if not captions:
        raise AlfredError(
            f"alfred project {project.get('_id') or project.get('id')} "
            f"({project.get('name')!r}) has no captions to burn in; "
            "alfred_captions_transcribe makes them"
        )
    return captions, {**(style or {}), "enabled": True}


def project_summary(project: dict[str, Any]) -> dict[str, Any]:
    """One project as `alfred_projects` lists it."""
    edl = project.get("edl") or {}
    return {
        "id": project.get("_id") or project.get("id"),
        "name": project.get("name"),
        "clips": len(edl.get("clips") or []),
        "duration": edl.get("duration"),
    }


def link_expiry(link: str | None) -> str | None:
    """When a signed alfred link stops working, from its `exp`, as ISO 8601 UTC;
    None when it carries no `exp`."""
    if not link:
        return None
    try:
        seconds = int(parse_qs(urlsplit(link).query)["exp"][0])
        return datetime.fromtimestamp(seconds, UTC).isoformat()
    except (KeyError, IndexError, ValueError, OverflowError, OSError):
        return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool_registry.register(
    name="alfred_projects",
    description=(
        "List the alfred projects the configured account can see: each one's id, name, "
        "the number of clips on its timeline, and the duration saved with it (null when "
        "none was saved)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "skip": {
                "type": "integer",
                "minimum": 0,
                "description": "Projects to skip, for paging",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "The most projects to return; alfred's own default when unset",
            },
        },
    },
)
def alfred_projects(params: dict[str, Any]) -> list[dict[str, Any]]:
    query = {key: params[key] for key in ("skip", "limit") if params.get(key) is not None}
    projects = client().get("/project", params=query or None)
    return [project_summary(p) for p in _expect(projects, list, "GET", "/project")]


@tool_registry.register(
    name="alfred_templates",
    description=(
        "List the render templates alfred accepts for a template clip: each one's name, "
        "description, the parameters it takes, and whether it is deprecated."
    ),
    input_schema={"type": "object", "properties": {}},
)
def alfred_templates(params: dict[str, Any]) -> list[dict[str, Any]]:
    return _expect(client().get("/templates"), list, "GET", "/templates")


@tool_registry.register(
    name="alfred_render_enqueue",
    description=(
        "Queue a render in alfred, of a saved project's timeline sent exactly as it is "
        "stored (project_id) or of an explicit EDL (edl); give one of the two. A project "
        "whose caption style is enabled has its captions burned in. Returns the render's "
        "id, the settings it was queued with, and how many captions went with it; "
        "alfred_render_status follows it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {
                "type": "string",
                "description": "Render this project's saved timeline, unchanged",
            },
            "edl": {
                "type": "object",
                "description": 'Render this EDL instead: {"clips": [...], "duration": seconds}',
            },
            "name": {
                "type": "string",
                "description": "What the render is called; a project's name by default",
            },
            "description": {"type": "string", "description": "A note kept with the render"},
            "width": {
                "type": "integer",
                "description": "Frame width in pixels; a project's saved size, else alfred's",
            },
            "height": {
                "type": "integer",
                "description": "Frame height in pixels; a project's saved size, else alfred's",
            },
            "fps": {
                "type": "number",
                "description": "Frames per second; a project's saved rate, else alfred's",
            },
            "quality": {
                "type": "string",
                "description": "An encoding preset alfred offers: draft, standard or high",
            },
            "captions": {
                "type": "boolean",
                "description": (
                    "Burn a project's captions in: true sends them with its caption style "
                    "switched on, false sends none; unset sends them when the project's "
                    "caption style is enabled"
                ),
            },
        },
    },
)
def alfred_render_enqueue(params: dict[str, Any]) -> dict[str, Any]:
    project_id = params.get("project_id") or None
    edl = params.get("edl")
    if (project_id is None) == (edl is None):
        raise ValueError("give exactly one of project_id, to render a saved project, or edl")
    wanted = params.get("captions")
    if wanted is not None and not isinstance(wanted, bool):
        raise ValueError("captions is true, false, or left out")
    if wanted and project_id is None:
        raise ValueError("captions come from a saved project: give project_id to burn them in")

    alfred = client()
    burned: tuple[list[Any], dict[str, Any]] | None = None
    if project_id is not None:
        path = f"/project/{_path_id(project_id)}"
        project = _expect(alfred.get(path), dict, "GET", path)
        edl = project.get("edl")
        if not isinstance(edl, dict) or not edl.get("clips"):
            raise AlfredError(
                f"alfred project {project_id} ({project.get('name')!r}) has no clips on its "
                "timeline to render"
            )
        burned = project_captions(project, wanted)
        name = project.get("name")
        body: dict[str, Any] = {"project": name, "project_id": str(project_id), "name": name}
        saved = project.get("render_settings") or {}
    else:
        if not isinstance(edl, dict) or not isinstance(edl.get("clips"), list):
            raise ValueError('edl must be an object with a list of clips: {"clips": [...]}')
        body, saved = {}, {}

    # The EDL goes as alfred gave it or the caller sent it: a null is a value
    # (`fadeIn: null` is no fade, where an absent `fadeIn` is the engine's).
    body["edl"] = edl
    for key in ("name", "description"):
        if params.get(key):
            body[key] = params[key]
    for key in RENDER_OPTIONS:
        value = params[key] if params.get(key) is not None else saved.get(key)
        if value is not None:
            body[key] = value
    duration = edl.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        body["duration"] = duration
    if burned is not None:
        body["captions"], body["caption_style"] = burned

    answer = alfred.post("/render", body)
    render_id = answer.strip('"') if isinstance(answer, str) else ""
    if not render_id:
        raise AlfredError(f"alfred answered POST /render with {answer!r}, not a render id")
    return {
        "render_id": render_id,
        "project_id": body.get("project_id"),
        "name": body.get("name"),
        **{key: body.get(key) for key in RENDER_OPTIONS},
        "captions": len(burned[0]) if burned else 0,
    }


@tool_registry.register(
    name="alfred_render_status",
    description=(
        "Where an alfred render has got to: its state (queued, rendering, rendered or "
        "failed), progress, the error when it failed, and once it is rendered an absolute "
        "download link and when that link expires, when this alfred issues one."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "render_id": {
                "type": "string",
                "description": "The id alfred_render_enqueue returned",
            },
        },
        "required": ["render_id"],
    },
)
def alfred_render_status(params: dict[str, Any]) -> dict[str, Any]:
    render_id = params.get("render_id")
    if not render_id:
        raise ValueError("render_id is required")
    alfred = client()
    path = f"/render/{_path_id(render_id)}/status"
    status = _expect(alfred.get(path), dict, "GET", path)
    link = status.get("download_url")
    return {
        "render_id": status.get("id") or str(render_id),
        "state": status.get("state"),
        "progress": status.get("progress"),
        "error": status.get("error"),
        "download_url": alfred.url_for(link) if link else None,
        "download_expires_at": link_expiry(link),
    }


@tool_registry.register(
    name="alfred_render_deliver",
    description=(
        "Send a finished alfred render to a destination alfred offers, and say where it "
        "went: the destination, and the location as an absolute link. Unset, the "
        "destination is the one the render was last sent to, else alfred's local store."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "render_id": {
                "type": "string",
                "description": "The id alfred_render_enqueue returned",
            },
            "destination": {
                "type": "string",
                "description": "A destination alfred offers, such as local; alfred's choice "
                "when unset",
            },
        },
        "required": ["render_id"],
    },
)
def alfred_render_deliver(params: dict[str, Any]) -> dict[str, Any]:
    render_id = _required(params, "render_id")
    destination = params.get("destination") or None
    alfred = client()
    path = f"/render/{_path_id(render_id)}/deliver"
    query = {"destination": destination} if destination else None
    delivered = _expect(alfred.request("POST", path, params=query), dict, "POST", path)
    location = delivered.get("location")
    return {
        "render_id": str(render_id),
        "destination": delivered.get("destination"),
        "location": alfred.url_for(location) if location else None,
    }


_PROJECT_ID = {
    "type": "object",
    "properties": {
        "project_id": {"type": "string", "description": "The alfred project's id"},
    },
    "required": ["project_id"],
}


@tool_registry.register(
    name="alfred_captions",
    description=(
        "An alfred project's captions -- each one's start and end in timeline seconds and "
        "its text -- the caption style its renders use, with whether it is enabled, and its "
        "last transcription job: queued, transcribing, done or failed, and why it failed."
    ),
    input_schema=_PROJECT_ID,
)
def alfred_captions(params: dict[str, Any]) -> dict[str, Any]:
    project_id = _required(params, "project_id")
    path = f"/project/{_path_id(project_id)}/captions"
    return captions_view(project_id, _expect(client().get(path), dict, "GET", path))


@tool_registry.register(
    name="alfred_captions_transcribe",
    description=(
        "Start transcribing an alfred project's saved timeline with alfred's speech engine. "
        "alfred stores the captions on the project when it is done; alfred_captions follows "
        "the job. An error says so when alfred has no speech engine, or when a transcription "
        "of the project is already running."
    ),
    input_schema=_PROJECT_ID,
)
def alfred_captions_transcribe(params: dict[str, Any]) -> dict[str, Any]:
    project_id = _required(params, "project_id")
    path = f"/project/{_path_id(project_id)}/captions/transcribe"
    try:
        started = client().request("POST", path)
    except AlfredError as refused:
        opening = {503: NO_SPEECH_ENGINE, 409: TRANSCRIPTION_RUNNING}.get(refused.status)
        if opening is None:
            raise
        raise AlfredError(
            f"{opening}: {refused}", status=refused.status, detail=refused.detail
        ) from refused
    started = _expect(started, dict, "POST", path)
    return {"project_id": str(project_id), "job": started.get("job")}


@tool_registry.register(
    name="alfred_captions_set",
    description=(
        "Store an alfred project's captions, its caption style, or both: a person's "
        "correction of a transcript. alfred checks them -- times in order, text not empty -- "
        "and stores only these; one left out is sent back as alfred holds it. Returns the "
        "captions, style and job as alfred then holds them."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string", "description": "The alfred project's id"},
            "captions": {
                "type": "array",
                "description": "Every caption, in order: start and end in timeline seconds, "
                "and the text",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "text": {"type": "string"},
                    },
                    "required": ["start", "end", "text"],
                },
            },
            "style": {
                "type": "object",
                "description": "The caption style: enabled, and the text box's fontsize, "
                "color, bg, position, align and font",
            },
        },
        "required": ["project_id"],
    },
)
def alfred_captions_set(params: dict[str, Any]) -> dict[str, Any]:
    project_id = _required(params, "project_id")
    captions, style = params.get("captions"), params.get("style")
    if captions is None and style is None:
        raise ValueError("give captions, style, or both")
    if captions is not None and not isinstance(captions, list):
        raise ValueError("captions must be a list of {start, end, text}")
    if style is not None and not isinstance(style, dict):
        raise ValueError("style must be an object")

    alfred = client()
    path = f"/project/{_path_id(project_id)}/captions"
    if captions is None or style is None:
        held = _expect(alfred.get(path), dict, "GET", path)
        captions = held.get("captions") if captions is None else captions
        style = held.get("style") if style is None else style
    body = {key: value for key, value in (("captions", captions), ("style", style))
            if value is not None}
    stored = alfred.put(path, body)
    if not (isinstance(stored, dict) and isinstance(stored.get("captions"), list)
            and "style" in stored):
        stored = _expect(alfred.get(path), dict, "GET", path)
    return captions_view(project_id, stored)


@tool_registry.register(
    name="alfred_health",
    description=(
        "alfred's report on itself, as alfred answers it: whether its API and database "
        "answer, how many render workers are running, how many renders are waiting (null "
        "when alfred could not count them), and whether the audio engine it transcribes "
        "captions with is configured and answering, with why not. An alfred without the "
        "report answers this with an error."
    ),
    input_schema={"type": "object", "properties": {}},
)
def alfred_health(params: dict[str, Any]) -> dict[str, Any]:
    return _expect(client().get("/health"), dict, "GET", "/health")


__all__ = [
    "NO_SPEECH_ENGINE",
    "TRANSCRIPTION_RUNNING",
    "AlfredClient",
    "AlfredConfig",
    "AlfredError",
    "AlfredNotConfiguredError",
    "captions_view",
    "client",
    "link_expiry",
    "project_captions",
    "project_summary",
]
