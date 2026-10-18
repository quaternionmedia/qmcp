"""The alfred tools: configuration, the token, each tool's request and answer,
and how a refusal reaches the caller.

alfred is a stand-in served through `httpx.MockTransport`: one account, tokens
shaped as alfred's JWTs are, and canned answers per route. Nothing here needs a
running alfred.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

import qmcp.config
import qmcp.tools.alfred as alfred
from qmcp.tools.alfred import (
    RENEW_MARGIN,
    AlfredClient,
    AlfredConfig,
    AlfredError,
    AlfredNotConfiguredError,
)

URL = "http://alfred.test"
EMAIL, PASSWORD = "editor@example.com", "correct horse"


def jwt(lifetime: float = 3600, *, iat: float | None = None, n: int = 0) -> str:
    """A token shaped as alfred's: `iat` and `exp` in an unsigned payload."""
    iat = time.time() if iat is None else iat

    def part(data: dict) -> str:
        raw = json.dumps(data).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    claims = {"sub": "user", "aud": ["fastapi-users:auth"], "iat": int(iat),
              "exp": int(iat + lifetime), "n": n}
    return f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part(claims)}.signature"


Route = Callable[[httpx.Request], httpx.Response]


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
            return httpx.Response(404, json={"detail": "Not Found"})
        return answer(request)


@pytest.fixture
def fake() -> FakeAlfred:
    return FakeAlfred()


def connect(fake: FakeAlfred, **config) -> AlfredClient:
    """A client for the stand-in, logging in as its account unless told otherwise."""
    config = {"email": EMAIL, "password": PASSWORD, **config}
    return AlfredClient(AlfredConfig(url=URL, **config), transport=fake.transport())


@pytest.fixture
def tools(fake, monkeypatch) -> FakeAlfred:
    """The tools, pointed at the stand-in through `qmcp.tools.alfred.client`."""
    connected = connect(fake)
    monkeypatch.setattr(alfred, "client", lambda: connected)
    return fake


# --- configuration -----------------------------------------------------------


def settings(**fields) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def test_no_url_is_not_configured_and_says_which_setting():
    """Mutation: fall back to a default address when none is set -- red."""
    with pytest.raises(AlfredNotConfiguredError, match="QMCP_ALFRED_URL"):
        AlfredConfig.from_settings(settings(alfred_url=None))


def test_settings_without_the_alfred_fields_are_an_unconfigured_alfred():
    with pytest.raises(AlfredNotConfiguredError):
        AlfredConfig.from_settings(settings(database_url="sqlite://"))


@pytest.mark.parametrize("given,missing", [
    ({"alfred_email": EMAIL}, "QMCP_ALFRED_PASSWORD"),
    ({"alfred_password": PASSWORD}, "QMCP_ALFRED_EMAIL"),
])
def test_half_an_account_is_refused_by_name(given, missing):
    """Mutation: accept an email without a password -- red, both ways round."""
    with pytest.raises(AlfredNotConfiguredError, match=missing):
        AlfredConfig.from_settings(settings(alfred_url=URL, **given))


def test_the_settings_are_read_from_qmcp_environment_names(monkeypatch):
    monkeypatch.setenv("QMCP_ALFRED_URL", URL + "/")
    monkeypatch.setenv("QMCP_ALFRED_EMAIL", EMAIL)
    monkeypatch.setenv("QMCP_ALFRED_PASSWORD", PASSWORD)
    monkeypatch.setenv("QMCP_ALFRED_TOKEN", "a.token.here")
    monkeypatch.setenv("QMCP_ALFRED_TIMEOUT", "4.5")
    read = AlfredConfig.from_settings(qmcp.config.Settings(_env_file=None))
    assert (read.url, read.email, read.password, read.token, read.timeout) == (
        URL, EMAIL, PASSWORD, "a.token.here", 4.5)


def test_the_secrets_stay_out_of_every_repr(monkeypatch):
    monkeypatch.setenv("QMCP_ALFRED_PASSWORD", PASSWORD)
    monkeypatch.setenv("QMCP_ALFRED_TOKEN", "a.token.here")
    loaded = qmcp.config.Settings(_env_file=None)
    read = AlfredConfig(url=URL, email=EMAIL, password=PASSWORD, token="a.token.here")
    for shown in (repr(loaded), repr(read)):
        assert PASSWORD not in shown and "a.token.here" not in shown


def test_the_client_is_kept_while_the_settings_name_the_same_alfred(monkeypatch):
    """Mutation: build a client on every call -- red."""
    current = settings(alfred_url=URL, alfred_email=EMAIL, alfred_password=PASSWORD)
    monkeypatch.setattr(qmcp.config, "get_settings", lambda: current)
    monkeypatch.setattr(alfred, "_current", None)
    first = alfred.client()
    assert alfred.client() is first
    current.alfred_url = "http://elsewhere.test"
    second = alfred.client()
    assert second is not first and second.config.url == "http://elsewhere.test"
    assert first._http.is_closed


def test_a_tool_says_alfred_is_not_configured(monkeypatch):
    monkeypatch.setattr(qmcp.config, "get_settings", lambda: settings(alfred_url=""))
    monkeypatch.setattr(alfred, "_current", None)
    with pytest.raises(AlfredNotConfiguredError, match="not configured"):
        alfred.alfred_projects({})


# --- the token ---------------------------------------------------------------


def test_logs_in_on_first_use_and_reuses_the_token(fake):
    """Mutation: log in before every request -- red."""
    fake.route("GET", "/templates", [])
    fake.route("GET", "/project", [])
    client = connect(fake)
    assert fake.requests == []
    client.get("/templates")
    client.get("/project")
    client.get("/templates")
    assert len(fake.calls("POST", "/auth/jwt/login")) == 1
    sent = {r.headers["authorization"] for r in fake.requests if r.url.path != "/auth/jwt/login"}
    assert sent == {f"Bearer {next(iter(fake.valid))}"}


def test_the_login_is_a_form_with_username_and_password(fake):
    fake.route("GET", "/templates", [])
    connect(fake).get("/templates")
    login = fake.calls("POST", "/auth/jwt/login")[0]
    assert login.headers["content-type"] == "application/x-www-form-urlencoded"
    assert parse_qs(login.content.decode()) == {"username": [EMAIL], "password": [PASSWORD]}


def test_a_refused_token_is_replaced_by_one_login_and_the_request_sent_again(fake):
    """Mutation: drop the retry after a 401 -- red."""
    fake.route("GET", "/templates", [{"name": "title"}])
    client = connect(fake)
    client.get("/templates")
    first = next(iter(fake.valid))
    fake.valid.clear()  # alfred no longer accepts it

    assert client.get("/templates") == [{"name": "title"}]
    assert len(fake.calls("POST", "/auth/jwt/login")) == 2
    refused, retried = fake.calls("GET", "/templates")[1:]
    assert refused.headers["authorization"] == f"Bearer {first}"
    assert retried.headers["authorization"] == f"Bearer {next(iter(fake.valid))}"


def test_a_401_after_a_fresh_login_is_an_error_not_a_loop(fake):
    """Mutation: log in and send once more after the second 401 -- red."""
    fake.route("GET", "/project", lambda r: httpx.Response(401, json={"detail": "Unauthorized"}))
    with pytest.raises(AlfredError, match="after a fresh login") as refused:
        connect(fake).get("/project")
    assert refused.value.status == 401
    assert len(fake.calls("POST", "/auth/jwt/login")) == 2
    assert len(fake.calls("GET", "/project")) == 2


def test_wrong_credentials_surface_alfred_s_detail(fake):
    client = connect(fake, password="wrong")
    with pytest.raises(AlfredError, match=r"refused the login for editor@example.com "
                                          r"\(400\): LOGIN_BAD_CREDENTIALS"):
        client.get("/project")


def test_a_configured_token_is_sent_and_no_login_happens(fake):
    token = jwt()
    fake.valid.add(token)
    fake.route("GET", "/project", [])
    connect(fake, email=None, password=None, token=token).get("/project")
    assert fake.calls("POST", "/auth/jwt/login") == []
    assert fake.calls("GET", "/project")[0].headers["authorization"] == f"Bearer {token}"


def test_a_refused_configured_token_says_how_to_let_qmcp_log_in(fake):
    with pytest.raises(AlfredError, match="QMCP_ALFRED_TOKEN.*QMCP_ALFRED_EMAIL"):
        connect(fake, email=None, password=None, token=jwt()).get("/project")
    assert fake.calls("POST", "/auth/jwt/login") == []


def test_a_refused_configured_token_with_an_account_falls_back_to_logging_in(fake):
    fake.route("GET", "/project", [])
    connect(fake, token=jwt()).get("/project")
    assert len(fake.calls("POST", "/auth/jwt/login")) == 1
    assert len(fake.calls("GET", "/project")) == 2


def test_a_configured_token_about_to_lapse_is_renewed_before_it_is_sent(fake):
    """Mutation: never renew -- red."""
    lapsing = jwt(RENEW_MARGIN / 2, iat=time.time())
    fake.valid.add(lapsing)
    fake.route("GET", "/project", [])
    connect(fake, email=None, password=None, token=lapsing).get("/project")
    refresh = fake.calls("POST", "/auth/jwt/refresh")
    assert [r.headers["authorization"] for r in refresh] == [f"Bearer {lapsing}"]
    assert fake.calls("GET", "/project")[0].headers["authorization"] != f"Bearer {lapsing}"


def test_an_issued_token_is_renewed_by_alfred_s_lifetime_on_this_machine_s_clock(fake):
    """Mutation: read an issued token's `exp` against this machine's clock -- red,
    with alfred's clock behind it."""
    now = [1000.0]
    fake.skew = -1000.0  # a lifetime read against this machine's clock looks shorter
    fake.route("GET", "/project", [])
    client = AlfredClient(AlfredConfig(url=URL, email=EMAIL, password=PASSWORD),
                          transport=fake.transport(), clock=lambda: now[0])
    client.get("/project")
    now[0] += fake.lifetime - RENEW_MARGIN - 1
    client.get("/project")
    assert fake.calls("POST", "/auth/jwt/refresh") == []
    now[0] += 2
    client.get("/project")
    assert len(fake.calls("POST", "/auth/jwt/refresh")) == 1
    assert len(fake.calls("POST", "/auth/jwt/login")) == 1


def test_a_token_that_cannot_be_renewed_is_sent_as_it_is_and_renewal_not_retried(fake):
    """Mutation: try to renew again on every request after a failed renewal -- red."""
    lapsing = jwt(RENEW_MARGIN / 2)
    fake.valid.add(lapsing)
    fake.refresh_works = False
    fake.route("GET", "/project", [])
    client = connect(fake, email=None, password=None, token=lapsing)
    client.get("/project")
    client.get("/project")
    assert len(fake.calls("POST", "/auth/jwt/refresh")) == 1
    assert {r.headers["authorization"] for r in fake.calls("GET", "/project")} == {
        f"Bearer {lapsing}"}


def test_a_token_that_cannot_be_renewed_is_replaced_by_a_login_when_there_is_an_account(fake):
    fake.refresh_works = False
    lapsing = jwt(RENEW_MARGIN / 2)
    fake.valid.add(lapsing)
    fake.route("GET", "/project", [])
    connect(fake, token=lapsing).get("/project")
    assert len(fake.calls("POST", "/auth/jwt/refresh")) == 1
    assert len(fake.calls("POST", "/auth/jwt/login")) == 1
    assert fake.calls("GET", "/project")[0].headers["authorization"] != f"Bearer {lapsing}"


def test_without_an_account_requests_carry_no_token(fake):
    fake.open = True
    fake.route("GET", "/templates", [])
    connect(fake, email=None, password=None).get("/templates")
    assert "authorization" not in fake.calls("GET", "/templates")[0].headers


def test_without_an_account_a_401_says_which_settings_to_set(fake):
    with pytest.raises(AlfredError, match="asks for an account: set QMCP_ALFRED_EMAIL"):
        connect(fake, email=None, password=None).get("/templates")


# --- what a refusal says -----------------------------------------------------


def test_a_string_detail_is_surfaced_with_the_status(fake):
    """Mutation: leave alfred's detail out of the message -- red."""
    fake.route("POST", "/render", lambda r: httpx.Response(
        422, json={"detail": "quality must be one of draft, standard, high, not 'ultra'"}))
    with pytest.raises(AlfredError) as refused:
        connect(fake).post("/render", {"edl": {"clips": []}})
    assert str(refused.value) == ("alfred refused POST /render (422): quality must be one of "
                                  "draft, standard, high, not 'ultra'")
    assert refused.value.status == 422


def test_a_validation_list_is_surfaced_as_where_and_what(fake):
    """Mutation: show the list as JSON -- red."""
    fake.route("POST", "/render", lambda r: httpx.Response(422, json={"detail": [
        {"loc": ["body", "edl"], "msg": "field required", "type": "value_error.missing"},
        {"loc": ["body", "width"], "msg": "value is not a valid integer"},
    ]}))
    with pytest.raises(AlfredError, match="body.edl: field required; "
                                          "body.width: value is not a valid integer"):
        connect(fake).post("/render", {})


def test_alfred_s_own_not_found_is_surfaced_without_asking_who_answers(fake):
    fake.route("GET", "/render/abc/status",
               lambda r: httpx.Response(404, json={"detail": "Item not found"}))
    with pytest.raises(AlfredError, match=r"\(404\): Item not found$"):
        connect(fake).get("/render/abc/status")
    assert fake.calls("GET", "/openapi.json") == []


def test_a_missing_route_names_the_server_that_answered(fake):
    """Mutation: leave out what the server calls itself -- red."""
    def someone_else(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"title": "Apothecary API"}})
        return httpx.Response(404, json={"detail": "Not Found"})

    client = AlfredClient(AlfredConfig(url=URL), transport=httpx.MockTransport(someone_else))
    with pytest.raises(AlfredError, match="calls itself 'Apothecary API'"):
        client.get("/project")


def test_an_answer_that_is_not_json_is_an_error(fake):
    fake.route("GET", "/project", lambda r: httpx.Response(
        200, text="<!doctype html>", headers={"content-type": "text/html"}))
    with pytest.raises(AlfredError, match="with text/html, not JSON"):
        connect(fake).get("/project")


def test_an_alfred_that_does_not_answer_is_named(fake):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = AlfredClient(AlfredConfig(url=URL), transport=httpx.MockTransport(down))
    with pytest.raises(AlfredError, match="alfred at http://alfred.test could not be reached "
                                          "for GET /project: connection refused"):
        client.get("/project")


def test_an_alfred_that_answers_too_slowly_is_named(fake):
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = AlfredClient(AlfredConfig(url=URL, timeout=2), transport=httpx.MockTransport(slow))
    with pytest.raises(AlfredError, match=r"did not answer GET /project in time \(ReadTimeout"):
        client.get("/project")


def test_the_timeouts_are_short_and_connecting_shorter():
    """Mutation: let connecting take the whole timeout -- red."""
    timeout = AlfredClient(AlfredConfig(url=URL, timeout=10))._http.timeout
    assert (timeout.read, timeout.connect) == (10, alfred.CONNECT_TIMEOUT)
    assert AlfredClient(AlfredConfig(url=URL, timeout=1))._http.timeout.connect == 1


# --- the tools ---------------------------------------------------------------


PROJECTS = [
    {"_id": "6a770760de7add77551a53df", "name": "Magnussens", "logic": {}, "edl": None},
    {"_id": "6a7707680d24a1f8f772191f", "name": "Demo",
     "edl": {"clips": [{"name": "title"}, {"name": "x"}, {"name": "y"}], "duration": 9.0}},
    {"_id": "6a78af6d0eea42d905943612", "name": "save test",
     "edl": {"clips": [{"name": "videos/a.mp4"}], "duration": None}},
]


def test_projects_are_listed_by_id_name_clips_and_duration(tools):
    tools.route("GET", "/project", PROJECTS)
    assert alfred.alfred_projects({}) == [
        {"id": "6a770760de7add77551a53df", "name": "Magnussens", "clips": 0, "duration": None},
        {"id": "6a7707680d24a1f8f772191f", "name": "Demo", "clips": 3, "duration": 9.0},
        {"id": "6a78af6d0eea42d905943612", "name": "save test", "clips": 1, "duration": None},
    ]
    assert tools.calls("GET", "/project")[0].url.query == b""


def test_projects_page_through_skip_and_limit(tools):
    tools.route("GET", "/project", [])
    alfred.alfred_projects({"skip": 100, "limit": 50})
    assert tools.calls("GET", "/project")[0].url.params == httpx.QueryParams(skip=100, limit=50)


def test_templates_are_alfred_s_list_as_it_answers(tools):
    templates = [{"name": "title", "description": "Render a large Title block of text",
                  "parameters": ["text", "color"], "deprecated": False}]
    tools.route("GET", "/templates", templates)
    assert alfred.alfred_templates({}) == templates


SAVED_EDL = {
    "clips": [
        {"duration": 3.0, "type": "template", "name": "title", "start": 0.0, "fadeIn": 0,
         "fadeOut": None, "inpoint": None, "data": {"text": "Alfred", "method": "label"}},
        {"duration": 2.0, "type": "video", "name": "videos/demo-bars.mp4", "start": 3.0,
         "fadeIn": 1.0, "fadeOut": 0},
    ],
    "duration": 5.0,
}


def render_body(fake: FakeAlfred) -> dict:
    return json.loads(fake.calls("POST", "/render")[0].content)


def test_a_project_renders_its_saved_edl_exactly_as_stored(tools):
    """Mutation: drop the nulls from the clips, ignore the saved render settings,
    or keep the quotes of a double-encoded id -- each red."""
    tools.route("GET", "/project/6a78b0420eea42d905943615", {
        "_id": "6a78b0420eea42d905943615", "name": "Timeline demo", "edl": SAVED_EDL,
        "render_settings": {"width": 1280, "height": 720, "fps": 30.0, "quality": None}})
    tools.route("POST", "/render", '"6a7900000000000000000001"')

    queued = alfred.alfred_render_enqueue({"project_id": "6a78b0420eea42d905943615"})

    body = render_body(tools)
    assert body["edl"] == SAVED_EDL  # nulls and zeros included, nothing added
    assert body == {"edl": SAVED_EDL, "project": "Timeline demo",
                    "project_id": "6a78b0420eea42d905943615", "name": "Timeline demo",
                    "width": 1280, "height": 720, "fps": 30.0, "duration": 5.0}
    assert queued == {"render_id": "6a7900000000000000000001",
                      "project_id": "6a78b0420eea42d905943615", "name": "Timeline demo",
                      "width": 1280, "height": 720, "fps": 30.0, "quality": None}


def test_what_the_caller_says_overrides_what_the_project_saved(tools):
    tools.route("GET", "/project/p1", {"_id": "p1", "name": "Demo", "edl": SAVED_EDL,
                                       "render_settings": {"width": 1280, "height": 720}})
    tools.route("POST", "/render", "r1")
    alfred.alfred_render_enqueue({"project_id": "p1", "width": 640, "height": 360,
                                  "quality": "draft", "name": "proof", "description": "v2"})
    body = render_body(tools)
    assert (body["width"], body["height"], body["quality"]) == (640, 360, "draft")
    assert (body["name"], body["project"], body["description"]) == ("proof", "Demo", "v2")


def test_a_project_without_a_timeline_is_refused_before_anything_is_queued(tools):
    tools.route("GET", "/project/p2", {"_id": "p2", "name": "Magnussens", "edl": None})
    with pytest.raises(AlfredError, match="p2 \\('Magnussens'\\) has no clips"):
        alfred.alfred_render_enqueue({"project_id": "p2"})
    assert tools.calls("POST", "/render") == []


def test_a_project_id_cannot_leave_its_path_segment(tools):
    """Mutation: put the id into the path unescaped -- red."""
    with pytest.raises(AlfredError):
        alfred.alfred_render_enqueue({"project_id": "../users/me"})
    asked = [r.url.raw_path for r in tools.requests if r.url.path.startswith("/project")]
    assert asked == [b"/project/..%2Fusers%2Fme"]


def test_an_explicit_edl_is_sent_with_only_what_was_given(tools):
    tools.route("POST", "/render", "r2")
    edl = {"clips": [{"type": "template", "name": "makeColor", "duration": 2}]}
    queued = alfred.alfred_render_enqueue({"edl": edl, "quality": "high"})
    assert render_body(tools) == {"edl": edl, "quality": "high"}
    assert queued["render_id"] == "r2" and queued["project_id"] is None
    assert not any(r.url.path.startswith("/project") for r in tools.requests)


@pytest.mark.parametrize("params", [{}, {"project_id": "p1", "edl": {"clips": []}}])
def test_enqueue_takes_exactly_one_of_a_project_or_an_edl(tools, params):
    with pytest.raises(ValueError, match="exactly one of project_id"):
        alfred.alfred_render_enqueue(params)
    assert tools.requests == []


def test_an_edl_without_a_clip_list_is_refused_here(tools):
    with pytest.raises(ValueError, match="list of clips"):
        alfred.alfred_render_enqueue({"edl": {"clips": "title"}})


def test_a_refused_render_carries_alfred_s_detail(tools):
    tools.route("POST", "/render", lambda r: httpx.Response(
        422, json={"detail": "clip 1 ('videos/x.mp4'): not in the media library"}))
    with pytest.raises(AlfredError, match="clip 1 \\('videos/x.mp4'\\): not in the media"):
        alfred.alfred_render_enqueue({"edl": {"clips": [{"name": "videos/x.mp4"}]}})


def test_an_answer_that_is_not_a_render_id_is_an_error(tools):
    tools.route("POST", "/render", {"id": "r3"})
    with pytest.raises(AlfredError, match="not a render id"):
        alfred.alfred_render_enqueue({"edl": {"clips": [{}]}})


STATUS = {"id": "r4", "state": "rendering", "status": None, "progress": 41.5,
          "filename": "r4.mp4", "destination": None, "error": None}


def test_status_maps_state_progress_and_error(tools):
    tools.route("GET", "/render/r4/status", STATUS)
    assert alfred.alfred_render_status({"render_id": "r4"}) == {
        "render_id": "r4", "state": "rendering", "progress": 41.5, "error": None,
        "download_url": None, "download_expires_at": None}


def test_a_failed_render_says_why(tools):
    tools.route("GET", "/render/r5/status", {**STATUS, "id": "r5", "state": "failed",
                                             "status": "failed", "error": "clip 2: no such file"})
    status = alfred.alfred_render_status({"render_id": "r5"})
    assert (status["state"], status["error"]) == ("failed", "clip 2: no such file")


def test_a_signed_download_path_is_made_absolute_with_its_expiry(tools):
    """Mutation: return alfred's path as it is -- red."""
    tools.route("GET", "/render/r6/status", {
        **STATUS, "id": "r6", "state": "rendered", "progress": 100.0,
        "download_url": "/render/r6/download?exp=1792000000&sig=abc_-12"})
    status = alfred.alfred_render_status({"render_id": "r6"})
    assert status["download_url"] == f"{URL}/render/r6/download?exp=1792000000&sig=abc_-12"
    assert status["download_expires_at"] == "2026-10-14T17:46:40+00:00"


def test_an_absolute_download_url_is_returned_as_it_is(tools):
    link = "https://media.example/render/r7/download?exp=1792000000&sig=x"
    tools.route("GET", "/render/r7/status", {**STATUS, "state": "rendered",
                                             "download_url": link})
    assert alfred.alfred_render_status({"render_id": "r7"})["download_url"] == link


def test_an_alfred_behind_a_path_keeps_the_path_in_requests_and_links(fake, monkeypatch):
    """Mutation: join the link to the host alone -- red."""
    fake.prefix = "/alfred"
    connected = AlfredClient(AlfredConfig(url="http://host.test/alfred", email=EMAIL,
                                          password=PASSWORD), transport=fake.transport())
    monkeypatch.setattr(alfred, "client", lambda: connected)
    fake.route("GET", "/render/r8/status", {
        **STATUS, "state": "rendered", "download_url": "/render/r8/download?exp=1&sig=s"})
    status = alfred.alfred_render_status({"render_id": "r8"})
    assert [r.url.path for r in fake.requests] == ["/alfred/auth/jwt/login",
                                                   "/alfred/render/r8/status"]
    assert status["download_url"] == "http://host.test/alfred/render/r8/download?exp=1&sig=s"


def test_status_needs_a_render_id(tools):
    with pytest.raises(ValueError, match="render_id is required"):
        alfred.alfred_render_status({})


def test_health_is_alfred_s_report_as_it_answers(tools):
    report = {"api": "ok", "db": "ok", "renderer": {"ok": False, "workers": 0},
              "queue": {"pending": 2}}
    tools.route("GET", "/health", report)
    assert alfred.alfred_health({}) == report


def test_an_alfred_without_a_health_report_is_an_error_naming_the_server(tools):
    with pytest.raises(AlfredError, match=r"GET /health \(404\): Not Found "
                                          r"\(the server at http://alfred.test calls itself "
                                          r"'Alfred API'\)"):
        alfred.alfred_health({})


@pytest.mark.parametrize("value,expected", [
    ("/render/r/download?exp=0&sig=s", "1970-01-01T00:00:00+00:00"),
    ("/render/r/download?sig=s", None),
    ("/render/r/download?exp=soon&sig=s", None),
    (None, None),
])
def test_link_expiry_reads_exp_or_says_nothing(value, expected):
    assert alfred.link_expiry(value) == expected


# --- through the server --------------------------------------------------------


def test_the_server_lists_the_alfred_tools(client):
    names = {tool["name"] for tool in client.get("/v1/tools").json()["tools"]}
    assert {"alfred_projects", "alfred_templates", "alfred_render_enqueue",
            "alfred_render_status", "alfred_health"} <= names


def test_an_unconfigured_alfred_is_the_invocation_s_error(client, monkeypatch):
    monkeypatch.setattr(alfred, "_current", None)
    answer = client.post("/v1/tools/alfred_projects", json={"input": {}}).json()
    assert answer["result"] is None
    assert answer["error"].startswith("Tool execution failed: alfred is not configured")


def test_a_refusal_reaches_the_invocation_record_with_alfred_s_detail(client, tools):
    tools.route("POST", "/render", lambda r: httpx.Response(
        422, json={"detail": "width must be between 16 and 7680 pixels"}))
    answer = client.post("/v1/tools/alfred_render_enqueue", json={
        "input": {"edl": {"clips": [{"name": "title"}]}, "width": 9000}}).json()
    assert answer["error"] == ("Tool execution failed: alfred refused POST /render (422): "
                               "width must be between 16 and 7680 pixels")
    recorded = client.get(f"/v1/invocations/{answer['invocation_id']}").json()
    assert recorded["status"] == "failed"
    assert "width must be between 16 and 7680 pixels" in recorded["error"]


def test_a_render_queued_through_the_server_returns_its_id(client, tools):
    tools.route("POST", "/render", '"r9"')
    answer = client.post("/v1/tools/alfred_render_enqueue", json={
        "input": {"edl": {"clips": [{"name": "title"}]}}}).json()
    assert answer["error"] is None
    assert answer["result"]["render_id"] == "r9"
