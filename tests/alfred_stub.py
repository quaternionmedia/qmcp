"""A stand-in alfred for the alfred tools' tests, served through `httpx.MockTransport`.

One account, tokens shaped as alfred's JWTs are, and canned answers per route,
so nothing needs a running alfred. Not collected: the name does not start
with `test_`.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
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
            return httpx.Response(404, json={"detail": "Not Found"})
        return answer(request)


def connect(fake: FakeAlfred, **config) -> AlfredClient:
    """A client for the stand-in, logging in as its account unless told otherwise."""
    config = {"email": EMAIL, "password": PASSWORD, **config}
    return AlfredClient(AlfredConfig(url=URL, **config), transport=fake.transport())
