"""A handler written for `httpx.MockTransport` answers the same on a real socket.

`qmcp.testing.loopback.serve` is what puts the suite's stand-ins where code that
builds its own client can reach them. These show a request arrives whole and an
answer leaves whole, and that a stand-in that raises is heard as a failure.
"""

from __future__ import annotations

import httpx

from qmcp.testing.loopback import serve


def test_a_request_arrives_whole_and_the_answer_leaves_whole() -> None:
    # Mutation seen red: the request body read dropped (`body = b""`) -- the
    # handler saw an empty body and the echo came back without it.
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"echo": request.read().decode()},
                              headers={"x-stand-in": "yes"})

    with serve(handler) as url:
        answer = httpx.post(f"{url}/v1/thing?x=1", content=b"hello",
                            headers={"x-caller": "test"})

    assert answer.status_code == 201
    assert answer.json() == {"echo": "hello"}
    assert answer.headers["x-stand-in"] == "yes"
    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/v1/thing" and request.url.params["x"] == "1"
    assert request.headers["x-caller"] == "test"


def test_a_stand_in_that_raises_answers_500_and_names_the_exception() -> None:
    # Mutation seen red: the `except` branch removed -- the bridge's thread
    # died mid-request and the client got no answer at all.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("asked for a route the script does not hold")

    with serve(handler) as url:
        answer = httpx.get(f"{url}/anything")

    assert answer.status_code == 500
    assert "AssertionError: asked for a route the script does not hold" in answer.text


def test_every_method_the_stand_ins_answer_is_bridged() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(204)

    with serve(handler) as url, httpx.Client(base_url=url) as client:
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            assert client.request(method, "/x").status_code == 204

    assert methods == ["GET", "POST", "PUT", "PATCH", "DELETE"]
