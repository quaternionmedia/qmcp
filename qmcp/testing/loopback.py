"""A stand-in on a real socket: any httpx handler, served on 127.0.0.1, port 0.

The suite's stand-ins -- `tests/alfred_stub.py`'s `FakeAlfred`, the local model
service in `tests/test_agents_local.py` -- are handlers from an
`httpx.Request` to an `httpx.Response`, written for `httpx.MockTransport`. A
transport reaches only the client it is handed. `serve(handler)` puts the same
handler behind a socket, so code that builds its own client from a configured
URL -- the alfred tools, the local runtime, a qmcp server in a process of its
own -- reaches the stand-in as it would reach the service.

What this cannot do: stream. A handler's response is read whole and sent with
a `Content-Length`, which is what every stand-in here answers with.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

Handler = Callable[[httpx.Request], httpx.Response]

# Framing the bridge sets itself, from the body it actually sends.
_FRAMING = {"content-length", "transfer-encoding", "connection"}


@contextlib.contextmanager
def serve(handler: Handler) -> Iterator[str]:
    """Serve `handler` on 127.0.0.1, port 0, in a thread. Yields its base URL.

    A handler that raises answers 500 with the exception named in the body, so
    a stand-in's own assertion reaches the test as a failure it can read.
    """

    class _Bridge(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _answer(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            port = self.server.server_address[1]
            request = httpx.Request(
                self.command, f"http://127.0.0.1:{port}{self.path}",
                headers=list(self.headers.items()), content=body)
            try:
                response = handler(request)
                content = response.read()
            except Exception as exc:  # noqa: BLE001 -- reported to the caller, not swallowed
                response = httpx.Response(500)
                content = f"the stand-in raised {type(exc).__name__}: {exc}".encode()
            self.send_response(response.status_code)
            for name, value in response.headers.multi_items():
                if name.lower() not in _FRAMING:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        # The names `http.server` dispatches to.
        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _answer  # noqa: N815

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Quiet: a check's output is its verdicts."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Bridge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
