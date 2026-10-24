"""The alfred tools reach alfred the way they do in use: from settings, over a socket.

The other alfred tests hand the tools a client over `httpx.MockTransport`, so
the settings that name alfred, the login and the client the tools build for
themselves are passed over. Here the stand-in in `tests/alfred_stub.py` is
served on a real socket (`qmcp.testing.loopback.serve`), the server is started
with `QMCP_ALFRED_*` naming it, and a tool is called over HTTP as a flow calls
it.
"""

from __future__ import annotations

import pytest

from qmcp.client import MCPClient
from qmcp.testing import throwaway_server
from qmcp.testing.loopback import serve
from tests.alfred_stub import EMAIL, PASSWORD, FakeAlfred

pytestmark = pytest.mark.integration


@pytest.fixture
def alfred(monkeypatch):
    fake = FakeAlfred()
    with serve(fake.handle) as url:
        monkeypatch.setenv("QMCP_ALFRED_URL", url)
        monkeypatch.setenv("QMCP_ALFRED_EMAIL", EMAIL)
        monkeypatch.setenv("QMCP_ALFRED_PASSWORD", PASSWORD)
        yield fake


def test_a_tool_logs_in_from_the_settings_and_answers_from_alfred(alfred, tmp_path) -> None:
    # Mutation seen red: `QMCP_ALFRED_PASSWORD` set to another password -- the
    # stand-in refused the login and the tool answered with alfred's refusal.
    alfred.route("GET", "/templates", [{"name": "title"}, {"name": "lower-third"}])

    with throwaway_server(tmp_path / "queue.db") as url, MCPClient(base_url=url) as client:
        called = client.invoke_tool("alfred_templates", {})
        again = client.invoke_tool("alfred_templates", {})

    assert called.error is None, called.error
    assert called.result == again.result == [{"name": "title"}, {"name": "lower-third"}]
    assert len(alfred.calls("POST", "/auth/jwt/login")) == 1
    assert len(alfred.calls("GET", "/templates")) == 2


def test_alfreds_refusal_reaches_the_caller_naming_the_route_and_the_server(
        alfred, tmp_path) -> None:
    # Nothing routes `/templates`, so the stand-in answers as FastAPI does.
    with throwaway_server(tmp_path / "queue.db") as url, MCPClient(base_url=url) as client:
        called = client.invoke_tool("alfred_templates", {})

    assert called.result is None
    assert "alfred refused GET /templates (404): Not Found" in called.error
    assert "calls itself 'Alfred API'" in called.error
