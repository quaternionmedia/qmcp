"""The client library and the server, over a real socket, against a database of their own.

Everything else that tests the routes goes through `TestClient`, in process and
without a socket; everything else that tests `MCPClient` hands it a transport.
Here the client is the one a flow constructs from a URL, and the server is
uvicorn on port 0 (`qmcp.testing.throwaway_server`), so what is shown is what a
flow on this machine gets: a tool call recorded and readable afterwards, and a
human request asked, refused when answered off its options, and answered.
"""

from __future__ import annotations

import httpx
import pytest

from qmcp.client import MCPClient, ToolNotFoundError
from qmcp.testing import throwaway_server

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path):
    with throwaway_server(tmp_path / "queue.db") as url, MCPClient(base_url=url) as connected:
        yield connected


def test_a_tool_call_is_answered_and_recorded_with_its_correlation_id(client) -> None:
    # Mutation seen red: `correlation_id` left out of the payload `invoke_tool`
    # sends -- the recorded invocation carried none.
    assert "echo" in {tool.name for tool in client.list_tools()}

    called = client.invoke_tool("echo", {"message": "over the wire"}, correlation_id="flow-7")

    assert called.error is None
    assert called.result == "over the wire"
    recorded = client.get_invocation(called.invocation_id)
    assert recorded["tool_name"] == "echo"
    assert recorded["status"] == "success"
    assert recorded["correlation_id"] == "flow-7"
    assert [row["id"] for row in client.list_invocations(tool_name="echo")] == [
        called.invocation_id]


def test_a_tool_the_server_does_not_hold_is_named_as_not_found(client) -> None:
    with pytest.raises(ToolNotFoundError, match="nonexistent"):
        client.invoke_tool("nonexistent", {})


def test_a_human_request_is_asked_refused_off_its_options_and_answered(client) -> None:
    client.create_human_request(
        "deploy-7", "approval", "Deploy build 7?", options=["approve", "hold"],
        context={"build": 7}, timeout_seconds=600)
    (pending,) = client.list_human_requests(status_filter="pending")
    assert (pending.id, pending.options, pending.context) == (
        "deploy-7", ["approve", "hold"], {"build": 7})

    with pytest.raises(httpx.HTTPStatusError) as off_the_options:
        client.submit_human_response("deploy-7", "maybe", responded_by="tester")
    assert off_the_options.value.response.status_code == 400
    assert client.get_human_request("deploy-7")[1] is None

    client.submit_human_response("deploy-7", "approve", responded_by="tester")
    request, response = client.get_human_request("deploy-7")
    assert request.status == "responded"
    assert (response.response, response.responded_by) == ("approve", "tester")
    assert client.wait_for_human_response("deploy-7", timeout=5).response == "approve"
    assert client.list_human_requests(status_filter="pending") == []


def test_the_metrics_count_what_crossed_the_wire(client) -> None:
    client.invoke_tool("echo", {"message": "one"})
    client.invoke_tool("echo", {"message": "two"})

    metrics = httpx.get(f"{client.base_url}/metrics/json").json()

    assert "echo" in str(metrics)
    exposition = httpx.get(f"{client.base_url}/metrics").text
    assert 'tool="echo"' in exposition
