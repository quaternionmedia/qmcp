"""A PydanticAI agent calls the server's tools through `QMCPToolset`, offline.

`QMCPToolset` was not a PydanticAI toolset: `Agent(toolsets=[QMCPToolset(url)])`
raised "'QMCPToolset' object is not callable", and the tests only constructed
it. Here an agent on PydanticAI's `TestModel` -- which calls the tools it is
told to, with arguments drawn from their schemas, and needs no model -- calls
`echo` through the toolset on a server on port 0, and the server's record of
the call is read back.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from qmcp.integrations.pydantic_ai import QMCPToolset  # noqa: E402
from qmcp.testing import throwaway_server  # noqa: E402

pytestmark = pytest.mark.integration


def _echo_calls(url: str) -> list[dict]:
    response = httpx.get(f"{url}/v1/invocations", params={"tool_name": "echo"})
    return response.json()["invocations"]


def test_an_agent_calls_a_server_tool_through_the_toolset(tmp_path) -> None:
    # Seen red before QMCPToolset was an AbstractToolset: the Agent refused it
    # with "'QMCPToolset' object is not callable".
    async def run(url: str) -> list[str]:
        outputs = []
        async with QMCPToolset(url, correlation_id="pai-1") as toolset:
            agent = Agent(TestModel(call_tools=["echo"]), toolsets=[toolset])
            for _ in range(2):
                outputs.append((await agent.run("Echo something.")).output)
        return outputs

    with throwaway_server(tmp_path / "queue.db") as url:
        outputs = asyncio.run(run(url))
        calls = _echo_calls(url)

    assert len(outputs) == 2 and all("echo" in output for output in outputs)
    assert [(c["status"], c["correlation_id"]) for c in calls] == [("success", "pai-1")] * 2


def test_the_client_survives_the_agents_own_entries(tmp_path) -> None:
    """An agent enters its toolsets on every run. Mutation seen red: the entry
    count removed, so every exit closed the client -- the second run inside
    the caller's `async with` failed on a closed client."""
    async def run(url: str) -> bool:
        async with QMCPToolset(url) as toolset:
            agent = Agent(TestModel(call_tools=["echo"]), toolsets=[toolset])
            await agent.run("First.")
            await agent.run("Second.")
            return toolset._client.is_closed

    with throwaway_server(tmp_path / "queue.db") as url:
        closed_inside = asyncio.run(run(url))

    assert closed_inside is False


def test_a_prefixed_tool_reaches_the_server_by_its_own_name(tmp_path) -> None:
    async def run(url: str) -> None:
        async with QMCPToolset(url, tool_prefix="qmcp_") as toolset:
            assert "qmcp_echo" in await toolset.get_tools()
            agent = Agent(TestModel(call_tools=["qmcp_echo"]), toolsets=[toolset])
            await agent.run("Echo something.")

    with throwaway_server(tmp_path / "queue.db") as url:
        asyncio.run(run(url))
        calls = _echo_calls(url)

    assert [c["status"] for c in calls] == ["success"]
