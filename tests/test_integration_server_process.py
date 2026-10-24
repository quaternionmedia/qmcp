"""qmcp as its own process: what a restart keeps, and what a second program sees.

Every other server the suite starts runs in the test's own process. Here
`qmcp serve --port 0` runs as the command a person types
(`qmcp.testing.process.server_process`), found through the run-file it
leaves, and stopped the way a person stops it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from qmcp.client import MCPClient
from qmcp.db.models import Instruction, InstructionSource
from qmcp.instructions.act import act, rows_at
from qmcp.integrations.agents.scripted import ScriptedRuntime
from qmcp.spend import Budget
from qmcp.testing.process import server_process

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent


def test_a_server_process_states_what_its_run_file_says_and_takes_it_away(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with server_process(tmp_path / "queue.db", run_dir, session="identity") as served:
        assert served.run_file.exists()
        assert served.identity["database"] == str((tmp_path / "queue.db").resolve())
        assert served.identity["port"] not in (0, 3141)

    # Stopped as a person stops it, it shut down cleanly.
    assert not served.run_file.exists()


def test_the_queue_survives_a_restart(tmp_path) -> None:
    database, run_dir = tmp_path / "queue.db", tmp_path / "run"
    with server_process(database, run_dir, session="first") as first, \
            MCPClient(base_url=first.url) as client:
        client.create_human_request("still-waiting", "approval", "Ship?",
                                    options=["approve", "hold"], timeout_seconds=600)
        client.create_human_request("answered", "approval", "Tag?",
                                    options=["approve", "hold"], timeout_seconds=600)
        client.submit_human_response("answered", "hold", responded_by="tester")

    with server_process(database, run_dir, session="second") as second, \
            MCPClient(base_url=second.url) as client:
        assert second.identity["pid"] != first.identity["pid"]
        pending = [r.id for r in client.list_human_requests(status_filter="pending")]
        _, response = client.get_human_request("answered")

    assert pending == ["still-waiting"]
    assert (response.response, response.responded_by) == ("hold", "tester")


class _Hears:
    def __init__(self, text: str) -> None:
        self.text = text

    def listen(self, duration: float = 5.0, *, pause_ms: int | None = None):
        return self.text, "take.wav"


class _Says:
    def speak(self, text: str, out_path: str | None = None) -> str:
        return "out.wav"


class _Another(ScriptedRuntime):
    name = "another"


def _instruct(client, rows, instruction_id: str, text: str, runtime, cwd=None):
    with rows() as session:
        session.add(Instruction(id=instruction_id, text=text, project="rad-godot",
                                source=InstructionSource.VOICE))
        session.commit()
    return act(instruction_id, runtime, Budget(authorised=1), client=client, rows=rows,
               cwd=cwd, stt=_Hears("approve"), tts=_Says(), poll_interval=0.05)


def test_a_second_instruction_is_told_what_the_first_found_after_the_server_restarts(
        tmp_path) -> None:
    """ROADMAP Phase 10: "A second instruction in a project is told what the
    first one found, by a different runtime, after every process has
    restarted -- the history is in the record, not in the runtime." The
    server restarts between the two, and a different runtime takes the second.
    Mutation seen red: the act handed no history -- the second brief carried
    nothing."""
    database, run_dir = tmp_path / "queue.db", tmp_path / "run"
    clone = tmp_path / "rad-godot"
    clone.mkdir()
    rows = rows_at(database)

    with server_process(database, run_dir, session="first") as first, \
            MCPClient(base_url=first.url) as client:
        done = _instruct(client, rows, "pin", "Pin rad godot to the vectors.",
                         ScriptedRuntime(text="pinned to 2c10fd1"), cwd=clone)
    assert done.status == "done"

    another = _Another(text="the vectors are pinned to 2c10fd1")
    with server_process(database, run_dir, session="second") as second, \
            MCPClient(base_url=second.url) as client:
        again = _instruct(client, rows, "say", "Say which commit the vectors are pinned to.",
                          another)
        recorded = client.get_instruction("say")

    assert (again.status, again.carried) == ("done", ("pin",))
    (turn,) = another.briefs[0].history
    assert (turn.instruction, turn.runtime, turn.outcome) == (
        "Pin rad godot to the vectors.", "scripted", "pinned to 2c10fd1")
    assert another.briefs[0].cwd == clone
    assert recorded["detail"]["continuity"] == ["pin"]


def test_the_mcp_server_reads_a_servers_queue_over_stdio(tmp_path) -> None:
    """`qmcp_mcp.py` as an assistant runs it: a process on stdio, pointed at a
    qmcp in another process by `QMCP_SERVER_URL`."""
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def listed(url: str) -> list[dict]:
        parameters = StdioServerParameters(
            command=sys.executable, args=[str(ROOT / "qmcp_mcp.py")], cwd=str(ROOT),
            env={**os.environ, "QMCP_SERVER_URL": url})
        async with stdio_client(parameters) as (read, write), \
                ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert "list_human_requests" in tools
            assert "submit_human_response" not in tools
            result = await session.call_tool("list_human_requests", {"server_url": url})
            return [json.loads(block.text) for block in result.content]

    with server_process(tmp_path / "queue.db", tmp_path / "run", session="mcp") as served, \
            MCPClient(base_url=served.url) as client:
        client.create_human_request("asked-over-http", "approval", "Ship?",
                                    options=["approve", "hold"], timeout_seconds=600)
        rows = asyncio.run(listed(served.url))

    assert [row["id"] for row in rows] == ["asked-over-http"]
