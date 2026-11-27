"""qmcp's Agent Host Protocol host, driven by the protocol's published client.

`tests/published_client/turn.mjs` runs `@microsoft/agent-host-protocol`, the
TypeScript client the protocol's own repository publishes, against a qmcp
server on a socket of its own with a database of its own. The client
negotiates its version, starts a turn and answers its consent; its reducers
are given every action the host sends, and the state they hold is compared
with the host's own snapshots during the turn and after it.

This is the one check of the host that qmcp did not write both sides of.
`tests/test_agents_host.py` holds the host to values written by hand.

It needs Node and the package, installed with

    npm ci --ignore-scripts --prefix tests/published_client

Without them the tests are skipped, and a skip is not a pass: with
`QMCP_PUBLISHED_CLIENT=required` in the environment, as the workflow sets it,
a missing Node or a missing package fails them.

What this cannot do: hold the host's messages to the protocol's JSON schemas,
which the package does not carry; or say anything of qmcp's client role.

Each test names the mutation it was seen to fail under.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from qmcp.client import MCPClient
from qmcp.testing import throwaway_server

pytestmark = pytest.mark.integration

HERE = Path(__file__).parent / "published_client"
PACKAGE = HERE / "node_modules" / "@microsoft" / "agent-host-protocol" / "package.json"
INSTALL = "npm ci --ignore-scripts --prefix tests/published_client"


def missing(node: str | None, package: Path) -> str | None:
    """What the check lacks on this machine, or None when it can run."""
    if node is None:
        return "Node is not on the PATH"
    if not package.is_file():
        return f"the published client is not installed; run `{INSTALL}`"
    return None


def refuse(lack: str, required: bool) -> None:
    """Skip for what is missing, or fail when the check is required."""
    if required:
        pytest.fail(f"QMCP_PUBLISHED_CLIENT=required, and {lack}")
    pytest.skip(lack)


@pytest.fixture
def turn(tmp_path):
    """Run one turn with the published client; returns its verdict and the queue's row."""
    lack = missing(shutil.which("node"), PACKAGE)
    if lack is not None:
        refuse(lack, os.environ.get("QMCP_PUBLISHED_CLIENT") == "required")

    def run(answer: str) -> tuple[dict, object]:
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        with throwaway_server(tmp_path / "queue.db") as url:
            done = subprocess.run(
                [shutil.which("node"), "turn.mjs",
                 url.replace("http://", "ws://") + "/v1/ahp", work.as_uri(), "scripted", answer],
                cwd=HERE, capture_output=True, text=True, timeout=180)
            lines = done.stdout.strip().splitlines()
            assert lines, done.stderr
            verdict = json.loads(lines[-1])
            assert done.returncode == 0, json.dumps(verdict, indent=1) + done.stderr
            with MCPClient(base_url=url) as queue:
                _, response = queue.get_human_request(verdict["request"]["id"])
        return verdict, response

    return run


def test_the_published_client_and_the_host_hold_one_state_through_an_approved_turn(turn):
    """Mutation seen red: a chat's snapshot sent with no turns, as
    `Chat.state` returning `{**held, "turns": []}` -- the client's reducers
    held the turn and the host's snapshot did not."""
    verdict, response = turn("approve")

    assert verdict["negotiated"] == verdict["package"]
    assert "scripted" in verdict["agents"]
    assert verdict["agree"] == {"during": True, "after": True, "session": True}
    assert verdict["problems"] == []
    assert verdict["turn"]["state"] == "complete"
    assert verdict["turn"]["response"] == "accept"
    assert verdict["turn"]["answers"]["answer"]["value"]["value"] == "approve"
    assert "The runtime finished." in verdict["turn"]["text"]
    assert (response.response, response.responded_by) == ("approve", "agent-host-protocol")


def test_a_turn_the_published_client_holds_runs_nothing(turn):
    """Mutation seen red: the relayed answer written as `approve` whatever
    the client chose -- the runtime ran."""
    verdict, response = turn("hold")

    assert verdict["agree"] == {"during": True, "after": True, "session": True}
    assert verdict["turn"]["state"] == "complete"
    assert verdict["turn"]["answers"]["answer"]["value"]["value"] == "hold"
    assert "The runtime finished." not in verdict["turn"]["text"]
    assert "nothing ran" in verdict["turn"]["text"]
    assert (response.response, response.responded_by) == ("hold", "agent-host-protocol")


@pytest.mark.parametrize("node,installed,lack", [
    (None, True, "Node is not on the PATH"),
    ("node", False, "is not installed"),
])
def test_a_missing_node_or_package_fails_the_check_where_it_is_required(
        tmp_path, node, installed, lack):
    """Mutation seen red: `refuse` skipping whatever `required` says."""
    package = tmp_path / "package.json"
    if installed:
        package.write_text("{}", encoding="utf-8")

    said = missing(node, package)

    assert lack in said
    package.write_text("{}", encoding="utf-8")
    assert missing("node", package) is None

    def outcome(required: bool) -> str:
        # Caught by name: a skip raised here would otherwise skip this test,
        # which reports green.
        try:
            refuse(said, required)
        except pytest.fail.Exception:
            return "failed"
        except pytest.skip.Exception:
            return "skipped"
        return "neither"

    assert (outcome(True), outcome(False)) == ("failed", "skipped")
