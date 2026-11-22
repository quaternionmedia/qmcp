"""`qmcp tools list` lists the tools `GET /v1/tools` publishes.

The command runs in a process of its own: the registry is shared within a
process, so once anything in the test session has imported the server, an
in-process run lists every tool whatever the command imports.

Mutation seen red: the command importing `qmcp.tools.builtin` alone, where it
reads `published_registry()` -- the `alfred_*` tools were missing from its
list.
"""

import re
import subprocess
import sys


def test_tools_list_names_every_tool_the_server_publishes(client):
    published = {tool["name"] for tool in client.get("/v1/tools").json()["tools"]}

    run = subprocess.run([sys.executable, "-m", "qmcp", "tools", "list"],
                         capture_output=True, text=True, timeout=120)

    assert run.returncode == 0, run.stderr
    listed = set(re.findall(r"^  (\S+)$", run.stdout, flags=re.MULTILINE))
    assert any(name.startswith("alfred_") for name in published)
    assert listed == published
