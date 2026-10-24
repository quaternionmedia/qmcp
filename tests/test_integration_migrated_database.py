"""A database the migration chain built is one the server runs on.

`tests/test_db_migrations.py` shows the chain builds every table and keeps
rows. The server builds its own tables with `create_all` when it starts, so
nothing there shows that a server started over a migrated database -- the
database a deployment has -- takes requests, writes rows the schema accepts,
and leaves the file still at the head revision with nothing drifted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from qmcp.client import MCPClient
from qmcp.db.schema import drift
from qmcp.testing import throwaway_server

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent


def alembic(*args: str, database: Path) -> subprocess.CompletedProcess:
    environment = {**os.environ,
                   "QMCP_DATABASE_URL": f"sqlite+aiosqlite:///{database.as_posix()}"}
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args], cwd=str(ROOT), env=environment,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


def test_a_server_over_a_migrated_database_takes_requests_and_leaves_it_at_head(
        tmp_path) -> None:
    # Mutation seen red: the upgrade stopped one revision short of head -- the
    # `project` columns the last migration adds were missing, and drift named
    # them.
    database = tmp_path / "deployed.db"
    upgraded = alembic("upgrade", "head", database=database)
    assert upgraded.returncode == 0, upgraded.stdout + upgraded.stderr
    head = alembic("heads", database=database).stdout.split()[0]

    with throwaway_server(database) as url, MCPClient(base_url=url) as client:
        called = client.invoke_tool("echo", {"message": "on the deployed schema"})
        client.create_human_request("migrated-1", "approval", "Still answerable?",
                                    options=["yes", "no"], timeout_seconds=600)
        client.submit_human_response("migrated-1", "yes", responded_by="tester")
        _, response = client.get_human_request("migrated-1")
        client.create_instruction(text="What is qmcp?", project="qmcp")

    assert called.error is None
    assert response.response == "yes"
    assert drift(database).clean, drift(database).lines()
    current = alembic("current", database=database)
    assert current.returncode == 0, current.stderr
    assert head in current.stdout
