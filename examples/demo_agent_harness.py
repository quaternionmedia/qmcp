#!/usr/bin/env python3
"""Local demo: qmcp as an agent harness, end to end, on this machine.

    uv run python examples/demo_agent_harness.py

WHAT IT SHOWS. A running server, the tools it registers, an agent-shaped
sequence through them -- planner, executor, reviewer -- and the audit log
growing by exactly the number of calls made. The last part is the point: every
invocation is recorded, so a harness that ran an agent can be asked afterwards
what it ran.

WHY THE AUDIT COUNT IS THE ASSERTION AND NOT THE TOOL OUTPUT. Recording an
invocation is the code path that was broken on `main`: `ToolInvocation.execution_id`
was `UUID NOT NULL` against an insert that supplies no id, so every call raised
`sqlite3.IntegrityError` and 19 tests failed. A demo that only printed a
planner's reply would have passed against that. This one exercises the fix.

WHY IT IS A FILE AND NOT A SESSION TRANSCRIPT. The same demo was run by hand
once and written into a handoff page. Nobody else could reproduce it, and a
result nobody can reproduce is the same defect as a hand-run check reported as
CI. `tests/test_demo_agent_harness.py` runs this module, so the demo is covered
by the suite rather than by a memory of it having worked.

WHAT IT DOES NOT TOUCH. The operator's `./qmcp.db`. Every run builds its own
database in a temporary directory and deletes it, because a demo that writes to
the working database is one nobody runs twice.

WHAT IT CANNOT SHOW ON THIS PLATFORM. The Metaflow flow layer. `import metaflow`
fails on `import fcntl`, which is POSIX-only, so the flows in `examples/flows/`
need Docker and are outside this demo by construction rather than by omission.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

TOOLS_EXPECTED = ("echo", "planner", "executor", "reviewer")


@contextmanager
def harness() -> Iterator[Any]:
    """A server on a scratch database, torn down afterwards.

    The settings object is replaced rather than the environment variable set:
    `qmcp.config.get_settings` is cached, and an env var read after the cache
    is warm changes nothing -- which looks like a demo writing to the scratch
    database while it writes to the real one.
    """
    import qmcp.config
    import qmcp.db.engine

    directory = tempfile.mkdtemp(prefix="qmcp-demo-")
    database = Path(directory) / f"demo-{uuid.uuid4().hex}.db"

    class DemoSettings:
        database_url = f"sqlite+aiosqlite:///{database.as_posix()}"
        debug = False
        host = "127.0.0.1"
        port = 8931
        log_level = "WARNING"

    settings = DemoSettings()
    real_config_settings = qmcp.config.get_settings
    real_engine_settings = qmcp.db.engine.get_settings

    qmcp.config.get_settings.cache_clear()
    qmcp.config.get_settings = lambda: settings
    qmcp.db.engine.get_settings = lambda: settings
    qmcp.db.engine._engine = None

    try:
        import logging

        from fastapi.testclient import TestClient

        from qmcp.logging import configure_logging
        from qmcp.server import create_app

        app = create_app()
        # After `create_app`, not before: it calls `configure_logging` itself,
        # with `level="INFO"` hardcoded from `settings.debug` -- `log_level` on
        # the settings object is never read. Setting it and expecting quiet is
        # a setting that is not the setting that gets read.
        configure_logging(json_format=False, level="WARNING")
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        with TestClient(app) as client:
            yield client
    finally:
        qmcp.config.get_settings = real_config_settings
        qmcp.db.engine.get_settings = real_engine_settings
        qmcp.db.engine._engine = None
        qmcp.config.get_settings.cache_clear()
        database.unlink(missing_ok=True)
        for leftover in Path(directory).glob("*"):
            leftover.unlink(missing_ok=True)
        os.rmdir(directory)


def audit_log(client: Any) -> tuple[int, list[dict]]:
    """The recorded invocations, and the server's own count of them.

    `count` is read rather than recomputed from the list: the endpoint states
    it, and a demo that recounted would be checking its own arithmetic instead
    of the server's answer.
    """
    response = client.get("/v1/invocations")
    response.raise_for_status()
    body = response.json()
    return body["count"], body["invocations"]


def run(out=print) -> dict[str, Any]:
    """The demo. Returns what it established, so a test can assert on it."""
    with harness() as client:
        identity = client.get("/openapi.json").json()["info"]
        out(f"server           {identity['title']} {identity['version']}")
        out(f"health           {client.get('/health').json().get('status')}")

        listed = client.get("/v1/tools").json()
        names = [tool["name"] for tool in listed["tools"]]
        out(f"tools registered {names}")

        before, _ = audit_log(client)
        out(f"audit log        {before} invocation(s) before")
        out("")

        # An agent-shaped sequence rather than three unrelated calls: the plan
        # the planner returns is what the executor is given, and the executor's
        # result is what the reviewer judges.
        #
        # `error` is read and `status` is not, because `ToolInvokeResponse`
        # carries no status field -- result, error, invocation_id, and that is
        # all. Printing `status=None` from a key that does not exist reads as
        # three failed calls, which is what the first draft of this demo did.
        plan = client.post("/v1/tools/planner", json={
            "input": {"goal": "ship the local demo", "context": "one machine, no network"},
        }).json()
        out(f"planner          error={plan.get('error')}  id={plan.get('invocation_id')}")

        executed = client.post("/v1/tools/executor", json={
            "input": {"plan": plan.get("result") or {}, "dry_run": True},
        }).json()
        out(f"executor         error={executed.get('error')}  id={executed.get('invocation_id')}")

        reviewed = client.post("/v1/tools/reviewer", json={
            "input": {"result": executed.get("result") or {},
                      "criteria": ["ran locally", "left no state behind"]},
        }).json()
        out(f"reviewer         error={reviewed.get('error')}  id={reviewed.get('invocation_id')}")

        after, recorded = audit_log(client)
        out("")
        out(f"audit log        {before} -> {after} invocation(s)")
        for entry in recorded:
            out(f"                 {entry.get('tool_name'):<10} "
                f"{entry.get('status')}  {entry.get('duration_ms')}ms")

        return {
            "title": identity["title"],
            "tools": names,
            "invocations_before": before,
            "invocations_after": after,
            "recorded": recorded,
            "errors": [plan.get("error"), executed.get("error"), reviewed.get("error")],
        }


def main() -> int:
    result = run()
    print()
    recorded = result["invocations_after"] - result["invocations_before"]
    print(f"{recorded} of 3 calls reached the audit log.")
    print("The audit log is the claim. A harness that cannot say what it ran "
          "afterwards is not one.")
    return 0 if recorded == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
