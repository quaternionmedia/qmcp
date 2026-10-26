# Contributing

A change to qmcp is done when its tests show what it does, the gates the
hosted runner applies have been run here first, and the next person can tell
from the change alone why it is shaped as it is. This page is the setup and the
loop for making one. `AGENTS.md` carries the rules every change is held to,
and [`roles.md`](roles.md) says which part of qmcp a change belongs to.

## Set up

Do the shared part in [`setup/README.md`](setup/README.md). Nothing else is
needed to change qmcp and prove the change; the other setup pages are for
running it.

## The loop

```bash
uv run pytest tests/test_hitl.py -q          # the tests nearest the change
uv run pytest tests walkthrough -q           # the suite, walkthrough pages included
uv run ruff check .
uv run qmcp preflight --event pull_request --base-ref main
```

`uv run qmcp test` runs the same suite and leaves the configured database
alone; `--clean` deletes that database before and after the run.
`walkthrough/` is one ordered set of pages the suite executes, so an example a
reader reads is an example that ran: a behaviour worth showing is shown there,
by a page that runs it, rather than in prose beside the code.

The suite runs on a hosted runner in `.github/workflows/tests.yml`, and that
runner sees what Windows cannot: Metaflow does not import on Windows, so every
flow test skips here and runs there. A skipped test is not a passing one.

`preflight` runs the steps of every workflow under `.github/workflows/` as the
hosted runner would. It does not reproduce `uses:` steps, the runner image or
secrets. On Windows, after a change to `pyproject.toml`, run the runner by path
instead, as [`setup/harness.md`](setup/harness.md) shows.

## A server of your own

Another qmcp may already be serving on this machine, and the clone's
`qmcp.db` is somebody's queue. Give a server started for development its own
port and its own database:

```bash
export QMCP_SESSION="dev-$(date +%s)"
export QMCP_DATABASE_URL="sqlite+aiosqlite:///./$QMCP_SESSION.db"
uv run qmcp serve --port 0
```

The database lands in the clone, named for the session, where `.gitignore`
already ignores `*.db`.

`serve` prints the port the system picked, and its run-file in
`~/.qmcp/run/qmcp-<pid>.json` names the same port, the resolved database and
the start time. Before measuring anything against it, ask it which server it
is: `curl -s http://127.0.0.1:<port>/health` states its port, start time, pid,
session and a fingerprint of its database. In the shell that runs the CLI
against it, set `QMCP_PORT=<port>` and the same `QMCP_DATABASE_URL`.

`--reload` restarts on a fixed port, so it takes an explicit one:
`uv run qmcp serve --port <a free port> --reload`. On Windows a running server
holds `Scripts/qmcp.exe`, so stop it before a sync that reinstalls qmcp.

## Where a change goes

| Part | Package |
|---|---|
| the HTTP server, its routes, identity | `qmcp/server.py`, `qmcp/instance.py`, `qmcp/middleware.py` |
| tools | `qmcp/tools/` |
| the database, its models and migrations | `qmcp/db/`, `alembic/` |
| the human queue | `qmcp/server.py`, `qmcp/client/` |
| speech | `qmcp/integrations/voice/`, with `vendor/vox` |
| instructions, consent and runtimes | `qmcp/instructions/`, `qmcp/integrations/agents/`, `qmcp/spend.py` |
| the local model | `qmcp/localmodel.py` |
| gates, the dashboard, deltas | `qmcp/selfcheck.py`, `qmcp/dashboard.py`, `qmcp/cookbook/delta.py`, `qmcp/addresses.py` |
| the thread archive | `qmcp/threads/` |
| topologies and the orchestration plane | `qmcp/topology_*.py`, `qmcp/orchestration*.py`, `qmcp/governed.py` |
| cookbook flows | `qmcp/cookbook/`, `examples/flows/`, `docker/` |
| agent schemas and PydanticAI | `qmcp/agentframework/`, `qmcp/integrations/pydantic_ai/` |
| the test harness every test runs under | `qmcp/testing/` |
| the CLI | `qmcp/cli.py` |

## Adding a tool

1. Add it to `qmcp/tools/builtin.py`, or a module of its own imported where
   the server imports its tools.
2. Register it with `@tool_registry.register()`.
3. Keep it to one responsibility, with no side effects it does not need, and
   no calls to other tools.

```python
@tool_registry.register(
    name="my_tool",
    description="What this tool does",
    input_schema={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."}
        },
        "required": ["param"],
    },
)
def my_tool(params: dict) -> Any:
    return params.get("param")
```

## What gets a second look

New orchestration, new persistence models, new long-lived state, and anything
called an agent. Each tends to move qmcp across a line [`roles.md`](roles.md)
draws: from recording what happens to deciding it.

## Commits and pull requests

A commit's subject states the change as a sentence about what is now true, and
its body says what a reader of the diff alone would not know. Commits are
signed, and carry the contributor's name only: no co-author trailer, and no
tool named as an author. A change arrives as a pull request, one change each,
and `AGENTS.md` says how they stack and when they merge.
