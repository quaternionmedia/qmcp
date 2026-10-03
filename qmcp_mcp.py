"""Meta-MCP server for interacting with the qmcp repository.

Exposes tools for running flows, querying the persistence database,
interacting with the live qmcp server, putting a question on the human queue
and waiting for its answer, and running tests — all accessible from any MCP
client (Claude Desktop, Claude Code, etc.).

An agent asks through three tools. `create_human_request` puts a question on
the queue; `await_human_response` waits for it to be answered or to expire;
`ask_human` is the two composed, and is the call an agent makes. The wait
polls the pending listing, which has no side effects, and reads the request
itself exactly once after it has left that listing: reading a pending request
past its expiry is what expires it, so a loop over the detail route would
expire the questions it was waiting on.

Usage (stdio transport):
    uv run python qmcp_mcp.py

Claude Code / Claude Desktop config:
    {
      "mcpServers": {
        "qmcp-repo": {
          "command": "uv",
          "args": ["run", "--project", "<path to your qmcp clone>", "python", "qmcp_mcp.py"],
          "cwd": "<path to your qmcp clone>"
        }
      }
    }

Environment variables:
    QMCP_SERVER_URL  - URL of the running qmcp HTTP server (default: the port `qmcp/config.py` allocates)
    QMCP_DB_PATH     - Path to the flow persistence SQLite DB (default: .qmcp_devflows.db)
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("qmcp-repo")

REPO_ROOT = Path(__file__).parent.resolve()
# **NOT A FOURTH COPY OF THE PORT.** `qmcp/config.py` owns the allocation --
# 3141 the harness, 1618 the panel, 2718 the maps -- and this file carried
# `3333` from before it existed, so the MCP server's default addressed a port
# nothing serves. Read rather than restated: a number repeated in a second
# place is one nothing updates.
try:
    from qmcp.config import Settings as _Settings
    _DEFAULT_PORT = _Settings().port
except Exception:                                  # noqa: BLE001
    _DEFAULT_PORT = 3141

DEFAULT_SERVER_URL = os.getenv("QMCP_SERVER_URL", f"http://localhost:{_DEFAULT_PORT}")
DEFAULT_DB_PATH = os.getenv("QMCP_DB_PATH", str(REPO_ROOT / ".qmcp_devflows.db"))

_RECIPES: dict[str, dict[str, Any]] = {
    "simple-plan": {
        "description": "Plan -> execute -> review using MCP tools",
        "flow": "examples/flows/simple_plan.py",
        "required_flags": [],
    },
    "approved-deploy": {
        "description": "HITL approval workflow for deployments",
        "flow": "examples/flows/approved_deploy.py",
        "required_flags": ["--service"],
    },
    "local-agent-chain": {
        "description": "Local LLM plan -> review -> refine chain",
        "flow": "examples/flows/local_agent_chain.py",
        "required_flags": ["--goal"],
    },
    "local-qc-gauntlet": {
        "description": "Local LLM QC checklist + tasks + gate",
        "flow": "examples/flows/local_qc_gauntlet.py",
        "required_flags": ["--change-summary"],
    },
    "local-release-notes": {
        "description": "Local LLM release notes + doc updates",
        "flow": "examples/flows/local_release_notes.py",
        "required_flags": ["--change-summary"],
    },
    "council-deliberation": {
        "description": "Multi-agent council deliberation for decisions",
        "flow": "examples/flows/council_deliberation.py",
        "required_flags": ["--question"],
    },
    "qc-release": {
        "description": "QC gauntlet + release notes compound pipeline",
        "flow": "examples/flows/qc_release.py",
        "required_flags": ["--change-summary"],
    },
    "plan-council": {
        "description": "Plan + council deliberation + refinement",
        "flow": "examples/flows/plan_council.py",
        "required_flags": ["--goal"],
    },
    "change-impact": {
        "description": "Full change impact analysis pipeline",
        "flow": "examples/flows/change_impact.py",
        "required_flags": ["--change-summary"],
    },
}


# ---------------------------------------------------------------------------
# Repo info
# ---------------------------------------------------------------------------


@mcp.tool()
def get_repo_info() -> dict[str, Any]:
    """Get information about the qmcp repository: structure, recipes, and git status.

    Returns dict with repo_root, git_branch, git_status, flow_files, recipes,
    default_server_url, and default_db_path.
    """
    flow_files = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "examples" / "flows").glob("*.py")
        if not p.name.startswith("_")
    )

    def _git(args: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git", *args], capture_output=True, text=True,
                cwd=str(REPO_ROOT), timeout=5,
            )
            return r.stdout.strip()
        except Exception:
            return "unavailable"

    return {
        "repo_root": str(REPO_ROOT),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "flow_files": flow_files,
        "recipes": list(_RECIPES.keys()),
        "default_server_url": DEFAULT_SERVER_URL,
        "default_db_path": DEFAULT_DB_PATH,
    }


# ---------------------------------------------------------------------------
# Recipe tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_recipes() -> list[dict[str, Any]]:
    """List all available qmcp cookbook recipes.

    Returns name, description, flow script path, and required CLI flags for each recipe.
    """
    return [
        {
            "name": name,
            "description": r["description"],
            "flow": r["flow"],
            "required_flags": r["required_flags"],
        }
        for name, r in _RECIPES.items()
    ]


@mcp.tool()
def run_recipe_local(
    recipe: str,
    flow_args: list[str] | None = None,
    mcp_url: str = DEFAULT_SERVER_URL,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run a qmcp recipe locally via Metaflow (not Docker).

    Args:
        recipe: Recipe name (e.g. "local-agent-chain"). Use list_recipes to see options.
        flow_args: Extra CLI arguments for the flow (e.g. ["--goal", "Deploy service"]).
        mcp_url: URL of the running qmcp server (injected as --mcp-url if not in flow_args).
        timeout_seconds: Subprocess timeout in seconds.

    Returns:
        Dict with status, returncode, stdout (last 4000 chars), and stderr (last 2000 chars).
    """
    name = recipe.lower().replace("_", "-")
    if name not in _RECIPES:
        return {"error": f"Unknown recipe '{recipe}'. Use list_recipes to see options."}

    flow_path = REPO_ROOT / _RECIPES[name]["flow"]
    if not flow_path.exists():
        return {"error": f"Flow script not found: {flow_path}"}

    args = flow_args or []
    cmd = [sys.executable, str(flow_path), "run", *args]
    if "--mcp-url" not in args:
        cmd.extend(["--mcp-url", mcp_url])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_seconds, cwd=str(REPO_ROOT),
        )
        return {
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Timed out after {timeout_seconds}s"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Flow persistence database tools
# ---------------------------------------------------------------------------


def _db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def list_flow_runs(
    db_path: str = DEFAULT_DB_PATH,
    flow_name: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List Metaflow run records from the local flow persistence database.

    Args:
        db_path: Path to the SQLite database (default: .qmcp_devflows.db in repo root).
        flow_name: Optional filter by flow name (e.g. "LocalAgentChain").
        limit: Max records to return.

    Returns:
        List of flow run records, newest first.
    """
    if not Path(db_path).exists():
        return [{"info": f"No database at {db_path} — run a flow first."}]

    sql = "SELECT * FROM flowrun"
    params: list[Any] = []
    if flow_name:
        sql += " WHERE flow_name = ?"
        params.append(flow_name)
    sql += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)

    with _db(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


@mcp.tool()
def get_flow_run_details(
    flow_run_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Get full details for a flow run: agent runs, artifacts, and MCP invocations.

    Args:
        flow_run_id: The flow run ID (from list_flow_runs).
        db_path: Path to the SQLite database.

    Returns:
        Dict with keys: flow_run, agent_runs, artifacts, mcp_invocations.
    """
    if not Path(db_path).exists():
        return {"error": f"No database at {db_path}"}

    with _db(db_path) as conn:
        run = conn.execute("SELECT * FROM flowrun WHERE id = ?", [flow_run_id]).fetchone()
        if run is None:
            return {"error": f"Flow run '{flow_run_id}' not found"}

        agent_runs = conn.execute(
            "SELECT * FROM agentrun WHERE flow_run_id = ? ORDER BY created_at",
            [flow_run_id],
        ).fetchall()

        artifacts = conn.execute(
            "SELECT * FROM artifact WHERE flow_run_id = ? ORDER BY created_at",
            [flow_run_id],
        ).fetchall()

        mcp_calls = conn.execute(
            "SELECT * FROM mcpinvocation WHERE flow_run_id = ? ORDER BY created_at",
            [flow_run_id],
        ).fetchall()

        return {
            "flow_run": dict(run),
            "agent_runs": [dict(r) for r in agent_runs],
            "artifacts": [dict(a) for a in artifacts],
            "mcp_invocations": [dict(m) for m in mcp_calls],
        }


@mcp.tool()
def list_checklist_items(
    flow_run_id: str,
    db_path: str = DEFAULT_DB_PATH,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    """List QC checklist items for a flow run.

    Args:
        flow_run_id: The flow run ID (from list_flow_runs).
        db_path: Path to the SQLite database.
        status_filter: Optional status filter: "pending", "passed", or "failed".

    Returns:
        List of checklist item records ordered by creation time.
    """
    if not Path(db_path).exists():
        return [{"error": f"No database at {db_path}"}]

    sql = "SELECT * FROM checklistitem WHERE flow_run_id = ?"
    params: list[Any] = [flow_run_id]
    if status_filter:
        sql += " AND status = ?"
        params.append(status_filter)
    sql += " ORDER BY created_at"

    with _db(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


# ---------------------------------------------------------------------------
# Live qmcp server tools
# ---------------------------------------------------------------------------


@mcp.tool()
def server_health(server_url: str = DEFAULT_SERVER_URL) -> dict[str, Any]:
    """Check if the qmcp HTTP server is running and healthy.

    Args:
        server_url: Base URL of the qmcp server (default: the port `qmcp/config.py` allocates).

    Returns:
        Health response dict, or an error dict if unreachable.
    """
    try:
        r = httpx.get(f"{server_url}/health", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"status": "unreachable", "error": f"Cannot connect to {server_url}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def list_server_tools(server_url: str = DEFAULT_SERVER_URL) -> list[dict[str, Any]]:
    """List tools registered on the running qmcp server.

    Args:
        server_url: Base URL of the qmcp server.

    Returns:
        List of tool definitions (name, description, input_schema).
    """
    try:
        r = httpx.get(f"{server_url}/v1/tools", timeout=5.0)
        r.raise_for_status()
        return r.json()["tools"]
    except httpx.ConnectError:
        return [{"error": f"Cannot connect to {server_url} — is the server running?"}]
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def invoke_server_tool(
    tool_name: str,
    input_params: dict[str, Any],
    server_url: str = DEFAULT_SERVER_URL,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Invoke a tool on the running qmcp HTTP server.

    Args:
        tool_name: Name of the tool (e.g. "planner", "reviewer", "executor").
        input_params: Input parameters dict for the tool.
        server_url: Base URL of the qmcp server.
        correlation_id: Optional correlation ID for tracing.

    Returns:
        Dict with result, error, and invocation_id.
    """
    payload: dict[str, Any] = {"input": input_params}
    if correlation_id:
        payload["correlation_id"] = correlation_id

    try:
        r = httpx.post(f"{server_url}/v1/tools/{tool_name}", json=payload, timeout=30.0)
        if r.status_code == 404:
            return {"error": f"Tool '{tool_name}' not found on server"}
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to {server_url}"}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_server_invocations(
    tool_name: str | None = None,
    status: str | None = None,
    limit: int = 20,
    server_url: str = DEFAULT_SERVER_URL,
) -> list[dict[str, Any]]:
    """List tool invocation history from the running qmcp server.

    Args:
        tool_name: Optional filter by tool name.
        status: Optional filter by status ("success" or "failed").
        limit: Max records to return.
        server_url: Base URL of the qmcp server.

    Returns:
        List of invocation records, newest first.
    """
    params: dict[str, Any] = {"limit": limit}
    if tool_name:
        params["tool_name"] = tool_name
    if status:
        params["status"] = status

    try:
        r = httpx.get(f"{server_url}/v1/invocations", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()["invocations"]
    except httpx.ConnectError:
        return [{"error": f"Cannot connect to {server_url}"}]
    except Exception as exc:
        return [{"error": str(exc)}]


@mcp.tool()
def submit_human_response(
    request_id: str,
    response: str,
    responded_by: str | None = None,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Submit a human response to a pending HITL request on the qmcp server.

    Args:
        request_id: The HITL request ID to respond to.
        response: The response value (must match allowed options if set).
        responded_by: Optional identifier for who is responding.
        server_url: Base URL of the qmcp server.

    Returns:
        The created response record or an error dict.
    """
    payload: dict[str, Any] = {"request_id": request_id, "response": response}
    if responded_by:
        payload["responded_by"] = responded_by

    try:
        r = httpx.post(f"{server_url}/v1/human/responses", json=payload, timeout=10.0)
        if r.status_code == 404:
            return {"error": f"Request '{request_id}' not found"}
        if r.status_code == 410:
            return {"error": f"Request '{request_id}' has expired"}
        if r.status_code == 409:
            return {"error": "Request has already been responded to"}
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to {server_url}"}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def list_human_requests(
    status_filter: str | None = None,
    limit: int = 20,
    server_url: str = DEFAULT_SERVER_URL,
) -> list[dict[str, Any]]:
    """List pending (or all) HITL requests from the qmcp server.

    Args:
        status_filter: Optional status filter: "pending", "responded", or "expired".
        limit: Max records to return.
        server_url: Base URL of the qmcp server.

    Returns:
        List of human request records.
    """
    params: dict[str, Any] = {"limit": limit}
    if status_filter:
        params["status"] = status_filter

    try:
        r = httpx.get(f"{server_url}/v1/human/requests", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()["requests"]
    except httpx.ConnectError:
        return [{"error": f"Cannot connect to {server_url}"}]
    except Exception as exc:
        return [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# An agent asks
# ---------------------------------------------------------------------------


def _request_id() -> str:
    """An id for a question whose caller gave none.

    Readable in a listing and ordered by creation. The milliseconds keep two
    questions asked within one second apart, which the server would otherwise
    refuse as a duplicate id.
    """
    return "ask-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")[:-3]


# One page of the pending listing. Any size the server accepts is correct,
# because paging stops on a short page rather than on this number; a size the
# server refuses fails loudly in `_pending_ids` rather than reading as empty.
_PENDING_PAGE = 100


def _pending_ids(server_url: str) -> set[str]:
    """Every id the server lists as pending, across every page.

    The listing applies no expiry and changes nothing, so it can be read as
    often as a wait needs. It is paged to the end: a queue deeper than one
    page would otherwise leave a still-pending id off the first page, and the
    wait would read that as the question having been answered.
    """
    ids: set[str] = set()
    offset = 0
    while True:
        r = httpx.get(
            f"{server_url}/v1/human/requests",
            params={"status": "pending", "limit": _PENDING_PAGE, "offset": offset},
            timeout=10.0,
        )
        r.raise_for_status()
        page = r.json()["requests"]
        ids.update(item["id"] for item in page)
        if len(page) < _PENDING_PAGE:
            return ids
        offset += _PENDING_PAGE


@mcp.tool()
def create_human_request(
    prompt: str,
    options: list[str] | None = None,
    request_type: str = "approval",
    request_id: str | None = None,
    expires_in_seconds: int = 600,
    context: dict[str, Any] | None = None,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Put a question on the qmcp human queue, for a person to answer.

    Args:
        prompt: The question, as it will be shown or spoken.
        options: The allowed answers. Leave unset for an open question, whose
                 answer is whatever the person says or types.
        request_type: "approval", "input" or "review".
        request_id: An id of the caller's choosing; a readable, time-derived
                    one is made when none is given.
        expires_in_seconds: How long the question stays answerable. The
                            server holds the floor and the ceiling.
        context: Anything else to show beside the prompt.
        server_url: Base URL of the qmcp server.

    Returns:
        The created request as the server returns it (id, request_type,
        prompt, status, created_at, expires_at), or an error dict.
    """
    payload: dict[str, Any] = {
        "id": request_id or _request_id(),
        "request_type": request_type,
        "prompt": prompt,
        "timeout_seconds": expires_in_seconds,
        "context": context or {},
    }
    if options is not None:
        payload["options"] = options

    try:
        r = httpx.post(f"{server_url}/v1/human/requests", json=payload, timeout=10.0)
        if r.status_code == 409:
            return {"error": f"Request '{payload['id']}' already exists"}
        if r.status_code == 422:
            # The server's own words for what it refused -- the expiry outside
            # its bounds, usually -- rather than the bare status line.
            return {"error": str(r.json().get("detail", r.text))}
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        return {"error": f"Cannot connect to {server_url}"}
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def await_human_response(
    request_id: str,
    timeout_seconds: float = 600,
    poll_seconds: float = 2.0,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Wait for a question on the human queue to be answered, or to expire.

    While the id is in the pending listing the wait sleeps and looks again,
    and reads nothing else: `GET /v1/human/requests/{id}` expires a pending
    request that is past its expiry, so polling it would expire the question
    being waited on. Once the id has left the listing, the request is read
    once, for its answer or its expired state. A timeout reads nothing.

    Args:
        request_id: The id returned by create_human_request.
        timeout_seconds: How long to wait before giving up.
        poll_seconds: How long to sleep between looks at the listing.
        server_url: Base URL of the qmcp server.

    Returns:
        Dict with status -- "answered", "expired" or "timeout", or the
        server's own word for any other state -- the request_id, and
        response: the response record when answered, else None.
    """
    deadline = time.monotonic() + timeout_seconds
    try:
        while request_id in _pending_ids(server_url):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "timeout", "request_id": request_id, "response": None}
            time.sleep(min(poll_seconds, remaining))

        r = httpx.get(f"{server_url}/v1/human/requests/{request_id}", timeout=10.0)
        if r.status_code == 404:
            return {"status": "error", "error": f"Request '{request_id}' not found"}
        r.raise_for_status()
        body = r.json()
        status = body["request"]["status"]
        return {
            "status": "answered" if status == "responded" else status,
            "request_id": request_id,
            "response": body.get("response"),
        }
    except httpx.ConnectError:
        return {"status": "error", "error": f"Cannot connect to {server_url}"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def ask_human(
    prompt: str,
    options: list[str] | None = None,
    timeout_seconds: int = 600,
    poll_seconds: float = 2.0,
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """Ask a person a question and wait for the answer.

    create_human_request followed by await_human_response, with the question
    answerable for as long as the caller waits. With options it is an
    "approval"; without, an "input" whose answer is free text.

    Args:
        prompt: The question.
        options: The allowed answers, or unset for an open question.
        timeout_seconds: How long to wait, and how long the question stays
                         answerable. The server holds the floor on the latter.
        poll_seconds: How long to sleep between looks at the listing.
        server_url: Base URL of the qmcp server.

    Returns:
        await_human_response's dict, or create_human_request's error dict.
    """
    created = create_human_request(
        prompt,
        options=options,
        request_type="approval" if options else "input",
        expires_in_seconds=timeout_seconds,
        server_url=server_url,
    )
    if "error" in created:
        return created
    return await_human_response(
        created["id"],
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        server_url=server_url,
    )


# ---------------------------------------------------------------------------
# Dev tools
# ---------------------------------------------------------------------------


@mcp.tool()
def run_tests(
    test_path: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the qmcp test suite via pytest.

    Args:
        test_path: Optional specific path (e.g. "tests/test_cookbook.py" or
                   "tests/test_cookbook_steps.py::TestAgentStep").
        verbose: Enable verbose pytest output (-v).

    Returns:
        Dict with passed (bool), returncode, stdout, and stderr.
    """
    cmd = [sys.executable, "-m", "pytest"]
    if verbose:
        cmd.append("-v")
    if test_path:
        cmd.append(test_path)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=180, cwd=str(REPO_ROOT),
        )
        return {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout[-6000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": "Tests timed out after 180s"}
    except Exception as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run()
