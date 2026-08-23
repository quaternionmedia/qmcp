"""Meta-MCP server for interacting with the qmcp repository.

Exposes tools for running flows, querying the persistence database,
interacting with the live qmcp server, and running tests — all accessible
from any MCP client (Claude Desktop, Claude Code, etc.).

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
