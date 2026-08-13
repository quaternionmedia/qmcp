"""Tests for the qmcp meta-MCP server (qmcp_mcp.py).

All network and subprocess calls are mocked so the suite runs fully offline.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# qmcp_mcp imports the MCP SDK at module level, and that SDK is an optional
# extra. Without this guard a machine that has not installed it fails
# collection for the whole suite rather than skipping this one module.
#
# Guard the submodule actually imported, not the top-level package: mcp 2.x
# installs as `mcp` but removed `mcp.server.fastmcp`, so a top-level check
# passes and the import below still dies at collection.
pytest.importorskip(
    "mcp.server.fastmcp",
    reason="requires the 'mcp' extra (<2.0): uv sync --extra mcp",
)

import qmcp_mcp as mcp_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> str:
    """Create a minimal flow-persistence SQLite DB for testing."""
    db_path = str(tmp_path / "test_flows.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE flowrun (
            id TEXT PRIMARY KEY,
            flow_name TEXT,
            run_id TEXT,
            meta TEXT,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE TABLE agentrun (
            id TEXT PRIMARY KEY,
            flow_run_id TEXT,
            agent_name TEXT,
            input_summary TEXT,
            output TEXT,
            created_at TEXT
        );
        CREATE TABLE artifact (
            id TEXT PRIMARY KEY,
            flow_run_id TEXT,
            kind TEXT,
            content TEXT,
            created_at TEXT
        );
        CREATE TABLE mcpinvocation (
            id TEXT PRIMARY KEY,
            flow_run_id TEXT,
            tool_name TEXT,
            invocation_id TEXT,
            correlation_id TEXT,
            payload TEXT,
            created_at TEXT
        );
        CREATE TABLE checklistitem (
            id TEXT PRIMARY KEY,
            flow_run_id TEXT,
            area TEXT,
            "check" TEXT,
            command TEXT,
            expected TEXT,
            status TEXT,
            notes TEXT,
            created_at TEXT
        );
        INSERT INTO flowrun VALUES ('run-1', 'TestFlow', 'mf-1', '{}', '2025-01-01', NULL);
        INSERT INTO agentrun VALUES ('ar-1', 'run-1', 'planner', 'plan', '{}', '2025-01-01');
        INSERT INTO artifact VALUES ('art-1', 'run-1', 'plan', '{}', '2025-01-01');
        INSERT INTO mcpinvocation VALUES ('mcp-1', 'run-1', 'executor', 'inv-1', NULL, '{}', '2025-01-01');
        INSERT INTO checklistitem VALUES ('ci-1', 'run-1', 'tests', 'run unit tests', NULL, NULL, 'pending', NULL, '2025-01-01');
    """)
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# get_repo_info
# ---------------------------------------------------------------------------


class TestGetRepoInfo:
    def test_returns_required_keys(self):
        result = mcp_mod.get_repo_info()
        for key in ("repo_root", "git_branch", "git_status", "flow_files", "recipes"):
            assert key in result

    def test_recipes_non_empty(self):
        result = mcp_mod.get_repo_info()
        assert len(result["recipes"]) > 0

    def test_flow_files_are_strings(self):
        result = mcp_mod.get_repo_info()
        assert all(isinstance(f, str) for f in result["flow_files"])


# ---------------------------------------------------------------------------
# list_recipes
# ---------------------------------------------------------------------------


class TestListRecipes:
    def test_returns_all_recipes(self):
        recipes = mcp_mod.list_recipes()
        names = {r["name"] for r in recipes}
        assert "local-agent-chain" in names
        assert "council-deliberation" in names
        assert "plan-council" in names

    def test_recipe_has_required_fields(self):
        for recipe in mcp_mod.list_recipes():
            assert "name" in recipe
            assert "description" in recipe
            assert "flow" in recipe
            assert "required_flags" in recipe

    def test_nine_recipes_total(self):
        assert len(mcp_mod.list_recipes()) == 9


# ---------------------------------------------------------------------------
# run_recipe_local
# ---------------------------------------------------------------------------


class TestRunRecipeLocal:
    def test_unknown_recipe_returns_error(self):
        result = mcp_mod.run_recipe_local("nonexistent-recipe")
        assert "error" in result

    def test_missing_flow_script_returns_error(self):
        # Patch REPO_ROOT so the flow path won't exist
        with patch.dict(mcp_mod._RECIPES, {"fake": {"flow": "no/such/file.py", "description": "x", "required_flags": []}}):
            result = mcp_mod.run_recipe_local("fake")
        assert "error" in result

    def test_successful_run(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Metaflow run complete"
        mock_result.stderr = ""

        with patch("qmcp_mcp.subprocess.run", return_value=mock_result) as mock_run:
            # local-agent-chain flow file must exist for this path
            with patch("qmcp_mcp.Path.exists", return_value=True):
                result = mcp_mod.run_recipe_local(
                    "local-agent-chain",
                    flow_args=["--goal", "Test goal"],
                )

        assert result["status"] == "completed"
        assert result["returncode"] == 0

    def test_failed_run(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error!"

        with patch("qmcp_mcp.subprocess.run", return_value=mock_result):
            with patch("qmcp_mcp.Path.exists", return_value=True):
                result = mcp_mod.run_recipe_local("local-agent-chain", flow_args=["--goal", "x"])

        assert result["status"] == "failed"
        assert result["returncode"] == 1

    def test_timeout_returns_error(self):
        with patch("qmcp_mcp.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            with patch("qmcp_mcp.Path.exists", return_value=True):
                result = mcp_mod.run_recipe_local("local-agent-chain", flow_args=["--goal", "x"])

        assert result["status"] == "timeout"

    def test_mcp_url_injected_if_absent(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        captured_cmd = []

        def capture(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_result

        with patch("qmcp_mcp.subprocess.run", side_effect=capture):
            with patch("qmcp_mcp.Path.exists", return_value=True):
                mcp_mod.run_recipe_local("local-agent-chain", flow_args=["--goal", "x"])

        assert "--mcp-url" in captured_cmd

    def test_mcp_url_not_duplicated_if_present(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        captured_cmd = []

        def capture(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_result

        with patch("qmcp_mcp.subprocess.run", side_effect=capture):
            with patch("qmcp_mcp.Path.exists", return_value=True):
                mcp_mod.run_recipe_local(
                    "local-agent-chain",
                    flow_args=["--goal", "x", "--mcp-url", "http://custom:9999"],
                )

        assert captured_cmd.count("--mcp-url") == 1


# ---------------------------------------------------------------------------
# list_flow_runs
# ---------------------------------------------------------------------------


class TestListFlowRuns:
    def test_missing_db_returns_info(self):
        result = mcp_mod.list_flow_runs(db_path="/nonexistent/path.db")
        assert len(result) == 1
        assert "info" in result[0]

    def test_returns_rows(self, tmp_path):
        db = _make_db(tmp_path)
        rows = mcp_mod.list_flow_runs(db_path=db)
        assert len(rows) == 1
        assert rows[0]["flow_name"] == "TestFlow"

    def test_filter_by_flow_name(self, tmp_path):
        db = _make_db(tmp_path)
        rows = mcp_mod.list_flow_runs(db_path=db, flow_name="TestFlow")
        assert len(rows) == 1

        rows_none = mcp_mod.list_flow_runs(db_path=db, flow_name="OtherFlow")
        assert len(rows_none) == 0

    def test_limit_respected(self, tmp_path):
        db = _make_db(tmp_path)
        rows = mcp_mod.list_flow_runs(db_path=db, limit=1)
        assert len(rows) <= 1


# ---------------------------------------------------------------------------
# get_flow_run_details
# ---------------------------------------------------------------------------


class TestGetFlowRunDetails:
    def test_missing_db_returns_error(self):
        result = mcp_mod.get_flow_run_details("any-id", db_path="/no/db.db")
        assert "error" in result

    def test_missing_run_id_returns_error(self, tmp_path):
        db = _make_db(tmp_path)
        result = mcp_mod.get_flow_run_details("not-a-real-id", db_path=db)
        assert "error" in result

    def test_returns_full_details(self, tmp_path):
        db = _make_db(tmp_path)
        result = mcp_mod.get_flow_run_details("run-1", db_path=db)
        assert result["flow_run"]["id"] == "run-1"
        assert len(result["agent_runs"]) == 1
        assert len(result["artifacts"]) == 1
        assert len(result["mcp_invocations"]) == 1


# ---------------------------------------------------------------------------
# list_checklist_items
# ---------------------------------------------------------------------------


class TestListChecklistItems:
    def test_missing_db_returns_error(self):
        result = mcp_mod.list_checklist_items("any-id", db_path="/no/db.db")
        assert "error" in result[0]

    def test_returns_items(self, tmp_path):
        db = _make_db(tmp_path)
        items = mcp_mod.list_checklist_items("run-1", db_path=db)
        assert len(items) == 1
        assert items[0]["status"] == "pending"

    def test_status_filter(self, tmp_path):
        db = _make_db(tmp_path)
        pending = mcp_mod.list_checklist_items("run-1", db_path=db, status_filter="pending")
        assert len(pending) == 1

        passed = mcp_mod.list_checklist_items("run-1", db_path=db, status_filter="passed")
        assert len(passed) == 0


# ---------------------------------------------------------------------------
# server_health
# ---------------------------------------------------------------------------


class TestServerHealth:
    def test_healthy_server(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "healthy", "version": "0.1.0"}
        mock_resp.raise_for_status = MagicMock()

        with patch("qmcp_mcp.httpx.get", return_value=mock_resp):
            result = mcp_mod.server_health("http://localhost:3333")

        assert result["status"] == "healthy"

    def test_unreachable_server(self):
        import httpx

        with patch("qmcp_mcp.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.server_health("http://localhost:3333")

        assert result["status"] == "unreachable"
        assert "error" in result


# ---------------------------------------------------------------------------
# list_server_tools
# ---------------------------------------------------------------------------


class TestListServerTools:
    def test_returns_tools(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"tools": [{"name": "echo", "description": "Echo"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("qmcp_mcp.httpx.get", return_value=mock_resp):
            tools = mcp_mod.list_server_tools()

        assert tools[0]["name"] == "echo"

    def test_unreachable_returns_error(self):
        import httpx

        with patch("qmcp_mcp.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.list_server_tools()

        assert "error" in result[0]


# ---------------------------------------------------------------------------
# invoke_server_tool
# ---------------------------------------------------------------------------


class TestInvokeServerTool:
    def test_successful_invocation(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "hello", "invocation_id": "inv-1"}
        mock_resp.raise_for_status = MagicMock()

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.invoke_server_tool("echo", {"message": "hello"})

        assert result["result"] == "hello"

    def test_tool_not_found(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.invoke_server_tool("no-such-tool", {})

        assert "error" in result
        assert "not found" in result["error"]

    def test_unreachable_returns_error(self):
        import httpx

        with patch("qmcp_mcp.httpx.post", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.invoke_server_tool("echo", {"message": "x"})

        assert "error" in result


# ---------------------------------------------------------------------------
# list_server_invocations
# ---------------------------------------------------------------------------


class TestListServerInvocations:
    def test_returns_invocations(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"invocations": [{"id": "inv-1", "tool_name": "echo"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("qmcp_mcp.httpx.get", return_value=mock_resp):
            result = mcp_mod.list_server_invocations()

        assert result[0]["id"] == "inv-1"

    def test_unreachable_returns_error(self):
        import httpx

        with patch("qmcp_mcp.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.list_server_invocations()

        assert "error" in result[0]


# ---------------------------------------------------------------------------
# submit_human_response
# ---------------------------------------------------------------------------


class TestSubmitHumanResponse:
    def test_successful_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {"id": "resp-1", "request_id": "req-1"}
        mock_resp.raise_for_status = MagicMock()

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.submit_human_response("req-1", "approve")

        assert result["id"] == "resp-1"

    def test_not_found(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.submit_human_response("bad-id", "approve")

        assert "error" in result

    def test_expired(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 410

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.submit_human_response("old-req", "approve")

        assert "error" in result
        assert "expired" in result["error"]

    def test_already_responded(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 409

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.submit_human_response("dup-req", "approve")

        assert "error" in result


# ---------------------------------------------------------------------------
# list_human_requests
# ---------------------------------------------------------------------------


class TestListHumanRequests:
    def test_returns_requests(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"requests": [{"id": "req-1", "status": "pending"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("qmcp_mcp.httpx.get", return_value=mock_resp):
            result = mcp_mod.list_human_requests()

        assert result[0]["id"] == "req-1"

    def test_unreachable_returns_error(self):
        import httpx

        with patch("qmcp_mcp.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.list_human_requests()

        assert "error" in result[0]


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------


class TestRunTests:
    def test_passing_suite(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "5 passed"
        mock_result.stderr = ""

        with patch("qmcp_mcp.subprocess.run", return_value=mock_result):
            result = mcp_mod.run_tests()

        assert result["passed"] is True
        assert result["returncode"] == 0

    def test_failing_suite(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "1 failed"
        mock_result.stderr = ""

        with patch("qmcp_mcp.subprocess.run", return_value=mock_result):
            result = mcp_mod.run_tests()

        assert result["passed"] is False

    def test_specific_path_forwarded(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        captured = []

        def capture(cmd, **kwargs):
            captured.extend(cmd)
            return mock_result

        with patch("qmcp_mcp.subprocess.run", side_effect=capture):
            mcp_mod.run_tests(test_path="tests/test_cookbook.py")

        assert "tests/test_cookbook.py" in captured

    def test_verbose_flag_forwarded(self):
        mock_result = MagicMock(returncode=0, stdout="", stderr="")
        captured = []

        def capture(cmd, **kwargs):
            captured.extend(cmd)
            return mock_result

        with patch("qmcp_mcp.subprocess.run", side_effect=capture):
            mcp_mod.run_tests(verbose=True)

        assert "-v" in captured

    def test_timeout_returns_error(self):
        with patch("qmcp_mcp.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            result = mcp_mod.run_tests()

        assert "error" in result
        assert "timed out" in result["error"]
