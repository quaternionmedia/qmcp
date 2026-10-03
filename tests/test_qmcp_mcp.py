"""Tests for the qmcp meta-MCP server (qmcp_mcp.py).

All network and subprocess calls are mocked so the suite runs fully offline.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
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
# An agent asks: create_human_request, await_human_response, ask_human
# ---------------------------------------------------------------------------


_LISTING = "/v1/human/requests"


class _Queue:
    """A scripted server for the asking tools, standing in for httpx.get.

    `looks` is what the pending listing holds at each look, in order; the last
    entry repeats, so a question that stays pending stays pending. A look is
    one page, not one poll: the listing honours `offset` and `limit` the way
    the server does, so paging is real, and a queue that changes between two
    pages of one poll is scripted as two looks. `detail` is what reading the
    one request returns -- one body, or a list of bodies served in order with
    the last repeating. Every GET is recorded with its URL, so a test can say
    which routes were read and how often. A `listing_status` of 4xx or 5xx is
    a listing the server refuses, and its `raise_for_status` raises the way
    httpx's does -- a MagicMock's raises nothing, which is why it is set here.
    """

    def __init__(self, looks, detail=None, detail_status=200, listing_status=200):
        self.looks = list(looks)
        self.details = list(detail) if isinstance(detail, list) else [detail]
        self.detail_status = detail_status
        self.listing_status = listing_status
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if url.endswith(_LISTING):
            pending = self.looks.pop(0) if len(self.looks) > 1 else self.looks[0]
            offset, limit = params["offset"], params["limit"]
            page = [{"id": rid, "status": "pending"} for rid in pending[offset:offset + limit]]
            resp.status_code = self.listing_status
            resp.json.return_value = {"requests": page, "count": len(page)}
            if self.listing_status >= 400:
                resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                    f"{self.listing_status} on the listing", request=MagicMock(), response=resp
                )
        else:
            resp.status_code = self.detail_status
            served = self.details.pop(0) if len(self.details) > 1 else self.details[0]
            resp.json.return_value = served
        return resp

    def listing_reads(self) -> int:
        return sum(1 for url, _ in self.calls if url.endswith(_LISTING))

    def detail_reads(self) -> int:
        return sum(1 for url, _ in self.calls if not url.endswith(_LISTING))


def _answered(request_id: str, answer: str, by: str = "vox") -> dict:
    return {
        "request": {"id": request_id, "status": "responded"},
        "response": {"request_id": request_id, "response": answer, "responded_by": by},
    }


def _expired(request_id: str) -> dict:
    return {"request": {"id": request_id, "status": "expired"}, "response": None}


def _pending(request_id: str) -> dict:
    return {"request": {"id": request_id, "status": "pending"}, "response": None}


class TestCreateHumanRequest:
    def test_body_sent(self):
        # Seen to fail: with the body's "timeout_seconds" key spelt
        # "expires_in_seconds" instead, the equality below went red.
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"id": "q-1", "status": "pending"}

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp) as post:
            result = mcp_mod.create_human_request(
                "Merge the pin bump?",
                options=["approve", "hold"],
                request_id="q-1",
                expires_in_seconds=120,
                context={"pr": 7},
                server_url="http://localhost:3333",
            )

        assert result["id"] == "q-1"
        assert post.call_args.args == ("http://localhost:3333/v1/human/requests",)
        assert post.call_args.kwargs["json"] == {
            "id": "q-1",
            "request_type": "approval",
            "prompt": "Merge the pin bump?",
            "timeout_seconds": 120,
            "context": {"pr": 7},
            "options": ["approve", "hold"],
        }

    def test_open_question_sends_no_options(self):
        # Seen to fail: with options sent unconditionally (as null), the
        # "not in" below went red; with `context` sent as given (null when
        # none was given, which the server refuses as a 422), the equality
        # went red.
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"id": "q-2"}

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp) as post:
            mcp_mod.create_human_request("Which branch?", request_id="q-2")

        assert "options" not in post.call_args.kwargs["json"]
        assert post.call_args.kwargs["json"]["context"] == {}

    def test_an_empty_option_list_is_an_open_question(self):
        # Seen to fail: with options sent whenever not None, the body carried
        # "options": [] and the "not in" below went red.
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {"id": "q-5"}

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp) as post:
            mcp_mod.create_human_request("Which branch?", options=[], request_id="q-5")

        assert "options" not in post.call_args.kwargs["json"]

    def test_id_made_when_none_given(self):
        mock_resp = MagicMock(status_code=201)
        mock_resp.json.return_value = {}

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp) as post:
            mcp_mod.create_human_request("Continue?")

        made = post.call_args.kwargs["json"]["id"]
        assert made.startswith("ask-")
        # Date, time and milliseconds: readable in a listing, and apart for
        # two questions asked within one second.
        date, clock, millis = made.removeprefix("ask-").split("-")
        assert (len(date), len(clock), len(millis)) == (8, 6, 3)

    def test_duplicate_id_is_an_error(self):
        mock_resp = MagicMock(status_code=409)

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.create_human_request("Again?", request_id="q-1")

        assert "already exists" in result["error"]

    def test_refused_body_carries_the_servers_words(self):
        # Seen to fail: with the 422 branch removed, raise_for_status on a
        # MagicMock raises nothing and the result had no "error" key.
        mock_resp = MagicMock(status_code=422)
        mock_resp.json.return_value = {
            "detail": [{"loc": ["body", "timeout_seconds"], "msg": "too short"}]
        }

        with patch("qmcp_mcp.httpx.post", return_value=mock_resp):
            result = mcp_mod.create_human_request("Quick?", request_id="q-3", expires_in_seconds=1)

        assert "timeout_seconds" in result["error"]

    def test_unreachable_returns_error(self):
        import httpx

        with patch("qmcp_mcp.httpx.post", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.create_human_request("Anyone?", request_id="q-4")

        assert "error" in result


class TestAwaitHumanResponse:
    def test_polls_the_listing_only_while_pending(self):
        # Seen to fail: with the detail route read on every look (the loop
        # polling GET /v1/human/requests/{id}), detail_reads() was 3 and the
        # ordering assertion went red first.
        queue = _Queue(looks=[["q-1"], ["q-1"], []], detail=_answered("q-1", "approve"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep") as sleep:
            result = mcp_mod.await_human_response("q-1", timeout_seconds=60, poll_seconds=2.0)

        assert result["status"] == "answered"
        assert result["response"]["response"] == "approve"
        # Three looks at the listing, then one read -- and nothing read
        # while the id was still listed.
        routes = [url.rsplit("/v1/", 1)[1] for url, _ in queue.calls]
        assert routes == [
            "human/requests", "human/requests", "human/requests", "human/requests/q-1",
        ]
        assert sleep.call_count == 2

    def test_the_single_read_happens_exactly_once(self):
        # Seen to fail: with the detail read duplicated after the loop,
        # detail_reads() was 2.
        queue = _Queue(looks=[[]], detail=_answered("q-1", "hold", by="walkthrough"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=60)

        assert queue.detail_reads() == 1
        assert queue.listing_reads() == 1
        assert result["response"]["responded_by"] == "walkthrough"

    def test_expired(self):
        # Seen to fail: with every non-pending status reported as
        # "answered", the status below read "answered".
        queue = _Queue(looks=[["q-1"], []], detail=_expired("q-1"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=60)

        assert result == {"status": "expired", "request_id": "q-1", "response": None}
        assert queue.detail_reads() == 1

    def test_timeout_reads_nothing(self):
        # Seen to fail: with the detail route read on the way out of a
        # timeout, detail_reads() was 1 -- and that read is the one that
        # would expire a question somebody could still have answered.
        queue = _Queue(looks=[["q-1"]], detail=_answered("q-1", "approve"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep") as sleep:
            result = mcp_mod.await_human_response("q-1", timeout_seconds=0)

        assert result == {"status": "timeout", "request_id": "q-1", "response": None}
        assert queue.detail_reads() == 0
        assert sleep.call_count == 0

    def test_a_sleep_never_outlasts_the_deadline(self):
        # Seen to fail: with `poll_seconds` slept as given rather than the
        # smaller of it and the time remaining, the sleep was 2.0.
        queue = _Queue(looks=[["q-1"]], detail=_answered("q-1", "approve"))
        clock = iter([100.0, 100.0, 100.5, 101.0])

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep") as sleep, \
             patch("qmcp_mcp.time.monotonic", side_effect=lambda: next(clock)):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=0.5, poll_seconds=2.0)

        assert result["status"] == "timeout"
        assert sleep.call_args.args == (0.5,)

    def test_pages_past_the_first_page(self):
        # Seen to fail: with `_pending_ids` returning after its first page,
        # an id listed on the second page was taken as answered, the detail
        # was read, and the status below read "answered".
        deep = [f"other-{n}" for n in range(mcp_mod._PENDING_PAGE)] + ["q-1"]
        queue = _Queue(looks=[deep], detail=_answered("q-1", "approve"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=0)

        assert result["status"] == "timeout"
        assert queue.detail_reads() == 0
        offsets = [params["offset"] for _, params in queue.calls]
        assert offsets == [0, mcp_mod._PENDING_PAGE]

    def test_an_id_on_the_first_page_of_a_deep_listing_is_kept(self):
        # Seen to fail: with `_pending_ids` building its set afresh from each
        # page (`ids = {...}` for `ids.update(...)`), so that every page but
        # the last was dropped, the id on page one was taken as answered, the
        # detail was read, and the status below read "answered". The test
        # above cannot see that: its only datum sits on page two.
        deep = ["q-1"] + [f"other-{n}" for n in range(mcp_mod._PENDING_PAGE)]
        queue = _Queue(looks=[deep], detail=_answered("q-1", "approve"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=0)

        assert result["status"] == "timeout"
        assert queue.detail_reads() == 0
        offsets = [params["offset"] for _, params in queue.calls]
        assert offsets == [0, mcp_mod._PENDING_PAGE]

    def test_an_id_that_slips_between_two_pages_is_still_waited_on(self):
        # Seen to fail: with the one read's "pending" returned as the result,
        # the status below read "pending" and the response was None -- the
        # wait had ended early and handed an unanswered question back as its
        # final word.
        #
        # The queue is deeper than one page and an older question is answered
        # between the wait's first page and its second, so the awaited id
        # moves from the first slot of page two to the last slot of page one
        # and is on neither page the wait read.
        others = [f"other-{n}" for n in range(mcp_mod._PENDING_PAGE)]
        looks = [others + ["q-1"], others[1:] + ["q-1"], []]
        queue = _Queue(looks=looks, detail=[_pending("q-1"), _answered("q-1", "approve")])

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep") as sleep:
            result = mcp_mod.await_human_response("q-1", timeout_seconds=60, poll_seconds=2.0)

        assert result["status"] == "answered"
        assert result["response"]["response"] == "approve"
        # Two pages, the read that found it still pending, a sleep, one more
        # look at the listing (now empty) and the read that found the answer.
        routes = [url.rsplit("/v1/", 1)[1] for url, _ in queue.calls]
        assert routes == [
            "human/requests", "human/requests", "human/requests/q-1",
            "human/requests", "human/requests/q-1",
        ]
        assert sleep.call_count == 1

    def test_a_pending_read_at_the_deadline_is_a_timeout(self):
        # Seen to fail: with the wait resuming after a "pending" read without
        # looking at the clock, the listing was read again after the deadline
        # and the sleep count was 1.
        others = [f"other-{n}" for n in range(mcp_mod._PENDING_PAGE)]
        looks = [others + ["q-1"], others[1:] + ["q-1"]]
        queue = _Queue(looks=looks, detail=_pending("q-1"))

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep") as sleep:
            result = mcp_mod.await_human_response("q-1", timeout_seconds=0)

        assert result == {"status": "timeout", "request_id": "q-1", "response": None}
        assert queue.detail_reads() == 1
        assert sleep.call_count == 0

    def test_a_listing_the_server_refuses_is_an_error(self):
        # Seen to fail: with `raise_for_status` dropped from `_pending_ids`,
        # the refused page was read as if the server had accepted it and the
        # wait ran on to its deadline, so the status below read "timeout" --
        # the comment at `_PENDING_PAGE` says a refusal fails loudly, and
        # this is what holds it to that.
        queue = _Queue(looks=[["q-1"]], detail=_answered("q-1", "approve"), listing_status=422)

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=0)

        assert result["status"] == "error"
        assert "422" in result["error"]
        assert queue.detail_reads() == 0

    def test_not_found(self):
        queue = _Queue(looks=[[]], detail=None, detail_status=404)

        with patch("qmcp_mcp.httpx.get", side_effect=queue.get), patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.await_human_response("nobody", timeout_seconds=60)

        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_unreachable_returns_error(self):
        import httpx

        with patch("qmcp_mcp.httpx.get", side_effect=httpx.ConnectError("refused")):
            result = mcp_mod.await_human_response("q-1", timeout_seconds=60)

        assert result["status"] == "error"


class TestAskHuman:
    def test_creates_then_waits(self):
        # Seen to fail: with the created request's expiry fixed at the
        # tool's default instead of the caller's timeout, the body's
        # "timeout_seconds" read 600.
        created = MagicMock(status_code=201)
        created.json.return_value = {"id": "ask-x", "status": "pending"}
        queue = _Queue(looks=[["ask-x"], []], detail=_answered("ask-x", "approve"))

        with patch("qmcp_mcp.httpx.post", return_value=created) as post, \
             patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.ask_human("Merge?", options=["approve", "hold"], timeout_seconds=90)

        body = post.call_args.kwargs["json"]
        assert body["request_type"] == "approval"
        assert body["options"] == ["approve", "hold"]
        assert body["timeout_seconds"] == 90
        assert result["status"] == "answered"
        assert result["request_id"] == "ask-x"
        assert queue.detail_reads() == 1

    def test_an_open_question_is_an_input(self):
        # Seen to fail: with request_type fixed at "approval", the body's
        # type read "approval" for a question with no options.
        created = MagicMock(status_code=201)
        created.json.return_value = {"id": "ask-y"}
        queue = _Queue(looks=[[]], detail=_answered("ask-y", "the pin, not the lock"))

        with patch("qmcp_mcp.httpx.post", return_value=created) as post, \
             patch("qmcp_mcp.httpx.get", side_effect=queue.get), \
             patch("qmcp_mcp.time.sleep"):
            result = mcp_mod.ask_human("Which one?")

        body = post.call_args.kwargs["json"]
        assert body["request_type"] == "input"
        assert "options" not in body
        assert result["response"]["response"] == "the pin, not the lock"

    def test_a_refused_question_is_not_waited_on(self):
        refused = MagicMock(status_code=409)

        with patch("qmcp_mcp.httpx.post", return_value=refused), \
             patch("qmcp_mcp.httpx.get") as get:
            result = mcp_mod.ask_human("Again?")

        assert "error" in result
        assert get.call_count == 0


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
