"""The approval-gated alfred render: queueing, following the render, the
approval's prompt, waiting for the answer, and one run end to end through a
qmcp server and its human queue.

alfred is the stand-in in `tests/alfred_stub.py`. The flow in
`examples/flows/alfred_render.py` cannot be imported where Metaflow cannot
(`import fcntl`), so its steps are tested here through
`qmcp.cookbook.alfred_render`, which it runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import qmcp.tools.alfred as alfred
from qmcp.client import MCPClient
from qmcp.cookbook import alfred_render as flow
from qmcp.cookbook.alfred_render import (
    FAILED,
    NO_RENDERER,
    RENDERED,
    TIMED_OUT,
    ToolFailedError,
    Waited,
)
from tests.alfred_stub import FakeAlfred, connect

REPO = Path(__file__).resolve().parents[1]


# --- the module and the flow file ----------------------------------------------


def test_the_steps_import_nothing_from_metaflow():
    """Asserted on the source, because the failure is an import line existing."""
    source = Path(flow.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]
    assert not [line for line in imports if "metaflow" in line]


def test_the_flow_file_compiles_and_runs_the_module_s_steps():
    """Compiled rather than imported: importing it imports Metaflow."""
    path = REPO / "examples" / "flows" / "alfred_render.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    for name in ("queue_render", "wait_for_render", "approval_request", "ask",
                 "wait_for_answer", "summary"):
        assert f"{name}(" in source, name


# --- calling a tool --------------------------------------------------------------


def test_the_invoker_returns_a_result_and_raises_a_tool_s_error():
    answers = {"good": SimpleNamespace(result={"ok": 1}, error=None),
               "bad": SimpleNamespace(result=None, error="Tool execution failed: no")}
    sent = []

    class Client:
        def invoke_tool(self, tool, params, correlation_id=None):
            sent.append((tool, params, correlation_id))
            return answers[tool]

    invoke = flow.invoker(Client(), "run-1")
    assert invoke("good", {"a": 1}) == {"ok": 1}
    with pytest.raises(ToolFailedError, match="Tool execution failed: no"):
        invoke("bad", {})
    assert sent == [("good", {"a": 1}, "run-1"), ("bad", {}, "run-1")]


def test_queueing_sends_only_the_options_given():
    sent = []

    def invoke(tool, params):
        sent.append((tool, params))
        return {"render_id": "r1"}

    assert flow.queue_render(invoke, "p1", width=640, height=None, quality="draft") == {
        "render_id": "r1"}
    assert sent == [("alfred_render_enqueue",
                     {"project_id": "p1", "width": 640, "quality": "draft"})]


# --- following the render ----------------------------------------------------------


class Clock:
    """Seconds that pass only when slept through."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Tools:
    """Tool answers in order: each status read takes the next state, and the
    last one repeats; `health` is alfred's report, or an error to raise."""

    def __init__(self, *states, health=None) -> None:
        self.states = list(states)
        self.health = health
        self.calls: list[str] = []

    def __call__(self, tool: str, params: dict):
        self.calls.append(tool)
        if tool == "alfred_render_status":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            status = {"render_id": params["render_id"], "state": state, "progress": None,
                      "error": None, "download_url": None, "download_expires_at": None}
            return {**status, **state} if isinstance(state, dict) else status
        if tool == "alfred_health":
            if isinstance(self.health, Exception):
                raise self.health
            return self.health
        raise AssertionError(f"unexpected tool {tool}")


def follow(tools: Tools, **limits) -> tuple[Waited, Clock]:
    clock = Clock()
    waited = flow.wait_for_render(tools, "r1", poll_interval=5, sleep=clock.sleep,
                                  clock=clock, **limits)
    return waited, clock


def test_a_render_is_followed_until_it_is_rendered():
    tools = Tools("queued", "rendering", "rendered")
    waited, clock = follow(tools)
    assert (waited.outcome, waited.rendered, waited.seconds) == (RENDERED, True, 10)
    assert clock.slept == [5, 5]
    assert "alfred_health" not in tools.calls


def test_a_failed_render_ends_the_wait_with_alfred_s_reason():
    waited, _ = follow(Tools("rendering", {"state": "failed", "error": "clip 2: no such file"}))
    assert (waited.outcome, waited.why) == (FAILED, "clip 2: no such file")


def test_a_render_queued_with_no_worker_running_ends_early_as_no_renderer():
    """Mutation: never ask `alfred_health` -- red, the wait runs to its timeout."""
    tools = Tools("queued", health={"renderer": {"ok": False, "workers": 0}})
    waited, _ = follow(tools, queued_timeout=60, timeout=1800)
    assert (waited.outcome, waited.seconds) == (NO_RENDERER, 60)
    assert waited.why == "still queued after 60 s, and alfred reports no render worker running"
    assert tools.calls.count("alfred_health") == 1


def test_an_alfred_without_a_health_report_is_waited_on_then_read_as_no_renderer():
    """Mutation: read a failed health call as no workers -- red, it ends at 60 s."""
    tools = Tools("queued", health=ToolFailedError("alfred refused GET /health (404): Not Found"))
    waited, _ = follow(tools, queued_timeout=60, timeout=120)
    assert (waited.outcome, waited.seconds) == (NO_RENDERER, 120)
    assert waited.why.endswith("alfred does not report its workers")
    assert tools.calls.count("alfred_health") == 1


def test_a_render_queued_behind_running_workers_times_out_rather_than_blaming_them():
    waited, _ = follow(Tools("queued", health={"renderer": {"ok": True, "workers": 2}}),
                       queued_timeout=60, timeout=120)
    assert (waited.outcome, waited.why) == (TIMED_OUT, "still queued after 120 s")


def test_a_render_still_rendering_at_the_timeout_says_how_far_it_got():
    waited, _ = follow(Tools({"state": "rendering", "progress": 40.4}), timeout=30)
    assert (waited.outcome, waited.why) == (TIMED_OUT, "still rendering at 40% after 30 s")


def test_a_status_the_tool_cannot_read_is_raised():
    def refuses(tool, params):
        raise ToolFailedError("alfred refused GET /render/r1/status (404): Item not found")

    with pytest.raises(ToolFailedError, match="Item not found"):
        flow.wait_for_render(refuses, "r1", sleep=lambda s: None)


# --- what a person is asked ----------------------------------------------------------


QUEUED = {"render_id": "r1", "project_id": "p1", "name": "Demo", "width": 1280,
          "height": 720, "fps": 30.0, "quality": None}
LINK = "http://alfred.test/render/r1/download?exp=1792000000&sig=s"


def rendered(**status) -> Waited:
    return Waited(RENDERED, {"render_id": "r1", "state": "rendered", **status}, 12.0)


def test_the_approval_carries_the_link_and_when_it_stops_working():
    request = flow.approval_request(
        QUEUED, rendered(download_url=LINK, download_expires_at="2026-10-14T17:46:40+00:00"),
        "1729")
    assert request["request_id"] == "alfred-render-r1-1729"
    assert request["options"] == ["approve", "reject"]
    assert request["prompt"] == (
        f"Approve the render of 'Demo'? Watch it at {LINK} -- the link works until "
        "2026-10-14T17:46:40+00:00; alfred_render_status for render r1 issues a fresh one.")
    assert request["context"] == {
        "render_id": "r1", "project_id": "p1", "name": "Demo", "download_url": LINK,
        "download_expires_at": "2026-10-14T17:46:40+00:00", "width": 1280, "height": 720,
        "quality": None, "flow_run_id": "1729"}


def test_without_a_link_the_approval_says_where_the_render_is():
    request = flow.approval_request(QUEUED, rendered(), "1729")
    assert request["prompt"] == ("Approve the render of 'Demo'? alfred issued no download "
                                 "link; it is render r1 in alfred.")
    assert request["context"]["download_url"] is None


# --- waiting for the answer ------------------------------------------------------------


class Queue:
    """The human queue as the wait sees it: the pending listing carries the
    request for its first `pending_rounds` reads, behind `ahead` others."""

    def __init__(self, answer: str | None, pending_rounds: int = 0, ahead: int = 0) -> None:
        self.answer = answer
        self.pending_rounds = pending_rounds
        self.ahead = ahead
        self.rounds = 0
        self.pages = 0
        self.reads = 0

    def list_human_requests(self, status_filter, limit, offset, **_):
        assert status_filter == "pending"
        self.pages += 1
        if offset == 0:
            self.rounds += 1
        listing = [SimpleNamespace(id=f"other-{n}") for n in range(self.ahead)]
        if self.rounds <= self.pending_rounds:
            listing.append(SimpleNamespace(id="ask-1"))
        return listing[offset:offset + limit]

    def get_human_request(self, request_id):
        self.reads += 1
        response = SimpleNamespace(response=self.answer) if self.answer else None
        return SimpleNamespace(id=request_id), response


def test_an_answer_is_read_once_the_request_leaves_the_pending_listing():
    """Mutation: poll `get_human_request` until it answers -- red, it is read
    on every round."""
    queue, clock = Queue("reject", pending_rounds=3), Clock()
    answer = flow.wait_for_answer(queue, "ask-1", timeout_seconds=600, poll_interval=5,
                                  sleep=clock.sleep, clock=clock)
    assert (answer, queue.rounds, queue.reads, clock.slept) == ("reject", 4, 1, [5, 5, 5])


def test_a_request_behind_a_full_page_is_still_found_pending():
    """Mutation: read only the first page -- red, the request is taken as answered."""
    queue = Queue("approve", pending_rounds=1, ahead=flow.PAGE)
    clock = Clock()
    flow.wait_for_answer(queue, "ask-1", timeout_seconds=600, poll_interval=5,
                         sleep=clock.sleep, clock=clock)
    assert clock.slept == [5]
    assert queue.pages == 4  # two pages a round, two rounds


def test_a_request_nobody_answers_ends_the_wait_after_its_expiry():
    """Mutation: drop the deadline -- red; the wait lasts as long as the listing
    carries the request, which on such a server is for ever."""
    queue, clock = Queue(None, pending_rounds=10**6), Clock()
    answer = flow.wait_for_answer(queue, "ask-1", timeout_seconds=60, poll_interval=5,
                                  sleep=clock.sleep, clock=clock)
    assert answer is None and queue.reads == 1
    assert clock.now == 75  # the expiry and two polls, then one more look


# --- how a run ends ----------------------------------------------------------------------


@pytest.mark.parametrize("waited,answer,line", [
    (rendered(download_url=LINK), "approve", f"Approved: render r1 of 'Demo', {LINK}."),
    (rendered(), "approve", "Approved: render r1 of 'Demo', in alfred."),
    (rendered(), "reject", "Rejected: render r1 of 'Demo'."),
    (rendered(), None, "Nobody answered before the approval of render r1 of 'Demo' expired."),
    (Waited(FAILED, {}, 3, "clip 2: no such file"), None,
     "Nothing was asked: render r1 of 'Demo' failed: clip 2: no such file."),
    (Waited(NO_RENDERER, {}, 60, "still queued after 60 s"), None,
     "Nothing was asked: render r1 of 'Demo' was not rendered: still queued after 60 s."),
    (Waited(TIMED_OUT, {}, 30, "still rendering after 30 s"), None,
     "Nothing was asked: render r1 of 'Demo' did not finish in time: "
     "still rendering after 30 s."),
])
def test_the_summary_says_how_the_run_ended(waited, answer, line):
    assert flow.summary(QUEUED, waited, answer) == line


# --- end to end, through a qmcp server and its queue ---------------------------------------


def test_a_rendered_project_reaches_a_person_and_their_answer_ends_the_run(
        tmp_path, monkeypatch):
    """Through a qmcp server on an ephemeral port over its own database: the
    tools are invoked over HTTP, so every call to alfred is in the server's
    record under the run's correlation id, and the approval is answered on the
    real queue."""
    from qmcp.integrations.voice.check import throwaway_server

    fake = FakeAlfred()
    fake.route("GET", "/project/p1", {"_id": "p1", "name": "Demo",
                                      "edl": {"clips": [{"name": "title"}], "duration": 3.0},
                                      "render_settings": {"width": 1280, "height": 720}})
    fake.route("POST", "/render", '"r1"')
    states = iter([{"state": "queued"}, {"state": "rendering", "progress": 50.0},
                   {"state": "rendered", "progress": 100.0,
                    "download_url": "/render/r1/download?exp=1792000000&sig=s"}])
    fake.route("GET", "/render/r1/status",
               lambda request: httpx.Response(200, json={"id": "r1", **next(states)}))
    connected = connect(fake)
    monkeypatch.setattr(alfred, "client", lambda: connected)

    with throwaway_server(tmp_path / "queue.db") as url, MCPClient(base_url=url) as client:
        invoke = flow.invoker(client, "alfred-render-run-1")
        queued = flow.queue_render(invoke, "p1")
        waited = flow.wait_for_render(invoke, queued["render_id"], poll_interval=0,
                                      sleep=lambda seconds: None)
        request = flow.approval_request(queued, waited, "run-1")
        flow.ask(client, request, timeout_seconds=600, correlation_id="alfred-render-run-1")

        answered = []

        def a_person_approves(seconds: float) -> None:
            if not answered:
                answered.append(client.submit_human_response(
                    request["request_id"], "approve", responded_by="editor@example.com"))

        answer = flow.wait_for_answer(client, request["request_id"], timeout_seconds=600,
                                      poll_interval=0, sleep=a_person_approves)
        recorded = client.list_invocations(limit=10)

    assert (queued["render_id"], queued["width"], queued["height"]) == ("r1", 1280, 720)
    assert (waited.outcome, answer) == (RENDERED, "approve")
    assert flow.summary(queued, waited, answer) == (
        "Approved: render r1 of 'Demo', "
        "http://alfred.test/render/r1/download?exp=1792000000&sig=s.")
    assert [(r["tool_name"], r["status"], r["correlation_id"]) for r in reversed(recorded)] == [
        ("alfred_render_enqueue", "success", "alfred-render-run-1"),
        ("alfred_render_status", "success", "alfred-render-run-1"),
        ("alfred_render_status", "success", "alfred-render-run-1"),
        ("alfred_render_status", "success", "alfred-render-run-1"),
    ]
