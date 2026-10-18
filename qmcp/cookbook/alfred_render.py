"""Rendering an alfred project behind a person's approval: the steps, apart from any flow.

`examples/flows/alfred_render.py` runs them as a Metaflow flow, one step each:

    queue    `alfred_render_enqueue` for a project's saved timeline
    wait     `alfred_render_status` until the render settles, sits queued
             with no renderer running, or runs out of time
    ask      an approval request on qmcp's human queue, carrying the
             render's download link
    answer   the pending listing watched until the request leaves it, then
             the request read once

Every call to alfred is a qmcp tool invocation, so it is in qmcp's record.
The steps take the tool caller (`Invoke`) and the queue client as arguments,
and a wait takes its `sleep` and `clock`, so a test drives them without a
flow. Nothing here imports Metaflow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Invoke = Callable[[str, dict[str, Any]], Any]
"""Calls one qmcp tool by name with its input and returns the tool's result;
raises `ToolFailedError` with the tool's error."""

OPTIONS = ["approve", "reject"]
APPROVE, REJECT = OPTIONS

# How a wait for a render ends.
RENDERED = "rendered"
FAILED = "failed"
NO_RENDERER = "no renderer"
TIMED_OUT = "timed out"

# How the pending listing is read: pages of the server's own cap, so a queue
# longer than one page is still searched to its end.
PAGE = 500


class ToolFailedError(RuntimeError):
    """A tool answered with an error; the message is the error the server recorded."""


def invoker(client: Any, correlation_id: str | None = None) -> Invoke:
    """`Invoke` through a `qmcp.client.MCPClient`, every call under `correlation_id`."""

    def invoke(tool: str, params: dict[str, Any]) -> Any:
        answer = client.invoke_tool(tool, params, correlation_id=correlation_id)
        if answer.error:
            raise ToolFailedError(answer.error)
        return answer.result

    return invoke


def queue_render(invoke: Invoke, project_id: str, **options: Any) -> dict[str, Any]:
    """Queue a render of a project's saved timeline; `options` left as None are
    the project's saved settings, else alfred's. Returns what the tool returned."""
    params = {"project_id": project_id}
    params.update({key: value for key, value in options.items() if value is not None})
    return invoke("alfred_render_enqueue", params)


@dataclass(frozen=True)
class Waited:
    """How a wait for a render ended: `outcome` is one of `RENDERED`, `FAILED`,
    `NO_RENDERER` and `TIMED_OUT`; `status` is the render's last status;
    `why` is one line for a person, empty when it rendered."""

    outcome: str
    status: dict[str, Any]
    seconds: float
    why: str = ""

    @property
    def rendered(self) -> bool:
        return self.outcome == RENDERED


def renderer_workers(invoke: Invoke) -> int | None:
    """How many render workers alfred reports running, or None when it does not say."""
    try:
        report = invoke("alfred_health", {})
    except ToolFailedError:
        return None
    renderer = report.get("renderer") if isinstance(report, dict) else None
    workers = renderer.get("workers") if isinstance(renderer, dict) else None
    if isinstance(workers, bool) or not isinstance(workers, int):
        return None
    return workers


def wait_for_render(
    invoke: Invoke,
    render_id: str,
    *,
    timeout: float = 1800.0,
    queued_timeout: float = 60.0,
    poll_interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Waited:
    """Follow a render with `alfred_render_status` until it is rendered or failed.

    A render still queued after `queued_timeout` seconds ends the wait as
    `NO_RENDERER` when `alfred_health` reports no worker running; a report of
    workers, or none, keeps it waiting. After `timeout` seconds the wait ends
    as `TIMED_OUT`, or as `NO_RENDERER` for a render that never left the queue
    of an alfred that does not report its workers. Each status is one tool
    invocation; a tool error raises `ToolFailedError`.
    """
    started = clock()
    workers: int | None = None
    asked = False
    while True:
        status = invoke("alfred_render_status", {"render_id": render_id})
        waited = clock() - started
        state = status.get("state")
        if state == RENDERED:
            return Waited(RENDERED, status, waited)
        if state == FAILED:
            return Waited(FAILED, status, waited, status.get("error") or "alfred gave no reason")
        if state == "queued" and waited >= queued_timeout and not asked:
            asked = True
            workers = renderer_workers(invoke)
            if workers == 0:
                return Waited(NO_RENDERER, status, waited,
                              f"still queued after {waited:.0f} s, and alfred reports "
                              "no render worker running")
        if waited >= timeout:
            if state == "queued" and workers is None:
                return Waited(NO_RENDERER, status, waited,
                              f"never left the queue in {waited:.0f} s, so no renderer "
                              "appears to be running; alfred does not report its workers")
            progress = status.get("progress")
            at = f" at {progress:.0f}%" if isinstance(progress, (int, float)) else ""
            return Waited(TIMED_OUT, status, waited, f"still {state}{at} after {waited:.0f} s")
        sleep(poll_interval)


def approval_request(queued: dict[str, Any], waited: Waited, run_id: str) -> dict[str, Any]:
    """What a person is asked about a rendered render: the request's id, prompt,
    options and context, as `MCPClient.create_human_request` takes them.

    The prompt carries the download link when alfred issued one, and when it
    stops working; `alfred_render_status` issues a fresh link after that.
    """
    render_id = queued["render_id"]
    name = queued.get("name") or queued.get("project_id") or render_id
    link = waited.status.get("download_url")
    expires = waited.status.get("download_expires_at")
    if link:
        where = f"Watch it at {link}"
        if expires:
            where += (f" -- the link works until {expires}; alfred_render_status for "
                      f"render {render_id} issues a fresh one")
    else:
        where = f"alfred issued no download link; it is render {render_id} in alfred"
    return {
        "request_id": f"alfred-render-{render_id}-{run_id}",
        "prompt": f"Approve the render of {name!r}? {where}.",
        "options": list(OPTIONS),
        "context": {
            "render_id": render_id,
            "project_id": queued.get("project_id"),
            "name": name,
            "download_url": link,
            "download_expires_at": expires,
            "width": queued.get("width"),
            "height": queued.get("height"),
            "quality": queued.get("quality"),
            "flow_run_id": run_id,
        },
    }


def ask(client: Any, request: dict[str, Any], *, timeout_seconds: int,
        correlation_id: str | None = None) -> None:
    """Put `request` (from `approval_request`) on the human queue as an approval
    that expires after `timeout_seconds`."""
    client.create_human_request(
        request_id=request["request_id"],
        request_type="approval",
        prompt=request["prompt"],
        options=request["options"],
        context=request["context"],
        timeout_seconds=timeout_seconds,
        correlation_id=correlation_id,
    )


def _is_pending(client: Any, request_id: str) -> bool:
    """Whether the pending listing still carries the request. Reads nothing else."""
    offset = 0
    while True:
        page = client.list_human_requests(status_filter="pending", limit=PAGE, offset=offset)
        if any(r.id == request_id for r in page):
            return True
        if len(page) < PAGE:
            return False
        offset += PAGE


def wait_for_answer(
    client: Any,
    request_id: str,
    *,
    timeout_seconds: float,
    poll_interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str | None:
    """The answer given to `request_id`, or None when nobody answered before it expired.

    Watches the pending listing, which expires nothing, until the request
    leaves it, then reads the request once: reading the request itself marks
    it expired once its time has passed (AGENTS.md, "One read in this API is a
    write"). A deadline two polls past `timeout_seconds` on this machine's
    clock ends the watch on a server whose listing never drops the request.
    """
    deadline = clock() + timeout_seconds + 2 * poll_interval
    while _is_pending(client, request_id):
        if clock() > deadline:
            break
        sleep(poll_interval)
    _, response = client.get_human_request(request_id)
    return response.response if response is not None else None


def summary(queued: dict[str, Any], waited: Waited, answer: str | None) -> str:
    """One line saying how the run ended."""
    render = f"render {queued['render_id']} of {queued.get('name')!r}"
    if not waited.rendered:
        ended = {FAILED: "failed", TIMED_OUT: "did not finish in time"}.get(
            waited.outcome, "was not rendered")
        return f"Nothing was asked: {render} {ended}: {waited.why}."
    if answer == APPROVE:
        link = waited.status.get("download_url") or "in alfred"
        return f"Approved: {render}, {link}."
    if answer == REJECT:
        return f"Rejected: {render}."
    return f"Nobody answered before the approval of {render} expired."


__all__ = [
    "APPROVE",
    "FAILED",
    "NO_RENDERER",
    "OPTIONS",
    "REJECT",
    "RENDERED",
    "TIMED_OUT",
    "Invoke",
    "ToolFailedError",
    "Waited",
    "approval_request",
    "ask",
    "invoker",
    "queue_render",
    "renderer_workers",
    "summary",
    "wait_for_answer",
    "wait_for_render",
]
