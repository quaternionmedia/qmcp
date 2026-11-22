"""Render an alfred project, and ask a person to approve what it rendered.

The project's saved timeline is queued in alfred, followed until it renders,
and put to a person on qmcp's human queue with its download link. Every call
to alfred is one of qmcp's alfred tools, made through the qmcp server, so the
server must be configured for alfred (docs/integrations/alfred.md). The steps
are `qmcp.cookbook.alfred_render`.

Usage:
    # The server, with alfred configured
    QMCP_ALFRED_URL=http://127.0.0.1:1732 uv run qmcp serve

    # In another terminal, the flow
    uv run python examples/flows/alfred_render.py run --project-id <id>

    # Once the render is ready the flow prints the request and waits:
    uv run qmcp human respond <request id> approve      # or reject
"""

import os

from metaflow import FlowSpec, Parameter, current, step

from qmcp.client import MCPClient
from qmcp.client.mcp_client import default_base_url
from qmcp.cookbook.alfred_render import (
    approval_request,
    ask,
    invoker,
    queue_render,
    summary,
    wait_for_answer,
    wait_for_render,
)


class AlfredRenderFlow(FlowSpec):
    """Queue an alfred render, follow it, and ask a person to approve it.

    Steps:
    1. start: check the qmcp server answers
    2. enqueue: queue the project's saved timeline in alfred
    3. follow: wait until the render is rendered or failed, or no renderer runs
    4. ask_person: put the render, with its link, to a person and wait
    5. end: say how it ended
    """

    project_id = Parameter("project-id", help="The alfred project to render", required=True)
    width = Parameter("width", help="Frame width; the project's saved size by default", type=int)
    height = Parameter("height", help="Frame height; the project's saved size by default",
                       type=int)
    quality = Parameter("quality", help="draft, standard or high; alfred's default if unset",
                        type=str)
    mcp_url = Parameter(
        "mcp-url",
        help="URL of the qmcp server",
        default=os.getenv("MCP_URL") or default_base_url(),
    )
    render_timeout = Parameter(
        "render-timeout", help="Seconds to wait for the render", default=1800, type=int
    )
    queued_timeout = Parameter(
        "queued-timeout",
        help="Seconds queued before asking alfred whether a renderer is running",
        default=60,
        type=int,
    )
    poll_interval = Parameter(
        "poll-interval", help="Seconds between status reads", default=5.0, type=float
    )
    approval_timeout = Parameter(
        "approval-timeout", help="Seconds the approval may wait for an answer", default=3600,
        type=int,
    )

    def _client(self) -> MCPClient:
        return MCPClient(self.mcp_url)

    def _correlation(self) -> str:
        return f"alfred-render-{self.run_id}"

    @step
    def start(self):
        """Check the qmcp server answers."""
        self.run_id = current.run_id
        with self._client() as client:
            print(f"qmcp at {self.mcp_url}: {client.health()['status']}")
        self.next(self.enqueue)

    @step
    def enqueue(self):
        """Queue the project's saved timeline in alfred."""
        with self._client() as client:
            self.queued = queue_render(
                invoker(client, self._correlation()),
                self.project_id,
                width=self.width,
                height=self.height,
                quality=self.quality,
            )
        print(f"Queued render {self.queued['render_id']} of {self.queued['name']!r}")
        self.next(self.follow)

    @step
    def follow(self):
        """Wait until the render is rendered or failed, or no renderer runs."""
        with self._client() as client:
            self.waited = wait_for_render(
                invoker(client, self._correlation()),
                self.queued["render_id"],
                timeout=self.render_timeout,
                queued_timeout=self.queued_timeout,
                poll_interval=self.poll_interval,
            )
        why = f" -- {self.waited.why}" if self.waited.why else ""
        print(f"After {self.waited.seconds:.0f} s the wait ended: {self.waited.outcome}{why}")
        self.next(self.ask_person)

    @step
    def ask_person(self):
        """Put a rendered render, with its link, to a person, and wait for the answer."""
        self.answer = None
        self.request_id = None
        if self.waited.rendered:
            request = approval_request(self.queued, self.waited, self.run_id)
            self.request_id = request["request_id"]
            with self._client() as client:
                ask(client, request, timeout_seconds=self.approval_timeout,
                    correlation_id=self._correlation())
                print(request["prompt"])
                print(f"Answer with: uv run qmcp human respond {self.request_id} approve"
                      " (or reject)")
                self.answer = wait_for_answer(
                    client,
                    self.request_id,
                    timeout_seconds=self.approval_timeout,
                    poll_interval=self.poll_interval,
                )
        self.next(self.end)

    @step
    def end(self):
        """Say how it ended."""
        self.outcome = summary(self.queued, self.waited, self.answer)
        print(self.outcome)


if __name__ == "__main__":
    AlfredRenderFlow()
