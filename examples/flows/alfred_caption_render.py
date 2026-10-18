"""Caption an alfred project, and render it once a person has read the captions.

alfred transcribes the project's saved timeline with its speech engine; a
person reads the transcript on qmcp's human queue and approves or rejects it;
an approved transcript is burned into a render, which is followed until it
renders and put to a person with its download link; and an approved render is
delivered, when the run is asked to deliver. Every call to alfred is one of
qmcp's alfred tools, made through the qmcp server, so the server must be
configured for alfred (docs/integrations/alfred.md). The steps are
`qmcp.cookbook.alfred_captions`.

Usage:
    # The server, with alfred configured
    QMCP_ALFRED_URL=http://localhost:8000 uv run qmcp serve

    # In another terminal, the flow
    uv run python examples/flows/alfred_caption_render.py run --project-id <id>
    uv run python examples/flows/alfred_caption_render.py run --project-id <id> \\
        --deliver True --destination local

    # The flow prints each request as it asks, and waits:
    uv run qmcp human respond <request id> approve      # or reject
"""

import os

from metaflow import FlowSpec, Parameter, current, step

from qmcp.client import MCPClient
from qmcp.client.mcp_client import default_base_url
from qmcp.cookbook.alfred_captions import (
    approval_request,
    deliver_render,
    review_request,
    summary,
    transcribe,
)
from qmcp.cookbook.alfred_render import (
    APPROVE,
    ask,
    invoker,
    queue_render,
    wait_for_answer,
    wait_for_render,
)


class AlfredCaptionRenderFlow(FlowSpec):
    """Transcribe an alfred project, have a person read it, render it with the
    captions burned in, have a person approve the render, and deliver it.

    Steps:
    1. start: check the qmcp server answers
    2. transcribe: transcribe the project's timeline, and wait for the captions
    3. review: put the transcript to a person and wait
    4. render: queue the render with the captions burned in, once approved
    5. follow: wait until the render is rendered or failed, or no renderer runs
    6. ask_person: put the render, with its link, to a person and wait
    7. delivery: deliver the approved render, when --deliver is set
    8. end: say how it ended
    """

    project_id = Parameter("project-id", help="The alfred project to caption", required=True)
    width = Parameter("width", help="Frame width; the project's saved size by default", type=int)
    height = Parameter("height", help="Frame height; the project's saved size by default",
                       type=int)
    quality = Parameter("quality", help="draft, standard or high; alfred's default if unset",
                        type=str)
    deliver = Parameter(
        "deliver", help="Deliver the render once a person approves it", type=bool, default=False
    )
    destination = Parameter(
        "destination", help="Where to deliver: a destination alfred offers; alfred's own "
        "default (where the render last went, else local) if unset", type=str,
    )
    mcp_url = Parameter(
        "mcp-url",
        help="URL of the qmcp server",
        default=os.getenv("MCP_URL") or default_base_url(),
    )
    transcribe_timeout = Parameter(
        "transcribe-timeout", help="Seconds to wait for the transcription", default=900,
        type=int,
    )
    render_timeout = Parameter(
        "render-timeout", help="Seconds to wait for the render", default=1800, type=int
    )
    queued_timeout = Parameter(
        "queued-timeout",
        help="Seconds queued before asking alfred whether a worker is running",
        default=60,
        type=int,
    )
    poll_interval = Parameter(
        "poll-interval", help="Seconds between reads of alfred and of the queue", default=5.0,
        type=float,
    )
    review_timeout = Parameter(
        "review-timeout", help="Seconds the transcript review may wait for an answer",
        default=3600, type=int,
    )
    approval_timeout = Parameter(
        "approval-timeout", help="Seconds the render's approval may wait for an answer",
        default=3600, type=int,
    )

    def _client(self) -> MCPClient:
        return MCPClient(self.mcp_url)

    def _correlation(self) -> str:
        return f"alfred-caption-render-{self.run_id}"

    def _ask(self, client: MCPClient, request: dict, timeout: int) -> str | None:
        """Put `request` to a person, print it, and wait for the answer."""
        ask(client, request, timeout_seconds=timeout, correlation_id=self._correlation())
        print(request["prompt"])
        print(f"Answer with: uv run qmcp human respond {request['request_id']} approve"
              " (or reject)")
        return wait_for_answer(client, request["request_id"], timeout_seconds=timeout,
                               poll_interval=self.poll_interval)

    @step
    def start(self):
        """Check the qmcp server answers."""
        self.run_id = current.run_id
        with self._client() as client:
            print(f"qmcp at {self.mcp_url}: {client.health()['status']}")
        self.next(self.transcribe)

    @step
    def transcribe(self):
        """Transcribe the project's timeline, and wait for the captions."""
        with self._client() as client:
            self.transcribed = transcribe(
                invoker(client, self._correlation()),
                self.project_id,
                timeout=self.transcribe_timeout,
                queued_timeout=self.queued_timeout,
                poll_interval=self.poll_interval,
            )
        joined = " (it was already running)" if self.transcribed.joined else ""
        why = f" -- {self.transcribed.why}" if self.transcribed.why else ""
        print(f"After {self.transcribed.seconds:.0f} s the transcription{joined} ended: "
              f"{self.transcribed.outcome}{why}")
        self.next(self.review)

    @step
    def review(self):
        """Put the transcript to a person, and wait for the answer."""
        self.reviewed = None
        if self.transcribed.done:
            request = review_request(self.project_id, self.transcribed, self.run_id)
            with self._client() as client:
                self.reviewed = self._ask(client, request, self.review_timeout)
        self.next(self.render)

    @step
    def render(self):
        """Queue the render with the captions burned in, once the transcript is approved."""
        self.queued = None
        if self.reviewed == APPROVE:
            with self._client() as client:
                self.queued = queue_render(
                    invoker(client, self._correlation()),
                    self.project_id,
                    captions=True,
                    width=self.width,
                    height=self.height,
                    quality=self.quality,
                )
            print(f"Queued render {self.queued['render_id']} of {self.queued['name']!r} "
                  f"with {self.queued['captions']} captions")
        self.next(self.follow)

    @step
    def follow(self):
        """Wait until the render is rendered or failed, or no renderer runs."""
        self.waited = None
        if self.queued is not None:
            with self._client() as client:
                self.waited = wait_for_render(
                    invoker(client, self._correlation()),
                    self.queued["render_id"],
                    timeout=self.render_timeout,
                    queued_timeout=self.queued_timeout,
                    poll_interval=self.poll_interval,
                )
            why = f" -- {self.waited.why}" if self.waited.why else ""
            print(f"After {self.waited.seconds:.0f} s the render ended: "
                  f"{self.waited.outcome}{why}")
        self.next(self.ask_person)

    @step
    def ask_person(self):
        """Put a rendered render, with its link, to a person, and wait for the answer."""
        self.answer = None
        if self.waited is not None and self.waited.rendered:
            request = approval_request(self.queued, self.waited, self.run_id,
                                       deliver=self.deliver, destination=self.destination)
            with self._client() as client:
                self.answer = self._ask(client, request, self.approval_timeout)
        self.next(self.delivery)

    @step
    def delivery(self):
        """Deliver the approved render, when the run was asked to."""
        self.delivered = None
        if self.answer == APPROVE and self.deliver:
            with self._client() as client:
                self.delivered = deliver_render(
                    invoker(client, self._correlation()),
                    self.queued["render_id"],
                    self.destination,
                )
        self.next(self.end)

    @step
    def end(self):
        """Say how it ended."""
        self.outcome = summary(self.project_id, self.transcribed, self.reviewed, self.queued,
                               self.waited, self.answer, self.delivered)
        print(self.outcome)


if __name__ == "__main__":
    AlfredCaptionRenderFlow()
