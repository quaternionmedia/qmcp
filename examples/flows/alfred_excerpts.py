"""Cut a long alfred project into excerpts a person chooses, and render each one.

alfred holds the project's transcript, made by its audio engine when it holds
none; the transcript's own timing proposes chapters and spans; a person is
asked about each span on qmcp's human queue, and approves or rejects it; each
approved span is rendered as an excerpt of its own, cut from the project's saved
timeline with its captions burned in, followed until it renders and put to a
person with its download link; and an approved excerpt is delivered, when the run
is asked to deliver. A local model, when the run names one, titles each
chapter and sorts each span into a category the run names. Every call to
alfred is one of qmcp's alfred tools, made through the qmcp server, so the
server must be configured for alfred (docs/integrations/alfred.md). The steps
are `qmcp.cookbook.alfred_excerpts`.

Usage:
    # The server, with alfred configured
    QMCP_ALFRED_URL=http://localhost:8000 uv run qmcp serve

    # In another terminal, the flow
    uv run python examples/flows/alfred_excerpts.py run --project-id <id>
    uv run python examples/flows/alfred_excerpts.py run --project-id <id> \\
        --label-with local --categories "song,story,announcement"
    uv run python examples/flows/alfred_excerpts.py run --project-id <id> \\
        --deliver True --destination local

    # The flow prints each request as it asks, and waits:
    uv run qmcp human respond <request id> approve      # or reject
"""

import os

from metaflow import FlowSpec, Parameter, current, step

from qmcp.client import MCPClient
from qmcp.client.mcp_client import default_base_url
from qmcp.cookbook.alfred_captions import TRANSCRIBE_TIMEOUT, deliver_render
from qmcp.cookbook.alfred_excerpts import (
    APPROVED,
    CHAPTER_PAUSE,
    DELIVERED,
    FAILED,
    LABEL_TIMEOUT,
    LONGEST,
    PAUSE,
    QUEUED,
    REJECTED,
    RENDERED,
    SHORTEST,
    UNANSWERED,
    WAITING,
    approval_request,
    label,
    local_model,
    mark,
    propose,
    queue_excerpt,
    read_transcript,
    review_request,
    summary,
)
from qmcp.cookbook.alfred_render import (
    APPROVE,
    REJECT,
    ask,
    invoker,
    wait_for_answer,
    wait_for_render,
)
from qmcp.localmodel import MODEL as LOCAL_MODEL


class AlfredExcerptsFlow(FlowSpec):
    """Propose excerpts from an alfred project's transcript, have a person choose
    them, render each one, have a person approve each render, and deliver them.

    Steps:
    1. start: check the qmcp server answers
    2. read: the project's saved timeline, and its transcript, made if it has none
    3. propose: chapters and spans from the transcript, labelled when asked
    4. review: put each span to a person, in order, and wait for each answer
    5. render: queue an excerpt of each approved span, cut from the timeline, with the
       captions alfred holds then
    6. follow: wait until each excerpt is rendered or failed, or no renderer runs
    7. ask_person: put each rendered excerpt, with its link, to a person and wait
    8. delivery: deliver each approved excerpt, when --deliver is set
    9. end: say how it ended

    After each step from propose on, the proposal is drawn on the project's
    timeline as cues (`alfred_cues_set`): each chapter, and each span with where
    it stands -- proposed, waiting, approved or rejected, queued, rendered,
    delivered -- so alfred's editor shows the run as it goes.
    """

    project_id = Parameter("project-id", help="The alfred project to cut", required=True)
    transcribe = Parameter(
        "transcribe", help="Transcribe the project when it holds no captions", type=bool,
        default=True,
    )
    pause = Parameter(
        "pause", help="Seconds of silence that end a span", default=PAUSE, type=float
    )
    chapter_pause = Parameter(
        "chapter-pause", help="Seconds of silence that end a chapter", default=CHAPTER_PAUSE,
        type=float,
    )
    shortest = Parameter(
        "shortest", help="The shortest an excerpt may run, in seconds", default=SHORTEST, type=float
    )
    longest = Parameter(
        "longest", help="The longest an excerpt may run, in seconds", default=LONGEST, type=float
    )
    label_with = Parameter(
        "label-with",
        help="none, or local: the model `qmcp localmodel` stands up titles each chapter "
        "and sorts each span into one of --categories",
        default="none", type=str,
    )
    categories = Parameter(
        "categories", help="Comma-separated categories a span is sorted into, with --label-with",
        default="", type=str,
    )
    llm_base_url = Parameter(
        "llm-base-url", help="Where the local model's chat route answers",
        default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"), type=str,
    )
    llm_model = Parameter("llm-model", help="The local model's name", default=LOCAL_MODEL,
                          type=str)
    llm_timeout = Parameter(
        "llm-timeout", help="Seconds one answer from the model may take", default=LABEL_TIMEOUT,
        type=float,
    )
    captions = Parameter(
        "captions", help="Burn each excerpt's captions into it", type=bool, default=True
    )
    width = Parameter("width", help="Frame width; the project's saved size by default", type=int)
    height = Parameter("height", help="Frame height; the project's saved size by default",
                       type=int)
    quality = Parameter("quality", help="draft, standard or high; alfred's default if unset",
                        type=str)
    deliver = Parameter(
        "deliver", help="Deliver each excerpt once a person approves it", type=bool, default=False
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
        "transcribe-timeout",
        help="Seconds to wait for a transcription; by default as long as alfred lets one run",
        default=int(TRANSCRIBE_TIMEOUT),
        type=int,
    )
    render_timeout = Parameter(
        "render-timeout", help="Seconds to wait for each render", default=1800, type=int
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
        "review-timeout", help="Seconds each span's review may wait for an answer",
        default=3600, type=int,
    )
    approval_timeout = Parameter(
        "approval-timeout", help="Seconds each excerpt's approval may wait for an answer",
        default=3600, type=int,
    )

    def _client(self) -> MCPClient:
        return MCPClient(self.mcp_url)

    def _correlation(self) -> str:
        return f"alfred-excerpts-{self.run_id}"

    def _ask(self, client: MCPClient, request: dict, timeout: int) -> str | None:
        """Put `request` to a person, print it, and wait for the answer."""
        ask(client, request, timeout_seconds=timeout, correlation_id=self._correlation())
        print(request["prompt"])
        print(f"Answer with: uv run qmcp human respond {request['request_id']} approve"
              " (or reject)")
        return wait_for_answer(client, request["request_id"], timeout_seconds=timeout,
                               poll_interval=self.poll_interval)

    def _mark(self, invoke) -> None:
        """Draw the proposal on the project's timeline as it stands. An alfred
        that holds no cues does not stop the run: the cues are a picture of it."""
        if self.proposal is None:
            return
        try:
            mark(invoke, self.project_id, self.proposal, self.states, self.refs)
        except Exception as exc:  # the picture is not the run
            print(f"The timeline was not marked: {exc}")

    @step
    def start(self):
        """Check the qmcp server answers."""
        self.run_id = current.run_id
        with self._client() as client:
            print(f"qmcp at {self.mcp_url}: {client.health()['status']}")
        self.next(self.read)

    @step
    def read(self):
        """The project's saved timeline, and its transcript, made if it has none."""
        with self._client() as client:
            invoke = invoker(client, self._correlation())
            self.project = invoke("alfred_project", {"project_id": self.project_id})
            self.transcribed = read_transcript(
                invoke,
                self.project_id,
                transcribe=self.transcribe,
                timeout=self.transcribe_timeout,
                queued_timeout=self.queued_timeout,
                poll_interval=self.poll_interval,
            )
        why = f" -- {self.transcribed.why}" if self.transcribed.why else ""
        print(f"{self.project['name']!r}: {self.project['clips']} clips, "
              f"{len(self.transcribed.captions)} captions ({self.transcribed.outcome}{why})")
        self.next(self.propose)

    @step
    def propose(self):
        """Chapters and spans from the transcript, labelled when asked."""
        self.proposal = None
        # Where each span stands, and the request or render it stands by, by number.
        self.states = {}
        self.refs = {}
        if self.transcribed.done:
            self.proposal = propose(
                self.transcribed.captions,
                pause=self.pause,
                chapter_pause=self.chapter_pause,
                shortest=self.shortest,
                longest=self.longest,
            )
            if self.label_with == "local":
                categories = [c.strip() for c in self.categories.split(",") if c.strip()]
                asker = local_model(self.llm_base_url, self.llm_model, timeout=self.llm_timeout)
                self.proposal = label(self.proposal, asker, categories=categories)
            elif self.label_with != "none":
                raise ValueError(f"--label-with is none or local, not {self.label_with!r}")
            for chapter in self.proposal.chapters:
                title = f" {chapter.title!r}" if chapter.title else ""
                print(f"chapter {chapter.number}{title}: {chapter.spans} excerpts, "
                      f"{chapter.opening}")
            print(f"{len(self.proposal.spans)} excerpts proposed, {self.proposal.left_out} "
                  "stretches too short to stand alone")
            with self._client() as client:
                self._mark(invoker(client, self._correlation()))
        self.next(self.review)

    @step
    def review(self):
        """Put each span to a person, in order, and wait for each answer."""
        self.reviewed = {}
        if self.proposal is not None:
            with self._client() as client:
                invoke = invoker(client, self._correlation())
                for span in self.proposal.spans:
                    request = review_request(self.project, span, self.proposal, self.run_id)
                    self.states[span.number] = WAITING
                    self.refs[span.number] = request["request_id"]
                    self._mark(invoke)
                    answer = self._ask(client, request, self.review_timeout)
                    self.reviewed[span.number] = answer
                    self.states[span.number] = (APPROVED if answer == APPROVE
                                                else REJECTED if answer == REJECT
                                                else UNANSWERED)
                    self._mark(invoke)
        self.next(self.render)

    @step
    def render(self):
        """Queue an excerpt of each approved span, cut from the project's timeline,
        with the captions alfred holds now: a correction saved in alfred during
        the review is in the excerpt."""
        self.queued = {}
        approved = [span for span in (self.proposal.spans if self.proposal else ())
                    if self.reviewed.get(span.number) == APPROVE]
        if approved:
            with self._client() as client:
                invoke = invoker(client, self._correlation())
                captions = style = None
                if self.captions:
                    held = invoke("alfred_captions", {"project_id": self.project_id})
                    captions, style = held.get("captions") or [], held.get("style")
                for span in approved:
                    self.queued[span.number] = queue_excerpt(
                        invoke,
                        self.project,
                        span,
                        captions=captions,
                        style=style,
                        width=self.width,
                        height=self.height,
                        quality=self.quality,
                    )
                    queued = self.queued[span.number]
                    print(f"Queued render {queued['render_id']} of {queued['name']!r} "
                          f"with {queued['captions']} captions")
                    self.states[span.number] = QUEUED
                    self.refs[span.number] = queued["render_id"]
                self._mark(invoke)
        self.next(self.follow)

    @step
    def follow(self):
        """Wait until each excerpt is rendered or failed, or no renderer runs."""
        self.waited = {}
        with self._client() as client:
            invoke = invoker(client, self._correlation())
            for number, queued in self.queued.items():
                self.waited[number] = wait_for_render(
                    invoke,
                    queued["render_id"],
                    timeout=self.render_timeout,
                    queued_timeout=self.queued_timeout,
                    poll_interval=self.poll_interval,
                )
                waited = self.waited[number]
                why = f" -- {waited.why}" if waited.why else ""
                print(f"After {waited.seconds:.0f} s excerpt {number} ended: {waited.outcome}{why}")
                self.states[number] = RENDERED if waited.rendered else FAILED
            if self.waited:
                self._mark(invoke)
        self.next(self.ask_person)

    @step
    def ask_person(self):
        """Put each rendered excerpt, with its link, to a person, and wait for each answer."""
        self.answers = {}
        spans = {span.number: span for span in (self.proposal.spans if self.proposal else ())}
        with self._client() as client:
            for number, waited in self.waited.items():
                if not waited.rendered:
                    continue
                request = approval_request(self.queued[number], waited, spans[number],
                                           self.run_id, deliver=self.deliver,
                                           destination=self.destination)
                self.answers[number] = self._ask(client, request, self.approval_timeout)
                if self.answers[number] == REJECT:
                    self.states[number] = REJECTED
            if self.answers:
                self._mark(invoker(client, self._correlation()))
        self.next(self.delivery)

    @step
    def delivery(self):
        """Deliver each approved excerpt, when the run was asked to."""
        self.delivered = {}
        if self.deliver:
            with self._client() as client:
                invoke = invoker(client, self._correlation())
                for number, answer in self.answers.items():
                    if answer == APPROVE:
                        self.delivered[number] = deliver_render(
                            invoke, self.queued[number]["render_id"], self.destination)
                        if self.delivered[number].ok:
                            self.states[number] = DELIVERED
                if self.delivered:
                    self._mark(invoke)
        self.next(self.end)

    @step
    def end(self):
        """Say how it ended."""
        self.outcome = summary(self.project, self.transcribed, self.proposal, self.reviewed,
                               self.queued, self.waited, self.answers, self.delivered,
                               deliver=self.deliver)
        print(self.outcome)


if __name__ == "__main__":
    AlfredExcerptsFlow()
