"""The caption flow: transcribing and waiting for the captions, the transcript
put to a person, the captioned render's approval, delivery, how a run ends,
and each of its branches end to end through a qmcp server and its queue.

alfred is `CaptionedAlfred` in `tests/alfred_stub.py`. The steps are tested
through `qmcp.cookbook.alfred_captions`, and the branches by running the steps
of `examples/flows/alfred_caption_render.py` itself, over a stand-in for
Metaflow, which cannot be imported where `import fcntl` fails.
"""

from __future__ import annotations

import importlib.util
import pickle
import re
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

import qmcp.tools.alfred as alfred
from qmcp.client import MCPClient
from qmcp.cookbook import alfred_captions as flow
from qmcp.cookbook import alfred_render
from qmcp.cookbook.alfred_captions import (
    DONE,
    FAILED,
    NO_ENGINE,
    NO_SPEECH,
    Delivered,
    Transcribed,
)
from qmcp.cookbook.alfred_render import (
    APPROVE,
    NO_RENDERER,
    REJECT,
    RENDERED,
    TIMED_OUT,
    ToolFailedError,
    Waited,
)
from qmcp.tools.alfred import NO_AUDIO_ENGINE, TRANSCRIPTION_RUNNING
from tests.alfred_stub import NO_ENGINE_DETAIL, URL, CaptionedAlfred, connect

REPO = Path(__file__).resolve().parents[1]

HEARD = [
    {"start": 0.0, "end": 2.5, "text": "Hello and welcome."},
    {"start": 2.5, "end": 5.04, "text": "This is the demo."},
]


# --- the module and the flow file ----------------------------------------------


def test_the_steps_import_nothing_from_metaflow():
    """Asserted on the source, because the failure is an import line existing."""
    source = Path(flow.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]
    assert not [line for line in imports if "metaflow" in line]


def test_the_flow_file_compiles_and_runs_the_module_s_steps():
    """Compiled rather than imported: importing it imports Metaflow."""
    path = REPO / "examples" / "flows" / "alfred_caption_render.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    for name in ("transcribe", "review_request", "queue_render", "wait_for_render",
                 "approval_request", "ask", "wait_for_answer", "deliver_render", "summary"):
        assert f"{name}(" in source, name
    assert "captions=True" in source
    assert '"deliver", help="Deliver the render once a person approves it", type=bool, ' \
           "default=False" in source


# --- transcribing -------------------------------------------------------------------


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
    """Tool answers in order: each look at the captions takes the next job
    state, the last one repeating; `start` is an error starting raises;
    `health` is alfred's report, or an error to raise."""

    def __init__(self, *states, captions=HEARD, start=None, health=None) -> None:
        self.states = list(states)
        self.captions = captions
        self.start = start
        self.health = health
        self.calls: list[str] = []

    def __call__(self, tool: str, params: dict):
        self.calls.append(tool)
        if tool == "alfred_captions_transcribe":
            if self.start is not None:
                raise self.start
            return {"project_id": params["project_id"], "job": {"state": "queued"}}
        if tool == "alfred_captions":
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            job = state if isinstance(state, dict) else {"state": state, "error": None}
            done = job.get("state") == DONE
            return {"project_id": params["project_id"], "style": None, "job": job,
                    "captions": self.captions if done else []}
        if tool == "alfred_health":
            if isinstance(self.health, Exception):
                raise self.health
            return self.health
        raise AssertionError(f"unexpected tool {tool}")


def transcribed(tools: Tools, **limits) -> tuple[Transcribed, Clock]:
    clock = Clock()
    return flow.transcribe(tools, "p1", poll_interval=5, sleep=clock.sleep, clock=clock,
                           **limits), clock


def refusal(opening: str, status: int, detail: str) -> ToolFailedError:
    return ToolFailedError(f"Tool execution failed: {opening}: alfred refused POST "
                           f"/project/p1/captions/transcribe ({status}): {detail}")


def test_a_transcription_is_followed_until_its_captions_are_there():
    tools = Tools("queued", "transcribing", "done")
    ended, clock = transcribed(tools)
    assert (ended.outcome, ended.done, ended.captions, ended.seconds) == (DONE, True, HEARD, 10)
    assert tools.calls == ["alfred_captions_transcribe"] + ["alfred_captions"] * 3
    assert clock.slept == [5, 5] and not ended.joined


def test_no_audio_engine_ends_it_at_once_in_alfred_s_words():
    """Mutation: treat the 503 as any other refusal -- red, it raises."""
    tools = Tools("queued", start=refusal(NO_AUDIO_ENGINE, 503, NO_ENGINE_DETAIL))
    ended, clock = transcribed(tools)
    assert ended.outcome == NO_ENGINE and not ended.done
    assert ended.why == (f"{NO_AUDIO_ENGINE}: alfred refused POST "
                         f"/project/p1/captions/transcribe (503): {NO_ENGINE_DETAIL}")
    assert tools.calls == ["alfred_captions_transcribe"] and clock.slept == []


def test_a_transcription_already_running_is_followed_not_refused():
    """Mutation: raise on the 409 -- red."""
    tools = Tools("transcribing", "done",
                  start=refusal(TRANSCRIPTION_RUNNING, 409, "already running"))
    ended, _ = transcribed(tools)
    assert (ended.outcome, ended.joined, ended.captions) == (DONE, True, HEARD)


def test_any_other_refusal_to_start_is_raised():
    tools = Tools("queued", start=ToolFailedError(
        "Tool execution failed: alfred refused POST /project/p1/captions/transcribe (404): "
        "Item not found"))
    with pytest.raises(ToolFailedError, match="Item not found"):
        transcribed(tools)


def test_a_failed_transcription_ends_with_alfred_s_reason():
    why = "the audio engine at http://joe.test:8010: timed out"
    ended, _ = transcribed(Tools("transcribing", {"state": "failed", "error": why}))
    assert (ended.outcome, ended.why) == (FAILED, why)


def test_a_transcription_that_hears_nothing_is_not_put_to_anyone():
    ended, _ = transcribed(Tools("done", captions=[]))
    assert (ended.outcome, ended.done) == (NO_SPEECH, False)


def test_a_job_queued_with_no_worker_running_ends_early_as_no_renderer():
    """Mutation: never ask `alfred_health` -- red, the wait runs to its timeout."""
    tools = Tools("queued", health={"renderer": {"ok": False, "workers": 0}})
    ended, _ = transcribed(tools, queued_timeout=60, timeout=900)
    assert (ended.outcome, ended.seconds) == (NO_RENDERER, 60)
    assert ended.why == ("still queued after 60 s, and alfred reports no worker running to "
                         "transcribe it")
    assert tools.calls.count("alfred_health") == 1


def test_an_alfred_without_a_health_report_is_waited_on_then_read_as_no_renderer():
    tools = Tools("queued", health=ToolFailedError("alfred refused GET /health (404)"))
    ended, _ = transcribed(tools, queued_timeout=60, timeout=120)
    assert (ended.outcome, ended.seconds) == (NO_RENDERER, 120)
    assert ended.why.endswith("alfred does not report its workers")


def test_a_job_queued_behind_running_workers_times_out_rather_than_blaming_them():
    ended, _ = transcribed(Tools("queued", health={"renderer": {"ok": True, "workers": 1}}),
                           queued_timeout=60, timeout=120)
    assert (ended.outcome, ended.why) == (TIMED_OUT, "still queued after 120 s")


def test_a_transcription_still_running_at_the_timeout_says_so():
    ended, _ = transcribed(Tools("transcribing"), timeout=30)
    assert (ended.outcome, ended.why) == (TIMED_OUT, "still transcribing after 30 s")


# --- what a person reads ---------------------------------------------------------------


def done(captions=HEARD) -> Transcribed:
    return Transcribed(DONE, captions, {"state": "done", "engine": "stub engine",
                                        "updated_at": "2026-10-18T10:00:00"}, 10.0)


@pytest.mark.parametrize("seconds,shown", [
    (0, "0:00.0"), (2.5, "0:02.5"), (59.96, "1:00.0"), (61.25, "1:01.2"), (3725.5, "62:05.5"),
    (None, "?"), (True, "?"),
])
def test_timeline_seconds_read_as_minutes_and_seconds(seconds, shown):
    assert flow.timecode(seconds) == shown


def test_the_review_lists_the_transcript_and_says_what_each_answer_does():
    request = flow.review_request("p1", done(), "1729")
    assert request["request_id"] == "alfred-captions-p1-1729"
    assert request["options"] == ["approve", "reject"]
    assert request["prompt"] == (
        "Burn these 2 captions into a render of alfred project p1?\n"
        "  0:00.0-0:02.5  Hello and welcome.\n"
        "  0:02.5-0:05.0  This is the demo.\n"
        "Answer approve to render it with the captions alfred holds when the render is "
        "queued -- a correction saved in alfred before then is included -- or reject to end "
        "the run without rendering.")
    assert request["context"] == {"project_id": "p1", "captions": HEARD,
                                  "engine": "stub engine",
                                  "transcribed_at": "2026-10-18T10:00:00",
                                  "flow_run_id": "1729"}


def test_a_long_transcript_is_listed_up_to_a_bound_and_kept_whole_in_the_context():
    """Mutation: list every caption -- red."""
    shown = flow.REVIEW_LINES
    many = [{"start": float(n), "end": n + 0.5, "text": f"line {n}"} for n in range(shown + 3)]
    lines = flow.review_request("p1", done(many), "1")["prompt"].splitlines()
    assert lines[0] == f"Burn these {shown + 3} captions into a render of alfred project p1?"
    last = shown - 1  # the last caption listed, on the line before the count of the rest
    assert lines[shown] == f"  {flow.timecode(last)}-{flow.timecode(last + 0.5)}  line {last}"
    assert lines[shown + 1] == "  ... and 3 more, in this request's context"
    assert len(lines) == shown + 3  # the question, the lines, the rest, what each answer does
    assert flow.review_request("p1", done(many), "1")["context"]["captions"] == many


QUEUED = {"render_id": "r1", "project_id": "p1", "name": "Talk", "width": 1280,
          "height": 720, "fps": 30.0, "quality": None, "captions": 2}
LINK = f"{URL}/render/r1/download?exp=1893456000&sig=c2ln"


def rendered() -> Waited:
    return Waited(RENDERED, {"render_id": "r1", "state": "rendered", "download_url": LINK,
                             "download_expires_at": "2030-01-01T00:00:00+00:00"}, 12.0)


@pytest.mark.parametrize("deliver,destination,then", [
    (True, "local", "Answer approve to deliver it to local, or reject to leave it undelivered."),
    (True, None, "Answer approve to deliver it to alfred's default destination, or reject to "
                 "leave it undelivered."),
    (False, None, "Nothing is delivered either way: this run was not asked to deliver."),
])
def test_the_render_s_approval_carries_its_link_and_says_what_approving_does(
        deliver, destination, then):
    request = flow.approval_request(QUEUED, rendered(), "1729", deliver=deliver,
                                    destination=destination)
    assert request["request_id"] == "alfred-caption-render-r1-1729"
    assert request["prompt"] == (
        f"Approve the render of 'Talk'? Watch it at {LINK} -- the link works until "
        "2030-01-01T00:00:00+00:00; alfred_render_status for render r1 issues a fresh one. "
        + then)
    assert request["context"]["captions"] == 2
    assert (request["context"]["deliver"], request["context"]["destination"]) == (
        deliver, destination)


# --- delivering ---------------------------------------------------------------------------


def test_delivery_names_a_destination_only_when_given():
    sent = []

    def invoke(tool, params):
        sent.append((tool, params))
        return {"render_id": "r1", "destination": "local", "location": f"{URL}/render/r1/download"}

    assert flow.deliver_render(invoke, "r1") == Delivered("local", f"{URL}/render/r1/download")
    flow.deliver_render(invoke, "r1", "local")
    assert sent == [("alfred_render_deliver", {"render_id": "r1"}),
                    ("alfred_render_deliver", {"render_id": "r1", "destination": "local"})]


def test_a_refused_delivery_is_said_rather_than_raised():
    """Mutation: let the refusal raise -- red; the render is fine, and the run
    should say what became of it."""
    def invoke(tool, params):
        raise ToolFailedError("Tool execution failed: alfred refused POST /render/r1/deliver "
                              "(502): upload failed: no credentials")

    delivered = flow.deliver_render(invoke, "r1", "bucket")
    assert not delivered.ok
    assert delivered.error == ("alfred refused POST /render/r1/deliver (502): upload failed: "
                               "no credentials")


# --- how a run ends -------------------------------------------------------------------------


STILL_QUEUED = Transcribed(NO_RENDERER, [], {"state": "queued"}, 60,
                           "still queued after 60 s, and alfred reports no worker running to "
                           "transcribe it")


@pytest.mark.parametrize("ending,line", [
    ((Transcribed(NO_ENGINE, [], None, 0, f"{NO_AUDIO_ENGINE}: ... (503): none set"),),
     f"Nothing was asked: the transcription of alfred project p1 could not start: "
     f"{NO_AUDIO_ENGINE}: ... (503): none set."),
    ((STILL_QUEUED,),
     "Nothing was asked: the transcription of alfred project p1 was not run: still queued "
     "after 60 s, and alfred reports no worker running to transcribe it."),
    ((Transcribed(NO_SPEECH, [], None, 9, "the speech engine heard nothing to caption"),),
     "Nothing was asked: the transcription of alfred project p1 found no speech: the speech "
     "engine heard nothing to caption."),
    ((Transcribed(FAILED, [], None, 9, "the engine timed out"),),
     "Nothing was asked: the transcription of alfred project p1 failed: the engine timed out."),
    ((done(), REJECT),
     "The transcript of alfred project p1 was rejected; nothing was rendered."),
    ((done(), None),
     "Nobody reviewed the transcript of alfred project p1 before the request expired; nothing "
     "was rendered."),
    ((done(), APPROVE, QUEUED, Waited(alfred_render.FAILED, {}, 3, "clip 1: no such file")),
     "The transcript was approved, and render r1 of 'Talk' with 2 captions failed: clip 1: no "
     "such file."),
    ((done(), APPROVE, QUEUED, rendered(), REJECT),
     "Rejected: render r1 of 'Talk' with 2 captions; nothing was delivered."),
    ((done(), APPROVE, QUEUED, rendered(), None),
     "Nobody answered before the approval of render r1 of 'Talk' with 2 captions expired; "
     "nothing was delivered."),
    ((done(), APPROVE, QUEUED, rendered(), APPROVE),
     f"Approved: render r1 of 'Talk' with 2 captions, {LINK}; not delivered."),
    ((done(), APPROVE, QUEUED, rendered(), APPROVE,
      Delivered("local", f"{URL}/render/r1/download")),
     f"Approved and delivered: render r1 of 'Talk' with 2 captions, to local at "
     f"{URL}/render/r1/download."),
    ((done(), APPROVE, QUEUED, rendered(), APPROVE, Delivered("bucket", None, "alfred ... (502)")),
     f"Approved: render r1 of 'Talk' with 2 captions, {LINK}; the delivery failed: "
     "alfred ... (502)."),
])
def test_the_summary_says_how_the_run_ended(ending, line):
    assert flow.summary("p1", *ending) == line


# --- end to end: the flow file's steps, through a qmcp server and its queue ------------------

FLOW_FILE = REPO / "examples" / "flows" / "alfred_caption_render.py"
CORRELATION = "alfred-caption-render-run-1"


class _FlowSpec:
    """Metaflow's `FlowSpec` as far as the steps use it: `next` names the next step."""

    def next(self, step) -> None:
        self._next = step


def _parameter(name, help=None, default=None, type=None, required=False):
    """Metaflow's `Parameter` as far as a run uses it: its default, unless a run sets one."""
    return default


@pytest.fixture
def flow_file(monkeypatch):
    """`examples/flows/alfred_caption_render.py`, loaded over a stand-in for
    Metaflow's classes and decorators. Its steps run in this process, in the
    order `self.next` names; what Metaflow adds -- a process per step, and
    parameters read from a command line -- is not exercised here, and the
    examples smoke test imports the file where Metaflow runs."""
    metaflow = types.ModuleType("metaflow")
    metaflow.FlowSpec, metaflow.Parameter = _FlowSpec, _parameter
    metaflow.step = lambda function: function
    metaflow.current = SimpleNamespace(run_id="run-1")
    monkeypatch.setitem(sys.modules, "metaflow", metaflow)
    spec = importlib.util.spec_from_file_location("alfred_caption_render_flow", FLOW_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(monkeypatch) -> CaptionedAlfred:
    stand_in = CaptionedAlfred()
    stand_in.add_project("p1", "Talk", heard=HEARD,
                         render_settings={"width": 1280, "height": 720})
    connected = connect(stand_in)
    monkeypatch.setattr(alfred, "client", lambda: connected)
    return stand_in


@pytest.fixture
def server(tmp_path):
    from qmcp.integrations.voice.check import throwaway_server

    with throwaway_server(tmp_path / "queue.db") as url:
        yield url


def run(module, monkeypatch, url: str, *, review: str | None = APPROVE,
        approval: str | None = APPROVE, **parameters) -> SimpleNamespace:
    """One run of the flow file's steps, against the qmcp server at `url`.

    Time passes only when a wait sleeps. A person answers each request at the
    first sleep of its wait -- the transcript with `review`, the render with
    `approval` -- and None is nobody answering. `parameters` are set on the run,
    the rest keep their defaults. After each step its artifacts are pickled, as
    Metaflow keeps them between steps.
    """
    clock = Clock()
    asked: list[dict] = []
    answers = {"alfred-captions-": review, "alfred-caption-render-": approval}

    def asking(client, request, **options):
        asked.append(request)
        alfred_render.ask(client, request, **options)

    def a_person_answers(client, request_id, **options):
        answer = next(a for prefix, a in answers.items() if request_id.startswith(prefix))
        answered: list[str] = []

        def waiting(seconds: float) -> None:
            if answer is not None and not answered:
                answered.append(answer)
                client.submit_human_response(request_id, answer,
                                             responded_by="editor@example.com")
            clock.sleep(seconds)

        return alfred_render.wait_for_answer(client, request_id, sleep=waiting, clock=clock,
                                             **options)

    monkeypatch.setattr(module, "transcribe",
                        partial(flow.transcribe, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(module, "wait_for_render",
                        partial(alfred_render.wait_for_render, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(module, "ask", asking)
    monkeypatch.setattr(module, "wait_for_answer", a_person_answers)

    ran = module.AlfredCaptionRenderFlow()
    ran.project_id, ran.mcp_url = "p1", url
    for name, value in parameters.items():
        setattr(ran, name, value)
    step = ran.start
    while step is not None:
        ran._next = None
        step()
        pickle.dumps({k: v for k, v in vars(ran).items() if not k.startswith("_")})
        step = ran._next

    with MCPClient(base_url=url) as client:
        tools = sorted(r["tool_name"] for r in client.list_invocations(limit=200)
                       if r["correlation_id"] == CORRELATION)
        pending = [r.id for r in client.list_human_requests(status_filter="pending")]
    return SimpleNamespace(flow=ran, line=ran.outcome, asked=asked, tools=tools,
                           pending=pending)


# The stand-in moves a job and a render one state per look: two looks each.
TRANSCRIBE_TOOLS = ["alfred_captions_transcribe"] + ["alfred_captions"] * 2
RENDER_TOOLS = ["alfred_render_enqueue"] + ["alfred_render_status"] * 2
SIGNED = f"{URL}/render/r1/download?exp=1893456000&sig=c2ln"


def test_an_approved_transcript_and_render_are_delivered_when_asked(
        flow_file, monkeypatch, server, fake):
    """Through a qmcp server on an ephemeral port over its own database: every
    call to alfred is an invocation it recorded under the run's correlation id,
    and both requests are answered on its queue."""
    ran = run(flow_file, monkeypatch, server, deliver=True)
    assert ran.line == ("Approved and delivered: render r1 of 'Talk' with 2 captions, to "
                        f"local at {URL}/render/r1/download.")
    assert ran.tools == sorted(TRANSCRIBE_TOOLS + RENDER_TOOLS + ["alfred_render_deliver"])
    review, approval = ran.asked
    assert review["request_id"] == "alfred-captions-p1-run-1"
    assert "  0:02.5-0:05.0  This is the demo." in review["prompt"].splitlines()
    assert approval["request_id"] == "alfred-caption-render-r1-run-1"
    assert approval["prompt"].endswith("Answer approve to deliver it to alfred's default "
                                       "destination, or reject to leave it undelivered.")
    # What alfred was asked to render: the saved size, and the captions with a style on.
    sent = fake.queued[0]
    assert (sent["captions"], sent["caption_style"]) == (HEARD, {"enabled": True})
    assert (sent["width"], sent["height"]) == (1280, 720)
    assert fake.renders["r1"]["destination"] == "local"
    assert ran.pending == []


def test_an_approved_render_is_not_delivered_unless_the_run_was_asked_to(
        flow_file, monkeypatch, server, fake):
    """Mutation: deliver whenever the render is approved -- red."""
    ran = run(flow_file, monkeypatch, server)
    assert ran.line == f"Approved: render r1 of 'Talk' with 2 captions, {SIGNED}; not delivered."
    assert "alfred_render_deliver" not in ran.tools
    assert fake.renders["r1"]["destination"] is None
    assert ran.asked[1]["prompt"].endswith("Nothing is delivered either way: this run was not "
                                           "asked to deliver.")


def test_a_named_destination_is_where_it_goes(flow_file, monkeypatch, server, fake):
    """Mutation: drop --destination on the way to the tool -- red, alfred is
    asked for its own choice."""
    ran = run(flow_file, monkeypatch, server, deliver=True, destination="local")
    assert ran.line.startswith("Approved and delivered:")
    assert fake.calls("POST", "/render/r1/deliver")[0].url.params.get("destination") == "local"


def test_a_destination_alfred_refuses_is_said_in_the_last_line(
        flow_file, monkeypatch, server, fake):
    ran = run(flow_file, monkeypatch, server, deliver=True, destination="dropbox")
    assert re.fullmatch(r"Approved: render r1 of 'Talk' with 2 captions, \S+; the delivery "
                        r"failed: alfred refused POST /render/r1/deliver \(502\): unknown "
                        r"destination 'dropbox'; known: \['bucket', 'local'\]\.", ran.line)


def test_a_rejected_transcript_renders_nothing(flow_file, monkeypatch, server, fake):
    """Mutation: render whatever the review's answer -- red."""
    ran = run(flow_file, monkeypatch, server, review=REJECT, deliver=True)
    assert ran.line == "The transcript of alfred project p1 was rejected; nothing was rendered."
    assert ran.tools == sorted(TRANSCRIBE_TOOLS)
    assert fake.queued == [] and len(ran.asked) == 1


def test_a_transcript_nobody_reviews_renders_nothing(flow_file, monkeypatch, server, fake):
    ran = run(flow_file, monkeypatch, server, review=None, review_timeout=60)
    assert ran.line == ("Nobody reviewed the transcript of alfred project p1 before the request "
                        "expired; nothing was rendered.")
    assert fake.queued == []


def test_a_rejected_render_is_not_delivered(flow_file, monkeypatch, server, fake):
    """Mutation: deliver once the transcript is approved -- red."""
    ran = run(flow_file, monkeypatch, server, approval=REJECT, deliver=True)
    assert ran.line == "Rejected: render r1 of 'Talk' with 2 captions; nothing was delivered."
    assert "alfred_render_deliver" not in ran.tools and len(ran.asked) == 2


def test_no_audio_engine_asks_nobody_anything(flow_file, monkeypatch, server, fake):
    fake.without_an_audio_engine()
    ran = run(flow_file, monkeypatch, server, deliver=True)
    assert ran.flow.transcribed.outcome == NO_ENGINE
    assert ran.line == (  # alfred's detail ends the sentence
        "Nothing was asked: the transcription of alfred project p1 could not start: "
        f"{NO_AUDIO_ENGINE}: alfred refused POST /project/p1/captions/transcribe (503): "
        f"{NO_ENGINE_DETAIL}")
    assert ran.tools == ["alfred_captions_transcribe"] and ran.asked == []


def test_no_worker_running_asks_nobody_anything(flow_file, monkeypatch, server, fake):
    fake.workers = 0
    ran = run(flow_file, monkeypatch, server)
    assert (ran.flow.transcribed.outcome, ran.flow.transcribed.seconds) == (NO_RENDERER, 60)
    assert ran.line == ("Nothing was asked: the transcription of alfred project p1 was not run: "
                        "still queued after 60 s, and alfred reports no worker running to "
                        "transcribe it.")
    assert "alfred_health" in ran.tools and ran.asked == []


def test_a_transcription_already_running_is_followed_to_the_same_end(
        flow_file, monkeypatch, server, fake):
    alfred.client().request("POST", "/project/p1/captions/transcribe")  # someone else's
    ran = run(flow_file, monkeypatch, server, deliver=True)
    assert ran.flow.transcribed.joined and ran.flow.transcribed.outcome == DONE
    assert ran.line.startswith("Approved and delivered: render r1")


def test_a_correction_saved_in_alfred_during_the_review_is_what_is_rendered(
        flow_file, monkeypatch, server, fake):
    """The render reads the captions when it is queued, as the review says."""
    corrected = [HEARD[0], {**HEARD[1], "text": "This is our demo."}]
    asking = flow_file.review_request

    def a_person_corrects_while_asked(project_id, transcribed, run_id):
        request = asking(project_id, transcribed, run_id)
        alfred.alfred_captions_set({"project_id": project_id, "captions": corrected})
        return request

    monkeypatch.setattr(flow_file, "review_request", a_person_corrects_while_asked)
    ran = run(flow_file, monkeypatch, server)
    assert "  0:02.5-0:05.0  This is the demo." in ran.asked[0]["prompt"].splitlines()
    assert fake.queued[0]["captions"] == corrected
    assert ran.line.startswith("Approved: render r1 of 'Talk' with 2 captions")


