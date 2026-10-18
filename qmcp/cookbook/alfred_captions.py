"""Captioning an alfred project behind a person's review: the steps, apart from any flow.

`examples/flows/alfred_caption_render.py` runs them as a Metaflow flow, one
step each:

    transcribe  `alfred_captions_transcribe` for the project's saved timeline,
                then `alfred_captions` until the job is done or failed, sits
                queued with no worker running, or runs out of time; an alfred
                with no speech engine ends it at once
    review      a request on qmcp's human queue carrying the transcript:
                approve renders the project with its captions, reject ends
                the run
    render      `alfred_render_enqueue` with the captions burned in, then
                `alfred_render_status` until the render settles
                (`qmcp.cookbook.alfred_render`)
    ask         a request carrying the render's download link and when it
                stops working, saying what approval does
    deliver     `alfred_render_deliver` once the render is approved, when the
                flow was asked to deliver

Every call to alfred is a qmcp tool invocation, so the run is in qmcp's
record. The steps take the tool caller (`Invoke`) and the queue client as
arguments, and a wait takes its `sleep` and `clock`, so a test drives them
without a flow. Nothing here imports Metaflow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qmcp.cookbook import alfred_render
from qmcp.cookbook.alfred_render import (
    APPROVE,
    NO_RENDERER,
    OPTIONS,
    REJECT,
    TIMED_OUT,
    Invoke,
    ToolFailedError,
    Waited,
    renderer_workers,
)
from qmcp.tools.alfred import NO_SPEECH_ENGINE, TRANSCRIPTION_RUNNING

# How a transcription ends. NO_RENDERER and TIMED_OUT are the render wait's.
DONE = "done"
NO_SPEECH = "no speech"
FAILED = "failed"
NO_ENGINE = "no speech engine"

# The most captions a review prompt lists; the request's context holds them all.
REVIEW_LINES = 200


@dataclass(frozen=True)
class Transcribed:
    """How a transcription ended.

    `outcome` is `DONE`, `NO_SPEECH` (done, with no caption to show),
    `FAILED`, `NO_ENGINE`, `NO_RENDERER` or `TIMED_OUT`. `captions` are the
    project's once the job is done; `job` is alfred's last word on it; `why`
    is one line for a person, empty when it is done. `joined` is true when
    the run followed a transcription that was already running.
    """

    outcome: str
    captions: list[dict[str, Any]]
    job: dict[str, Any] | None
    seconds: float
    why: str = ""
    joined: bool = False

    @property
    def done(self) -> bool:
        return self.outcome == DONE


@dataclass(frozen=True)
class Delivered:
    """Where an approved render went: alfred's destination and an absolute
    location, or `error`, the tool's, when alfred would not deliver it."""

    destination: str | None
    location: str | None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _from(phrase: str, error: Exception) -> str:
    """A tool's error from `phrase` on: the server's own opening dropped."""
    text = str(error)
    return text[text.find(phrase):] if phrase in text else text


def transcribe(
    invoke: Invoke,
    project_id: str,
    *,
    timeout: float = 900.0,
    queued_timeout: float = 60.0,
    poll_interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Transcribed:
    """Start transcribing the project's saved timeline and follow the job.

    alfred with no speech engine ends it at once as `NO_ENGINE`. A
    transcription already running is followed rather than refused, since it
    makes the same captions. A job still queued after `queued_timeout` seconds
    ends the wait as `NO_RENDERER` when `alfred_health` reports no worker
    running, the worker that renders being the one that transcribes; after
    `timeout` seconds the wait ends as `TIMED_OUT`, or as `NO_RENDERER` for a
    job that never left the queue of an alfred that does not report its
    workers. Any other tool error raises `ToolFailedError`.
    """
    started = clock()
    joined = False
    try:
        invoke("alfred_captions_transcribe", {"project_id": project_id})
    except ToolFailedError as refused:
        if NO_SPEECH_ENGINE in str(refused):
            return Transcribed(NO_ENGINE, [], None, clock() - started,
                               _from(NO_SPEECH_ENGINE, refused))
        if TRANSCRIPTION_RUNNING not in str(refused):
            raise
        joined = True

    workers: int | None = None
    asked = False
    while True:
        held = invoke("alfred_captions", {"project_id": project_id})
        waited = clock() - started
        job = held.get("job") if isinstance(held.get("job"), dict) else None
        state = (job or {}).get("state")
        if state == DONE:
            captions = held.get("captions") or []
            if not captions:
                return Transcribed(NO_SPEECH, [], job, waited,
                                   "the speech engine heard nothing to caption", joined)
            return Transcribed(DONE, captions, job, waited, "", joined)
        if state == FAILED:
            return Transcribed(FAILED, [], job, waited,
                               job.get("error") or "alfred gave no reason", joined)
        if state == "queued" and waited >= queued_timeout and not asked:
            asked = True
            workers = renderer_workers(invoke)
            if workers == 0:
                return Transcribed(NO_RENDERER, [], job, waited,
                                   f"still queued after {waited:.0f} s, and alfred reports "
                                   "no worker running to transcribe it", joined)
        if waited >= timeout:
            if state == "queued" and workers is None:
                return Transcribed(NO_RENDERER, [], job, waited,
                                   f"never left the queue in {waited:.0f} s, so no worker "
                                   "appears to be running; alfred does not report its workers",
                                   joined)
            return Transcribed(TIMED_OUT, [], job, waited,
                               f"still {state or 'without a job'} after {waited:.0f} s", joined)
        sleep(poll_interval)


def timecode(seconds: Any) -> str:
    """Timeline seconds as `m:ss.s`; `?` for anything that is not a number."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return "?"
    minutes, tenths = divmod(round(seconds * 10), 600)
    return f"{minutes}:{tenths // 10:02d}.{tenths % 10}"


def transcript(captions: list[dict[str, Any]], limit: int = REVIEW_LINES) -> list[str]:
    """The captions as lines a person reads, `start-end  text`, at most `limit`
    of them, the last saying how many more there are."""
    lines = [f"{timecode(c.get('start'))}-{timecode(c.get('end'))}  {c.get('text')}"
             for c in captions[:limit]]
    if len(captions) > limit:
        lines.append(f"... and {len(captions) - limit} more, in this request's context")
    return lines


def review_request(project_id: str, transcribed: Transcribed, run_id: str) -> dict[str, Any]:
    """What a person is asked about a finished transcription: the request's id,
    prompt, options and context, as `MCPClient.create_human_request` takes them.

    The prompt lists the transcript and says what each answer does. The render
    carries the captions alfred holds when it is queued, so a correction saved
    in alfred before the answer is rendered too.
    """
    captions = transcribed.captions
    count = f"{len(captions)} caption{'' if len(captions) == 1 else 's'}"
    lines = "\n".join(f"  {line}" for line in transcript(captions))
    prompt = (
        f"Burn these {count} into a render of alfred project {project_id}?\n{lines}\n"
        "Answer approve to render it with the captions alfred holds when the render is "
        "queued -- a correction saved in alfred before then is included -- or reject to end "
        "the run without rendering."
    )
    job = transcribed.job or {}
    return {
        "request_id": f"alfred-captions-{project_id}-{run_id}",
        "prompt": prompt,
        "options": list(OPTIONS),
        "context": {
            "project_id": project_id,
            "captions": captions,
            "engine": job.get("engine"),
            "transcribed_at": job.get("updated_at"),
            "flow_run_id": run_id,
        },
    }


def approval_request(
    queued: dict[str, Any],
    waited: Waited,
    run_id: str,
    *,
    deliver: bool = False,
    destination: str | None = None,
) -> dict[str, Any]:
    """What a person is asked about the captioned render: `alfred_render`'s
    request, with the link and when it stops working, saying what approval
    does -- deliver the render, when the run was asked to, or nothing more."""
    request = alfred_render.approval_request(queued, waited, run_id)
    if deliver:
        where = destination or "alfred's default destination"
        then = f" Answer approve to deliver it to {where}, or reject to leave it undelivered."
    else:
        then = " Nothing is delivered either way: this run was not asked to deliver."
    request["request_id"] = f"alfred-caption-render-{queued['render_id']}-{run_id}"
    request["prompt"] += then
    request["context"].update(captions=queued.get("captions"), deliver=deliver,
                              destination=destination)
    return request


def deliver_render(invoke: Invoke, render_id: str, destination: str | None = None) -> Delivered:
    """Deliver an approved render with `alfred_render_deliver`. A refusal is
    returned as `Delivered.error` rather than raised: the render is fine, and
    the run says what became of it."""
    params = {"render_id": render_id}
    if destination:
        params["destination"] = destination
    try:
        answer = invoke("alfred_render_deliver", params)
    except ToolFailedError as refused:
        return Delivered(destination, None, _from("alfred", refused))
    return Delivered(answer.get("destination"), answer.get("location"))


_TRANSCRIPTION_ENDED = {
    NO_ENGINE: "could not start",
    NO_SPEECH: "found no speech",
    FAILED: "failed",
    NO_RENDERER: "was not run",
    TIMED_OUT: "did not finish in time",
}
_RENDER_ENDED = {alfred_render.FAILED: "failed", TIMED_OUT: "did not finish in time"}


def summary(
    project_id: str,
    transcribed: Transcribed,
    reviewed: str | None = None,
    queued: dict[str, Any] | None = None,
    waited: Waited | None = None,
    answer: str | None = None,
    delivered: Delivered | None = None,
) -> str:
    """One line saying how the run ended."""
    if not transcribed.done:
        ended = _TRANSCRIPTION_ENDED.get(transcribed.outcome, "ended")
        return (f"Nothing was asked: the transcription of alfred project {project_id} "
                f"{ended}: {transcribed.why}.")
    if reviewed == REJECT:
        return f"The transcript of alfred project {project_id} was rejected; nothing was rendered."
    if reviewed != APPROVE:
        return (f"Nobody reviewed the transcript of alfred project {project_id} before the "
                "request expired; nothing was rendered.")
    render = (f"render {queued['render_id']} of {queued.get('name')!r} with "
              f"{queued.get('captions')} captions")
    if not waited.rendered:
        ended = _RENDER_ENDED.get(waited.outcome, "was not rendered")
        return f"The transcript was approved, and {render} {ended}: {waited.why}."
    if answer == REJECT:
        return f"Rejected: {render}; nothing was delivered."
    if answer != APPROVE:
        return (f"Nobody answered before the approval of {render} expired; nothing was "
                "delivered.")
    link = waited.status.get("download_url") or "in alfred"
    if delivered is None:
        return f"Approved: {render}, {link}; not delivered."
    if not delivered.ok:
        return f"Approved: {render}, {link}; the delivery failed: {delivered.error}."
    return (f"Approved and delivered: {render}, to {delivered.destination} at "
            f"{delivered.location}.")


__all__ = [
    "DONE",
    "FAILED",
    "NO_ENGINE",
    "NO_SPEECH",
    "REVIEW_LINES",
    "Delivered",
    "Transcribed",
    "approval_request",
    "deliver_render",
    "review_request",
    "summary",
    "timecode",
    "transcribe",
    "transcript",
]
