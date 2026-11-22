"""Cutting a long alfred project into excerpts a person chooses: the steps, apart from any flow.

`examples/flows/alfred_excerpts.py` runs them as a Metaflow flow, one step each:

    read      `alfred_project` for the saved timeline, and `alfred_captions`
              for the transcript -- made first, as the caption flow makes it,
              when the project holds none, since alfred cuts by the words
    propose   `alfred_drafts` with alfred's `excerpts` recipe, followed with
              `alfred_drafts_status` until alfred's worker is done: each draft
              is a project of its own, cut from the source with its captions
              and its words, read back with `alfred_captions`; a labeller,
              when the run has one, sorts each draft into a category
    review    one request on qmcp's human queue per draft, carrying its words:
              approve renders it as an excerpt of its own, reject drops it
              (`alfred_drafts_drop`)
    render    `alfred_render_enqueue` of each approved draft's own timeline,
              its captions burned in when asked, then `alfred_render_status`
              until each settles (`alfred_render`)
    ask       a request per rendered excerpt, with its link, saying what
              approval does
    deliver   `alfred_render_deliver` for each approved excerpt, when the run was
              asked to deliver
    mark      after each of those, `alfred_cues_set` with where each draft
              stands as cues on the project's timeline, beside the cues
              alfred's own run left, which alfred's editor draws under the clips

Every call to alfred is a qmcp tool invocation, so the run is in qmcp's
record. The steps take the tool caller (`Invoke`) and the queue client as
arguments, a wait takes its `sleep` and `clock`, and a labeller is a function
from a prompt to an answer, so a test drives them without a flow, a model or
alfred. Nothing here imports Metaflow.

What a draft is comes from alfred (its docs/drafts.md): the `excerpts` recipe
cuts the transcript at pauses and at the ends of sentences, within the length
an excerpt may have, and alfred's worker leaves each stretch as a project
carrying its source. Nothing here reads a pause or cuts a timeline.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import httpx

from qmcp.cookbook import alfred_captions, alfred_render
from qmcp.cookbook.alfred_captions import (
    DONE,
    NO_CAPTIONS,
    Delivered,
    Transcribed,
    timecode,
    transcript,
)
from qmcp.cookbook.alfred_render import (
    APPROVE,
    OPTIONS,
    REJECT,
    Invoke,
    Waited,
    sentence,
)

# alfred's recipe that proposes excerpts, by the name every account reads it by.
RECIPE = "excerpts"
# The recipe's knobs, as the run sets them: seconds of nothing said that end a
# stretch, and the shortest and longest an excerpt may run, in seconds.
PAUSE = 1.0
SHORTEST = 10.0
LONGEST = 90.0
# Seconds alfred's worker has to leave the drafts, and how often it is asked.
DRAFTS_TIMEOUT = 600.0
DRAFTS_POLL = 5.0
# How much of a draft a prompt quotes, in characters.
OPENING_CHARS = 80
LABEL_CHARS = 4000
# What a labeller may say back, in tokens; a category needs few.
LABEL_TOKENS = 32
# Seconds one answer from a model may take before the draft stays unlabelled.
LABEL_TIMEOUT = 30.0
# What alfred's worker says a run is, from queued to done or failed.
JOB_DONE = "done"
JOB_FAILED = "failed"

_FIRST_SENTENCE = re.compile(r"^(.*?[.!?…][\"')”’»\]]*)(?:\s|$)")


class DraftsFailedError(RuntimeError):
    """alfred's run of the recipe failed, or did not finish in time: one line."""


@dataclass(frozen=True)
class Draft:
    """A draft alfred left, offered as an excerpt: `number` counts drafts from 1
    in timeline order, `id` is the draft's own project id, `name` and
    `description` are alfred's, `start` and `end` are the stretch of the
    source it took in the source's seconds, `captions` what is said in it in
    its own seconds, `scores` what the recipe scored it by, and `label` the
    category a labeller gave it, empty until one does."""

    number: int
    id: str
    name: str
    description: str
    start: float
    end: float
    captions: tuple[dict[str, Any], ...]
    scores: dict[str, float]
    label: str = ""

    @property
    def seconds(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def text(self) -> str:
        return " ".join(str(c.get("text") or "").strip() for c in self.captions).strip()

    @property
    def opening(self) -> str:
        return opening(self.text)

    @property
    def title(self) -> str:
        """What alfred's name says beyond the source's: the topic the language
        engine had live over the draft, or the recipe's label and the number."""
        return self.name.split(" – ", 1)[-1] if " – " in self.name else self.name


@dataclass(frozen=True)
class Proposal:
    """What alfred's run left: the drafts in timeline order, the run's `notes`,
    and the `settings` the recipe was run with."""

    drafts: tuple[Draft, ...]
    notes: tuple[str, ...]
    settings: dict[str, float]


# --- the transcript ------------------------------------------------------------------


def read_transcript(
    invoke: Invoke,
    project_id: str,
    *,
    transcribe: bool = True,
    **limits: Any,
) -> Transcribed:
    """The project's captions as alfred holds them, or made by its audio engine.

    Captions alfred holds are used as they are, typed or transcribed, unless a
    transcription of the project is running now, which is followed instead. A
    project with none is transcribed as the caption flow transcribes it
    (`alfred_captions.transcribe`, which `limits` reach) when `transcribe` is
    true, and ends the run as `NO_CAPTIONS` otherwise. alfred cuts a source by
    its words, which a transcription leaves beside the captions.
    """
    held = invoke("alfred_captions", {"project_id": project_id})
    captions = held.get("captions") or []
    job = held.get("job") if isinstance(held.get("job"), dict) else None
    state = (job or {}).get("state")
    if captions and state not in ("queued", "transcribing"):
        return Transcribed(DONE, list(captions), job, 0.0)
    if not transcribe:
        return Transcribed(NO_CAPTIONS, [], job, 0.0,
                           "the project holds no captions, and this run was not asked to "
                           "transcribe it")
    return alfred_captions.transcribe(invoke, project_id, **limits)


# --- the proposal ----------------------------------------------------------------------


def _time(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def opening(text: str, limit: int = OPENING_CHARS) -> str:
    """The first sentence of `text`, or its first `limit` characters when the
    sentence runs longer, with an ellipsis where it was cut."""
    text = " ".join(text.split())
    found = _FIRST_SENTENCE.match(text)
    first = found.group(1) if found else text
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def _span_of(listed: Mapping[str, Any]) -> tuple[float, float]:
    """Where a listed draft lies in its source: from its first span's start to
    its last span's end; 0 to 0 for one alfred lists without spans."""
    spans = [s for s in (listed.get("source") or {}).get("spans") or []
             if isinstance(s, (list, tuple)) and len(s) == 2
             and _time(s[0]) is not None and _time(s[1]) is not None]
    if not spans:
        return 0.0, 0.0
    return float(min(s[0] for s in spans)), float(max(s[1] for s in spans))


def _draft_ids(status: Mapping[str, Any]) -> set[str]:
    return {str(d.get("id")) for d in status.get("drafts") or [] if d.get("id")}


def propose(
    invoke: Invoke,
    project_id: str,
    *,
    pause: float = PAUSE,
    shortest: float = SHORTEST,
    longest: float = LONGEST,
    timeout: float = DRAFTS_TIMEOUT,
    poll_interval: float = DRAFTS_POLL,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Proposal:
    """The drafts alfred's `excerpts` recipe leaves of the project, run with
    `pause`, `shortest` and `longest` as its knobs.

    `alfred_drafts` queues the run; `alfred_drafts_status` is asked every
    `poll_interval` seconds until alfred's worker says done, and the drafts
    the run added, told from those the project held before, are the proposal,
    in timeline order, each with its captions read by `alfred_captions`. A
    run that fails, or is not done within `timeout` seconds, raises
    `DraftsFailedError` with alfred's reason; a tool error raises as it came.
    """
    project_id = str(project_id)
    before = _draft_ids(invoke("alfred_drafts_status", {"project_id": project_id}))
    values = {"pause": f"{pause:g}", "shortest": f"{shortest:g}", "longest": f"{longest:g}"}
    invoke("alfred_drafts", {"project_id": project_id,
                             "recipes": [{"recipe": RECIPE, "values": values}]})
    started = clock()
    while True:
        status = invoke("alfred_drafts_status", {"project_id": project_id})
        job = status.get("job") if isinstance(status.get("job"), dict) else {}
        state = job.get("state")
        if state == JOB_DONE:
            break
        if state == JOB_FAILED:
            raise DraftsFailedError(f"alfred's run of the {RECIPE} recipe failed: "
                               f"{job.get('error') or 'alfred gave no reason'}")
        waited = clock() - started
        if waited >= timeout:
            raise DraftsFailedError(f"alfred's run of the {RECIPE} recipe was still "
                               f"{state or 'unreported'} after {waited:.0f} s")
        sleep(poll_interval)
    added = [d for d in status.get("drafts") or []
             if str(d.get("id")) not in before and d.get("id")]
    added.sort(key=lambda d: _span_of(d))
    drafts = []
    for number, listed in enumerate(added, start=1):
        start, end = _span_of(listed)
        held = invoke("alfred_captions", {"project_id": str(listed["id"])})
        drafts.append(Draft(
            number=number,
            id=str(listed["id"]),
            name=str(listed.get("name") or f"draft {number}"),
            description=str(listed.get("description") or ""),
            start=start,
            end=end,
            captions=tuple(dict(c) for c in held.get("captions") or []),
            scores={str(k): float(v) for k, v in ((listed.get("source") or {}).get("scores")
                                                  or {}).items()
                    if _time(v) is not None},
        ))
    return Proposal(tuple(drafts), tuple(str(n) for n in job.get("notes") or []),
                    {"pause": pause, "shortest": shortest, "longest": longest})


# --- the labeller ----------------------------------------------------------------------

Ask = Callable[[str], str]
"""A labeller: a model's answer to one prompt, or an exception when it gives none."""


def category_prompt(draft_text: str, categories: Sequence[str]) -> str:
    return (f"Which one of these categories fits this excerpt best: {', '.join(categories)}? "
            f"Answer with the category alone.\n\n{draft_text[:LABEL_CHARS]}")


def category_from(answer: str, categories: Sequence[str]) -> str:
    """The category `answer` names: the first of `categories`, longest first,
    found in it whatever the case; empty when it names none."""
    lowered = answer.lower()
    for category in sorted(categories, key=len, reverse=True):
        if category.lower() in lowered:
            return category
    return ""


def label(proposal: Proposal, ask: Ask, *, categories: Sequence[str] = ()) -> Proposal:
    """`proposal` with each draft sorted into one of `categories` by `ask`, when
    any are given; a draft's title is alfred's, from its name. An answer that
    names no category, or an `ask` that raises, leaves that label empty, so a
    model that stalls on one prompt leaves the rest of the proposal readable."""
    if not categories:
        return proposal

    def answer(prompt: str) -> str:
        try:
            return str(ask(prompt) or "")
        except Exception:  # noqa: BLE001 -- whatever a model's client raises
            return ""

    drafts = [replace(draft, label=category_from(answer(category_prompt(draft.text, categories)),
                                                 categories))
              for draft in proposal.drafts]
    return replace(proposal, drafts=tuple(drafts))


def local_model(
    base_url: str,
    model: str,
    *,
    timeout: float = LABEL_TIMEOUT,
    max_tokens: int = LABEL_TOKENS,
    api_key: str = "ollama",
    transport: httpx.BaseTransport | None = None,
) -> Ask:
    """An `Ask` over the OpenAI-style chat route a local model serves, as
    `qmcp localmodel` stands one up: one short answer per prompt, at a
    temperature of 0 and at most `max_tokens`, within `timeout` seconds."""
    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport,
                          headers={"Authorization": f"Bearer {api_key}"})

    def ask(prompt: str) -> str:
        response = client.post("/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return ask


# --- the excerpt ------------------------------------------------------------------------


def excerpt_description(project_name: str | None, draft: Draft) -> str:
    """What alfred's Deliver stage shows under the excerpt's name: where it lies
    in the recording, and its label when it has one."""
    where = f"{timecode(draft.start)}-{timecode(draft.end)} of {project_name or 'the recording'!r}"
    return f"{where}: {draft.label}" if draft.label else where


def queue_excerpt(
    invoke: Invoke,
    project: dict[str, Any],
    draft: Draft,
    *,
    captions: bool = True,
    **options: Any,
) -> dict[str, Any]:
    """Queue a render of the draft as an excerpt of its own: `alfred_render_enqueue`
    of the draft's own saved timeline, under alfred's name for it, with the
    draft's captions burned in when `captions` is true and it has any -- the
    captions alfred holds on the draft when the excerpt is queued, so a
    correction saved in alfred's editor during the review is in the excerpt.
    `options` left as None are the draft's saved settings, which are its
    source's, else alfred's. Returns what the tool returned."""
    params: dict[str, Any] = {
        "project_id": draft.id,
        "name": draft.name,
        "description": excerpt_description(project.get("name"), draft),
        "captions": bool(captions and draft.captions),
    }
    for key in ("width", "height", "fps", "quality"):
        if options.get(key) is not None:
            params[key] = options[key]
    return invoke("alfred_render_enqueue", params)


def drop(invoke: Invoke, project_id: Any, drafts: Sequence[Draft]) -> dict[str, Any]:
    """Let go of `drafts`, each with its cue on the source: `alfred_drafts_drop`.
    Nothing is asked for none. Returns what the tool returned."""
    if not drafts:
        return {"project_id": str(project_id), "dropped": []}
    return invoke("alfred_drafts_drop", {"project_id": str(project_id),
                                         "drafts": [draft.id for draft in drafts]})


# --- the cues on alfred's timeline -----------------------------------------------------

# Whose the cues are: this flow's, so a run replaces the last run's and leaves
# what anything else marked, alfred's own `alfred-drafts` cues among them.
CUE_SOURCE = "alfred-excerpts"
# Where a draft stands, as its cue says: proposed and not yet asked about;
# waiting for an answer on the queue; approved or rejected by it, or
# unanswered when the wait ran out; queued as a render; rendered, or failed;
# and delivered.
PROPOSED = "proposed"
WAITING = "waiting"
APPROVED = "approved"
REJECTED = "rejected"
UNANSWERED = "unanswered"
QUEUED = "queued"
RENDERED = "rendered"
FAILED = "failed"
DELIVERED = "delivered"


def cues_of(
    proposal: Proposal,
    states: Mapping[int, str] | None = None,
    refs: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """The proposal as cues on the project's timeline: one per draft, over the
    stretch of the source it took, labelled by its number and its category,
    with its `state` and `ref` from `states` and `refs` by draft number --
    `proposed`, and no reference, until they say otherwise -- and the opening
    of what is said in it as its note."""
    states = states or {}
    refs = refs or {}
    cues: list[dict[str, Any]] = []
    for draft in proposal.drafts:
        label = f"Excerpt {draft.number}" + (f": {draft.label}" if draft.label else "")
        cue: dict[str, Any] = {
            "start": draft.start,
            "end": draft.end,
            "label": label,
            "kind": "excerpt",
            "state": states.get(draft.number, PROPOSED),
            "note": draft.opening,
        }
        if refs.get(draft.number):
            cue["ref"] = refs[draft.number]
        cues.append(cue)
    return cues


def mark(
    invoke: Invoke,
    project_id: Any,
    proposal: Proposal,
    states: Mapping[int, str] | None = None,
    refs: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Draw the proposal on the project's timeline as this flow's cues,
    replacing the last run's: `alfred_cues_set` with `cues_of`. Returns what
    the tool returned."""
    return invoke("alfred_cues_set", {
        "project_id": str(project_id),
        "cues": cues_of(proposal, states, refs),
        "source": CUE_SOURCE,
    })


# --- what a person is asked --------------------------------------------------------------


def _placed(draft: Draft) -> list[dict[str, Any]]:
    """The draft's captions in the source's seconds, for a person reading the
    recording's clock."""
    return [{**c, "start": round(float(c.get("start") or 0.0) + draft.start, 3),
             "end": round(float(c.get("end") or 0.0) + draft.start, 3)}
            for c in draft.captions]


def review_request(project: dict[str, Any], draft: Draft, proposal: Proposal,
                   run_id: str) -> dict[str, Any]:
    """What a person is asked about one draft: the request's id, prompt, options
    and context, as `MCPClient.create_human_request` takes them. The prompt
    places the draft in the recording, names it as alfred did and quotes its
    words; the context's `spoken` is the same question in a breath, for a
    voice reader, which speaks it in place of the prompt."""
    project_id = str(project.get("id"))
    name = project.get("name") or project_id
    labelled = f", {draft.label}" if draft.label else ""
    lines = "\n".join(f"  {line}" for line in transcript(_placed(draft)))
    prompt = (
        f"Excerpt {draft.number} of {len(proposal.drafts)} from {name!r}, "
        f"{timecode(draft.start)}-{timecode(draft.end)} ({draft.seconds:.0f} s{labelled}), "
        f"alfred's {draft.title!r}:\n{lines}\n"
        "Answer approve to render it as an excerpt of its own to review, or reject to drop it."
    )
    spoken = (
        f"Excerpt {draft.number} of {len(proposal.drafts)} from {name}, "
        f"{draft.seconds:.0f} seconds{labelled}, {draft.title}. It opens: {draft.opening} "
        "Approve to render it as an excerpt of its own, or reject to drop it."
    )
    return {
        "request_id": f"alfred-excerpt-{project_id}-{draft.number}-{run_id}",
        "prompt": prompt,
        "options": list(OPTIONS),
        "context": {
            "spoken": spoken,
            "project_id": project_id,
            "name": name,
            "excerpt": draft.number,
            "of": len(proposal.drafts),
            "draft": draft.id,
            "draft_name": draft.name,
            "start": draft.start,
            "end": draft.end,
            "seconds": draft.seconds,
            "label": draft.label,
            "captions": list(draft.captions),
            "flow_run_id": run_id,
        },
    }


def approval_request(
    queued: dict[str, Any],
    waited: Waited,
    draft: Draft,
    run_id: str,
    *,
    deliver: bool = False,
    destination: str | None = None,
) -> dict[str, Any]:
    """What a person is asked about a rendered excerpt: `alfred_render`'s request,
    with the link and when it stops working, placing the excerpt in the recording
    and saying what approval does -- deliver the excerpt, when the run was asked
    to, or nothing more."""
    request = alfred_render.approval_request(queued, waited, run_id)
    where = f" It is {timecode(draft.start)}-{timecode(draft.end)} of the recording"
    where += f", {draft.label}." if draft.label else "."
    if deliver:
        then = (f" Answer approve to deliver it to {destination or 'alfred\'s default destination'}"
                ", or reject to leave it undelivered.")
    else:
        then = " Nothing is delivered either way: this run was not asked to deliver."
    request["request_id"] = f"alfred-excerpt-render-{queued['render_id']}-{run_id}"
    request["prompt"] += where + then
    # The prompt carries a link, which is for a page; the ear gets the place.
    name = queued.get("name") or queued.get("render_id")
    spoken = f"{name} is rendered.{where}{then}"
    request["context"].update(spoken=spoken, excerpt=draft.number, draft=draft.id,
                              start=draft.start, end=draft.end, label=draft.label,
                              captions=queued.get("captions"), deliver=deliver,
                              destination=destination)
    return request


# --- how the run ended -------------------------------------------------------------------


def counted(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def summary(
    project: dict[str, Any],
    transcribed: Transcribed,
    proposal: Proposal | None = None,
    reviewed: dict[int, str | None] | None = None,
    queued: dict[int, dict[str, Any]] | None = None,
    waited: dict[int, Waited] | None = None,
    answers: dict[int, str | None] | None = None,
    delivered: dict[int, Delivered] | None = None,
    *,
    deliver: bool = False,
    failure: str | None = None,
) -> str:
    """One line saying how the run ended."""
    project_id = str(project.get("id"))
    name = project.get("name") or project_id
    if not transcribed.done:
        return alfred_captions.summary(project_id, transcribed)
    if failure:
        return sentence(f"Nothing was asked: {failure}")
    if proposal is None or not proposal.drafts:
        notes = f": {'; '.join(proposal.notes)}" if proposal and proposal.notes else ""
        return sentence(f"Nothing was asked: the {RECIPE} recipe left no draft of {name!r} "
                        f"({counted(len(transcribed.captions), 'caption')}){notes}")
    reviewed, queued, waited = reviewed or {}, queued or {}, waited or {}
    answers, delivered = answers or {}, delivered or {}
    approved = sum(answer == APPROVE for answer in reviewed.values())
    dropped = sum(answer == REJECT for answer in reviewed.values())
    unanswered = sum(answer is None for answer in reviewed.values())
    rendered = sum(w.rendered for w in waited.values())
    kept = sum(answer == APPROVE for answer in answers.values())
    parts = [f"{counted(len(proposal.drafts), 'excerpt')} of {name!r} offered",
             f"{approved} approved"]
    if unanswered:
        parts[-1] += f" ({unanswered} unanswered)"
    if dropped:
        parts.append(f"{dropped} dropped")
    if approved:
        parts.append(f"{rendered} rendered")
        failed = [f"excerpt {n} {w.why}" for n, w in sorted(waited.items()) if not w.rendered]
        if failed:
            parts[-1] += f" ({'; '.join(failed)})"
        parts.append(f"{kept} approved as excerpts")
        if deliver:
            went = sum(d.ok for d in delivered.values())
            parts.append(f"{went} delivered")
            refused = [f"excerpt {n}: {d.error}" for n, d in sorted(delivered.items()) if not d.ok]
            if refused:
                parts[-1] += f" ({'; '.join(refused)})"
    return sentence(", ".join(parts))


__all__ = [
    "APPROVE",
    "DRAFTS_POLL",
    "DRAFTS_TIMEOUT",
    "LABEL_TIMEOUT",
    "LABEL_TOKENS",
    "LONGEST",
    "PAUSE",
    "RECIPE",
    "REJECT",
    "SHORTEST",
    "Ask",
    "Draft",
    "DraftsFailedError",
    "Proposal",
    "approval_request",
    "category_from",
    "cues_of",
    "drop",
    "excerpt_description",
    "label",
    "local_model",
    "mark",
    "opening",
    "propose",
    "queue_excerpt",
    "read_transcript",
    "review_request",
    "summary",
]
