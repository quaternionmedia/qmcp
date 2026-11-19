"""Cutting a long alfred project into clips a person chooses: the steps, apart from any flow.

`examples/flows/alfred_excerpts.py` runs them as a Metaflow flow, one step each:

    read      `alfred_project` for the saved timeline, and `alfred_captions`
              for the transcript -- made first, as the caption flow makes it,
              when the project holds none
    propose   chapters and spans from the transcript's own timing: a chapter
              ends at a long pause, a span at a pause or at a sentence's end,
              within the length a clip may have; a labeller, when the run has
              one, titles each chapter and sorts each span into a category
    review    one request on qmcp's human queue per span, carrying its words:
              approve renders it as a clip of its own, reject leaves it out
    render    `alfred_render_enqueue` with a timeline cut from the project's
              for each approved span, its captions burned in when asked, then
              `alfred_render_status` until each settles (`alfred_render`)
    ask       a request per rendered clip, with its link, saying what
              approval does
    deliver   `alfred_render_deliver` for each approved clip, when the run was
              asked to deliver

Every call to alfred is a qmcp tool invocation, so the run is in qmcp's
record. The steps take the tool caller (`Invoke`) and the queue client as
arguments, a wait takes its `sleep` and `clock`, and a labeller is a function
from a prompt to an answer, so a test drives them without a flow, a model or
alfred. Nothing here imports Metaflow.

What a span is comes from the transcript alone: when the words were said and
whether a caption ends a sentence. Nothing here hears the sound, so a song the
engine wrote no words for is a pause, and where it falls is a chapter's edge.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
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

# Seconds of nothing said that end a span: a breath between thoughts.
PAUSE = 1.0
# Seconds of nothing said that end a chapter: a change of subject, or a song.
CHAPTER_PAUSE = 4.0
# The shortest and longest a span may run, in seconds. A stretch shorter than
# `SHORTEST` is joined to a neighbour when that fits, else left out; one
# longer than `LONGEST` is cut at the ends of sentences.
SHORTEST = 10.0
LONGEST = 90.0
# How much of a span or a chapter a prompt quotes, in characters.
OPENING_CHARS = 80
TITLE_CHARS = 60
LABEL_CHARS = 4000
# What a labeller may say back, in tokens; a title or a category needs few.
LABEL_TOKENS = 32
# Seconds one answer from a model may take before the span stays unlabelled.
LABEL_TIMEOUT = 30.0

# What ends a sentence, once closing quotes and brackets are set aside.
_SENTENCE_END = (".", "!", "?", "…")
_CLOSERS = "\"')”’»]"
_FIRST_SENTENCE = re.compile(r"^(.*?[.!?…][\"')”’»\]]*)(?:\s|$)")


@dataclass(frozen=True)
class Span:
    """A stretch of the transcript offered as a clip: `number` counts spans
    from 1 in timeline order, `chapter` is the chapter's number from 1,
    `start` and `end` are timeline seconds, `captions` what is said in it, and
    `label` the category a labeller gave it, empty until one does."""

    number: int
    chapter: int
    start: float
    end: float
    captions: tuple[dict[str, Any], ...]
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


@dataclass(frozen=True)
class Chapter:
    """A part of the recording between long pauses: `number` from 1, its
    `start` and `end` in timeline seconds, how many `spans` it holds, the
    `opening` of what is said in it, and the `title` a labeller gave it,
    empty until one does."""

    number: int
    start: float
    end: float
    spans: int
    opening: str
    title: str = ""


@dataclass(frozen=True)
class Proposal:
    """What the transcript offers: its chapters, its spans in timeline order,
    how many stretches were `left_out` for being too short to stand alone,
    and the `settings` they were cut with."""

    chapters: tuple[Chapter, ...]
    spans: tuple[Span, ...]
    left_out: int
    settings: dict[str, float]

    def chapter_of(self, span: Span) -> Chapter:
        return self.chapters[span.chapter - 1]


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
    true, and ends the run as `NO_CAPTIONS` otherwise.
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


def spoken(captions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The captions that say something, each with a start and an end after it,
    in the order they start. Anything else is not a stretch of speech."""
    kept = []
    for caption in captions:
        start, end = _time(caption.get("start")), _time(caption.get("end"))
        text = str(caption.get("text") or "").strip()
        if start is None or end is None or end <= start or not text:
            continue
        kept.append({**caption, "start": start, "end": end, "text": text})
    return sorted(kept, key=lambda c: (c["start"], c["end"]))


def ends_sentence(text: str) -> bool:
    """Whether `text` ends a sentence: a full stop, a question or exclamation
    mark, or an ellipsis, with any closing quote or bracket after it."""
    return text.rstrip().rstrip(_CLOSERS).endswith(_SENTENCE_END)


def opening(text: str, limit: int = OPENING_CHARS) -> str:
    """The first sentence of `text`, or its first `limit` characters when the
    sentence runs longer, with an ellipsis where it was cut."""
    text = " ".join(text.split())
    found = _FIRST_SENTENCE.match(text)
    first = found.group(1) if found else text
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def stretches(captions: Sequence[dict[str, Any]], gap: float) -> list[list[dict[str, Any]]]:
    """`captions`, already `spoken`, cut wherever `gap` seconds or more pass
    between one caption's end and the next's start."""
    runs: list[list[dict[str, Any]]] = []
    for caption in captions:
        if runs and caption["start"] - runs[-1][-1]["end"] < gap:
            runs[-1].append(caption)
        else:
            runs.append([caption])
    return runs


def _length(run: Sequence[dict[str, Any]]) -> float:
    return run[-1]["end"] - run[0]["start"]


def pieces(stretch: Sequence[dict[str, Any]], longest: float) -> list[list[dict[str, Any]]]:
    """One stretch cut into runs of at most `longest` seconds, each ending at
    the last caption within that length that ends a sentence, or at the last
    caption within it when none does. A caption longer than `longest` on its
    own is a piece on its own."""
    cut: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(stretch):
        start = stretch[i]["start"]
        last_fit, last_sentence = i, None
        j = i
        while j < len(stretch) and stretch[j]["end"] - start <= longest:
            last_fit = j
            if ends_sentence(stretch[j]["text"]):
                last_sentence = j
            j += 1
        end = last_fit if last_sentence is None else last_sentence
        cut.append(list(stretch[i:end + 1]))
        i = end + 1
    return cut


def joined(pieces_: Sequence[Sequence[dict[str, Any]]], shortest: float,
           longest: float) -> tuple[list[list[dict[str, Any]]], int]:
    """The pieces of one stretch with each one shorter than `shortest` joined
    to the piece before it, else to the one after it, where the join stays
    within `longest`; what cannot be joined is left out, and counted."""
    kept: list[list[dict[str, Any]]] = []
    left_out = 0
    pending: list[dict[str, Any]] = []
    for piece in pieces_:
        piece = pending + list(piece)
        pending = []
        if _length(piece) >= shortest:
            kept.append(piece)
        elif kept and _length(kept[-1] + piece) <= longest:
            kept[-1] = kept[-1] + piece
        else:
            pending = piece
    if pending:
        left_out += 1
    return kept, left_out


def propose(
    captions: Sequence[dict[str, Any]],
    *,
    pause: float = PAUSE,
    chapter_pause: float = CHAPTER_PAUSE,
    shortest: float = SHORTEST,
    longest: float = LONGEST,
) -> Proposal:
    """Chapters and spans from the transcript's timing.

    A chapter is what lies between pauses of `chapter_pause` seconds or more.
    Within it, a stretch is what lies between pauses of `pause` seconds or
    more, and a stretch is a span when it runs `shortest` seconds or more and
    `longest` or fewer; one that runs longer is cut at the ends of sentences
    (`pieces`), and one that runs shorter is joined to a neighbour or left out
    (`joined`). Spans and chapters are numbered from 1 in timeline order.
    """
    if not 0 < shortest <= longest:
        raise ValueError(f"shortest ({shortest}) must be positive and at most longest ({longest})")
    if not 0 < pause <= chapter_pause:
        raise ValueError(f"pause ({pause}) must be positive and at most chapter_pause "
                         f"({chapter_pause})")
    chapters: list[Chapter] = []
    spans: list[Span] = []
    left_out = 0
    for part in stretches(spoken(captions), chapter_pause):
        number = len(chapters) + 1
        held = 0
        for stretch in stretches(part, pause):
            kept, dropped = joined(pieces(stretch, longest), shortest, longest)
            left_out += dropped
            for run in kept:
                spans.append(Span(len(spans) + 1, number, run[0]["start"], run[-1]["end"],
                                  tuple(run)))
                held += 1
        chapters.append(Chapter(number, part[0]["start"], part[-1]["end"], held,
                                opening(" ".join(c["text"] for c in part))))
    return Proposal(tuple(chapters), tuple(spans), left_out,
                    {"pause": pause, "chapter_pause": chapter_pause,
                     "shortest": shortest, "longest": longest})


# --- the labeller ----------------------------------------------------------------------

Ask = Callable[[str], str]
"""A labeller: a model's answer to one prompt, or an exception when it gives none."""


def title_prompt(chapter_text: str) -> str:
    return ("Give a title of at most six words to this part of a recording. Answer with "
            f"the title alone.\n\n{chapter_text[:LABEL_CHARS]}")


def category_prompt(span_text: str, categories: Sequence[str]) -> str:
    return (f"Which one of these categories fits this clip best: {', '.join(categories)}? "
            f"Answer with the category alone.\n\n{span_text[:LABEL_CHARS]}")


def title_from(answer: str) -> str:
    """The first line of `answer`, its quotes dropped, at most `TITLE_CHARS`."""
    for line in answer.splitlines():
        line = line.strip().strip("\"'“”‘’").strip()
        if line:
            return line if len(line) <= TITLE_CHARS else line[: TITLE_CHARS - 1].rstrip() + "…"
    return ""


def category_from(answer: str, categories: Sequence[str]) -> str:
    """The category `answer` names: the first of `categories`, longest first,
    found in it whatever the case; empty when it names none."""
    lowered = answer.lower()
    for category in sorted(categories, key=len, reverse=True):
        if category.lower() in lowered:
            return category
    return ""


def label(proposal: Proposal, ask: Ask, *, categories: Sequence[str] = ()) -> Proposal:
    """`proposal` with each chapter titled by `ask`, and each span sorted into
    one of `categories` by it, when any are given. An answer that names no
    category, or an `ask` that raises, leaves that title or label empty, so a
    model that stalls on one prompt leaves the rest of the proposal readable."""
    def answer(prompt: str) -> str:
        try:
            return str(ask(prompt) or "")
        except Exception:  # noqa: BLE001 -- whatever a model's client raises
            return ""

    chapters = []
    for chapter in proposal.chapters:
        words = " ".join(span.text for span in proposal.spans if span.chapter == chapter.number)
        chapters.append(replace(chapter, title=title_from(answer(title_prompt(words or
                                                                               chapter.opening)))))
    spans = list(proposal.spans)
    if categories:
        spans = [replace(span, label=category_from(answer(category_prompt(span.text, categories)),
                                                   categories))
                 for span in spans]
    return replace(proposal, chapters=tuple(chapters), spans=tuple(spans))


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


# --- the clip's timeline ---------------------------------------------------------------


def _number(value: Any, default: float = 0.0) -> float:
    found = _time(value)
    return default if found is None else found


def excerpt_edl(edl: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    """The part of `edl` between `start` and `end`, in timeline seconds, as a
    timeline of its own that begins at 0.

    Each clip that plays in the span is kept, with every field it had, and
    re-timed as otto reads a clip (`otto.render.clipTiming`): its `start`
    moved by the span's, its `inpoint` and `outpoint` set to the part of its
    source the span holds, and `duration` their difference; a deprecated
    `offset` is folded into them. An image keeps a `duration` alone, being
    one picture throughout. An edge the span cut gets no fade (`fadeIn` or
    `fadeOut` 0), as alfred's own cuts have none; an edge the span did not
    cut keeps the clip's. The span is held to the timeline's `duration` when
    it has one, and a span with nothing in it is refused.
    """
    duration = _time(edl.get("duration"))
    if duration is not None and end > duration:
        end = duration
    if not 0 <= start < end:
        raise ValueError(f"nothing to cut between {start} and {end} seconds")
    clips = []
    for index, clip in enumerate(edl.get("clips") or []):
        offset = _number(clip.get("offset"))
        clip_start = _number(clip.get("start")) + max(offset, 0.0)
        inpoint = _number(clip.get("inpoint")) + max(-offset, 0.0)
        outpoint = _time(clip.get("outpoint"))
        length = outpoint - inpoint if outpoint is not None else _number(clip.get("duration"), 5.0)
        if length <= 0:
            raise ValueError(f"clip {index} ({clip.get('name')!r}) has no length: "
                             "otto would refuse the timeline")
        clip_end = clip_start + length
        low, high = max(clip_start, start), min(clip_end, end)
        if high <= low:
            continue
        cut = {key: value for key, value in clip.items() if key != "offset"}
        cut["start"] = round(low - start, 3)
        cut["duration"] = round(high - low, 3)
        if clip.get("type") == "image":
            cut.pop("inpoint", None)
            cut.pop("outpoint", None)
        else:
            cut["inpoint"] = round(inpoint + (low - clip_start), 3)
            cut["outpoint"] = round(cut["inpoint"] + (high - low), 3)
        if low > clip_start:
            cut["fadeIn"] = 0
        if high < clip_end:
            cut["fadeOut"] = 0
        clips.append(cut)
    if not clips:
        raise ValueError(f"no clip of the timeline plays between {start} and {end} seconds")
    return {"clips": clips, "duration": round(end - start, 3)}


def excerpt_captions(captions: Sequence[dict[str, Any]], start: float,
                     end: float) -> list[dict[str, Any]]:
    """The captions that lie between `start` and `end`, in the clip's own
    seconds: moved by `start`, trimmed where the span cuts one, and dropped
    where nothing of one is left."""
    kept = []
    for caption in spoken(captions):
        low, high = max(caption["start"], start), min(caption["end"], end)
        if high <= low:
            continue
        kept.append({**caption, "start": round(low - start, 3), "end": round(high - start, 3)})
    return kept


def clip_name(project_name: str | None, span: Span) -> str:
    return f"{project_name or 'alfred project'} - clip {span.number}"


def clip_description(project_id: str, span: Span) -> str:
    where = (f"{timecode(span.start)}-{timecode(span.end)} of alfred project {project_id}, "
             f"chapter {span.chapter}")
    return f"{where}: {span.label}" if span.label else where


def queue_clip(
    invoke: Invoke,
    project: dict[str, Any],
    span: Span,
    *,
    captions: Sequence[dict[str, Any]] | None = None,
    style: dict[str, Any] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Queue a render of the span as a clip of its own: `alfred_render_enqueue`
    with the timeline cut from the project's saved one, named for the project
    and the span, with the project's captions that fall in it burned in when
    `captions` are given, in `style`. `options` left as None are the project's
    saved settings, else alfred's. Returns what the tool returned."""
    edl = excerpt_edl(project["edl"], span.start, span.end)
    params: dict[str, Any] = {
        "edl": edl,
        "name": clip_name(project.get("name"), span),
        "description": clip_description(str(project.get("id")), span),
    }
    saved = project.get("render_settings") or {}
    for key in ("width", "height", "fps", "quality"):
        value = options.get(key)
        value = value if value is not None else saved.get(key)
        if value is not None:
            params[key] = value
    if captions is not None:
        # Held to the clip, as the timeline was: a caption past its end is left.
        burned = excerpt_captions(captions, span.start, span.start + edl["duration"])
        if burned:
            params["captions"] = burned
            params["caption_style"] = dict(style or {})
    return invoke("alfred_render_enqueue", params)


# --- what a person is asked --------------------------------------------------------------


def review_request(project: dict[str, Any], span: Span, proposal: Proposal,
                   run_id: str) -> dict[str, Any]:
    """What a person is asked about one span: the request's id, prompt, options
    and context, as `MCPClient.create_human_request` takes them. The prompt
    places the span in the recording and quotes its words."""
    project_id = str(project.get("id"))
    name = project.get("name") or project_id
    chapter = proposal.chapter_of(span)
    place = f"chapter {span.chapter} of {len(proposal.chapters)}"
    if chapter.title:
        place += f" ({chapter.title})"
    labelled = f", {span.label}" if span.label else ""
    lines = "\n".join(f"  {line}" for line in transcript(list(span.captions)))
    prompt = (
        f"Clip {span.number} of {len(proposal.spans)} from {name!r}, {place}, "
        f"{timecode(span.start)}-{timecode(span.end)} ({span.seconds:.0f} s{labelled}):\n{lines}\n"
        "Answer approve to render it as a clip of its own to review, or reject to leave it out."
    )
    return {
        "request_id": f"alfred-excerpt-{project_id}-{span.number}-{run_id}",
        "prompt": prompt,
        "options": list(OPTIONS),
        "context": {
            "project_id": project_id,
            "name": name,
            "clip": span.number,
            "of": len(proposal.spans),
            "chapter": span.chapter,
            "chapter_title": chapter.title,
            "start": span.start,
            "end": span.end,
            "seconds": span.seconds,
            "label": span.label,
            "captions": list(span.captions),
            "flow_run_id": run_id,
        },
    }


def approval_request(
    queued: dict[str, Any],
    waited: Waited,
    span: Span,
    run_id: str,
    *,
    deliver: bool = False,
    destination: str | None = None,
) -> dict[str, Any]:
    """What a person is asked about a rendered clip: `alfred_render`'s request,
    with the link and when it stops working, placing the clip in the recording
    and saying what approval does -- deliver the clip, when the run was asked
    to, or nothing more."""
    request = alfred_render.approval_request(queued, waited, run_id)
    where = f" It is {timecode(span.start)}-{timecode(span.end)} of the recording"
    where += f", {span.label}." if span.label else "."
    if deliver:
        then = (f" Answer approve to deliver it to {destination or 'alfred\'s default destination'}"
                ", or reject to leave it undelivered.")
    else:
        then = " Nothing is delivered either way: this run was not asked to deliver."
    request["request_id"] = f"alfred-excerpt-render-{queued['render_id']}-{run_id}"
    request["prompt"] += where + then
    request["context"].update(clip=span.number, start=span.start, end=span.end, label=span.label,
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
) -> str:
    """One line saying how the run ended."""
    project_id = str(project.get("id"))
    name = project.get("name") or project_id
    if not transcribed.done:
        return alfred_captions.summary(project_id, transcribed)
    if proposal is None or not proposal.spans:
        shortest = (proposal.settings if proposal else {}).get("shortest", SHORTEST)
        return (f"Nothing was asked: the transcript of {name!r} "
                f"({counted(len(transcribed.captions), 'caption')}) has no stretch of "
                f"{shortest:g} s or more to offer as a clip.")
    reviewed, queued, waited = reviewed or {}, queued or {}, waited or {}
    answers, delivered = answers or {}, delivered or {}
    approved = sum(answer == APPROVE for answer in reviewed.values())
    unanswered = sum(answer is None for answer in reviewed.values())
    rendered = sum(w.rendered for w in waited.values())
    kept = sum(answer == APPROVE for answer in answers.values())
    parts = [f"{counted(len(proposal.spans), 'clip')} in "
             f"{counted(len(proposal.chapters), 'chapter')} of {name!r} offered",
             f"{approved} approved"]
    if unanswered:
        parts[-1] += f" ({unanswered} unanswered)"
    if approved:
        parts.append(f"{rendered} rendered")
        failed = [f"clip {n} {w.why}" for n, w in sorted(waited.items()) if not w.rendered]
        if failed:
            parts[-1] += f" ({'; '.join(failed)})"
        parts.append(f"{kept} approved as clips")
        if deliver:
            went = sum(d.ok for d in delivered.values())
            parts.append(f"{went} delivered")
            refused = [f"clip {n}: {d.error}" for n, d in sorted(delivered.items()) if not d.ok]
            if refused:
                parts[-1] += f" ({'; '.join(refused)})"
    return sentence(", ".join(parts))


__all__ = [
    "APPROVE",
    "CHAPTER_PAUSE",
    "LABEL_TIMEOUT",
    "LABEL_TOKENS",
    "LONGEST",
    "PAUSE",
    "REJECT",
    "SHORTEST",
    "Ask",
    "Chapter",
    "Proposal",
    "Span",
    "approval_request",
    "category_from",
    "clip_description",
    "clip_name",
    "ends_sentence",
    "excerpt_captions",
    "excerpt_edl",
    "joined",
    "label",
    "local_model",
    "opening",
    "pieces",
    "propose",
    "queue_clip",
    "read_transcript",
    "review_request",
    "spoken",
    "stretches",
    "summary",
    "title_from",
]
