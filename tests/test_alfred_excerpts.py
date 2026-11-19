"""The excerpt flow: a transcript cut into chapters and spans, a labeller over
them, the clip's timeline cut from the project's, the transcript read or made,
each request a person gets, how a run ends, and the flow file's steps end to
end through a qmcp server and its queue.

alfred is `CaptionedAlfred` in `tests/alfred_stub.py`. The steps are tested
through `qmcp.cookbook.alfred_excerpts`, and the branches by running the steps
of `examples/flows/alfred_excerpts.py` itself, over a stand-in for Metaflow,
which cannot be imported where `import fcntl` fails.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import re
import sys
import types
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import qmcp.tools.alfred as alfred
from qmcp.client import MCPClient
from qmcp.cookbook import alfred_captions, alfred_render
from qmcp.cookbook import alfred_excerpts as flow
from qmcp.cookbook.alfred_captions import DONE, NO_CAPTIONS, Delivered, Transcribed
from qmcp.cookbook.alfred_excerpts import (
    APPROVE,
    REJECT,
    Chapter,
    Proposal,
    Span,
    category_from,
    ends_sentence,
    excerpt_captions,
    excerpt_edl,
    label,
    local_model,
    opening,
    propose,
    queue_clip,
    read_transcript,
    review_request,
    spoken,
    summary,
    title_from,
)
from qmcp.cookbook.alfred_render import RENDERED, ToolFailedError, Waited
from tests.alfred_stub import CAPTIONS, URL, CaptionedAlfred, connect, render_id

REPO = Path(__file__).resolve().parents[1]


def caption(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


def said(*lines: tuple[float, str], gap: float = 0.4, at: float = 0.0) -> list[dict]:
    """Captions one after another from `at`, each `(seconds, text)`, `gap`
    seconds of nothing between them."""
    captions, now = [], at
    for seconds, text in lines:
        captions.append(caption(round(now, 3), round(now + seconds, 3), text))
        now += seconds + gap
    return captions


# A recording in three parts: a welcome too short to stand alone, a long story
# that must be cut, and a goodbye too short to stand alone.
WELCOME = said((2.0, "Welcome to the show."), (3.0, "Tonight we have three songs for you."))
STORY = said(*[(7.0, f"This is sentence {n} of the story, which goes on for a while")
               if n % 2 == 0 else (2.0, "and ends here.") for n in range(48)], at=11.0)
GOODBYE = said((2.0, "Thanks for coming."), (1.5, "Good night!"), at=STORY[-1]["end"] + 6.0)
RECORDING = WELCOME + STORY + GOODBYE


# --- the module and the flow file ----------------------------------------------


def test_the_steps_import_nothing_from_metaflow():
    """Asserted on the source, because the failure is an import line existing."""
    source = Path(flow.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]
    assert not [line for line in imports if "metaflow" in line]


def test_the_flow_file_compiles_and_runs_the_module_s_steps():
    """Compiled rather than imported: importing it imports Metaflow."""
    path = REPO / "examples" / "flows" / "alfred_excerpts.py"
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")
    for name in ("read_transcript", "propose", "label", "local_model", "review_request",
                 "queue_clip", "wait_for_render", "approval_request", "ask", "wait_for_answer",
                 "deliver_render", "summary"):
        assert f"{name}(" in source, name
    assert '"alfred_project"' in source


# --- what is spoken -------------------------------------------------------------------


def test_spoken_keeps_what_says_something_in_the_order_it_starts():
    """Mutation: keep a caption that ends when it starts -- red."""
    kept = spoken([caption(5.0, 6.0, "later"), caption(1.0, 1.0, "no length"),
                   caption(0.0, 2.0, "  "), {"start": "0", "end": 1, "text": "not a number"},
                   {"start": True, "end": 2, "text": "a bool"}, caption(0.0, 1.0, " first ")])
    assert kept == [caption(0.0, 1.0, "first"), caption(5.0, 6.0, "later")]


@pytest.mark.parametrize("text,ends", [
    ("A sentence.", True), ("Really?", True), ("Go!", True), ("and so on…", True),
    ('He said "stop."', True), ("(It was late.)", True), ("but then", False),
    ("a comma,", False), ("", False),
])
def test_a_sentence_ends_at_its_mark_whatever_closes_it(text, ends):
    assert ends_sentence(text) is ends


def test_an_opening_is_the_first_sentence_or_the_first_characters():
    assert opening("Good evening. We have songs.") == "Good evening."
    assert opening('He said "no." Then left.') == 'He said "no."'
    assert opening("a " * 60, limit=20) == "a a a a a a a a a a…"
    assert opening("no full stop here") == "no full stop here"


# --- the proposal ----------------------------------------------------------------------


def test_a_long_pause_ends_a_chapter_and_a_shorter_one_a_span():
    """Mutation: cut chapters at `pause` too -- red, every stretch is a chapter."""
    story = said((6.0, "One thing."), (6.0, "Another thing."), gap=2.0)
    more = said((12.0, "A new subject."), at=story[-1]["end"] + 5.0)
    proposed = propose(story + more, pause=1.0, chapter_pause=4.0, shortest=5.0, longest=60.0)
    assert [(c.number, c.start, c.end, c.spans) for c in proposed.chapters] == [
        (1, 0.0, 14.0, 2), (2, 19.0, 31.0, 1)]
    assert [(s.number, s.chapter, s.start, s.end) for s in proposed.spans] == [
        (1, 1, 0.0, 6.0), (2, 1, 8.0, 14.0), (3, 2, 19.0, 31.0)]
    assert proposed.left_out == 0
    assert proposed.settings == {"pause": 1.0, "chapter_pause": 4.0, "shortest": 5.0,
                                 "longest": 60.0}


def test_a_stretch_longer_than_longest_is_cut_at_the_last_sentence_end_that_fits():
    """Mutation: cut at the last caption that fits instead -- red, a span ends
    mid-sentence."""
    proposed = propose(RECORDING)
    story = [s for s in proposed.spans if s.chapter == 2]
    assert [(s.start, s.end) for s in story] == [(11.0, 98.8), (99.2, 187.0), (187.4, 245.8)]
    for span in story:
        assert span.seconds <= flow.LONGEST and ends_sentence(span.captions[-1]["text"])
    assert story[0].text.startswith("This is sentence 0 of the story")
    assert story[1].text.startswith("This is sentence 18 of the story")
    assert story[2].text.startswith("This is sentence 36 of the story")


def test_a_run_without_a_sentence_end_is_cut_at_the_last_caption_that_fits():
    run = said(*[(10.0, f"words {n} and more words") for n in range(5)])
    proposed = propose(run, shortest=5.0, longest=25.0)
    assert [(s.start, s.end) for s in proposed.spans] == [(0.0, 20.4), (20.8, 41.2), (41.6, 51.6)]


def test_a_caption_longer_than_longest_is_a_span_on_its_own():
    """Mutation: skip a caption that does not fit -- red, the loop never ends
    or the caption is lost."""
    proposed = propose(said((40.0, "One very long caption."), (3.0, "Then a short one.")),
                       shortest=2.0, longest=30.0)
    assert [(s.start, s.end) for s in proposed.spans] == [(0.0, 40.0), (40.4, 43.4)]


def test_a_tail_too_short_to_stand_alone_joins_the_span_before_it():
    """Mutation: leave the tail out -- red, it is counted as left out."""
    proposed = propose(said((20.0, "A whole thought."), (2.0, "And a tag.")),
                       shortest=5.0, longest=25.0)
    assert [(s.start, s.end) for s in proposed.spans] == [(0.0, 22.4)]
    assert proposed.left_out == 0


def test_a_head_too_short_to_stand_alone_joins_the_span_after_it():
    proposed = propose(said((2.0, "A tag."), (20.0, "Then a whole thought.")),
                       shortest=5.0, longest=25.0)
    assert [(s.start, s.end) for s in proposed.spans] == [(0.0, 22.4)]


def test_what_is_too_short_and_cannot_be_joined_is_left_out_and_counted():
    """Mutation: join it however long the result -- red, a span runs past `longest`."""
    proposed = propose(said((24.0, "A whole thought."), (2.0, "And a tag.")),
                       shortest=5.0, longest=25.0)
    assert [(s.start, s.end) for s in proposed.spans] == [(0.0, 24.0)]
    assert proposed.left_out == 1
    whole = propose(RECORDING)
    assert whole.left_out == 2 and [c.spans for c in whole.chapters] == [0, 3, 0]
    assert whole.chapters[0].opening == "Welcome to the show."


def test_a_transcript_with_nothing_long_enough_proposes_nothing():
    proposed = propose(WELCOME)
    assert proposed.spans == () and proposed.left_out == 1 and len(proposed.chapters) == 1


@pytest.mark.parametrize("settings,why", [
    ({"shortest": 30.0, "longest": 20.0}, "shortest"),
    ({"shortest": 0.0}, "shortest"),
    ({"pause": 5.0, "chapter_pause": 4.0}, "pause"),
    ({"pause": 0.0}, "pause"),
])
def test_settings_that_describe_no_cut_are_refused(settings, why):
    with pytest.raises(ValueError, match=why):
        propose(RECORDING, **settings)


# --- the labeller ----------------------------------------------------------------------


def test_title_from_takes_the_first_line_without_its_quotes():
    assert title_from('\n"The Long Story"\nmore') == "The Long Story"
    assert title_from("  ‘Late Checkout’ ") == "Late Checkout"
    assert title_from("x" * 80) == "x" * 59 + "…"
    assert title_from("") == ""


def test_category_from_finds_the_longest_category_named_whatever_the_case():
    """Mutation: match shortest first -- red, 'song' wins over 'folk song'."""
    assert category_from("Category: Folk Song.", ["song", "folk song", "story"]) == "folk song"
    assert category_from("it is a STORY", ["song", "story"]) == "story"
    assert category_from("none of those", ["song", "story"]) == ""


def test_label_titles_each_chapter_and_sorts_each_span():
    asked: list[str] = []

    def model(prompt: str) -> str:
        asked.append(prompt)
        return "song" if prompt.startswith("Which one") else "A Title"

    proposed = label(propose(RECORDING), model, categories=["song", "story"])
    assert [c.title for c in proposed.chapters] == ["A Title"] * 3
    assert [s.label for s in proposed.spans] == ["song"] * 3
    assert sum(p.startswith("Give a title") for p in asked) == 3
    assert sum(p.startswith("Which one of these categories") for p in asked) == 3
    # A chapter with no span is titled from its opening.
    assert "Welcome to the show." in asked[0]
    assert "song, story" in asked[3]


def test_a_labeller_that_fails_leaves_that_label_empty_and_the_rest_labelled():
    """Mutation: let the exception out -- red, the proposal is lost."""
    def model(prompt: str) -> str:
        if "sentence 18" in prompt:
            raise httpx.ReadTimeout("stalled")
        return "story"

    proposed = label(propose(RECORDING), model, categories=["story"])
    assert [s.label for s in proposed.spans] == ["story", "", "story"]


def test_label_without_categories_titles_chapters_and_labels_no_span():
    proposed = label(propose(RECORDING), lambda prompt: "A Title")
    assert [c.title for c in proposed.chapters] == ["A Title"] * 3
    assert [s.label for s in proposed.spans] == ["", "", ""]


def test_the_local_model_is_asked_one_short_answer_at_a_time():
    """Mutation: drop max_tokens -- red; a model free to run on is the stall
    this guards against."""
    sent: list[httpx.Request] = []

    def answers(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": " story \n"}}]})

    ask = local_model("http://model.test/v1/", "a-model", timeout=7, max_tokens=5,
                      transport=httpx.MockTransport(answers))
    assert ask("Which?") == " story \n"
    request, = sent
    assert str(request.url) == "http://model.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer ollama"
    assert json.loads(request.content) == {
        "model": "a-model", "messages": [{"role": "user", "content": "Which?"}],
        "temperature": 0, "max_tokens": 5}


def test_a_model_that_refuses_raises_and_label_leaves_the_span_unlabelled():
    ask = local_model("http://model.test/v1", "a-model",
                      transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(httpx.HTTPStatusError):
        ask("Which?")
    proposed = label(propose(RECORDING), ask, categories=["story"])
    assert [s.label for s in proposed.spans] == ["", "", ""]


# --- the clip's timeline ---------------------------------------------------------------

TIMELINE = {"duration": 260.0, "clips": [
    {"type": "template", "name": "title", "start": 0, "duration": 5, "data": {"text": "Show"}},
    {"type": "video", "name": "videos/a.mp4", "start": 5, "inpoint": 10, "outpoint": 110,
     "fadeIn": 1, "fadeOut": 1},
    {"type": "video", "name": "videos/b.mp4", "start": 105, "duration": 155, "fadeIn": None},
    {"type": "audio", "name": "music.mp3", "start": 0, "duration": 260},
]}


def test_the_clips_in_a_span_are_kept_and_re_timed_from_zero():
    """Mutation: forget the inpoint when moving a clip's start -- red, the clip
    plays from its own beginning."""
    cut = excerpt_edl(TIMELINE, 99.0, 128.0)
    assert cut["duration"] == 29.0
    assert [c["name"] for c in cut["clips"]] == ["videos/a.mp4", "videos/b.mp4", "music.mp3"]
    a, b, music = cut["clips"]
    assert (a["start"], a["inpoint"], a["outpoint"], a["duration"]) == (0.0, 104.0, 110.0, 6.0)
    assert (b["start"], b["inpoint"], b["outpoint"], b["duration"]) == (6.0, 0.0, 23.0, 23.0)
    assert (music["start"], music["inpoint"], music["outpoint"]) == (0.0, 99.0, 128.0)


def test_an_edge_the_span_cut_has_no_fade_and_an_edge_it_did_not_keeps_the_clip_s():
    """Mutation: zero every fade -- red, a's own fade out is lost."""
    a, b, music = excerpt_edl(TIMELINE, 99.0, 128.0)["clips"]
    assert (a["fadeIn"], a["fadeOut"]) == (0, 1)
    assert (b["fadeIn"], b["fadeOut"]) == (None, 0)
    assert (music["fadeIn"], music["fadeOut"]) == (0, 0)
    title, = [c for c in excerpt_edl(TIMELINE, 0.0, 3.0)["clips"] if c["type"] == "template"]
    assert "fadeIn" not in title and title["fadeOut"] == 0
    assert (title["inpoint"], title["outpoint"], title["duration"]) == (0.0, 3.0, 3.0)


def test_an_image_keeps_a_duration_alone():
    """Mutation: give an image an inpoint -- red; it is one picture throughout."""
    edl = {"clips": [{"type": "image", "name": "still.png", "start": 2, "duration": 10}]}
    still, = excerpt_edl(edl, 5.0, 9.0)["clips"]
    assert still == {"type": "image", "name": "still.png", "start": 0.0, "duration": 4.0,
                     "fadeIn": 0, "fadeOut": 0}


def test_a_deprecated_offset_is_folded_into_the_timing():
    edl = {"clips": [{"type": "video", "name": "v.mp4", "offset": 3, "duration": 10},
                     {"type": "video", "name": "w.mp4", "start": 13, "offset": -4,
                      "duration": 10}]}
    v, w = excerpt_edl(edl, 4.0, 20.0)["clips"]
    assert "offset" not in v and "offset" not in w
    assert (v["start"], v["inpoint"], v["outpoint"]) == (0.0, 1.0, 10.0)
    assert (w["start"], w["inpoint"], w["outpoint"]) == (9.0, 4.0, 11.0)


def test_a_span_is_held_to_the_timeline_s_duration():
    """Mutation: ignore the timeline's duration -- red, the clip runs past it."""
    cut = excerpt_edl({"duration": 12.0, "clips": [{"type": "video", "name": "v.mp4",
                                                     "duration": 30}]}, 10.0, 20.0)
    assert cut["duration"] == 2.0 and cut["clips"][0]["outpoint"] == 12.0


@pytest.mark.parametrize("start,end,why", [
    (20.0, 10.0, "nothing to cut"), (-1.0, 5.0, "nothing to cut"), (150.0, 160.0, "no clip"),
])
def test_a_span_with_nothing_in_it_is_refused(start, end, why):
    edl = {"clips": [{"type": "video", "name": "v.mp4", "duration": 100}]}
    with pytest.raises(ValueError, match=why):
        excerpt_edl(edl, start, end)


def test_a_clip_otto_would_refuse_is_refused_here_by_its_place():
    with pytest.raises(ValueError, match="clip 1 \\('w.mp4'\\) has no length"):
        excerpt_edl({"clips": [{"type": "video", "name": "v.mp4", "duration": 5},
                               {"type": "video", "name": "w.mp4", "inpoint": 5, "outpoint": 5}]},
                    0.0, 3.0)


def test_the_captions_in_a_span_are_moved_trimmed_and_dropped():
    """Mutation: keep a caption's timeline seconds -- red, it is drawn late."""
    captions = [caption(0.0, 2.0, "before"), caption(9.0, 11.0, "across the start"),
                caption(12.0, 14.0, "inside"), caption(19.0, 21.0, "across the end"),
                caption(30.0, 31.0, "after")]
    assert excerpt_captions(captions, 10.0, 20.0) == [
        caption(0.0, 1.0, "across the start"), caption(2.0, 4.0, "inside"),
        caption(9.0, 10.0, "across the end")]


# --- the transcript, read or made --------------------------------------------------------


class Tools:
    """Tool answers: what `alfred_captions` holds, and a transcription that
    finishes with `heard` on the first look after it starts."""

    def __init__(self, captions=(), job=None, heard=()) -> None:
        self.held = {"captions": list(captions), "style": None, "job": job}
        self.heard = list(heard)
        self.calls: list[str] = []

    def __call__(self, tool: str, params: dict):
        self.calls.append(tool)
        if tool == "alfred_captions_transcribe":
            self.held["job"] = {"state": "transcribing", "error": None}
            return {"project_id": params["project_id"], "job": self.held["job"]}
        if tool == "alfred_captions":
            if self.held["job"] and self.held["job"]["state"] == "transcribing":
                self.held["job"] = {"state": DONE, "error": None, "engine": "joe", "segments": 1}
                self.held["captions"] = self.heard
            return {"project_id": params["project_id"], **self.held}
        raise AssertionError(f"unexpected tool {tool}")


def test_captions_alfred_holds_are_used_without_transcribing():
    """Mutation: transcribe whatever alfred holds -- red, a long recording is
    heard again."""
    tools = Tools(captions=WELCOME, job={"state": DONE})
    ended = read_transcript(tools, "p1")
    assert (ended.outcome, ended.captions, ended.seconds) == (DONE, WELCOME, 0.0)
    assert tools.calls == ["alfred_captions"]
    typed = read_transcript(Tools(captions=WELCOME, job=None), "p1")
    assert typed.done and typed.captions == WELCOME


def test_a_project_without_captions_is_transcribed_when_the_run_allows():
    tools = Tools(heard=WELCOME)
    ended = read_transcript(tools, "p1", sleep=lambda s: None)
    assert ended.done and ended.captions == WELCOME
    assert tools.calls == ["alfred_captions", "alfred_captions_transcribe", "alfred_captions"]


def test_a_project_without_captions_ends_the_run_when_it_may_not_transcribe():
    """Mutation: transcribe anyway -- red."""
    tools = Tools(heard=WELCOME)
    ended = read_transcript(tools, "p1", transcribe=False)
    assert ended.outcome == NO_CAPTIONS and not ended.done
    assert "not asked to transcribe" in ended.why and tools.calls == ["alfred_captions"]


def test_a_transcription_running_now_is_followed_rather_than_read_stale():
    tools = Tools(captions=WELCOME, job={"state": "transcribing"}, heard=GOODBYE)
    ended = read_transcript(tools, "p1", sleep=lambda s: None)
    assert ended.captions == GOODBYE


# --- queueing a clip --------------------------------------------------------------------

PROJECT_ID = "6a78b0420eea42d905943601"
PROJECT = {"id": PROJECT_ID, "name": "Set at the Hall", "edl": TIMELINE,
           "render_settings": {"width": 1280, "height": 720, "fps": None, "quality": "draft"}}


class Queue:
    def __init__(self) -> None:
        self.params: list[dict] = []

    def __call__(self, tool: str, params: dict):
        assert tool == "alfred_render_enqueue"
        self.params.append(params)
        return {"render_id": render_id(len(self.params)), "name": params["name"],
                "captions": len(params.get("captions") or [])}


def test_a_clip_is_queued_from_the_cut_timeline_with_its_captions():
    """Mutation: send the project's whole timeline -- red."""
    span = propose(RECORDING).spans[1]
    queue = Queue()
    queued = queue_clip(queue, PROJECT, span, captions=RECORDING,
                        style={**CAPTIONS["style"], "fontsize": 40.0}, quality="high")
    sent, = queue.params
    assert sent["edl"] == excerpt_edl(TIMELINE, span.start, span.end)
    assert sent["name"] == "Set at the Hall - clip 2"
    assert sent["description"] == f"1:39.2-3:07.0 of alfred project {PROJECT_ID}, chapter 2"
    assert (sent["width"], sent["height"], sent["quality"]) == (1280, 720, "high")
    assert "fps" not in sent
    assert sent["captions"] == excerpt_captions(RECORDING, span.start, span.end)
    assert sent["caption_style"] == {**CAPTIONS["style"], "fontsize": 40.0}
    assert queued["captions"] == len(sent["captions"]) > 0


def test_a_clip_queued_without_captions_sends_none():
    """Mutation: send the captions whenever there are any -- red."""
    span = propose(RECORDING).spans[0]
    queue = Queue()
    queue_clip(queue, PROJECT, span, captions=None, style=CAPTIONS["style"])
    assert "captions" not in queue.params[0] and "caption_style" not in queue.params[0]
    queue_clip(queue, PROJECT, span, captions=GOODBYE, style=CAPTIONS["style"])
    assert "captions" not in queue.params[1]


def test_a_labelled_clip_s_description_names_its_label():
    span = Span(1, 1, 0.0, 10.0, (), label="song")
    assert flow.clip_description("p1", span) == "0:00.0-0:10.0 of alfred project p1, chapter 1: song"
    assert flow.clip_name(None, span) == "alfred project - clip 1"


# --- what a person is asked ----------------------------------------------------------


def test_the_review_places_the_span_and_quotes_its_words():
    proposed = label(propose(RECORDING), lambda p: "The Story" if "title" in p else "story",
                     categories=["story"])
    request = review_request(PROJECT, proposed.spans[2], proposed, "run-1")
    assert request["request_id"] == f"alfred-excerpt-{PROJECT_ID}-3-run-1"
    assert request["options"] == [APPROVE, REJECT]
    lines = request["prompt"].splitlines()
    assert lines[0] == ("Clip 3 of 3 from 'Set at the Hall', chapter 2 of 3 (The Story), "
                        "3:07.4-4:05.8 (58 s, story):")
    assert lines[1] == "  3:07.4-3:14.4  This is sentence 36 of the story, which goes on for a while"
    assert lines[-1] == ("Answer approve to render it as a clip of its own to review, or reject "
                         "to leave it out.")
    assert request["context"]["clip"] == 3 and request["context"]["of"] == 3
    assert request["context"]["chapter_title"] == "The Story"
    assert request["context"]["captions"] == list(proposed.spans[2].captions)


def test_an_unlabelled_review_names_no_label():
    request = review_request(PROJECT, propose(RECORDING).spans[0], propose(RECORDING), "run-1")
    assert request["prompt"].splitlines()[0].endswith("chapter 2 of 3, 0:11.0-1:38.8 (88 s):")


def test_the_approval_places_the_clip_and_says_what_approval_does():
    """Mutation: say delivery follows whatever the run asked -- red."""
    span = Span(2, 1, 99.2, 187.0, (), label="story")
    queued = {"render_id": "r2", "name": "Set at the Hall - clip 2", "captions": 4}
    waited = Waited(RENDERED, {"download_url": "http://alfred.test/r2", "download_expires_at":
                               "2030-01-01T00:00:00+00:00"}, 10.0)
    request = flow.approval_request(queued, waited, span, "run-1", deliver=True)
    assert request["request_id"] == "alfred-excerpt-render-r2-run-1"
    assert request["prompt"].startswith("Approve the render of 'Set at the Hall - clip 2'? ")
    assert request["prompt"].endswith(
        " It is 1:39.2-3:07.0 of the recording, story. Answer approve to deliver it to "
        "alfred's default destination, or reject to leave it undelivered.")
    assert request["context"]["clip"] == 2 and request["context"]["captions"] == 4
    kept = flow.approval_request(queued, waited, Span(2, 1, 99.2, 187.0, ()), "run-1")
    assert kept["prompt"].endswith(" It is 1:39.2-3:07.0 of the recording. Nothing is "
                                   "delivered either way: this run was not asked to deliver.")


# --- how a run ends --------------------------------------------------------------------

HEARD = Transcribed(DONE, RECORDING, {"state": DONE}, 0.0)


def test_a_transcription_that_did_not_finish_ends_the_run_in_the_caption_flow_s_words():
    line = summary(PROJECT, Transcribed(NO_CAPTIONS, [], None, 0.0, "nothing to read"))
    assert line == (f"Nothing was asked: the transcription of alfred project {PROJECT_ID} "
                    "left nothing to read: nothing to read.")


def test_a_transcript_with_nothing_to_offer_says_so():
    line = summary(PROJECT, Transcribed(DONE, WELCOME, None, 0.0), propose(WELCOME))
    assert line == ("Nothing was asked: the transcript of 'Set at the Hall' (2 captions) has no "
                    "stretch of 10 s or more to offer as a clip.")


def test_the_last_line_counts_each_stage():
    """Mutation: count a rejected clip as approved -- red."""
    proposed = propose(RECORDING)
    reviewed = {1: APPROVE, 2: REJECT, 3: None}
    queued = {1: {"render_id": "r1"}}
    waited = {1: Waited(RENDERED, {}, 1.0)}
    line = summary(PROJECT, HEARD, proposed, reviewed, queued, waited, {1: APPROVE},
                   {1: Delivered("local", "http://alfred.test/r1")}, deliver=True)
    assert line == ("3 clips in 3 chapters of 'Set at the Hall' offered, 1 approved "
                    "(1 unanswered), 1 rendered, 1 approved as clips, 1 delivered.")
    line = summary(PROJECT, HEARD, proposed, {1: APPROVE, 2: APPROVE, 3: REJECT},
                   {1: {"render_id": "r1"}, 2: {"render_id": "r2"}},
                   {1: Waited(RENDERED, {}, 1.0), 2: Waited("failed", {}, 1.0, "out of disk")},
                   {1: REJECT}, {})
    assert line == ("3 clips in 3 chapters of 'Set at the Hall' offered, 2 approved, 1 rendered "
                    "(clip 2 out of disk), 0 approved as clips.")
    line = summary(PROJECT, HEARD, proposed, {1: REJECT, 2: REJECT, 3: REJECT})
    assert line == "3 clips in 3 chapters of 'Set at the Hall' offered, 0 approved."


def test_a_delivery_alfred_refused_is_said_in_the_last_line():
    proposed = propose(RECORDING)
    line = summary(PROJECT, HEARD, proposed, {1: APPROVE, 2: REJECT, 3: REJECT},
                   {1: {"render_id": "r1"}}, {1: Waited(RENDERED, {}, 1.0)}, {1: APPROVE},
                   {1: Delivered("dropbox", None, "alfred refused it")}, deliver=True)
    assert line.endswith("1 approved as clips, 0 delivered (clip 1: alfred refused it).")


def test_a_proposal_pickles_as_metaflow_keeps_artifacts():
    proposed = label(propose(RECORDING), lambda p: "x", categories=["x"])
    assert pickle.loads(pickle.dumps(proposed)) == proposed
    assert isinstance(proposed.chapter_of(proposed.spans[0]), Chapter)
    assert isinstance(proposed, Proposal)


# --- end to end: the flow file's steps, through a qmcp server and its queue ------------------

FLOW_FILE = REPO / "examples" / "flows" / "alfred_excerpts.py"
CORRELATION = "alfred-excerpts-run-1"
RENDER = render_id(1)


class _FlowSpec:
    """Metaflow's `FlowSpec` as far as the steps use it: `next` names the next step."""

    def next(self, step) -> None:
        self._next = step


def _parameter(name, help=None, default=None, type=None, required=False):
    """Metaflow's `Parameter` as far as a run uses it: its default, unless a run sets one."""
    return default


@pytest.fixture
def flow_file(monkeypatch):
    """`examples/flows/alfred_excerpts.py`, loaded over a stand-in for
    Metaflow's classes and decorators. Its steps run in this process, in the
    order `self.next` names; what Metaflow adds -- a process per step, and
    parameters read from a command line -- is not exercised here, and the
    examples smoke test imports the file where Metaflow runs."""
    metaflow = types.ModuleType("metaflow")
    metaflow.FlowSpec, metaflow.Parameter = _FlowSpec, _parameter
    metaflow.step = lambda function: function
    metaflow.current = SimpleNamespace(run_id="run-1")
    monkeypatch.setitem(sys.modules, "metaflow", metaflow)
    spec = importlib.util.spec_from_file_location("alfred_excerpts_flow", FLOW_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(monkeypatch) -> CaptionedAlfred:
    """The stand-in holding the recording's project, transcribed, with a caption
    style saved."""
    stand_in = CaptionedAlfred()
    stand_in.add_project(PROJECT_ID, "Set at the Hall", captions=RECORDING, edl=TIMELINE,
                         style={**CAPTIONS["style"], "enabled": False, "fontsize": 40.0},
                         render_settings={"width": 1280, "height": 720})
    connected = connect(stand_in)
    monkeypatch.setattr(alfred, "client", lambda: connected)
    return stand_in


@pytest.fixture
def server(tmp_path):
    from qmcp.testing import throwaway_server

    with throwaway_server(tmp_path / "queue.db") as url:
        yield url


def run(module, monkeypatch, url: str, *, review=APPROVE, approval=APPROVE,
        **parameters) -> SimpleNamespace:
    """One run of the flow file's steps, against the qmcp server at `url`.

    Time passes only when a wait sleeps. A person answers each request at the
    first sleep of its wait: each span's review with `review`, one answer or a
    dict by clip number, and each render with `approval`, likewise; None is
    nobody answering. `parameters` are set on the run, the rest keep their
    defaults. After each step its artifacts are pickled, as Metaflow keeps them
    between steps.
    """
    clock = alfred_captions_clock()
    asked: list[dict] = []

    def answer_to(request_id: str):
        found = re.match(r"alfred-excerpt-[0-9a-f]+-(\d+)-", request_id)
        number = int(found.group(1)) if found else None
        chosen = approval if "-render-" in request_id else review
        if isinstance(chosen, dict):
            if number is None:
                number = next(n for n, q in queued_by_clip(module).items()
                              if request_id.startswith(f"alfred-excerpt-render-{q}-"))
            return chosen.get(number)
        return chosen

    def asking(client, request, **options):
        asked.append(request)
        alfred_render.ask(client, request, **options)

    def a_person_answers(client, request_id, **options):
        answer = answer_to(request_id)
        answered: list[str] = []

        def waiting(seconds: float) -> None:
            if answer is not None and not answered:
                answered.append(answer)
                client.submit_human_response(request_id, answer,
                                             responded_by="editor@example.com")
            clock.sleep(seconds)

        return alfred_render.wait_for_answer(client, request_id, sleep=waiting, clock=clock,
                                             **options)

    def queued_by_clip(module) -> dict[int, str]:
        return {n: q["render_id"] for n, q in getattr(module._ran, "queued", {}).items()}

    monkeypatch.setattr(module, "read_transcript",
                        partial(flow.read_transcript, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(module, "wait_for_render",
                        partial(alfred_render.wait_for_render, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(module, "ask", asking)
    monkeypatch.setattr(module, "wait_for_answer", a_person_answers)

    ran = module.AlfredExcerptsFlow()
    module._ran = ran
    ran.project_id, ran.mcp_url = PROJECT_ID, url
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


def alfred_captions_clock():
    class Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    return Clock()


# Reading the project and its captions, then the style: three tools; each render
# moves one state per look, so two status reads follow each enqueue.
READ_TOOLS = ["alfred_project", "alfred_captions", "alfred_captions"]


def test_chosen_clips_are_cut_rendered_approved_and_delivered(
        flow_file, monkeypatch, server, fake):
    """Through a qmcp server on an ephemeral port over its own database: every
    call to alfred is an invocation it recorded under the run's correlation id,
    and every request is answered on its queue."""
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: APPROVE},
              approval={1: APPROVE, 3: REJECT}, deliver=True)
    assert ran.line == ("3 clips in 3 chapters of 'Set at the Hall' offered, 2 approved, "
                        "2 rendered, 1 approved as clips, 1 delivered.")
    assert ran.tools == sorted(READ_TOOLS + ["alfred_render_enqueue"] * 2
                               + ["alfred_render_status"] * 4 + ["alfred_render_deliver"])
    reviews, approvals = ran.asked[:3], ran.asked[3:]
    assert [r["request_id"] for r in reviews] == [
        f"alfred-excerpt-{PROJECT_ID}-{n}-run-1" for n in (1, 2, 3)]
    assert [a["request_id"] for a in approvals] == [
        f"alfred-excerpt-render-{render_id(n)}-run-1" for n in (1, 2)]
    # What alfred was asked to render: the first and third spans, cut from the
    # saved timeline, with the captions that fall in each and the saved style on.
    spans = propose(RECORDING).spans
    first, third = fake.queued
    assert first["edl"] == excerpt_edl(TIMELINE, spans[0].start, spans[0].end)
    assert third["edl"] == excerpt_edl(TIMELINE, spans[2].start, spans[2].end)
    assert first["name"] == "Set at the Hall - clip 1" and third["name"].endswith("clip 3")
    assert first["captions"] == excerpt_captions(RECORDING, spans[0].start, spans[0].end)
    assert first["caption_style"] == {**CAPTIONS["style"], "enabled": True, "fontsize": 40.0}
    assert (first["width"], first["height"]) == (1280, 720)
    assert fake.renders[render_id(1)]["destination"] == "local"
    assert fake.renders[render_id(2)]["destination"] is None
    assert ran.pending == []


def test_a_clip_approved_is_not_delivered_unless_the_run_was_asked_to(
        flow_file, monkeypatch, server, fake):
    """Mutation: deliver whenever a clip is approved -- red."""
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: REJECT})
    assert ran.line == ("3 clips in 3 chapters of 'Set at the Hall' offered, 1 approved, "
                        "1 rendered, 1 approved as clips.")
    assert "alfred_render_deliver" not in ran.tools
    assert ran.asked[3]["prompt"].endswith("Nothing is delivered either way: this run was not "
                                           "asked to deliver.")


def test_clips_nobody_chooses_render_nothing(flow_file, monkeypatch, server, fake):
    """Mutation: render whatever the review's answer -- red."""
    ran = run(flow_file, monkeypatch, server, review=REJECT)
    assert ran.line == "3 clips in 3 chapters of 'Set at the Hall' offered, 0 approved."
    assert ran.tools == sorted(READ_TOOLS) and fake.queued == [] and len(ran.asked) == 3


def test_a_review_nobody_answers_is_counted(flow_file, monkeypatch, server, fake):
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE}, review_timeout=60)
    assert ran.line == ("3 clips in 3 chapters of 'Set at the Hall' offered, 1 approved "
                        "(2 unanswered), 1 rendered, 1 approved as clips.")


def test_captions_are_left_off_the_clips_when_the_run_says(
        flow_file, monkeypatch, server, fake):
    """Mutation: burn them in regardless -- red."""
    run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: REJECT},
        captions=False)
    assert "captions" not in fake.queued[0] and "caption_style" not in fake.queued[0]


def test_a_project_without_captions_is_transcribed_first(
        flow_file, monkeypatch, server, fake):
    project = fake.projects[PROJECT_ID]
    project["captions"], project["caption_style"] = None, None
    fake.heard[PROJECT_ID] = RECORDING
    ran = run(flow_file, monkeypatch, server, review=REJECT)
    assert ran.line == "3 clips in 3 chapters of 'Set at the Hall' offered, 0 approved."
    # The captions looked at before transcribing, twice while the job moves,
    # and once more for the style.
    assert ran.tools == sorted(["alfred_project", "alfred_captions_transcribe"]
                               + ["alfred_captions"] * 4)


def test_a_project_the_run_may_not_transcribe_asks_nobody(
        flow_file, monkeypatch, server, fake):
    fake.projects[PROJECT_ID]["captions"] = None
    ran = run(flow_file, monkeypatch, server, transcribe=False)
    assert ran.line == (f"Nothing was asked: the transcription of alfred project {PROJECT_ID} "
                        "left nothing to read: the project holds no captions, and this run was "
                        "not asked to transcribe it.")
    assert ran.asked == [] and ran.tools == sorted(["alfred_project", "alfred_captions"])


def test_a_local_labeller_is_built_from_the_run_s_settings(
        flow_file, monkeypatch, server, fake):
    """Mutation: ignore --label-with -- red, no labeller is asked."""
    built: list[tuple] = []

    def stand_in(base_url, model, *, timeout):
        built.append((base_url, model, timeout))
        return lambda prompt: "song" if prompt.startswith("Which") else "A Song"

    monkeypatch.setattr(flow_file, "local_model", stand_in)
    ran = run(flow_file, monkeypatch, server, review=REJECT, label_with="local",
              categories="song, story", llm_base_url="http://model.test/v1", llm_model="m",
              llm_timeout=9.0)
    assert built == [("http://model.test/v1", "m", 9.0)]
    assert ran.asked[0]["prompt"].startswith("Clip 1 of 3 from 'Set at the Hall', chapter 2 "
                                             "of 3 (A Song), 0:11.0-1:38.8 (88 s, song):")


def test_an_unknown_labeller_is_refused_before_anybody_is_asked(
        flow_file, monkeypatch, server, fake):
    with pytest.raises(ValueError, match="--label-with is none or local"):
        run(flow_file, monkeypatch, server, label_with="remote")


def test_a_refusal_from_alfred_is_the_tool_s_error(flow_file, monkeypatch, server, fake):
    fake.workers = 0
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: REJECT},
              render_timeout=30, queued_timeout=10)
    assert ran.line.endswith("1 approved, 0 rendered (clip 1 still queued after 10 s, and "
                             "alfred reports no render worker running), 0 approved as clips.")


def test_a_tool_error_outside_the_known_refusals_is_raised(
        flow_file, monkeypatch, server, fake):
    del fake.projects[PROJECT_ID]
    with pytest.raises(ToolFailedError, match="404"):
        run(flow_file, monkeypatch, server)
