"""The excerpt flow: alfred's drafts proposed through the tools and followed to
done, a labeller over them, the transcript read or made, each request a person
gets, how a run ends, and the flow file's steps end to end through a qmcp
server and its queue.

alfred is `CaptionedAlfred` in `tests/alfred_stub.py`, whose worker leaves
drafts cut by an arithmetic of the stand-in's own (`tests/alfred_cuts.py`).
The steps are tested through `qmcp.cookbook.alfred_excerpts` over stand-ins
for the tools, and the branches by running the steps of
`examples/flows/alfred_excerpts.py` itself, over a stand-in for Metaflow,
which cannot be imported where `import fcntl` fails.
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
from qmcp.cookbook import alfred_excerpts as flow
from qmcp.cookbook import alfred_render
from qmcp.cookbook.alfred_captions import DONE, NO_CAPTIONS, Delivered, Transcribed
from qmcp.cookbook.alfred_excerpts import (
    APPROVE,
    REJECT,
    Draft,
    DraftsFailedError,
    Proposal,
    category_from,
    drop,
    label,
    opening,
    propose,
    queue_excerpt,
    read_transcript,
    review_request,
    summary,
)
from qmcp.cookbook.alfred_render import RENDERED, ToolFailedError, Waited
from tests import alfred_cuts as cuts
from tests.alfred_stub import CAPTIONS, CaptionedAlfred, connect, draft_id, render_id

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
# that alfred's recipe cuts into three, and a goodbye too short to stand alone.
WELCOME = said((2.0, "Welcome to the show."), (3.0, "Tonight we have three songs for you."))
STORY = said(*[(7.0, f"This is sentence {n} of the story, which goes on for a while")
               if n % 2 == 0 else (2.0, "and ends here.") for n in range(48)], at=11.0)
GOODBYE = said((2.0, "Thanks for coming."), (1.5, "Good night!"), at=STORY[-1]["end"] + 6.0)
RECORDING = WELCOME + STORY + GOODBYE

PROJECT_ID = "6a78b0420eea42d905943601"
TIMELINE = {"duration": 260.0, "clips": [
    {"type": "template", "name": "title", "start": 0, "duration": 5, "data": {"text": "Show"}},
    {"type": "video", "name": "videos/a.mp4", "start": 5, "inpoint": 10, "outpoint": 110,
     "fadeIn": 1, "fadeOut": 1},
    {"type": "video", "name": "videos/b.mp4", "start": 105, "duration": 155, "fadeIn": None},
    {"type": "audio", "name": "music.mp3", "start": 0, "duration": 260},
]}
PROJECT = {"id": PROJECT_ID, "name": "Set at the Hall", "edl": TIMELINE,
           "render_settings": {"width": 1280, "height": 720, "fps": None, "quality": "draft"}}
# What the stand-in's recipe leaves of the recording with the flow's defaults:
# three stretches of the story, in timeline order.
SPANS = cuts.spans_of(RECORDING, pause=1.0, chapter_pause=4.0, shortest=10.0, longest=90.0)


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
                 "queue_excerpt", "drop", "wait_for_render", "approval_request", "ask",
                 "wait_for_answer", "deliver_render", "summary"):
        assert f"{name}(" in source, name
    assert '"alfred_project"' in source


def test_nothing_here_cuts_a_timeline_or_reads_a_pause():
    """Mutation: bring the cut back into the module -- red; alfred cuts."""
    source = Path(flow.__file__).read_text(encoding="utf-8")
    assert "def excerpt_edl" not in source and "def stretches" not in source


# --- what is spoken -------------------------------------------------------------------


def test_an_opening_is_the_first_sentence_or_the_first_characters():
    assert opening("Good evening. We have a song.") == "Good evening."
    assert opening("A sentence that runs on and on " * 5, limit=20) == "A sentence that run…"
    assert opening("") == ""


def test_category_from_finds_the_longest_category_named_whatever_the_case():
    assert category_from("It is a STORY, mostly.", ["song", "story"]) == "story"
    assert category_from("a song story", ["song", "song story"]) == "song story"
    assert category_from("neither", ["song", "story"]) == ""


# --- the transcript -------------------------------------------------------------------


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


# --- the proposal, through the tools --------------------------------------------------


def listed_draft(n: int, start: float, end: float, captions=RECORDING, scores=None) -> dict:
    """A draft as `alfred_drafts_status` lists one, the `n`th the stand-in made."""
    return {"id": draft_id(n), "name": f"Set at the Hall – Excerpts {n}",
            "description": f"{start}-{end} of the source", "duration": round(end - start, 3),
            "source": {"project": PROJECT_ID, "recipe": "excerpts",
                       "values": {"pause": "1"}, "spans": [[start, end]],
                       "scores": scores or {"dialogue": 0.5}}}


class Drafting:
    """Tool answers for a run of recipes: `alfred_drafts` queues it, each look at
    `alfred_drafts_status` moves it on, and after `looks` looks it is done,
    with the drafts the stand-in's arithmetic cuts from `captions`, or failed
    with `error`. `held` are drafts the project had before the run, listed
    from the start; `order` reverses the listing."""

    def __init__(self, captions=RECORDING, *, looks=2, error=None, held=(), reverse=False):
        self.captions = list(captions)
        self.looks, self.error, self.held, self.reverse = looks, error, list(held), reverse
        self.calls: list[tuple[str, dict]] = []
        self.state: str | None = None
        self.seen = 0
        self.dropped: list[list[str]] = []
        self.made = [listed_draft(100 + n, start, end)
                     for n, (start, end, _) in enumerate(cuts.spans_of(
                         self.captions, pause=1.0, chapter_pause=4.0, shortest=10.0,
                         longest=90.0), start=1)]

    def status(self) -> dict:
        drafts = list(self.held) + (self.made if self.state == "done" else [])
        if self.reverse:
            drafts.reverse()
        job = None if self.state is None else {
            "state": self.state, "error": self.error if self.state == "failed" else None,
            "recipes": ["excerpts"], "made": len(self.made) if self.state == "done" else None,
            "notes": ["excerpts: label: nothing was titled: no reading"]
            if self.state == "done" else []}
        return {"project_id": PROJECT_ID, "job": job, "drafts": drafts}

    def __call__(self, tool: str, params: dict):
        self.calls.append((tool, params))
        if tool == "alfred_drafts":
            self.state = "queued"
            return {"project_id": params["project_id"], "job": {"state": "queued"}}
        if tool == "alfred_drafts_status":
            if self.state in ("queued", "running"):
                self.seen += 1
                if self.seen >= self.looks:
                    self.state = "failed" if self.error else "done"
                elif self.state == "queued":
                    self.state = "running"
            return self.status()
        if tool == "alfred_captions":
            for draft in self.made:
                if draft["id"] == params["project_id"]:
                    [[start, end]] = draft["source"]["spans"]
                    return {"project_id": params["project_id"],
                            "captions": cuts.cut_captions(self.captions, start, end),
                            "style": None, "job": None}
            raise AssertionError(f"no draft {params['project_id']}")
        if tool == "alfred_drafts_drop":
            self.dropped.append(list(params["drafts"]))
            return {"project_id": params["project_id"], "dropped": list(params["drafts"])}
        raise AssertionError(f"unexpected tool {tool}")


def no_wait(seconds: float) -> None:
    return None


def test_the_run_is_queued_with_its_knobs_as_text_and_followed_to_done():
    """Mutation: send the knobs as numbers -- red; a knob is typed."""
    tools = Drafting()
    proposed = propose(tools, PROJECT_ID, pause=1.5, shortest=12, longest=60, sleep=no_wait)
    names = [tool for tool, _ in tools.calls]
    assert names == (["alfred_drafts_status", "alfred_drafts"] + ["alfred_drafts_status"] * 2
                     + ["alfred_captions"] * 3)
    assert tools.calls[1][1] == {"project_id": PROJECT_ID, "recipes": [
        {"recipe": "excerpts", "values": {"pause": "1.5", "shortest": "12", "longest": "60"}}]}
    assert [d.number for d in proposed.drafts] == [1, 2, 3]
    assert [(d.start, d.end) for d in proposed.drafts] == [(s, e) for s, e, _ in SPANS]
    assert proposed.drafts[0].id == draft_id(101)
    assert proposed.drafts[0].name == "Set at the Hall – Excerpts 101"
    assert proposed.drafts[0].title == "Excerpts 101"
    assert proposed.drafts[0].captions[0] == cuts.cut_captions(RECORDING, SPANS[0][0],
                                                                 SPANS[0][1])[0]
    assert proposed.drafts[0].scores == {"dialogue": 0.5}
    assert proposed.notes == ("excerpts: label: nothing was titled: no reading",)
    assert proposed.settings == {"pause": 1.5, "shortest": 12, "longest": 60}


def test_drafts_the_project_held_before_the_run_are_not_the_proposal():
    """Mutation: take every draft alfred lists -- red; an earlier run's are offered again."""
    earlier = listed_draft(7, 0.0, 10.0)
    tools = Drafting(held=[earlier])
    proposed = propose(tools, PROJECT_ID, sleep=no_wait)
    assert [d.id for d in proposed.drafts] == [draft_id(101), draft_id(102), draft_id(103)]


def test_the_drafts_are_in_timeline_order_whatever_order_alfred_lists():
    proposed = propose(Drafting(reverse=True), PROJECT_ID, sleep=no_wait)
    assert [d.start for d in proposed.drafts] == sorted(d.start for d in proposed.drafts)
    assert [d.number for d in proposed.drafts] == [1, 2, 3]


def test_a_run_that_fails_raises_alfred_s_reason():
    with pytest.raises(DraftsFailedError, match="excerpts recipe failed: no recipe is named"):
        propose(Drafting(error="no recipe is named 'excerpts'"), PROJECT_ID, sleep=no_wait)


def test_a_run_not_done_in_time_raises_how_long_it_waited():
    """Mutation: wait for ever -- the test hangs; a clock bounds it."""
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds: float) -> None:
        clock.now += seconds

    with pytest.raises(DraftsFailedError, match="still running after 10 s"):
        propose(Drafting(looks=99), PROJECT_ID, timeout=10, poll_interval=5, sleep=sleep,
                clock=lambda: clock.now)


# --- the labeller ---------------------------------------------------------------------


def proposed_drafts() -> Proposal:
    return propose(Drafting(), PROJECT_ID, sleep=no_wait)


def test_label_sorts_each_draft_into_a_category_and_leaves_the_name_alfred_s():
    asked: list[str] = []

    def model(prompt: str) -> str:
        asked.append(prompt)
        return "That is a story about the road."

    labelled = label(proposed_drafts(), model, categories=["song", "story"])
    assert [d.label for d in labelled.drafts] == ["story"] * 3
    assert len(asked) == 3 and all(p.startswith("Which one of these categories") for p in asked)
    assert asked[0].endswith(labelled.drafts[0].text)
    assert labelled.drafts[0].name == "Set at the Hall – Excerpts 101"


def test_a_labeller_that_fails_leaves_that_label_empty_and_the_rest_labelled():
    def model(prompt: str) -> str:
        if "sentence 36" in prompt:
            raise TimeoutError("the model stalled")
        return "song"

    labelled = label(proposed_drafts(), model, categories=["song"])
    assert [d.label for d in labelled.drafts] == ["song", "song", ""]


def test_label_without_categories_asks_nothing():
    """Mutation: ask the model anyway -- red."""
    proposed = proposed_drafts()
    assert label(proposed, lambda p: pytest.fail("asked")) == proposed


# --- queueing an excerpt, and dropping one --------------------------------------------------


class Queue:
    def __init__(self) -> None:
        self.params: list[dict] = []

    def __call__(self, tool: str, params: dict):
        assert tool == "alfred_render_enqueue"
        self.params.append(params)
        return {"render_id": render_id(len(self.params)), "name": params["name"],
                "captions": 4 if params.get("captions") else 0}


def test_an_excerpt_is_the_draft_s_own_timeline_rendered_under_alfred_s_name():
    """Mutation: send the source's id, or a cut timeline -- red."""
    draft = proposed_drafts().drafts[1]
    queue = Queue()
    queued = queue_excerpt(queue, PROJECT, draft, quality="high")
    sent, = queue.params
    assert sent == {"project_id": draft_id(102), "name": "Set at the Hall – Excerpts 102",
                    "description": "1:39.2-3:07.0 of 'Set at the Hall'", "captions": True,
                    "quality": "high"}
    assert queued["captions"] == 4


def test_an_excerpt_queued_without_captions_asks_for_none():
    """Mutation: burn them in whenever the draft has any -- red."""
    draft = proposed_drafts().drafts[0]
    queue = Queue()
    queue_excerpt(queue, PROJECT, draft, captions=False)
    assert queue.params[0]["captions"] is False
    bare = Draft(1, draft_id(1), "x – Excerpts 1", "", 0.0, 10.0, (), {})
    queue_excerpt(queue, PROJECT, bare)
    assert queue.params[1]["captions"] is False


def test_a_labelled_excerpt_s_description_names_its_label():
    draft = Draft(1, draft_id(1), "A Set – Excerpts 1", "", 0.0, 10.0, (), {}, label="song")
    assert flow.excerpt_description("A Set", draft) == "0:00.0-0:10.0 of 'A Set': song"
    assert flow.excerpt_description(None, draft) == "0:00.0-0:10.0 of 'the recording': song"


def test_rejected_drafts_are_dropped_together_and_none_asks_nothing():
    """Mutation: ask alfred to drop an empty list -- red, alfred refuses one."""
    tools = Drafting()
    proposed = propose(tools, PROJECT_ID, sleep=no_wait)
    assert drop(tools, PROJECT_ID, []) == {"project_id": PROJECT_ID, "dropped": []}
    assert tools.dropped == []
    dropped = drop(tools, PROJECT_ID, proposed.drafts[1:])
    assert tools.dropped == [[draft_id(102), draft_id(103)]]
    assert dropped["dropped"] == [draft_id(102), draft_id(103)]


# --- what a person is asked ----------------------------------------------------------


def test_the_review_places_the_draft_names_it_and_quotes_its_words():
    proposed = label(proposed_drafts(), lambda p: "story", categories=["story"])
    request = review_request(PROJECT, proposed.drafts[2], proposed, "run-1")
    assert request["request_id"] == f"alfred-excerpt-{PROJECT_ID}-3-run-1"
    assert request["options"] == [APPROVE, REJECT]
    lines = request["prompt"].splitlines()
    assert lines[0] == ("Excerpt 3 of 3 from 'Set at the Hall', 3:07.4-4:05.8 (58 s, story), "
                        "alfred's 'Excerpts 103':")
    # The words in the recording's seconds, as a person reads its clock.
    assert lines[1] == ("  3:07.4-3:14.4  This is sentence 36 of the story, which goes on for "
                        "a while")
    assert lines[-1] == ("Answer approve to render it as an excerpt of its own to review, or "
                         "reject to drop it.")
    assert request["context"]["excerpt"] == 3 and request["context"]["of"] == 3
    assert request["context"]["draft"] == draft_id(103)
    assert request["context"]["captions"] == list(proposed.drafts[2].captions)
    assert request["context"]["captions"][0]["start"] == 0.0
    # The ear gets the place and the opening, not every line.
    assert request["context"]["spoken"] == (
        "Excerpt 3 of 3 from Set at the Hall, 58 seconds, story, Excerpts 103. "
        "It opens: This is sentence 36 of the story, which goes on for a while and ends here. "
        "Approve to render it as an excerpt of its own, or reject to drop it.")


def test_an_unlabelled_review_names_no_label():
    proposed = proposed_drafts()
    request = review_request(PROJECT, proposed.drafts[0], proposed, "run-1")
    assert request["prompt"].splitlines()[0] == (
        "Excerpt 1 of 3 from 'Set at the Hall', 0:11.0-1:38.8 (88 s), alfred's 'Excerpts 101':")


def test_the_approval_places_the_excerpt_and_says_what_approval_does():
    """Mutation: say delivery follows whatever the run asked -- red."""
    draft = Draft(2, draft_id(2), "Set at the Hall – Excerpts 2", "", 99.2, 187.0, (), {},
                  label="story")
    queued = {"render_id": "r2", "name": "Set at the Hall – Excerpts 2", "captions": 4}
    waited = Waited(RENDERED, {"download_url": "http://alfred.test/r2", "download_expires_at":
                               "2030-01-01T00:00:00+00:00"}, 10.0)
    request = flow.approval_request(queued, waited, draft, "run-1", deliver=True)
    assert request["request_id"] == "alfred-excerpt-render-r2-run-1"
    assert request["prompt"].startswith("Approve the render of 'Set at the Hall – Excerpts 2'? ")
    assert request["prompt"].endswith(
        " It is 1:39.2-3:07.0 of the recording, story. Answer approve to deliver it to "
        "alfred's default destination, or reject to leave it undelivered.")
    assert request["context"]["excerpt"] == 2 and request["context"]["captions"] == 4
    assert request["context"]["draft"] == draft_id(2)
    assert request["context"]["spoken"] == (
        "Set at the Hall – Excerpts 2 is rendered. It is 1:39.2-3:07.0 of the recording, "
        "story. Answer approve to deliver it to alfred's default destination, or reject to "
        "leave it undelivered.")
    kept = flow.approval_request(queued, waited, Draft(2, draft_id(2), "x", "", 99.2, 187.0,
                                                       (), {}), "run-1")
    assert kept["prompt"].endswith(" It is 1:39.2-3:07.0 of the recording. Nothing is "
                                   "delivered either way: this run was not asked to deliver.")


# --- the cues on the timeline ---------------------------------------------------------


def test_cues_of_draws_each_draft_with_where_it_stands():
    proposed = label(proposed_drafts(), lambda p: "story", categories=["story"])
    cues = flow.cues_of(proposed, {1: flow.APPROVED, 2: flow.WAITING}, {2: "req-2"})
    assert [(c["label"], c["kind"], c["state"], c.get("ref")) for c in cues] == [
        ("Excerpt 1: story", "excerpt", flow.APPROVED, None),
        ("Excerpt 2: story", "excerpt", flow.WAITING, "req-2"),
        ("Excerpt 3: story", "excerpt", flow.PROPOSED, None),
    ]
    assert (cues[0]["start"], cues[0]["end"]) == (SPANS[0][0], SPANS[0][1])
    assert cues[0]["note"] == proposed.drafts[0].opening


def test_mark_writes_the_cues_as_this_flow_s_source():
    sent = []

    def invoke(tool, params):
        sent.append((tool, params))
        return {"project_id": params["project_id"], "cues": params["cues"]}

    proposed = proposed_drafts()
    marked = flow.mark(invoke, PROJECT_ID, proposed, {1: flow.APPROVED})
    assert sent[0][0] == "alfred_cues_set"
    assert sent[0][1]["source"] == flow.CUE_SOURCE == "alfred-excerpts"
    assert sent[0][1]["cues"] == flow.cues_of(proposed, {1: flow.APPROVED})
    assert marked["cues"] == sent[0][1]["cues"]


# --- how a run ends --------------------------------------------------------------------

HEARD = Transcribed(DONE, RECORDING, {"state": DONE}, 0.0)


def test_a_transcription_that_did_not_finish_ends_the_run_in_the_caption_flow_s_words():
    line = summary(PROJECT, Transcribed(NO_CAPTIONS, [], None, 0.0, "nothing to read"))
    assert line == (f"Nothing was asked: the transcription of alfred project {PROJECT_ID} "
                    "left nothing to read: nothing to read.")


def test_a_run_alfred_failed_says_why():
    line = summary(PROJECT, HEARD, failure="alfred's run of the excerpts recipe failed: boom")
    assert line == "Nothing was asked: alfred's run of the excerpts recipe failed: boom."


def test_a_recipe_that_left_nothing_says_so_with_alfred_s_notes():
    left = Proposal((), ("excerpts: no stretch of the source was found to keep",), {})
    line = summary(PROJECT, Transcribed(DONE, WELCOME, None, 0.0), left)
    assert line == ("Nothing was asked: the excerpts recipe left no draft of 'Set at the Hall' "
                    "(2 captions): excerpts: no stretch of the source was found to keep.")


def test_the_last_line_counts_each_stage():
    """Mutation: count a rejected excerpt as approved -- red."""
    proposed = proposed_drafts()
    reviewed = {1: APPROVE, 2: REJECT, 3: None}
    queued = {1: {"render_id": "r1"}}
    waited = {1: Waited(RENDERED, {}, 1.0)}
    line = summary(PROJECT, HEARD, proposed, reviewed, queued, waited, {1: APPROVE},
                   {1: Delivered("local", "http://alfred.test/r1")}, deliver=True)
    assert line == ("3 excerpts of 'Set at the Hall' offered, 1 approved (1 unanswered), "
                    "1 dropped, 1 rendered, 1 approved as excerpts, 1 delivered.")
    line = summary(PROJECT, HEARD, proposed, {1: APPROVE, 2: APPROVE, 3: REJECT},
                   {1: {"render_id": "r1"}, 2: {"render_id": "r2"}},
                   {1: Waited(RENDERED, {}, 1.0), 2: Waited("failed", {}, 1.0, "out of disk")},
                   {1: REJECT}, {})
    assert line == ("3 excerpts of 'Set at the Hall' offered, 2 approved, 1 dropped, 1 rendered "
                    "(excerpt 2 out of disk), 0 approved as excerpts.")
    line = summary(PROJECT, HEARD, proposed, {1: REJECT, 2: REJECT, 3: REJECT})
    assert line == "3 excerpts of 'Set at the Hall' offered, 0 approved, 3 dropped."


def test_a_delivery_alfred_refused_is_said_in_the_last_line():
    proposed = proposed_drafts()
    line = summary(PROJECT, HEARD, proposed, {1: APPROVE, 2: REJECT, 3: REJECT},
                   {1: {"render_id": "r1"}}, {1: Waited(RENDERED, {}, 1.0)}, {1: APPROVE},
                   {1: Delivered("dropbox", None, "alfred refused it")}, deliver=True)
    assert line.endswith("1 approved as excerpts, 0 delivered (excerpt 1: alfred refused it).")


def test_a_proposal_pickles_as_metaflow_keeps_artifacts():
    proposed = label(proposed_drafts(), lambda p: "x", categories=["x"])
    assert pickle.loads(pickle.dumps(proposed)) == proposed
    assert isinstance(proposed.drafts[0], Draft)


# --- end to end: the flow file's steps, through a qmcp server and its queue ------------------

FLOW_FILE = REPO / "examples" / "flows" / "alfred_excerpts.py"
CORRELATION = "alfred-excerpts-run-1"


def marks_apart(tools):
    """The tools a run called, the marks on the timeline set aside."""
    return [tool for tool in tools if tool != "alfred_cues_set"]


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
    style saved, and a worker that cuts drafts."""
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
        on_review=None, **parameters) -> SimpleNamespace:
    """One run of the flow file's steps, against the qmcp server at `url`.

    Time passes only when a wait sleeps. A person answers each request at the
    first sleep of its wait: each draft's review with `review`, one answer or a
    dict by excerpt number, and each render with `approval`, likewise; None is
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
                number = next(n for n, q in queued_by_excerpt(module).items()
                              if request_id.startswith(f"alfred-excerpt-render-{q}-"))
            return chosen.get(number)
        return chosen

    def asking(client, request, **options):
        asked.append(request)
        if on_review is not None and "-render-" not in request["request_id"]:
            on_review(request)
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

    def queued_by_excerpt(module) -> dict[int, str]:
        return {n: q["render_id"] for n, q in getattr(module._ran, "queued", {}).items()}

    monkeypatch.setattr(module, "read_transcript",
                        partial(flow.read_transcript, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(module, "propose",
                        partial(flow.propose, sleep=clock.sleep, clock=clock))
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


# Reading the project and its captions: two tools. The run: the drafts looked
# at before it, queued, then looked at twice while the stand-in's worker moves
# it, then each of the three drafts' captions read. Each render moves one
# state per look, so two status reads follow each enqueue.
READ_TOOLS = ["alfred_project", "alfred_captions"]
PROPOSE_TOOLS = ["alfred_drafts_status"] * 3 + ["alfred_drafts"] + ["alfred_captions"] * 3


def flow_cues(fake) -> list[dict]:
    return [c for c in fake.projects[PROJECT_ID]["cues"] if c["source"] == "alfred-excerpts"]


def test_chosen_excerpts_are_rendered_approved_and_delivered_and_the_rest_dropped(
        flow_file, monkeypatch, server, fake):
    """Through a qmcp server on an ephemeral port over its own database: every
    call to alfred is an invocation it recorded under the run's correlation id,
    and every request is answered on its queue."""
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: APPROVE},
              approval={1: APPROVE, 3: REJECT}, deliver=True)
    assert ran.line == ("3 excerpts of 'Set at the Hall' offered, 2 approved, 1 dropped, "
                        "2 rendered, 1 approved as excerpts, 1 delivered.")
    assert marks_apart(ran.tools) == sorted(
        READ_TOOLS + PROPOSE_TOOLS + ["alfred_drafts_drop"] + ["alfred_render_enqueue"] * 2
        + ["alfred_render_status"] * 4 + ["alfred_render_deliver"])
    # The timeline is marked as the run goes: once proposed, twice per review,
    # once queued, once rendered, once answered, once delivered.
    assert ran.tools.count("alfred_cues_set") == 1 + 2 * 3 + 1 + 1 + 1 + 1
    assert [(c["label"], c["state"], c["ref"]) for c in flow_cues(fake)] == [
        ("Excerpt 1", flow.DELIVERED, render_id(1)),
        ("Excerpt 2", flow.REJECTED, f"alfred-excerpt-{PROJECT_ID}-2-run-1"),
        ("Excerpt 3", flow.REJECTED, render_id(2)),
    ]
    # alfred's own cues stay, less the draft dropped, which is gone with its cue.
    alfreds = [c for c in fake.projects[PROJECT_ID]["cues"] if c["source"] == "alfred-drafts"]
    assert [c["ref"] for c in alfreds] == [draft_id(1), draft_id(3)]
    assert draft_id(2) not in fake.projects
    reviews, approvals = ran.asked[:3], ran.asked[3:]
    assert [r["request_id"] for r in reviews] == [
        f"alfred-excerpt-{PROJECT_ID}-{n}-run-1" for n in (1, 2, 3)]
    assert [a["request_id"] for a in approvals] == [
        f"alfred-excerpt-render-{render_id(n)}-run-1" for n in (1, 2)]
    # What alfred was asked to render: the first and third drafts' own
    # timelines and captions, under alfred's names, with the saved style on.
    first, third = fake.queued
    assert first["edl"] == fake.projects[draft_id(1)]["edl"]
    assert third["edl"] == fake.projects[draft_id(3)]["edl"]
    assert first["name"] == "Set at the Hall – Excerpts 1"
    assert third["name"] == "Set at the Hall – Excerpts 3"
    assert first["captions"] == fake.projects[draft_id(1)]["captions"]
    assert first["captions"][0]["start"] == 0.0
    assert first["caption_style"] == {**CAPTIONS["style"], "enabled": True, "fontsize": 40.0}
    assert (first["width"], first["height"]) == (1280, 720)
    assert fake.renders[render_id(1)]["destination"] == "local"
    assert fake.renders[render_id(2)]["destination"] is None
    assert ran.pending == []


def test_an_excerpt_approved_is_not_delivered_unless_the_run_was_asked_to(
        flow_file, monkeypatch, server, fake):
    """Mutation: deliver whenever an excerpt is approved -- red."""
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: REJECT})
    assert ran.line == ("3 excerpts of 'Set at the Hall' offered, 1 approved, 2 dropped, "
                        "1 rendered, 1 approved as excerpts.")
    assert "alfred_render_deliver" not in ran.tools
    assert ran.asked[3]["prompt"].endswith("Nothing is delivered either way: this run was not "
                                           "asked to deliver.")


def test_excerpts_nobody_chooses_render_nothing_and_are_dropped(
        flow_file, monkeypatch, server, fake):
    """Mutation: render whatever the review's answer -- red."""
    ran = run(flow_file, monkeypatch, server, review=REJECT)
    assert ran.line == "3 excerpts of 'Set at the Hall' offered, 0 approved, 3 dropped."
    assert marks_apart(ran.tools) == sorted(READ_TOOLS + PROPOSE_TOOLS + ["alfred_drafts_drop"])
    assert fake.queued == [] and len(ran.asked) == 3
    assert [c["state"] for c in flow_cues(fake)] == [flow.REJECTED] * 3
    assert not [d for d in fake.projects.values() if d.get("source")]


def test_a_review_nobody_answers_is_counted_and_the_draft_kept(
        flow_file, monkeypatch, server, fake):
    """Mutation: drop an unanswered draft -- red."""
    ran = run(flow_file, monkeypatch, server, review={1: APPROVE}, review_timeout=60)
    assert ran.line == ("3 excerpts of 'Set at the Hall' offered, 1 approved "
                        "(2 unanswered), 1 rendered, 1 approved as excerpts.")
    assert "alfred_drafts_drop" not in ran.tools
    assert draft_id(2) in fake.projects and draft_id(3) in fake.projects


def test_captions_are_left_off_the_excerpts_when_the_run_says(
        flow_file, monkeypatch, server, fake):
    """Mutation: burn them in regardless -- red."""
    run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: REJECT},
        captions=False)
    assert "captions" not in fake.queued[0] and "caption_style" not in fake.queued[0]


def test_a_caption_corrected_on_the_draft_during_the_review_is_in_the_excerpt(
        flow_file, monkeypatch, server, fake):
    """Mutation: queue the excerpt with the captions read when the drafts were
    proposed -- red, the excerpt burns the misheard word in."""
    def correct(request):
        held = fake.projects[draft_id(1)]["captions"]
        held[0] = {**held[0], "text": "This is sentence 0 of the story, corrected"}

    run(flow_file, monkeypatch, server, review={1: APPROVE, 2: REJECT, 3: REJECT},
        on_review=correct)
    assert fake.queued[0]["captions"][0]["text"] == "This is sentence 0 of the story, corrected"


def test_a_project_without_captions_is_transcribed_first(
        flow_file, monkeypatch, server, fake):
    project = fake.projects[PROJECT_ID]
    project["captions"], project["caption_style"] = None, None
    fake.heard[PROJECT_ID] = RECORDING
    ran = run(flow_file, monkeypatch, server, review=REJECT)
    assert ran.line == "3 excerpts of 'Set at the Hall' offered, 0 approved, 3 dropped."
    # The captions looked at before transcribing, and twice while the job moves.
    assert marks_apart(ran.tools) == sorted(
        ["alfred_project", "alfred_captions_transcribe"] + ["alfred_captions"] * 3
        + PROPOSE_TOOLS + ["alfred_drafts_drop"])


def test_a_project_the_run_may_not_transcribe_asks_nobody(
        flow_file, monkeypatch, server, fake):
    fake.projects[PROJECT_ID]["captions"] = None
    ran = run(flow_file, monkeypatch, server, transcribe=False)
    assert ran.line == (f"Nothing was asked: the transcription of alfred project {PROJECT_ID} "
                        "left nothing to read: the project holds no captions, and this run was "
                        "not asked to transcribe it.")
    assert ran.asked == [] and ran.tools == sorted(["alfred_project", "alfred_captions"])


def test_the_knobs_reach_alfred_s_recipe(flow_file, monkeypatch, server, fake):
    """Mutation: run the recipe with its own defaults -- red."""
    run(flow_file, monkeypatch, server, review=REJECT, pause=2.0, shortest=20.0, longest=45.0)
    [asked] = fake.drafts_asked
    assert asked == {"recipes": [{"recipe": "excerpts",
                                  "values": {"pause": "2", "shortest": "20", "longest": "45"}}]}


def test_a_local_labeller_is_built_from_the_run_s_settings(
        flow_file, monkeypatch, server, fake):
    """Mutation: ignore --label-with -- red, no labeller is asked."""
    built: list[tuple] = []

    def stand_in(base_url, model, *, timeout):
        built.append((base_url, model, timeout))
        return lambda prompt: "song"

    monkeypatch.setattr(flow_file, "local_model", stand_in)
    ran = run(flow_file, monkeypatch, server, review=REJECT, label_with="local",
              categories="song, story", llm_base_url="http://model.test/v1", llm_model="m",
              llm_timeout=9.0)
    assert built == [("http://model.test/v1", "m", 9.0)]
    assert ran.asked[0]["prompt"].startswith(
        "Excerpt 1 of 3 from 'Set at the Hall', 0:11.0-1:38.8 (88 s, song), "
        "alfred's 'Excerpts 1':")


def test_an_unknown_labeller_is_refused_before_alfred_is_asked(
        flow_file, monkeypatch, server, fake):
    with pytest.raises(ValueError, match="--label-with is none or local"):
        run(flow_file, monkeypatch, server, label_with="remote")
    assert fake.drafts_asked == []


def test_a_worker_that_never_runs_ends_the_run_saying_how_long_it_waited(
        flow_file, monkeypatch, server, fake):
    fake.workers = 0
    ran = run(flow_file, monkeypatch, server, drafts_timeout=10, poll_interval=5)
    assert ran.line == ("Nothing was asked: alfred's run of the excerpts recipe was still "
                        "queued after 10 s.")
    assert ran.asked == []


def test_a_run_alfred_refuses_is_the_tool_s_error(flow_file, monkeypatch, server, fake):
    fake.broker = False
    with pytest.raises(ToolFailedError, match="could not be queued"):
        run(flow_file, monkeypatch, server)


def test_a_tool_error_outside_the_known_refusals_is_raised(
        flow_file, monkeypatch, server, fake):
    del fake.projects[PROJECT_ID]
    with pytest.raises(ToolFailedError, match="404"):
        run(flow_file, monkeypatch, server)
