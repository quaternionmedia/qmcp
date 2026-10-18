"""The caption tools, a render's captions, delivery, and alfred's health report.

alfred is `CaptionedAlfred` in `tests/alfred_stub.py`, which holds projects,
captions and renders as alfred's caption routes do; nothing here needs a
running alfred.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import qmcp.tools.alfred as alfred
from qmcp.tools.alfred import (
    NO_AUDIO_ENGINE,
    TRANSCRIPTION_NOT_QUEUED,
    TRANSCRIPTION_RUNNING,
    AlfredError,
)
from tests.alfred_stub import (
    CAPTIONS,
    HEALTH,
    INVALID_ID_DETAIL,
    NO_ENGINE_DETAIL,
    NO_OUTPUT_DETAIL,
    NO_SPEECH_ERROR,
    NOT_QUEUED_DETAIL,
    NOT_QUEUED_ERROR,
    RUNNING_DETAIL,
    URL,
    CaptionedAlfred,
    FakeAlfred,
    connect,
)

HEARD = [
    {"start": 0.0, "end": 2.5, "text": "Hello and welcome."},
    {"start": 2.5, "end": 5.0, "text": "This is the demo."},
]
# A whole style, as alfred holds one: its colours are #rrggbb, as it requires.
STYLE = {"enabled": True, "fontsize": 48, "color": "#ffffff", "bg": "#000000",
         "position": "bottom", "align": "center", "font": "DejaVu-Sans"}

# Projects named as alfred names them, by an ObjectId's 24 hex digits. No test
# adds UNKNOWN.
P1, P9 = "6a78b0420eea42d905943601", "6a78b0420eea42d905943609"
UNKNOWN = "6a78b0420eea42d9059436ff"


@pytest.fixture
def fake(monkeypatch) -> CaptionedAlfred:
    """The tools, pointed at a stand-in alfred that holds captions."""
    stand_in = CaptionedAlfred()
    connected = connect(stand_in)
    monkeypatch.setattr(alfred, "client", lambda: connected)
    return stand_in


def asked(fake: FakeAlfred) -> list[tuple[str, str]]:
    """Every request after the login, as (method, path)."""
    return [(r.method, r.url.path) for r in fake.requests if r.url.path != "/auth/jwt/login"]


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)


# --- alfred_captions -----------------------------------------------------------


def test_captions_are_the_projects_captions_style_and_job(fake):
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    fake.projects[P1]["captions_job"] = {"state": "done", "error": None, "engine": "joe",
                                         "segments": 2, "updated_at": "2026-10-18T10:00:00"}
    assert alfred.alfred_captions({"project_id": P1}) == {
        "project_id": P1, "captions": HEARD, "style": STYLE,
        "job": {"state": "done", "error": None, "engine": "joe", "segments": 2,
                "updated_at": "2026-10-18T10:00:00"}}
    assert asked(fake) == [("GET", f"/project/{P1}/captions")]


def test_a_project_never_captioned_is_answered_as_alfred_answers_one(fake):
    """Held to alfred's own answer, `tests/alfred_captions.json`: no captions, no
    job, and a whole style switched off. Mutation: answer the style as stored,
    null -- red."""
    fake.add_project(P1, "Talk")
    assert alfred.alfred_captions({"project_id": P1}) == {"project_id": P1, **CAPTIONS}


def test_a_stored_style_is_answered_whole_as_alfred_answers_it(fake):
    """Every field of alfred's style, the ones never set null. Mutation: answer
    only the fields stored -- red."""
    fake.add_project(P1, "Talk", style={"enabled": True, "fontsize": 48})
    assert alfred.alfred_captions({"project_id": P1})["style"] == {
        **CAPTIONS["style"], "enabled": True, "fontsize": 48}


def test_a_project_id_cannot_leave_its_path_segment_in_the_caption_routes(fake):
    """Mutation: put the id into the caption paths unescaped -- red."""
    for tool in (alfred.alfred_captions, alfred.alfred_captions_transcribe):
        with pytest.raises(AlfredError):
            tool({"project_id": "../render"})
    assert [r.url.raw_path for r in fake.requests if r.url.raw_path.startswith(b"/project")] == [
        b"/project/..%2Frender/captions", b"/project/..%2Frender/captions/transcribe"]


@pytest.mark.parametrize("tool", ["alfred_captions", "alfred_captions_transcribe",
                                  "alfred_captions_set"])
def test_the_caption_tools_need_a_project_id(fake, tool):
    with pytest.raises(ValueError, match="project_id is required"):
        getattr(alfred, tool)({"captions": HEARD})
    assert fake.requests == []


def test_a_project_alfred_does_not_hold_is_its_own_not_found(fake):
    with pytest.raises(AlfredError, match=rf"GET /project/{UNKNOWN}/captions \(404\): "
                                          r"Item not found$"):
        alfred.alfred_captions({"project_id": UNKNOWN})


@pytest.mark.parametrize("tool,params", [
    ("alfred_captions", {}),
    ("alfred_captions_transcribe", {}),
    ("alfred_captions_set", {"captions": HEARD, "style": STYLE}),
    ("alfred_render_enqueue", {}),
])
def test_an_id_that_is_not_an_objectid_is_refused_before_anything_is_looked_up(
        fake, tool, params):
    """As alfred refuses one: even a project held under that name is not
    reached. Mutation: look the project up before checking the id -- red."""
    fake.add_project("nope", "Talk", captions=HEARD, style=STYLE)
    with pytest.raises(AlfredError) as refused:
        getattr(alfred, tool)({"project_id": "nope", **params})
    assert (refused.value.status, refused.value.detail) == (422, INVALID_ID_DETAIL)
    assert fake.queued == []


@pytest.mark.parametrize("tool", ["alfred_render_status", "alfred_render_deliver"])
def test_a_render_id_that_is_not_an_objectid_is_refused_as_alfred_refuses_it(fake, tool):
    fake.renders["r1"] = {"id": "r1", "state": "rendered", "progress": 100.0,
                          "destination": None}
    with pytest.raises(AlfredError) as refused:
        getattr(alfred, tool)({"render_id": "r1"})
    assert (refused.value.status, refused.value.detail) == (422, INVALID_ID_DETAIL)


# --- alfred_captions_transcribe --------------------------------------------------


def test_transcribing_starts_a_job_and_returns_it(fake):
    fake.add_project(P1, "Talk", heard=HEARD)
    started = alfred.alfred_captions_transcribe({"project_id": P1})
    assert started["project_id"] == P1
    assert started["job"]["state"] == "queued"
    request = fake.calls("POST", f"/project/{P1}/captions/transcribe")[0]
    assert request.content == b""  # alfred is asked to transcribe what it saved


def test_no_audio_engine_is_said_first_and_alfred_s_detail_kept(fake):
    """Mutation: let alfred's 503 through as it came -- red, the flow could not
    tell it from any other refusal."""
    fake.add_project(P1, "Talk")
    fake.without_an_audio_engine()
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": P1})
    assert str(refused.value) == (
        f"{NO_AUDIO_ENGINE}: alfred refused POST /project/{P1}/captions/transcribe (503): "
        f"{NO_ENGINE_DETAIL}")
    assert (refused.value.status, refused.value.detail) == (503, NO_ENGINE_DETAIL)


def test_a_running_transcription_is_said_first_and_alfred_s_detail_kept(fake):
    """Mutation: let alfred's 409 through as it came -- red."""
    fake.add_project(P1, "Talk")
    alfred.alfred_captions_transcribe({"project_id": P1})
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": P1})
    assert str(refused.value) == (
        f"{TRANSCRIPTION_RUNNING}: alfred refused POST /project/{P1}/captions/transcribe "
        f"(409): {RUNNING_DETAIL}")
    assert refused.value.status == 409


def test_a_job_alfred_could_not_queue_is_said_as_that_not_as_no_audio_engine(fake):
    """alfred's other 503: its broker would not take the job, which alfred then
    holds as failed. Mutation: open every 503 with NO_AUDIO_ENGINE -- red."""
    fake.add_project(P1, "Talk")
    fake.broker = False
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": P1})
    assert str(refused.value) == (
        f"{TRANSCRIPTION_NOT_QUEUED}: alfred refused POST /project/{P1}/captions/transcribe "
        f"(503): {NOT_QUEUED_DETAIL}")
    assert (refused.value.status, refused.value.detail) == (503, NOT_QUEUED_DETAIL)
    assert alfred.alfred_captions({"project_id": P1})["job"]["error"] == NOT_QUEUED_ERROR


# Every reason alfred gives for having no audio engine to ask, as its
# audio_engine.problem() words them; the list of vox's fields is cut short.
NO_ENGINE_REASONS = [
    NO_ENGINE_DETAIL,
    "ALFRED_AUDIO_ENGINE_URL must be an http(s) URL, not 'engine:8010'.",
    "The vendored vox (alfred/vox) has no media client (vox.media), so the audio engine at "
    "http://joe.test:8010 cannot be asked anything.",
    "ALFRED_AUDIO_ENGINE_CONTRACT_MEDIA_UPLOADS: not a field of vox's EngineContract, so not "
    "something an engine can be told to spell differently; the fields are audio_path_key, ...",
    "ALFRED_AUDIO_ENGINE_CONTRACT_*: vox refused the contract: media_transcribe "
    "'/api/media/transcribe' must name {media_id} once: it is where an upload's id goes",
]


@pytest.mark.parametrize("detail", NO_ENGINE_REASONS)
def test_every_reason_alfred_gives_for_no_audio_engine_is_said_alike(fake, detail):
    """Mutation: recognise only the first of them -- red for the rest."""
    fake.route("POST", f"/project/{P1}/captions/transcribe",
               lambda request: httpx.Response(503, json={"detail": detail}))
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": P1})
    assert str(refused.value).startswith(f"{NO_AUDIO_ENGINE}: alfred refused POST ")
    assert refused.value.detail == detail


def test_a_503_saying_neither_is_passed_on_unopened(fake):
    """What a proxy in front of alfred answers while alfred is down. Mutation:
    open any 503 but the queue's with NO_AUDIO_ENGINE -- red."""
    fake.route("POST", f"/project/{P1}/captions/transcribe",
               lambda request: httpx.Response(503, text="no available server"))
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": P1})
    assert str(refused.value) == (f"alfred refused POST /project/{P1}/captions/transcribe (503): "
                                  "no available server")


def test_another_refusal_is_alfred_s_own(fake):
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": UNKNOWN})
    assert str(refused.value) == (f"alfred refused POST /project/{UNKNOWN}/captions/transcribe "
                                  "(404): Item not found")


def test_an_alfred_without_captions_is_named_by_what_answers(monkeypatch):
    """An alfred from before captions has no such route: FastAPI's own 404."""
    stand_in = FakeAlfred()
    connected = connect(stand_in)
    monkeypatch.setattr(alfred, "client", lambda: connected)
    with pytest.raises(AlfredError, match=r"\(404\): Not Found \(the server at "
                                          r"http://alfred.test calls itself 'Alfred API'\)"):
        alfred.alfred_captions_transcribe({"project_id": P1})


def test_a_job_moves_as_alfred_s_does_while_a_worker_runs(fake):
    """The stand-in's lifecycle, which the flow's tests lean on: one state a look."""
    fake.add_project(P1, "Talk", heard=HEARD)
    alfred.alfred_captions_transcribe({"project_id": P1})
    seen = [alfred.alfred_captions({"project_id": P1}) for _ in range(3)]
    assert [s["job"]["state"] for s in seen] == ["transcribing", "done", "done"]
    assert seen[-1]["captions"] == HEARD and seen[-1]["job"]["segments"] == 2


def test_a_job_that_hears_no_speech_fails_as_alfred_s_does(fake):
    """alfred says so as the job's error and keeps the captions the project had,
    rather than finishing with none."""
    fake.add_project(P1, "Talk", captions=HEARD)  # and nothing for the engine to hear
    alfred.alfred_captions_transcribe({"project_id": P1})
    seen = [alfred.alfred_captions({"project_id": P1}) for _ in range(2)]
    assert [s["job"]["state"] for s in seen] == ["transcribing", "failed"]
    assert (seen[-1]["job"]["error"], seen[-1]["captions"]) == (NO_SPEECH_ERROR, HEARD)


# --- alfred_captions_set -------------------------------------------------------------


def test_captions_and_style_are_stored_by_one_put_carrying_exactly_them(fake):
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    corrected = [{**HEARD[0], "text": "Hello, and welcome."}, HEARD[1]]
    stored = alfred.alfred_captions_set({"project_id": P1, "captions": corrected,
                                         "style": {**STYLE, "fontsize": 40}})
    assert asked(fake) == [("PUT", f"/project/{P1}/captions")]
    assert body(fake.calls("PUT", f"/project/{P1}/captions")[0]) == {
        "captions": corrected, "style": {**STYLE, "fontsize": 40}}
    assert stored == {"project_id": P1, "captions": corrected,
                      "style": {**STYLE, "fontsize": 40}, "job": None}


def test_a_correction_of_the_captions_sends_back_the_style_alfred_holds(fake):
    """Mutation: send only what was given -- red, a PUT that replaces both would
    reset the style."""
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    corrected = [HEARD[0], {**HEARD[1], "text": "This is our demo."}]
    alfred.alfred_captions_set({"project_id": P1, "captions": corrected})
    assert asked(fake) == [("GET", f"/project/{P1}/captions"), ("PUT", f"/project/{P1}/captions")]
    assert body(fake.calls("PUT", f"/project/{P1}/captions")[0]) == {
        "captions": corrected, "style": STYLE}


def test_a_new_style_sends_back_the_captions_alfred_holds(fake):
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    alfred.alfred_captions_set({"project_id": P1, "style": {"enabled": False}})
    assert body(fake.calls("PUT", f"/project/{P1}/captions")[0]) == {
        "captions": HEARD, "style": {"enabled": False}}


def test_a_style_never_set_goes_back_as_alfred_answers_it_switched_off(fake):
    """alfred answers a whole style for every project, so a correction of the
    captions alone sends that back."""
    fake.add_project(P1, "Talk", captions=HEARD)
    alfred.alfred_captions_set({"project_id": P1, "captions": HEARD[:1]})
    assert body(fake.calls("PUT", f"/project/{P1}/captions")[0]) == {
        "captions": HEARD[:1], "style": CAPTIONS["style"]}


def test_a_style_sent_without_enabled_keeps_the_one_alfred_holds(fake):
    """As alfred stores it: `enabled` as sent, else as it was, else false."""
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    stored = alfred.alfred_captions_set({"project_id": P1, "style": {"fontsize": 40}})
    assert stored["style"] == {**CAPTIONS["style"], "enabled": True, "fontsize": 40}


@pytest.mark.parametrize("refused,why", [
    ({"start": 4.0, "end": 3.0, "text": "Backwards."},
     "body.captions.1.__root__: a caption must end after it starts (start 4.0, end 3.0)"),
    ({"start": -1.0, "end": 3.0, "text": "Early."},
     "body.captions.1.start: start must not be negative"),
    ({"start": 6.0, "end": 7.0, "text": "   "}, "body.captions.1.text: a caption needs some text"),
])
def test_alfred_s_refusal_of_a_caption_is_surfaced(fake, refused, why):
    """In alfred's words: its validation error, where and what."""
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    with pytest.raises(AlfredError) as refusal:
        alfred.alfred_captions_set({"project_id": P1, "captions": [HEARD[0], refused]})
    assert str(refusal.value) == f"alfred refused PUT /project/{P1}/captions (422): {why}"
    assert fake.projects[P1]["captions"] == HEARD


def test_captions_sent_out_of_order_come_back_in_alfred_s_order(fake):
    """alfred puts them in order rather than refusing them, and the result is
    what alfred holds. Mutation: return the captions as sent -- red."""
    fake.add_project(P1, "Talk", captions=[], style=STYLE)
    stored = alfred.alfred_captions_set({"project_id": P1, "captions": HEARD[::-1]})
    assert body(fake.calls("PUT", f"/project/{P1}/captions")[0])["captions"] == HEARD[::-1]
    assert stored["captions"] == HEARD == fake.projects[P1]["captions"]


def test_more_captions_than_alfred_keeps_is_its_refusal(fake):
    fake.add_project(P1, "Talk", style=STYLE)
    many = [{"start": float(n), "end": n + 0.5, "text": f"line {n}"} for n in range(1001)]
    with pytest.raises(AlfredError, match=r"\(422\): 1001 captions is more than the 1000 a "
                                          r"project may carry$"):
        alfred.alfred_captions_set({"project_id": P1, "captions": many})


def test_the_body_is_refused_before_the_project_is_looked_up(fake):
    """As alfred does: FastAPI checks a PUT's captions before the route looks at
    the id or the project, and the route counts them only once it has found
    the project. Mutation: look the project up first -- red."""
    backwards = [{"start": 4.0, "end": 3.0, "text": "Backwards."}]
    for project_id in (UNKNOWN, "nope"):
        with pytest.raises(AlfredError) as refused:
            alfred.alfred_captions_set({"project_id": project_id, "captions": backwards,
                                        "style": STYLE})
        assert refused.value.status == 422
        assert refused.value.detail[0]["loc"] == ["body", "captions", 0, "__root__"]
    many = [{"start": float(n), "end": n + 0.5, "text": f"line {n}"} for n in range(1001)]
    with pytest.raises(AlfredError, match=r"\(404\): Item not found$"):
        alfred.alfred_captions_set({"project_id": UNKNOWN, "captions": many, "style": STYLE})


def test_a_field_alfred_does_not_take_is_refused_in_alfred_s_words(fake):
    fake.add_project(P1, "Talk")
    with pytest.raises(AlfredError, match=r"\(422\): body\.job: extra fields not permitted$"):
        connect(fake).put(f"/project/{P1}/captions", {"captions": [], "job": None})


@pytest.mark.parametrize("params,why", [
    ({}, "give captions, style, or both"),
    ({"captions": {"start": 0}}, "captions must be a list"),
    ({"style": "large"}, "style must be an object"),
])
def test_set_refuses_what_is_not_captions_or_a_style_before_asking(fake, params, why):
    with pytest.raises(ValueError, match=why):
        alfred.alfred_captions_set({"project_id": P1, **params})
    assert fake.requests == []


def test_an_answer_that_is_not_the_captions_is_read_back(fake):
    """Mutation: return the PUT's answer as it is -- red, the result says nothing
    alfred holds."""
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    fake.route("PUT", f"/project/{P1}/captions", lambda request: httpx.Response(200, json=True))
    stored = alfred.alfred_captions_set({"project_id": P1, "captions": HEARD, "style": STYLE})
    assert asked(fake) == [("PUT", f"/project/{P1}/captions"), ("GET", f"/project/{P1}/captions")]
    assert stored["captions"] == HEARD and stored["style"] == STYLE


# --- a render's captions ---------------------------------------------------------------


def queued_body(fake: CaptionedAlfred) -> dict:
    return fake.queued[-1]


def test_a_project_whose_caption_style_is_enabled_renders_with_its_captions(fake):
    """Mutation: never send the captions -- red."""
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    queued = alfred.alfred_render_enqueue({"project_id": P1})
    sent = queued_body(fake)
    assert (sent["captions"], sent["caption_style"]) == (HEARD, STYLE)  # as alfred stores them
    assert queued["captions"] == 2


@pytest.mark.parametrize("style", [
    {**STYLE, "enabled": False},
    {"fontsize": 48},  # says nothing about enabled
    None,
])
def test_a_project_whose_caption_style_is_not_enabled_renders_without_captions(fake, style):
    """Mutation: send captions whenever there are some -- red."""
    fake.add_project(P1, "Talk", captions=HEARD, style=style)
    queued = alfred.alfred_render_enqueue({"project_id": P1})
    assert "captions" not in queued_body(fake) and "caption_style" not in queued_body(fake)
    assert queued["captions"] == 0


def test_an_enabled_style_with_no_captions_sends_none(fake):
    fake.add_project(P1, "Talk", style=STYLE)
    assert alfred.alfred_render_enqueue({"project_id": P1})["captions"] == 0
    assert "captions" not in queued_body(fake)


def test_captions_true_burns_them_in_with_the_style_switched_on(fake):
    """Mutation: send the saved style as it is -- red, alfred skips a style that
    is not enabled."""
    fake.add_project(P1, "Talk", captions=HEARD, style={**STYLE, "enabled": False})
    queued = alfred.alfred_render_enqueue({"project_id": P1, "captions": True})
    assert queued_body(fake)["caption_style"] == STYLE
    assert queued_body(fake)["captions"] == HEARD and queued["captions"] == 2
    assert fake.projects[P1]["caption_style"]["enabled"] is False  # nothing was saved


def test_captions_true_with_no_style_saved_switches_one_on(fake):
    fake.add_project(P1, "Talk", captions=HEARD)
    alfred.alfred_render_enqueue({"project_id": P1, "captions": True})
    assert queued_body(fake)["caption_style"] == {"enabled": True}


def test_captions_false_sends_none_even_when_enabled(fake):
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    assert alfred.alfred_render_enqueue({"project_id": P1, "captions": False})["captions"] == 0
    assert "captions" not in queued_body(fake)


def test_captions_true_without_captions_is_refused_before_anything_is_queued(fake):
    fake.add_project(P1, "Talk", style=STYLE)
    with pytest.raises(AlfredError, match=rf"{P1} \('Talk'\) has no captions to burn in; "
                                          "alfred_captions_transcribe makes them"):
        alfred.alfred_render_enqueue({"project_id": P1, "captions": True})
    assert fake.queued == []


def test_captions_come_from_a_project_not_an_explicit_edl(fake):
    with pytest.raises(ValueError, match="captions come from a saved project"):
        alfred.alfred_render_enqueue({"edl": {"clips": [{"name": "title"}]}, "captions": True})
    assert fake.requests == []


def test_captions_is_a_boolean(fake):
    with pytest.raises(ValueError, match="captions is true, false, or left out"):
        alfred.alfred_render_enqueue({"project_id": P1, "captions": "yes"})
    assert fake.requests == []


def test_a_rerender_of_the_same_project_carries_the_captions_as_they_now_are(fake):
    """The render reads the project when it is queued, so a correction is in it."""
    fake.add_project(P1, "Talk", captions=HEARD, style=STYLE)
    alfred.alfred_captions_set({"project_id": P1, "captions": HEARD[:1]})
    alfred.alfred_render_enqueue({"project_id": P1})
    assert queued_body(fake)["captions"] == HEARD[:1]


# --- alfred_render_deliver -------------------------------------------------------------


def rendered(fake: CaptionedAlfred) -> str:
    """A render of a fresh project, followed until it is rendered."""
    fake.add_project(P9, "Talk")
    render_id = alfred.alfred_render_enqueue({"project_id": P9})["render_id"]
    while alfred.alfred_render_status({"render_id": render_id})["state"] != "rendered":
        pass
    return render_id


def test_a_render_is_delivered_where_alfred_chooses_and_the_location_made_absolute(fake):
    """Mutation: return alfred's location as it is -- red, it is a path on alfred."""
    render_id = rendered(fake)
    delivered = alfred.alfred_render_deliver({"render_id": render_id})
    assert delivered == {"render_id": render_id, "destination": "local",
                         "location": f"{URL}/render/{render_id}/download"}
    request = fake.calls("POST", f"/render/{render_id}/deliver")[0]
    assert request.url.query == b"" and request.content == b""


def test_a_named_destination_is_alfred_s_query_parameter(fake):
    """Mutation: send the destination in a JSON body -- red, alfred reads the query."""
    render_id = rendered(fake)
    alfred.alfred_render_deliver({"render_id": render_id, "destination": "local"})
    request = fake.calls("POST", f"/render/{render_id}/deliver")[0]
    assert request.url.params == httpx.QueryParams(destination="local")


def test_an_unfinished_render_is_refused_with_alfred_s_detail(fake):
    fake.add_project(P9, "Talk")
    render_id = alfred.alfred_render_enqueue({"project_id": P9})["render_id"]
    with pytest.raises(AlfredError, match=rf"deliver \(409\): {NO_OUTPUT_DETAIL}$"):
        alfred.alfred_render_deliver({"render_id": render_id})


def test_a_destination_alfred_cannot_reach_is_its_502(fake):
    render_id = rendered(fake)
    with pytest.raises(AlfredError, match=r"\(502\): unknown destination 'dropbox'"):
        alfred.alfred_render_deliver({"render_id": render_id, "destination": "dropbox"})


def test_a_delivery_is_given_its_own_longer_wait(fake):
    """alfred answers a delivery once the whole render is copied or uploaded.
    Mutation: send it with the client's own timeout -- red."""
    render_id = rendered(fake)
    alfred.alfred_render_deliver({"render_id": render_id})
    waits = {(r.method, r.url.path): r.extensions["timeout"]["read"] for r in fake.requests}
    assert waits[("POST", f"/render/{render_id}/deliver")] == alfred.DEFAULT_DELIVER_TIMEOUT
    assert waits[("GET", f"/render/{render_id}/status")] == alfred.DEFAULT_TIMEOUT


def test_a_delivery_that_outlasts_the_wait_says_alfred_carries_on_with_it(fake):
    render_id = rendered(fake)

    def copying(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    fake.route("POST", f"/render/{render_id}/deliver", copying)
    with pytest.raises(AlfredError) as failed:
        alfred.alfred_render_deliver({"render_id": render_id})
    assert str(failed.value) == (
        f"alfred at {URL} did not answer POST /render/{render_id}/deliver in time "
        f"(ReadTimeout, {alfred.DEFAULT_DELIVER_TIMEOUT:g} s); alfred answers a delivery once "
        "it has copied or uploaded the whole render, and carries on with one nobody waits "
        "for: QMCP_ALFRED_DELIVER_TIMEOUT is how long qmcp waits")


def test_delivery_needs_a_render_id(fake):
    with pytest.raises(ValueError, match="render_id is required"):
        alfred.alfred_render_deliver({})


# --- alfred_health ---------------------------------------------------------------------


def keys(value: Any, within: str = "") -> set[str]:
    """Every key of a JSON object, a nested one as `outer.inner`."""
    if not isinstance(value, dict):
        return set()
    found: set[str] = set()
    for key, inner in value.items():
        found |= {within + key} | keys(inner, f"{within}{key}.")
    return found


def test_the_stand_in_reports_health_as_alfred_does(fake):
    """Held to alfred's own answer, `tests/alfred_health.json`: with no audio
    engine the stand-in answers exactly that, and with one, the same keys.
    Mutation: report `speech` for `audio_engine`, or leave out its `detail` -- red."""
    fake.without_an_audio_engine()
    assert alfred.alfred_health({}) == HEALTH
    fake.audio_engine = {"configured": True, "ok": True, "detail": None}
    assert keys(alfred.alfred_health({})) == keys(HEALTH)


def test_health_carries_alfred_s_audio_engine_report(fake):
    why = "the audio engine at http://joe.test:8010: it failed (500): no model loaded"
    fake.audio_engine = {"configured": True, "ok": False, "detail": why}
    report = alfred.alfred_health({})
    assert report["audio_engine"] == {"configured": True, "ok": False, "detail": why}
    assert report["renderer"] == {"ok": True, "workers": 1}


# --- through the server ----------------------------------------------------------------


def test_the_server_lists_the_caption_tools(client):
    names = {tool["name"] for tool in client.get("/v1/tools").json()["tools"]}
    assert {"alfred_captions", "alfred_captions_transcribe", "alfred_captions_set",
            "alfred_render_deliver"} <= names


def test_no_audio_engine_reaches_the_invocation_record_in_so_many_words(client, fake):
    fake.add_project(P1, "Talk")
    fake.without_an_audio_engine()
    answer = client.post("/v1/tools/alfred_captions_transcribe",
                         json={"input": {"project_id": P1}}).json()
    assert answer["error"] == (
        f"Tool execution failed: {NO_AUDIO_ENGINE}: alfred refused POST "
        f"/project/{P1}/captions/transcribe (503): {NO_ENGINE_DETAIL}")
    recorded = client.get(f"/v1/invocations/{answer['invocation_id']}").json()
    assert recorded["status"] == "failed" and NO_ENGINE_DETAIL in recorded["error"]
