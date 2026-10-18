"""The caption tools, a render's captions, delivery, and alfred's speech report.

alfred is `CaptionedAlfred` in `tests/alfred_stub.py`, which holds projects,
captions and renders as alfred's caption routes do; nothing here needs a
running alfred.
"""

from __future__ import annotations

import json

import httpx
import pytest

import qmcp.tools.alfred as alfred
from qmcp.tools.alfred import NO_SPEECH_ENGINE, TRANSCRIPTION_RUNNING, AlfredError
from tests.alfred_stub import (
    NO_ENGINE_DETAIL,
    NO_OUTPUT_DETAIL,
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
STYLE = {"enabled": True, "fontsize": 48, "color": "white", "bg": "black",
         "position": "bottom", "align": "center", "font": "Roboto"}


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
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    fake.projects["p1"]["captions_job"] = {"state": "done", "error": None, "engine": "joe",
                                           "segments": 2, "updated_at": "2026-10-18T10:00:00"}
    assert alfred.alfred_captions({"project_id": "p1"}) == {
        "project_id": "p1", "captions": HEARD, "style": STYLE,
        "job": {"state": "done", "error": None, "engine": "joe", "segments": 2,
                "updated_at": "2026-10-18T10:00:00"}}
    assert asked(fake) == [("GET", "/project/p1/captions")]


def test_a_project_never_transcribed_has_no_job(fake):
    fake.add_project("p1", "Talk")
    assert alfred.alfred_captions({"project_id": "p1"}) == {
        "project_id": "p1", "captions": [], "style": None, "job": None}


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
    with pytest.raises(AlfredError, match=r"GET /project/nope/captions \(404\): Item not found$"):
        alfred.alfred_captions({"project_id": "nope"})


# --- alfred_captions_transcribe --------------------------------------------------


def test_transcribing_starts_a_job_and_returns_it(fake):
    fake.add_project("p1", "Talk", heard=HEARD)
    started = alfred.alfred_captions_transcribe({"project_id": "p1"})
    assert started["project_id"] == "p1"
    assert started["job"]["state"] == "queued"
    request = fake.calls("POST", "/project/p1/captions/transcribe")[0]
    assert request.content == b""  # alfred is asked to transcribe what it saved


def test_no_speech_engine_is_said_first_and_alfred_s_detail_kept(fake):
    """Mutation: let alfred's 503 through as it came -- red, the flow could not
    tell it from any other refusal."""
    fake.add_project("p1", "Talk")
    fake.speech = {"configured": False, "ok": False}
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": "p1"})
    assert str(refused.value) == (
        f"{NO_SPEECH_ENGINE}: alfred refused POST /project/p1/captions/transcribe (503): "
        f"{NO_ENGINE_DETAIL}")
    assert (refused.value.status, refused.value.detail) == (503, NO_ENGINE_DETAIL)


def test_a_running_transcription_is_said_first_and_alfred_s_detail_kept(fake):
    """Mutation: let alfred's 409 through as it came -- red."""
    fake.add_project("p1", "Talk")
    alfred.alfred_captions_transcribe({"project_id": "p1"})
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": "p1"})
    assert str(refused.value) == (
        f"{TRANSCRIPTION_RUNNING}: alfred refused POST /project/p1/captions/transcribe "
        f"(409): {RUNNING_DETAIL}")
    assert refused.value.status == 409


def test_another_refusal_is_alfred_s_own(fake):
    with pytest.raises(AlfredError) as refused:
        alfred.alfred_captions_transcribe({"project_id": "nope"})
    assert str(refused.value) == ("alfred refused POST /project/nope/captions/transcribe (404): "
                                  "Item not found")


def test_an_alfred_without_captions_is_named_by_what_answers(monkeypatch):
    """An alfred from before captions has no such route: FastAPI's own 404."""
    stand_in = FakeAlfred()
    connected = connect(stand_in)
    monkeypatch.setattr(alfred, "client", lambda: connected)
    with pytest.raises(AlfredError, match=r"\(404\): Not Found \(the server at "
                                          r"http://alfred.test calls itself 'Alfred API'\)"):
        alfred.alfred_captions_transcribe({"project_id": "p1"})


def test_a_job_moves_as_alfred_s_does_while_a_worker_runs(fake):
    """The stand-in's lifecycle, which the flow's tests lean on: one state a look."""
    fake.add_project("p1", "Talk", heard=HEARD)
    alfred.alfred_captions_transcribe({"project_id": "p1"})
    seen = [alfred.alfred_captions({"project_id": "p1"}) for _ in range(3)]
    assert [s["job"]["state"] for s in seen] == ["transcribing", "done", "done"]
    assert seen[-1]["captions"] == HEARD and seen[-1]["job"]["segments"] == 2


# --- alfred_captions_set -------------------------------------------------------------


def test_captions_and_style_are_stored_by_one_put_carrying_exactly_them(fake):
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    corrected = [{**HEARD[0], "text": "Hello, and welcome."}, HEARD[1]]
    stored = alfred.alfred_captions_set({"project_id": "p1", "captions": corrected,
                                         "style": {**STYLE, "fontsize": 40}})
    assert asked(fake) == [("PUT", "/project/p1/captions")]
    assert body(fake.calls("PUT", "/project/p1/captions")[0]) == {
        "captions": corrected, "style": {**STYLE, "fontsize": 40}}
    assert stored == {"project_id": "p1", "captions": corrected,
                      "style": {**STYLE, "fontsize": 40}, "job": None}


def test_a_correction_of_the_captions_sends_back_the_style_alfred_holds(fake):
    """Mutation: send only what was given -- red, a PUT that replaces both would
    reset the style."""
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    corrected = [HEARD[0], {**HEARD[1], "text": "This is our demo."}]
    alfred.alfred_captions_set({"project_id": "p1", "captions": corrected})
    assert asked(fake) == [("GET", "/project/p1/captions"), ("PUT", "/project/p1/captions")]
    assert body(fake.calls("PUT", "/project/p1/captions")[0]) == {
        "captions": corrected, "style": STYLE}


def test_a_new_style_sends_back_the_captions_alfred_holds(fake):
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    alfred.alfred_captions_set({"project_id": "p1", "style": {"enabled": False}})
    assert body(fake.calls("PUT", "/project/p1/captions")[0]) == {
        "captions": HEARD, "style": {"enabled": False}}


def test_a_style_never_set_is_left_out_rather_than_sent_as_null(fake):
    fake.add_project("p1", "Talk", captions=HEARD)
    alfred.alfred_captions_set({"project_id": "p1", "captions": HEARD[:1]})
    assert body(fake.calls("PUT", "/project/p1/captions")[0]) == {"captions": HEARD[:1]}


def test_alfred_s_refusal_of_a_caption_is_surfaced(fake):
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    backwards = [HEARD[0], {"start": 4.0, "end": 3.0, "text": "Backwards."}]
    with pytest.raises(AlfredError, match=r"PUT /project/p1/captions \(422\): "
                                          r"caption 2: end must be after start$"):
        alfred.alfred_captions_set({"project_id": "p1", "captions": backwards})
    assert fake.projects["p1"]["captions"] == HEARD


@pytest.mark.parametrize("params,why", [
    ({}, "give captions, style, or both"),
    ({"captions": {"start": 0}}, "captions must be a list"),
    ({"style": "large"}, "style must be an object"),
])
def test_set_refuses_what_is_not_captions_or_a_style_before_asking(fake, params, why):
    with pytest.raises(ValueError, match=why):
        alfred.alfred_captions_set({"project_id": "p1", **params})
    assert fake.requests == []


def test_an_answer_that_is_not_the_captions_is_read_back(fake):
    """Mutation: return the PUT's answer as it is -- red, the result says nothing
    alfred holds."""
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    fake.route("PUT", "/project/p1/captions", lambda request: httpx.Response(200, json=True))
    stored = alfred.alfred_captions_set({"project_id": "p1", "captions": HEARD, "style": STYLE})
    assert asked(fake) == [("PUT", "/project/p1/captions"), ("GET", "/project/p1/captions")]
    assert stored["captions"] == HEARD and stored["style"] == STYLE


# --- a render's captions ---------------------------------------------------------------


def queued_body(fake: CaptionedAlfred) -> dict:
    return fake.queued[-1]


def test_a_project_whose_caption_style_is_enabled_renders_with_its_captions(fake):
    """Mutation: never send the captions -- red."""
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    queued = alfred.alfred_render_enqueue({"project_id": "p1"})
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
    fake.add_project("p1", "Talk", captions=HEARD, style=style)
    queued = alfred.alfred_render_enqueue({"project_id": "p1"})
    assert "captions" not in queued_body(fake) and "caption_style" not in queued_body(fake)
    assert queued["captions"] == 0


def test_an_enabled_style_with_no_captions_sends_none(fake):
    fake.add_project("p1", "Talk", style=STYLE)
    assert alfred.alfred_render_enqueue({"project_id": "p1"})["captions"] == 0
    assert "captions" not in queued_body(fake)


def test_captions_true_burns_them_in_with_the_style_switched_on(fake):
    """Mutation: send the saved style as it is -- red, alfred skips a style that
    is not enabled."""
    fake.add_project("p1", "Talk", captions=HEARD, style={**STYLE, "enabled": False})
    queued = alfred.alfred_render_enqueue({"project_id": "p1", "captions": True})
    assert queued_body(fake)["caption_style"] == STYLE
    assert queued_body(fake)["captions"] == HEARD and queued["captions"] == 2
    assert fake.projects["p1"]["caption_style"]["enabled"] is False  # nothing was saved


def test_captions_true_with_no_style_saved_switches_one_on(fake):
    fake.add_project("p1", "Talk", captions=HEARD)
    alfred.alfred_render_enqueue({"project_id": "p1", "captions": True})
    assert queued_body(fake)["caption_style"] == {"enabled": True}


def test_captions_false_sends_none_even_when_enabled(fake):
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    assert alfred.alfred_render_enqueue({"project_id": "p1", "captions": False})["captions"] == 0
    assert "captions" not in queued_body(fake)


def test_captions_true_without_captions_is_refused_before_anything_is_queued(fake):
    fake.add_project("p1", "Talk", style=STYLE)
    with pytest.raises(AlfredError, match=r"p1 \('Talk'\) has no captions to burn in; "
                                          "alfred_captions_transcribe makes them"):
        alfred.alfred_render_enqueue({"project_id": "p1", "captions": True})
    assert fake.queued == []


def test_captions_come_from_a_project_not_an_explicit_edl(fake):
    with pytest.raises(ValueError, match="captions come from a saved project"):
        alfred.alfred_render_enqueue({"edl": {"clips": [{"name": "title"}]}, "captions": True})
    assert fake.requests == []


def test_captions_is_a_boolean(fake):
    with pytest.raises(ValueError, match="captions is true, false, or left out"):
        alfred.alfred_render_enqueue({"project_id": "p1", "captions": "yes"})
    assert fake.requests == []


def test_a_rerender_of_the_same_project_carries_the_captions_as_they_now_are(fake):
    """The render reads the project when it is queued, so a correction is in it."""
    fake.add_project("p1", "Talk", captions=HEARD, style=STYLE)
    alfred.alfred_captions_set({"project_id": "p1", "captions": HEARD[:1]})
    alfred.alfred_render_enqueue({"project_id": "p1"})
    assert queued_body(fake)["captions"] == HEARD[:1]


# --- alfred_render_deliver -------------------------------------------------------------


def rendered(fake: CaptionedAlfred) -> str:
    """A render of a fresh project, followed until it is rendered."""
    fake.add_project("p9", "Talk")
    render_id = alfred.alfred_render_enqueue({"project_id": "p9"})["render_id"]
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
    fake.add_project("p9", "Talk")
    render_id = alfred.alfred_render_enqueue({"project_id": "p9"})["render_id"]
    with pytest.raises(AlfredError, match=rf"deliver \(409\): {NO_OUTPUT_DETAIL}$"):
        alfred.alfred_render_deliver({"render_id": render_id})


def test_a_destination_alfred_cannot_reach_is_its_502(fake):
    render_id = rendered(fake)
    with pytest.raises(AlfredError, match=r"\(502\): unknown destination 'dropbox'"):
        alfred.alfred_render_deliver({"render_id": render_id, "destination": "dropbox"})


def test_delivery_needs_a_render_id(fake):
    with pytest.raises(ValueError, match="render_id is required"):
        alfred.alfred_render_deliver({})


# --- alfred_health ---------------------------------------------------------------------


def test_health_carries_alfred_s_speech_report(fake):
    fake.speech = {"configured": True, "ok": False}
    report = alfred.alfred_health({})
    assert report["speech"] == {"configured": True, "ok": False}
    assert report["renderer"] == {"ok": True, "workers": 1}


def test_health_from_an_alfred_that_does_not_transcribe_says_nothing_of_speech(fake):
    """Nothing is filled in for an alfred that does not report it."""
    fake.speech = None
    assert "speech" not in alfred.alfred_health({})


# --- through the server ----------------------------------------------------------------


def test_the_server_lists_the_caption_tools(client):
    names = {tool["name"] for tool in client.get("/v1/tools").json()["tools"]}
    assert {"alfred_captions", "alfred_captions_transcribe", "alfred_captions_set",
            "alfred_render_deliver"} <= names


def test_no_speech_engine_reaches_the_invocation_record_in_so_many_words(client, fake):
    fake.add_project("p1", "Talk")
    fake.speech = {"configured": False, "ok": False}
    answer = client.post("/v1/tools/alfred_captions_transcribe",
                         json={"input": {"project_id": "p1"}}).json()
    assert answer["error"] == (
        f"Tool execution failed: {NO_SPEECH_ENGINE}: alfred refused POST "
        f"/project/p1/captions/transcribe (503): {NO_ENGINE_DETAIL}")
    recorded = client.get(f"/v1/invocations/{answer['invocation_id']}").json()
    assert recorded["status"] == "failed" and NO_ENGINE_DETAIL in recorded["error"]
