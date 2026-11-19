# 12 — A long recording is offered as clips

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

A recording of a whole set, a talk or an interview is one alfred project, and
what is worth sharing from it is a few stretches that stand on their own. Page
11 has alfred's audio engine say what is said in a project, word by word and
when. This page takes that transcript and offers the stretches that stand
alone as clips: a person is asked about each one, each approved span is cut
from the project's saved timeline and rendered as a clip of its own with its
captions burned in, and each rendered clip is put to a person with its link.
The steps are `qmcp.cookbook.alfred_excerpts`, run by
`examples/flows/alfred_excerpts.py`; `docs/integrations/alfred.md` is the
reference.

## A qmcp server, and alfred stood in for

The server and the handler are as on pages 10 and 11. This handler holds one
project, transcribed already, whose timeline is a title and then the
recording, and keeps every request:

    >>> import json
    >>> import tempfile
    >>> from pathlib import Path
    >>> import httpx
    >>> import qmcp.tools.alfred as alfred
    >>> from qmcp.client import MCPClient
    >>> from qmcp.testing import throwaway_server
    >>> from qmcp.tools.alfred import AlfredClient, AlfredConfig

    >>> PROJECT, RENDER = "6a78b0420eea42d905943615", "6a7900000000000000000003"
    >>> TIMELINE = {"duration": 110.0, "clips": [
    ...     {"type": "template", "name": "title", "start": 0.0, "duration": 4.0,
    ...      "data": {"text": "Set at the Hall"}},
    ...     {"type": "video", "name": "videos/hall.mp4", "start": 4.0, "inpoint": 0.0,
    ...      "outpoint": 106.0, "fadeIn": 1, "fadeOut": 1}]}
    >>> HEARD = [
    ...     {"start": 5.0, "end": 8.2, "text": "Good evening, and thank you all for coming."},
    ...     {"start": 8.6, "end": 11.8, "text": "We have a new song for you tonight."},
    ...     {"start": 12.2, "end": 15.0, "text": "It is about a hotel in Leeds."},
    ...     {"start": 21.0, "end": 24.5, "text": "We were on the road for three weeks last spring."},
    ...     {"start": 24.9, "end": 27.6, "text": "The van broke down outside Leeds,"},
    ...     {"start": 28.0, "end": 31.2, "text": "so we wrote this in the hotel bar."},
    ...     {"start": 31.8, "end": 34.4, "text": "It is called Late Checkout."},
    ...     {"start": 100.0, "end": 103.2, "text": "Thank you, you have been wonderful."},
    ...     {"start": 103.6, "end": 105.0, "text": "Good night!"}]
    >>> STYLE = {"enabled": False, "fontsize": 40.0, "color": None, "bg": None,
    ...          "position": None, "align": None, "font": None}
    >>> held = {"captions": HEARD, "style": STYLE,
    ...         "job": {"state": "done", "error": None, "engine": "joe", "segments": 3}}
    >>> sent = []

    >>> def answers(request):
    ...     sent.append(request)
    ...     asked = request.method, request.url.path
    ...     if asked == ("POST", "/auth/jwt/login"):
    ...         return httpx.Response(200, json={"access_token": "issued"})
    ...     if request.headers.get("authorization") != "Bearer issued":
    ...         return httpx.Response(401, json={"detail": "Unauthorized"})
    ...     if asked == ("GET", f"/project/{PROJECT}"):
    ...         return httpx.Response(200, json={
    ...             "_id": PROJECT, "name": "Set at the Hall", "edl": TIMELINE,
    ...             "render_settings": {"width": 1280, "height": 720, "fps": None,
    ...                                 "quality": None}})
    ...     if asked == ("GET", f"/project/{PROJECT}/captions"):
    ...         return httpx.Response(200, json=held)
    ...     if asked == ("POST", "/render"):
    ...         return httpx.Response(200, json=RENDER)
    ...     if asked == ("GET", f"/render/{RENDER}/status"):
    ...         return httpx.Response(200, json={
    ...             "id": RENDER, "state": "rendered", "progress": 100.0, "error": None,
    ...             "download_url": f"/render/{RENDER}/download?exp=1893456000&sig=c2ln"})
    ...     return httpx.Response(404, json={"detail": "Not Found"})

    >>> configured = alfred.client
    >>> standin = AlfredClient(
    ...     AlfredConfig(url="http://alfred.test", email="editor@example.com",
    ...                  password="a password"),
    ...     transport=httpx.MockTransport(answers))
    >>> alfred.client = lambda: standin
    >>> server = throwaway_server(Path(tempfile.mkdtemp()) / "queue.db")
    >>> client = MCPClient(base_url=server.__enter__())

## The project, and the transcript alfred holds

`alfred_project` is the project as alfred holds it, its saved timeline
included. The flow needs the timeline to cut a clip from it:

    >>> from qmcp.cookbook import alfred_excerpts as flow
    >>> from qmcp.cookbook import alfred_render
    >>> invoke = alfred_render.invoker(client, "alfred-excerpts-page-12")
    >>> project = invoke("alfred_project", {"project_id": PROJECT})
    >>> project["name"], project["clips"], project["duration"]
    ('Set at the Hall', 2, 110.0)

A transcript alfred holds is read, not made again: a recording an hour long
is heard once. A project with none is transcribed as page 11 transcribes it:

    >>> transcribed = flow.read_transcript(invoke, PROJECT)
    >>> transcribed.outcome, len(transcribed.captions), transcribed.seconds
    ('done', 9, 0.0)

## What the transcript offers

The proposal comes from the transcript's own timing and nothing else. A pause
of four seconds or more ends a chapter; within one, a pause of a second or
more ends a span, as does the end of a sentence once a span is as long as a
clip may be. Here the song itself is a pause: the engine wrote no words for
it, so the applause after it opens the last chapter:

    >>> proposal = flow.propose(transcribed.captions)
    >>> for chapter in proposal.chapters:
    ...     print(chapter.number, f"{chapter.start}-{chapter.end}", chapter.spans, chapter.opening)
    1 5.0-15.0 1 Good evening, and thank you all for coming.
    2 21.0-34.4 1 We were on the road for three weeks last spring.
    3 100.0-105.0 0 Thank you, you have been wonderful.

A span runs ten seconds or more and ninety or fewer unless the run says
otherwise (`SHORTEST`, `LONGEST`), so the goodbye, five seconds, is left out
and counted:

    >>> for span in proposal.spans:
    ...     print(span.number, f"{span.start}-{span.end}", f"{span.seconds} s", span.opening)
    1 5.0-15.0 10.0 s Good evening, and thank you all for coming.
    2 21.0-34.4 13.4 s We were on the road for three weeks last spring.
    >>> proposal.left_out
    1

A labeller, when the run names one, titles each chapter and sorts each span
into a category the run names. It is a function from a prompt to an answer;
the flow's `--label-with local` builds one over the model `qmcp localmodel`
stands up, and this page stands one in. An answer that names no category, or
a model that stalls, leaves that label empty and the rest readable:

    >>> def model(prompt):
    ...     if prompt.startswith("Give a title"):
    ...         return "Late Checkout" if "hotel bar" in prompt else "The Welcome"
    ...     return "That is a story about the road."
    >>> proposal = flow.label(proposal, model, categories=["song", "story", "announcement"])
    >>> [chapter.title for chapter in proposal.chapters]
    ['The Welcome', 'Late Checkout', 'The Welcome']
    >>> [span.label for span in proposal.spans]
    ['story', 'story']

## Each span is put to a person

One request per span, in order, placing it in the recording and quoting its
words. Here the person approves the story and rejects the welcome:

    >>> review = flow.review_request(project, proposal.spans[1], proposal, "page-12")
    >>> print(review["prompt"])
    Clip 2 of 2 from 'Set at the Hall', chapter 2 of 3 (Late Checkout), 0:21.0-0:34.4 (13 s, story):
      0:21.0-0:24.5  We were on the road for three weeks last spring.
      0:24.9-0:27.6  The van broke down outside Leeds,
      0:28.0-0:31.2  so we wrote this in the hotel bar.
      0:31.8-0:34.4  It is called Late Checkout.
    Answer approve to render it as a clip of its own to review, or reject to leave it out.
    >>> alfred_render.ask(client, review, timeout_seconds=600,
    ...                   correlation_id="alfred-excerpts-page-12")
    >>> _ = client.submit_human_response(review["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> alfred_render.wait_for_answer(client, review["request_id"], timeout_seconds=600)
    'approve'

## The clip is cut from the timeline and rendered

An approved span is rendered as a clip of its own: `alfred_render_enqueue`
with a timeline cut from the project's saved one, and the captions that fall
in the span, moved to the clip's own seconds, burned in with the project's
style switched on. The title clip ends before the span begins, so only the
recording is in it, playing from the second the span starts, and neither edge
the cut made has a fade, where the clip's own fades were:

    >>> queued = flow.queue_clip(invoke, project, proposal.spans[1],
    ...                          captions=transcribed.captions, style=held["style"])
    >>> queued["render_id"], queued["name"], queued["captions"]
    ('6a7900000000000000000003', 'Set at the Hall - clip 2', 4)
    >>> body = json.loads([r for r in sent if r.url.path == "/render"][-1].content)
    >>> print(json.dumps(body["edl"], indent=1))
    {
     "clips": [
      {
       "type": "video",
       "name": "videos/hall.mp4",
       "start": 0.0,
       "inpoint": 17.0,
       "outpoint": 30.4,
       "fadeIn": 0,
       "fadeOut": 0,
       "duration": 13.4
      }
     ],
     "duration": 13.4
    }
    >>> body["captions"][0], body["caption_style"]["enabled"], body["description"]
    ({'start': 0.0, 'end': 3.5, 'text': 'We were on the road for three weeks last spring.'}, True, '0:21.0-0:34.4 of alfred project 6a78b0420eea42d905943615, chapter 2: story')
    >>> (body["width"], body["height"]) == (1280, 720)
    True

It is followed as a render is on page 10, and put to a person with its link.
This run was not asked to deliver, and the request says so:

    >>> waited = alfred_render.wait_for_render(invoke, queued["render_id"],
    ...                                        sleep=lambda seconds: None)
    >>> approval = flow.approval_request(queued, waited, proposal.spans[1], "page-12")
    >>> print(approval["prompt"])
    Approve the render of 'Set at the Hall - clip 2'? Watch it at http://alfred.test/render/6a7900000000000000000003/download?exp=1893456000&sig=c2ln -- the link works until 2030-01-01T00:00:00+00:00; alfred_render_status for render 6a7900000000000000000003 issues a fresh one. It is 0:21.0-0:34.4 of the recording, story. Nothing is delivered either way: this run was not asked to deliver.
    >>> alfred_render.ask(client, approval, timeout_seconds=600,
    ...                   correlation_id="alfred-excerpts-page-12")
    >>> _ = client.submit_human_response(approval["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> answer = alfred_render.wait_for_answer(client, approval["request_id"],
    ...                                        timeout_seconds=600)
    >>> print(flow.summary(project, transcribed, proposal, {1: "reject", 2: "approve"},
    ...                    {2: queued}, {2: waited}, {2: answer}))
    2 clips in 3 chapters of 'Set at the Hall' offered, 1 approved, 1 rendered, 1 approved as clips.

## In qmcp's record

The server recorded every call to alfred, the flow's under its correlation id.
Sorted, because the listing orders by creation time alone:

    >>> rows = client.list_invocations(limit=50)
    >>> for row in sorted((r["tool_name"], r["status"]) for r in rows
    ...                   if r["correlation_id"] == "alfred-excerpts-page-12"):
    ...     print(*row)
    alfred_captions success
    alfred_project success
    alfred_render_enqueue success
    alfred_render_status success

    >>> client.close()
    >>> _ = server.__exit__(None, None, None)
    >>> alfred.client = configured

## What this page does not claim

That alfred renders the clip, or that otto plays the cut timeline as a
person would expect: the handler answers as told, and a running alfred is a
person's check (`docs/integrations/alfred.md`). That the spans are the ones a
person would choose: they come from pauses and sentence ends, and a person
chooses each. Nor does it run the Metaflow flow. Its steps are the ones
above, and `tests/test_alfred_excerpts.py` runs the flow file's own steps,
over a stand-in for Metaflow, through each way a run can end.
