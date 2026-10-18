# 11 — Captions are read before they are burned in

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

alfred, the video editor of page 10, can caption a project: its speech engine
transcribes the project's saved timeline, alfred keeps the captions on the
project, and a render can burn them into the picture. qmcp reaches that through
three more tools -- `alfred_captions`, `alfred_captions_transcribe` and
`alfred_captions_set` -- and delivers a finished render with a fourth,
`alfred_render_deliver`; `docs/integrations/alfred.md` is their reference.
Words a speech engine got wrong stay in every copy of a captioned video, so the
caption flow puts the transcript to a person before anything is rendered with
it, and the render to a person before it goes anywhere. This page runs the
flow's steps.

## A qmcp server, and alfred stood in for

As on page 10, the server is a real one on an ephemeral port over a database
made for this page, and alfred is a handler answering each route the tools use
as alfred's API does. This one holds one project's captions, starts with no
audio engine, finishes a transcription the first time it is looked at, and
keeps every request:

    >>> import json
    >>> import tempfile
    >>> from pathlib import Path
    >>> import httpx
    >>> import qmcp.tools.alfred as alfred
    >>> from qmcp.client import MCPClient
    >>> from qmcp.integrations.voice.check import throwaway_server
    >>> from qmcp.tools.alfred import AlfredClient, AlfredConfig

    >>> PROJECT, RENDER = "6a78b0420eea42d905943615", "6a7900000000000000000002"
    >>> HEARD = [{"start": 0.0, "end": 2.4, "text": "Welcome to the interview."},
    ...          {"start": 2.4, "end": 5.0, "text": "This is all fred's caption demo."}]
    >>> held = {"captions": [], "style": {"enabled": False, "fontsize": 48}, "job": None}
    >>> engine = {"configured": False, "ok": False,
    ...           "detail": "No audio engine is configured: set ALFRED_AUDIO_ENGINE_URL to its "
    ...                     "address (docs/captions.md says how to run one)."}
    >>> sent = []

    >>> def answers(request):
    ...     sent.append(request)
    ...     asked = request.method, request.url.path
    ...     if asked == ("POST", "/auth/jwt/login"):
    ...         return httpx.Response(200, json={"access_token": "issued"})
    ...     if request.headers.get("authorization") != "Bearer issued":
    ...         return httpx.Response(401, json={"detail": "Unauthorized"})
    ...     if asked == ("GET", "/health"):
    ...         return httpx.Response(200, json={
    ...             "api": "ok", "db": "ok", "renderer": {"ok": True, "workers": 1},
    ...             "queue": {"pending": 0}, "audio_engine": engine})
    ...     if asked == ("GET", f"/project/{PROJECT}"):
    ...         return httpx.Response(200, json={
    ...             "_id": PROJECT, "name": "Interview", "captions": held["captions"],
    ...             "caption_style": held["style"], "edl": {"duration": 5.0, "clips": [
    ...                 {"type": "video", "name": "videos/interview.mp4", "duration": 5.0}]}})
    ...     if asked == ("POST", f"/project/{PROJECT}/captions/transcribe"):
    ...         if not engine["configured"]:
    ...             return httpx.Response(503, json={"detail": engine["detail"]})
    ...         held["job"] = {"state": "queued", "error": None}
    ...         return httpx.Response(202, json={"job": held["job"]})
    ...     if asked == ("GET", f"/project/{PROJECT}/captions"):
    ...         if held["job"] and held["job"]["state"] == "queued":
    ...             held["job"] = {"state": "done", "error": None, "engine": "joe", "segments": 2}
    ...             held["captions"] = HEARD
    ...         return httpx.Response(200, json=held)
    ...     if asked == ("PUT", f"/project/{PROJECT}/captions"):
    ...         held.update(json.loads(request.content))
    ...         return httpx.Response(200, json=held)
    ...     if asked == ("POST", "/render"):
    ...         return httpx.Response(200, json=RENDER)
    ...     if asked == ("GET", f"/render/{RENDER}/status"):
    ...         return httpx.Response(200, json={
    ...             "id": RENDER, "state": "rendered", "progress": 100.0, "error": None,
    ...             "download_url": f"/render/{RENDER}/download?exp=1893456000&sig=c2ln"})
    ...     if asked == ("POST", f"/render/{RENDER}/deliver"):
    ...         return httpx.Response(
    ...             200, json={"destination": "local", "location": f"/render/{RENDER}/download"})
    ...     return httpx.Response(404, json={"detail": "Not Found"})

    >>> configured = alfred.client
    >>> standin = AlfredClient(
    ...     AlfredConfig(url="http://alfred.test", email="editor@example.com",
    ...                  password="a password"),
    ...     transport=httpx.MockTransport(answers))
    >>> alfred.client = lambda: standin
    >>> server = throwaway_server(Path(tempfile.mkdtemp()) / "queue.db")
    >>> client = MCPClient(base_url=server.__enter__())

## An alfred without an audio engine says so

`alfred_health` passes on alfred's report, which says whether the audio engine
alfred transcribes with is configured and answering, and why not:

    >>> client.invoke_tool("alfred_health", {}).result["audio_engine"]
    {'configured': False, 'ok': False, 'detail': 'No audio engine is configured: set ALFRED_AUDIO_ENGINE_URL to its address (docs/captions.md says how to run one).'}

Asked to transcribe anyway, alfred refuses, and the tool's error opens with
what that means before alfred's own words:

    >>> print(client.invoke_tool("alfred_captions_transcribe", {"project_id": PROJECT}).error)
    Tool execution failed: alfred has no speech engine: alfred refused POST /project/6a78b0420eea42d905943615/captions/transcribe (503): No audio engine is configured: set ALFRED_AUDIO_ENGINE_URL to its address (docs/captions.md says how to run one).

The caption flow, `examples/flows/alfred_caption_render.py`, runs the steps in
`qmcp.cookbook.alfred_captions`, one Metaflow step each, with every call to
alfred a tool invocation through the server under one correlation id. Its
first step ends there, asking nobody anything:

    >>> from qmcp.cookbook import alfred_captions as flow
    >>> from qmcp.cookbook import alfred_render
    >>> invoke = alfred_render.invoker(client, "alfred-caption-render-page-11")
    >>> ended = flow.transcribe(invoke, PROJECT, sleep=lambda seconds: None)
    >>> ended.outcome
    'no speech engine'
    >>> print(flow.summary(PROJECT, ended))
    Nothing was asked: the transcription of alfred project 6a78b0420eea42d905943615 could not start: alfred has no speech engine: alfred refused POST /project/6a78b0420eea42d905943615/captions/transcribe (503): No audio engine is configured: set ALFRED_AUDIO_ENGINE_URL to its address (docs/captions.md says how to run one).

## Transcribed, and put to a person

With an engine configured, the step starts a transcription and reads the
project's captions until the job is done; the page does not sleep between
reads:

    >>> engine.update(configured=True, ok=True, detail=None)
    >>> transcribed = flow.transcribe(invoke, PROJECT, sleep=lambda seconds: None)
    >>> transcribed.outcome, transcribed.job["engine"]
    ('done', 'joe')

The transcript goes to a person as an approval, with what each answer does:

    >>> review = flow.review_request(PROJECT, transcribed, "page-11")
    >>> print(review["prompt"])
    Burn these 2 captions into a render of alfred project 6a78b0420eea42d905943615?
      0:00.0-0:02.4  Welcome to the interview.
      0:02.4-0:05.0  This is all fred's caption demo.
    Answer approve to render it with the captions alfred holds when the render is queued -- a correction saved in alfred before then is included -- or reject to end the run without rendering.
    >>> alfred_render.ask(client, review, timeout_seconds=600,
    ...                   correlation_id="alfred-caption-render-page-11")

The engine heard "alfred" as "all fred". The person corrects the line with
`alfred_captions_set`, and approves. Only the captions changed, so the tool
sends back the style alfred holds with them, and a route that stores both
cannot reset it:

    >>> fixed = [HEARD[0], {**HEARD[1], "text": "This is alfred's caption demo."}]
    >>> stored = client.invoke_tool("alfred_captions_set",
    ...                             {"project_id": PROJECT, "captions": fixed}).result
    >>> stored["captions"][1]["text"]
    "This is alfred's caption demo."
    >>> json.loads([r for r in sent if r.method == "PUT"][-1].content)["style"]
    {'enabled': False, 'fontsize': 48}

As on page 10, the answer is given before the wait begins, and the wait reads
the request once it has left the pending listing:

    >>> _ = client.submit_human_response(review["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> alfred_render.wait_for_answer(client, review["request_id"], timeout_seconds=600)
    'approve'

## Rendered with the captions burned in

The render is queued with the captions alfred holds now, the correction among
them. The project's caption style is switched off for its everyday renders, so
the flow asks for the captions explicitly, and the style goes with them
switched on. They go to alfred as alfred stores them: alfred turns them into
clips on the timeline, not qmcp:

    >>> queued = alfred_render.queue_render(invoke, PROJECT, captions=True)
    >>> queued["render_id"], queued["captions"]
    ('6a7900000000000000000002', 2)
    >>> body = json.loads([r for r in sent if r.url.path == "/render"][-1].content)
    >>> body["captions"] == fixed, body["caption_style"]
    (True, {'enabled': True, 'fontsize': 48})

It is followed as page 10's render was, and put to a person with its link.
This run was asked to deliver, so the request says that approving delivers it:

    >>> waited = alfred_render.wait_for_render(invoke, RENDER, sleep=lambda seconds: None)
    >>> approval = flow.approval_request(queued, waited, "page-11", deliver=True)
    >>> print(approval["prompt"])
    Approve the render of 'Interview'? Watch it at http://alfred.test/render/6a7900000000000000000002/download?exp=1893456000&sig=c2ln -- the link works until 2030-01-01T00:00:00+00:00; alfred_render_status for render 6a7900000000000000000002 issues a fresh one. Answer approve to deliver it to alfred's default destination, or reject to leave it undelivered.
    >>> alfred_render.ask(client, approval, timeout_seconds=600,
    ...                   correlation_id="alfred-caption-render-page-11")
    >>> _ = client.submit_human_response(approval["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> answer = alfred_render.wait_for_answer(client, approval["request_id"],
    ...                                        timeout_seconds=600)

## Delivered, because the run was asked to

Approval delivers it only when the flow runs with `--deliver True`; without it,
approval ends the run and nothing is delivered. alfred chooses the destination
when the run names none, and the tool makes the location it answers absolute:

    >>> delivered = flow.deliver_render(invoke, RENDER)
    >>> print(flow.summary(PROJECT, transcribed, "approve", queued, waited, answer, delivered))
    Approved and delivered: render 6a7900000000000000000002 of 'Interview' with 2 captions, to local at http://alfred.test/render/6a7900000000000000000002/download.

## In qmcp's record

Every call to alfred is an invocation the server recorded, the flow's under its
correlation id. Sorted, because the listing orders by creation time alone:

    >>> rows = client.list_invocations(limit=50)
    >>> for row in sorted((r["tool_name"], r["status"]) for r in rows
    ...                   if r["correlation_id"] == "alfred-caption-render-page-11"):
    ...     print(*row)
    alfred_captions success
    alfred_captions_transcribe failed
    alfred_captions_transcribe success
    alfred_render_deliver success
    alfred_render_enqueue success
    alfred_render_status success

    >>> client.close()
    >>> _ = server.__exit__(None, None, None)
    >>> alfred.client = configured

## What this page does not claim

That alfred transcribes anything, or that a speech engine hears what was said:
the handler answers as told, and the tools against a running alfred with an
engine are a person's check (`docs/integrations/alfred.md`). Nor does the page
run the Metaflow flow; its steps are the ones above, and
`tests/test_alfred_caption_flow.py` runs the flow file's own steps, over a
stand-in for Metaflow, through each way a run can end.
