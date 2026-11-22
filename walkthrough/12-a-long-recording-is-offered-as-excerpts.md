# 12 — A long recording is offered as excerpts

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

A recording of a whole set, a talk or an interview is one alfred project, and
what is worth sharing from it is a few stretches that stand on their own. Page
11 has alfred's audio engine say what is said in a project, word by word and
when. This page has alfred cut the project into drafts by its `excerpts`
recipe, each a project of its own, and offers each draft to a person: a
rejected draft is dropped, an approved one is rendered as an excerpt of its
own with its captions burned in, and each rendered excerpt is put to a person
with its link. The steps are `qmcp.cookbook.alfred_excerpts`, run by
`examples/flows/alfred_excerpts.py`; `docs/integrations/alfred.md` is the
reference.

## A qmcp server, and alfred stood in for

The server and the handler are as on pages 10 and 11. This handler holds one
project, transcribed already, whose timeline is a title and then the
recording; a worker that leaves two drafts of it, the second named by the
topic alfred's language engine had live over it; and it keeps every request:

    >>> import json
    >>> import tempfile
    >>> from pathlib import Path
    >>> import httpx
    >>> import qmcp.tools.alfred as alfred
    >>> from qmcp.client import MCPClient
    >>> from qmcp.testing import throwaway_server
    >>> from qmcp.tools.alfred import AlfredClient, AlfredConfig

    >>> PROJECT, RENDER = "6a78b0420eea42d905943615", "6a7900000000000000000003"
    >>> WELCOME, STORY = "6a7a00000000000000000001", "6a7a00000000000000000002"
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
    >>> SETTINGS = {"width": 1280, "height": 720, "fps": None, "quality": None}
    >>> held = {"captions": HEARD, "style": STYLE,
    ...         "job": {"state": "done", "error": None, "engine": "joe", "segments": 3}}

The drafts, as alfred's worker leaves them: each a project carrying `source`,
its captions moved into its own seconds, and its timeline cut from the
project's, which alfred's `GET /project/{id}` answers whole:

    >>> def draft(draft_id, name, start, end, captions, scores):
    ...     cut = [{**c, "start": round(c["start"] - start, 3), "end": round(c["end"] - start, 3)}
    ...            for c in captions]
    ...     return {"_id": draft_id, "name": name,
    ...             "description": f"{int(start) // 60}:{int(start) % 60:02d}-"
    ...                            f"{int(end) // 60}:{int(end) % 60:02d} of the source",
    ...             "edl": {"clips": [{"type": "video", "name": "videos/hall.mp4", "start": 0.0,
    ...                                "inpoint": round(start - 4.0, 3),
    ...                                "outpoint": round(end - 4.0, 3), "fadeIn": 0,
    ...                                "fadeOut": 0, "duration": round(end - start, 3)}],
    ...                     "duration": round(end - start, 3)},
    ...             "captions": cut, "caption_style": STYLE, "render_settings": SETTINGS,
    ...             "source": {"project": PROJECT, "recipe": "excerpts",
    ...                        "values": {"pause": "1", "shortest": "10", "longest": "90"},
    ...                        "spans": [[start, end]], "scores": scores,
    ...                        "made_at": "2026-11-21T15:02:11Z"}}
    >>> DRAFTS = {
    ...     WELCOME: draft(WELCOME, "Set at the Hall – Excerpts 1", 5.0, 15.0, HEARD[:3],
    ...                    {"dialogue": 0.62}),
    ...     STORY: draft(STORY, "Set at the Hall – Late Checkout 2", 21.0, 34.4, HEARD[3:7],
    ...                  {"dialogue": 0.71}),
    ... }
    >>> def listed(d):
    ...     return {"id": d["_id"], "name": d["name"], "description": d["description"],
    ...             "duration": d["edl"]["duration"], "source": d["source"]}
    >>> run = {"started": False, "looks": 0}
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
    ...             "render_settings": SETTINGS})
    ...     if asked == ("GET", f"/project/{PROJECT}/captions"):
    ...         return httpx.Response(200, json=held)
    ...     if asked == ("POST", f"/project/{PROJECT}/drafts"):
    ...         run["started"] = True
    ...         return httpx.Response(202, json={"job": {
    ...             "state": "queued", "error": None, "recipes": ["excerpts"], "made": None,
    ...             "notes": []}})
    ...     if asked == ("GET", f"/project/{PROJECT}/drafts"):
    ...         if not run["started"]:
    ...             return httpx.Response(200, json={"job": None, "drafts": []})
    ...         run["looks"] += 1
    ...         done = run["looks"] >= 2
    ...         job = {"state": "done" if done else "running", "error": None,
    ...                "recipes": ["excerpts"], "made": 2 if done else None, "notes": []}
    ...         return httpx.Response(200, json={
    ...             "job": job, "drafts": [listed(d) for d in DRAFTS.values()] if done else []})
    ...     if asked == ("DELETE", f"/project/{PROJECT}/drafts"):
    ...         dropped = json.loads(request.content)["drafts"]
    ...         for draft_id in dropped:
    ...             del DRAFTS[draft_id]
    ...         return httpx.Response(200, json={"dropped": dropped})
    ...     for draft_id, d in DRAFTS.items():
    ...         if asked == ("GET", f"/project/{draft_id}/captions"):
    ...             return httpx.Response(200, json={"captions": d["captions"],
    ...                                              "style": STYLE, "job": None})
    ...         if asked == ("GET", f"/project/{draft_id}"):
    ...             return httpx.Response(200, json=d)
    ...     if asked == ("PUT", f"/project/{PROJECT}/cues"):
    ...         body = json.loads(request.content)
    ...         return httpx.Response(200, json={"cues": [
    ...             {**cue, "source": body.get("source")} for cue in body["cues"]]})
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
included:

    >>> from qmcp.cookbook import alfred_excerpts as flow
    >>> from qmcp.cookbook import alfred_render
    >>> invoke = alfred_render.invoker(client, "alfred-excerpts-page-12")
    >>> project = invoke("alfred_project", {"project_id": PROJECT})
    >>> project["name"], project["clips"], project["duration"]
    ('Set at the Hall', 2, 110.0)

A transcript alfred holds is read, not made again: a recording an hour long
is heard once. A project with none is transcribed as page 11 transcribes it,
since alfred cuts a project by the words a transcription leaves:

    >>> transcribed = flow.read_transcript(invoke, PROJECT)
    >>> transcribed.outcome, len(transcribed.captions), transcribed.seconds
    ('done', 9, 0.0)

## alfred proposes the excerpts

The proposal is alfred's: `alfred_drafts` runs its `excerpts` recipe over the
project, with the run's knobs sent as text, since a knob is typed, and
`alfred_drafts_status` is asked until alfred's worker is done. Each draft the
run added is a project of its own, read back with `alfred_captions`; here the
song itself was a pause the engine wrote no words for, and the goodbye too
short to stand alone, so two drafts come back, the second named by the topic
alfred's language engine had live over it:

    >>> proposal = flow.propose(invoke, PROJECT, sleep=lambda seconds: None)
    >>> asked = [r for r in sent if r.method == "POST" and r.url.path.endswith("/drafts")]
    >>> json.loads(asked[-1].content)
    {'recipes': [{'recipe': 'excerpts', 'values': {'pause': '1', 'shortest': '10', 'longest': '90'}}]}
    >>> for draft in proposal.drafts:
    ...     print(draft.number, f"{draft.start}-{draft.end}", f"{draft.seconds} s",
    ...           repr(draft.title), "|", draft.opening)
    1 5.0-15.0 10.0 s 'Excerpts 1' | Good evening, and thank you all for coming.
    2 21.0-34.4 13.4 s 'Late Checkout 2' | We were on the road for three weeks last spring.
    >>> proposal.drafts[1].captions[0]
    {'start': 0.0, 'end': 3.5, 'text': 'We were on the road for three weeks last spring.'}

A labeller, when the run names one, sorts each draft into a category the run
names. It is a function from a prompt to an answer; the flow's
`--label-with local` builds one over the model `qmcp localmodel` stands up,
and this page stands one in. An answer that names no category, or a model
that stalls, leaves that label empty and the rest readable:

    >>> def model(prompt):
    ...     return "That is a story about the road." if "road" in prompt else "An announcement."
    >>> proposal = flow.label(proposal, model, categories=["song", "story", "announcement"])
    >>> [draft.label for draft in proposal.drafts]
    ['announcement', 'story']

## Each draft is put to a person

One request per draft, in order, placing it in the recording, naming it as
alfred did, and quoting its words in the recording's seconds. Here the person
approves the story:

    >>> review = flow.review_request(project, proposal.drafts[1], proposal, "page-12")
    >>> print(review["prompt"])
    Excerpt 2 of 2 from 'Set at the Hall', 0:21.0-0:34.4 (13 s, story), alfred's 'Late Checkout 2':
      0:21.0-0:24.5  We were on the road for three weeks last spring.
      0:24.9-0:27.6  The van broke down outside Leeds,
      0:28.0-0:31.2  so we wrote this in the hotel bar.
      0:31.8-0:34.4  It is called Late Checkout.
    Answer approve to render it as an excerpt of its own to review, or reject to drop it.

A voice reader speaks the request's shorter form in place of the prompt,
since every line of a draft is too long to listen to:

    >>> print(review["context"]["spoken"])
    Excerpt 2 of 2 from Set at the Hall, 13 seconds, story, Late Checkout 2. It opens: We were on the road for three weeks last spring. Approve to render it as an excerpt of its own, or reject to drop it.
    >>> alfred_render.ask(client, review, timeout_seconds=600,
    ...                   correlation_id="alfred-excerpts-page-12")
    >>> _ = client.submit_human_response(review["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> alfred_render.wait_for_answer(client, review["request_id"], timeout_seconds=600)
    'approve'

The welcome, rejected, is dropped: `alfred_drafts_drop` deletes the draft, a
project of its own, with its cue on the source:

    >>> dropped = flow.drop(invoke, PROJECT, [proposal.drafts[0]])
    >>> dropped["dropped"] == [WELCOME], sorted(DRAFTS) == [STORY]
    (True, True)

## The timeline is marked as it goes

Where each draft stands is drawn on the project's timeline as cues, beside
the cues alfred's own run left, which alfred's editor shows in a lane under
the clips. Here the first draft was rejected and the second is waiting on the
queue, by its request:

    >>> states = {1: flow.REJECTED, 2: flow.WAITING}
    >>> cues = flow.cues_of(proposal, states, {2: review["request_id"]})
    >>> for cue in cues:
    ...     print(cue["kind"], f"{cue['start']}-{cue['end']}", cue["label"], "|",
    ...           cue.get("state") or "-", cue.get("ref") or "-")
    excerpt 5.0-15.0 Excerpt 1: announcement | rejected -
    excerpt 21.0-34.4 Excerpt 2: story | waiting alfred-excerpt-6a78b0420eea42d905943615-2-page-12

They are written as this flow's, so a run replaces the last run's cues and
leaves whatever else marked the timeline:

    >>> marked = flow.mark(invoke, PROJECT, proposal, states, {2: review["request_id"]})
    >>> len(marked["cues"]), marked["cues"][1]["source"]
    (2, 'alfred-excerpts')

## The excerpt is the draft's own timeline, rendered

An approved draft is rendered as an excerpt of its own: `alfred_render_enqueue`
of the draft by its id, which renders the draft's saved timeline, cut by
alfred from the project's, with the draft's own captions burned in and its
style switched on; a caption corrected on the draft in alfred's editor during
the review is in the excerpt. The title clip ends before the stretch begins,
so only the recording is in it, playing from the second the stretch starts:

    >>> queued = flow.queue_excerpt(invoke, project, proposal.drafts[1])
    >>> queued["render_id"], queued["name"], queued["captions"]
    ('6a7900000000000000000003', 'Set at the Hall – Late Checkout 2', 4)
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
    ({'start': 0.0, 'end': 3.5, 'text': 'We were on the road for three weeks last spring.'}, True, "0:21.0-0:34.4 of 'Set at the Hall': story")
    >>> (body["width"], body["height"]) == (1280, 720)
    True

It is followed as a render is on page 10, and put to a person with its link.
This run was not asked to deliver, and the request says so:

    >>> waited = alfred_render.wait_for_render(invoke, queued["render_id"],
    ...                                        sleep=lambda seconds: None)
    >>> approval = flow.approval_request(queued, waited, proposal.drafts[1], "page-12")
    >>> print(approval["prompt"])
    Approve the render of 'Set at the Hall – Late Checkout 2'? Watch it at http://alfred.test/render/6a7900000000000000000003/download?exp=1893456000&sig=c2ln -- the link works until 2030-01-01T00:00:00+00:00; alfred_render_status for render 6a7900000000000000000003 issues a fresh one. It is 0:21.0-0:34.4 of the recording, story. Nothing is delivered either way: this run was not asked to deliver.
    >>> alfred_render.ask(client, approval, timeout_seconds=600,
    ...                   correlation_id="alfred-excerpts-page-12")
    >>> _ = client.submit_human_response(approval["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> answer = alfred_render.wait_for_answer(client, approval["request_id"],
    ...                                        timeout_seconds=600)
    >>> print(flow.summary(project, transcribed, proposal, {1: "reject", 2: "approve"},
    ...                    {2: queued}, {2: waited}, {2: answer}))
    2 excerpts of 'Set at the Hall' offered, 1 approved, 1 dropped, 1 rendered, 1 approved as excerpts.

## In qmcp's record

The server recorded every call to alfred, the flow's under its correlation id.
Sorted, because the listing orders by creation time alone:

    >>> rows = client.list_invocations(limit=50)
    >>> for row in sorted((r["tool_name"], r["status"]) for r in rows
    ...                   if r["correlation_id"] == "alfred-excerpts-page-12"):
    ...     print(*row)
    alfred_captions success
    alfred_captions success
    alfred_captions success
    alfred_cues_set success
    alfred_drafts success
    alfred_drafts_drop success
    alfred_drafts_status success
    alfred_drafts_status success
    alfred_drafts_status success
    alfred_project success
    alfred_render_enqueue success
    alfred_render_status success

    >>> client.close()
    >>> _ = server.__exit__(None, None, None)
    >>> alfred.client = configured

## What this page does not claim

That alfred's recipe cuts where a person would, or that it renders the
excerpt: the handler answers as told, and a running alfred is a person's
check (`docs/integrations/alfred.md`; alfred's own `docs/drafts.md` says how
the recipe cuts). A person chooses each draft, and alfred's editor is where a
draft is refined. Nor does it run the Metaflow flow. Its steps are the ones
above, and `tests/test_alfred_excerpts.py` runs the flow file's own steps,
over a stand-in for Metaflow and a stand-in alfred whose worker cuts drafts,
through each way a run can end.
