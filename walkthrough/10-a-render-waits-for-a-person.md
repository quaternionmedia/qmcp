# 10 — A render waits for a person

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

alfred is a self-hosted video editor that renders on its server. qmcp drives
it with tools, each a request or two to alfred's HTTP API;
`docs/integrations/alfred.md` is their reference. This page invokes
`alfred_projects`, `alfred_render_enqueue` and `alfred_render_status` through a
qmcp server, as an agent does. Then it runs the approval flow's steps: queue a
project's render, follow it, and put the result to a person, who decides.

## A qmcp server, and alfred stood in for

The server is real, on an ephemeral port over a database made for this page.
alfred is a handler that answers each route the tools use as alfred's API does.
It keeps every request, so the page can show what alfred was sent. It logs one
account in and accepts only the token it issued:

    >>> import json
    >>> import tempfile
    >>> from pathlib import Path
    >>> import httpx
    >>> import qmcp.tools.alfred as alfred
    >>> from qmcp.client import MCPClient
    >>> from qmcp.testing import throwaway_server
    >>> from qmcp.tools.alfred import AlfredClient, AlfredConfig

    >>> TIMELINE = {"clips": [
    ...     {"type": "template", "name": "title", "duration": 3.0, "start": 0.0,
    ...      "fadeIn": 0, "fadeOut": None, "data": {"text": "Alfred"}},
    ...     {"type": "video", "name": "videos/demo-bars.mp4", "duration": 2.0,
    ...      "start": 3.0, "fadeIn": 0, "fadeOut": 0}], "duration": 5.0}
    >>> PROJECTS = [
    ...     {"_id": "6a78b0420eea42d905943615", "name": "Timeline demo", "edl": TIMELINE,
    ...      "render_settings": {"width": 1280, "height": 720, "fps": 30.0, "quality": None}},
    ...     {"_id": "6a770760de7add77551a53df", "name": "Magnussens", "edl": None}]
    >>> RENDER = "6a7900000000000000000001"
    >>> states = iter(["queued", "rendering", "rendered"])
    >>> sent = []

It refuses as alfred's code does, from the file qmcp's tests hold their
stand-in alfred to: a status, and a body with alfred's sentence and its code:

    >>> refusals = json.loads(Path("tests/alfred_refusals.json").read_text(encoding="utf-8"))
    >>> refusals["frame_size_out_of_range"]
    {'status': 422, 'body': {'detail': 'width must be between 16 and 7680 pixels', 'code': 'frame_size_out_of_range'}}

    >>> def answers(request):
    ...     sent.append(request)
    ...     asked = request.method, request.url.path
    ...     if asked == ("POST", "/auth/jwt/login"):
    ...         return httpx.Response(200, json={"access_token": "issued"})
    ...     if request.headers.get("authorization") != "Bearer issued":
    ...         return httpx.Response(401, json={"detail": "Unauthorized"})
    ...     if asked == ("GET", "/project"):
    ...         return httpx.Response(200, json=PROJECTS)
    ...     for project in PROJECTS:
    ...         if asked == ("GET", f"/project/{project['_id']}"):
    ...             return httpx.Response(200, json=project)
    ...     if asked == ("POST", "/render"):
    ...         if not 16 <= json.loads(request.content).get("width", 1920) <= 7680:
    ...             too_wide = refusals["frame_size_out_of_range"]
    ...             return httpx.Response(too_wide["status"], json=too_wide["body"])
    ...         return httpx.Response(200, json=RENDER)
    ...     if asked == ("GET", f"/render/{RENDER}/status"):
    ...         state = next(states)
    ...         signed = f"/render/{RENDER}/download?exp=1893456000&sig=c2ln"
    ...         return httpx.Response(200, json={
    ...             "id": RENDER, "state": state, "error": None,
    ...             "progress": {"queued": None, "rendering": 50.0, "rendered": 100.0}[state],
    ...             "download_url": signed if state == "rendered" else None})
    ...     return httpx.Response(404, json={"detail": "Not Found"})

The tools reach alfred through `qmcp.tools.alfred.client`, which builds a client
from the `QMCP_ALFRED_*` settings. Here it hands out one built over the handler,
and the page puts the original back at its end:

    >>> configured = alfred.client
    >>> standin = AlfredClient(
    ...     AlfredConfig(url="http://alfred.test", email="editor@example.com",
    ...                  password="a password"),
    ...     transport=httpx.MockTransport(answers))
    >>> alfred.client = lambda: standin
    >>> server = throwaway_server(Path(tempfile.mkdtemp()) / "queue.db")
    >>> client = MCPClient(base_url=server.__enter__())

## What alfred holds

`alfred_projects` lists the projects the account can see. A project built from
a form rather than on the timeline has no clips:

    >>> for project in client.invoke_tool("alfred_projects", {}).result:
    ...     print(project)
    {'id': '6a78b0420eea42d905943615', 'name': 'Timeline demo', 'clips': 2, 'duration': 5.0}
    {'id': '6a770760de7add77551a53df', 'name': 'Magnussens', 'clips': 0, 'duration': None}

## A refusal is in alfred's own words

A render alfred refuses comes back as the invocation's error, in alfred's
`detail`, and the server records the invocation as failed. The code stays on
the `AlfredError` the tool raises, for a caller to switch on, as page 11 shows:

    >>> refused = client.invoke_tool(
    ...     "alfred_render_enqueue", {"project_id": "6a78b0420eea42d905943615", "width": 9000})
    >>> print(refused.result, refused.error, sep="\n")
    None
    Tool execution failed: alfred refused POST /render (422): width must be between 16 and 7680 pixels
    >>> client.get_invocation(refused.invocation_id)["status"]
    'failed'

A project with no timeline is refused before a render is queued:

    >>> print(client.invoke_tool(
    ...     "alfred_render_enqueue", {"project_id": "6a770760de7add77551a53df"}).error)
    Tool execution failed: alfred project 6a770760de7add77551a53df ('Magnussens') has no clips on its timeline to render

## Queued, followed, and put to a person

The approval flow, `examples/flows/alfred_render.py`, runs these steps from
`qmcp.cookbook.alfred_render`, one per Metaflow step. Each call to alfred is a
tool invocation through the server, under one correlation id:

    >>> from qmcp.cookbook import alfred_render as flow
    >>> invoke = flow.invoker(client, "alfred-render-page-10")
    >>> queued = flow.queue_render(invoke, "6a78b0420eea42d905943615")
    >>> for key, value in queued.items():
    ...     print(f"{key}: {value}")
    render_id: 6a7900000000000000000001
    project_id: 6a78b0420eea42d905943615
    name: Timeline demo
    width: 1280
    height: 720
    fps: 30.0
    quality: None
    captions: 0

The size and rate are the project's saved ones; `quality` None is alfred's
default. The project holds no captions, so none went. The timeline went to
alfred exactly as alfred stored it, `"fadeOut": None` included: no fade, where
a missing `fadeOut` would be the engine's default:

    >>> body = json.loads([r for r in sent if r.url.path == "/render"][-1].content)
    >>> body["edl"] == TIMELINE, body["width"], body["height"]
    (True, 1280, 720)

Following it reads its status until it settles; the page does not sleep between
reads. alfred hands out a path signed for a few minutes. The tool makes it a
link a person can open, and reads from the link when it stops working:

    >>> waited = flow.wait_for_render(invoke, queued["render_id"], sleep=lambda seconds: None)
    >>> waited.outcome
    'rendered'
    >>> print(waited.status["download_url"], waited.status["download_expires_at"], sep="\n")
    http://alfred.test/render/6a7900000000000000000001/download?exp=1893456000&sig=c2ln
    2030-01-01T00:00:00+00:00

The approval carries the link, and says how to get a fresh one:

    >>> request = flow.approval_request(queued, waited, "page-10")
    >>> request["request_id"], request["options"]
    ('alfred-render-6a7900000000000000000001-page-10', ['approve', 'reject'])
    >>> print(request["prompt"])
    Approve the render of 'Timeline demo'? Watch it at http://alfred.test/render/6a7900000000000000000001/download?exp=1893456000&sig=c2ln -- the link works until 2030-01-01T00:00:00+00:00; alfred_render_status for render 6a7900000000000000000001 issues a fresh one.
    >>> flow.ask(client, request, timeout_seconds=600, correlation_id="alfred-render-page-10")

A person answers on the queue, here before the wait begins, so the wait sees
the answer on its first look. The wait watches the pending listing and reads
the request once it has left, because reading a request is what expires it:

    >>> _ = client.submit_human_response(request["request_id"], "approve",
    ...                                  responded_by="editor@example.com")
    >>> answer = flow.wait_for_answer(client, request["request_id"], timeout_seconds=600)
    >>> print(flow.summary(queued, waited, answer))
    Approved: render 6a7900000000000000000001 of 'Timeline demo', http://alfred.test/render/6a7900000000000000000001/download?exp=1893456000&sig=c2ln.

## In qmcp's record

The server recorded every call to alfred, the flow's under its correlation id.
Sorted, because the listing orders by creation time alone:

    >>> rows = client.list_invocations(limit=20)
    >>> for row in sorted((r["tool_name"], r["status"]) for r in rows
    ...                   if r["correlation_id"] == "alfred-render-page-10"):
    ...     print(*row)
    alfred_render_enqueue success
    alfred_render_status success
    alfred_render_status success
    alfred_render_status success
    >>> for row in sorted((r["tool_name"], r["status"]) for r in rows
    ...                   if r["correlation_id"] is None):
    ...     print(*row)
    alfred_projects success
    alfred_render_enqueue failed
    alfred_render_enqueue failed

alfred was logged in to once, and its token reused for every request after:

    >>> [r.url.path for r in sent].count("/auth/jwt/login"), len(sent)
    (1, 10)

    >>> client.close()
    >>> _ = server.__exit__(None, None, None)
    >>> alfred.client = configured

## What this page does not claim

That alfred renders anything, or that its link opens a video: the handler
answers as told, and a running alfred is a person's check
(`docs/integrations/alfred.md`). Nor does it run the Metaflow flow, which
cannot be imported where `fcntl` is missing. Its steps are the ones above, and
the examples smoke test imports the flow where Metaflow runs.
