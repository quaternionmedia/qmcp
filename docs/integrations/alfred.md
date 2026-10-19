# alfred Integration

qmcp can drive alfred, a self-hosted video editor whose renders are made on its
server. Its tools let an agent see alfred's projects and templates, queue a
render, follow it to a link a person can open, and deliver it; and caption a
project -- transcribe its timeline with alfred's audio engine, read the
captions, and store a person's correction -- so that a render burns them in.
Two flows put the work in front of a person: one asks about a finished render
before anything else happens to it, the other asks about the transcript before
it is burned into a render, then about the render. Every call to alfred is a
tool invocation, so it is in qmcp's record.
`walkthrough/10-a-render-waits-for-a-person.md` and
`walkthrough/11-captions-are-read-before-they-are-burned-in.md` run the tools
and the flows' steps against an alfred stood in for.

The tools talk to alfred over its HTTP API and nothing else: no alfred or render
engine code is imported.

## Configuration

| Setting | Meaning |
|---|---|
| `QMCP_ALFRED_URL` | Where alfred's API answers, such as `http://localhost:8000`. Unset, every tool answers that alfred is not configured. A path is kept, so `https://host/alfred` sends `/project` to `https://host/alfred/project` |
| `QMCP_ALFRED_EMAIL`, `QMCP_ALFRED_PASSWORD` | An alfred account qmcp logs in as. Both or neither |
| `QMCP_ALFRED_TOKEN` | A token alfred issued, sent before any login |
| `QMCP_ALFRED_TIMEOUT` | Seconds alfred has to answer one request; the default is in `qmcp/config.py`. Opening a connection may take at most `CONNECT_TIMEOUT` (`qmcp.tools.alfred`) |
| `QMCP_ALFRED_DELIVER_TIMEOUT` | Seconds alfred has to answer a delivery, which it answers only once it has copied or uploaded the whole render; the default, longer, is in `qmcp/config.py` |

They are read as qmcp reads every setting, from the environment or from a `.env`
file in the directory the server starts in, once per server process. `.env` is
ignored by git; the password and token are kept as secrets and left out of
every `repr`.

**Who qmcp is to alfred.**

- **An account**: qmcp logs in on the first tool call (`POST /auth/jwt/login`)
  and reuses the token. A token with less than `RENEW_MARGIN` seconds left is
  renewed with `POST /auth/jwt/refresh` before it is sent, measured by the
  token's own `exp - iat`, so the two machines' clocks need not agree. A
  request alfred refuses with 401 is sent once more after a fresh login, and a
  second 401 is an error.
- **A token alone**: it is sent, and renewed the same way while it is in use,
  measured against this machine's clock. Once alfred refuses it, every tool
  says so; alfred's tokens expire, so an account is the setting that lasts.
- **Neither**: requests carry no token, which suits an alfred that asks for
  none, such as its development stack with `ALFRED_DEV_SKIP_AUTH=1`.

## Tools

| Tool | Input | Returns | alfred route |
|---|---|---|---|
| `alfred_projects` | `skip`, `limit` (optional) | `[{id, name, clips, duration}]` | `GET /project` |
| `alfred_templates` | none | alfred's list, `[{name, description, parameters, deprecated}]` | `GET /templates` |
| `alfred_render_enqueue` | `project_id` or `edl`, exactly one; `name`, `description`, `width`, `height`, `fps`, `quality`, `captions` (optional) | `{render_id, project_id, name, width, height, fps, quality, captions}` | `GET /project/{id}`, `POST /render` |
| `alfred_render_status` | `render_id` | `{render_id, state, progress, error, download_url, download_expires_at}` | `GET /render/{id}/status` |
| `alfred_render_deliver` | `render_id`; `destination` (optional) | `{render_id, destination, location}` | `POST /render/{id}/deliver` |
| `alfred_captions` | `project_id` | `{project_id, captions, style, job}` | `GET /project/{id}/captions` |
| `alfred_captions_transcribe` | `project_id` | `{project_id, job}` | `POST /project/{id}/captions/transcribe` |
| `alfred_captions_set` | `project_id`; `captions`, `style`, or both | `{project_id, captions, style, job}` | `PUT /project/{id}/captions`, and `GET` for what is left out |
| `alfred_health` | none | alfred's report as alfred answers it, `{api, db, renderer: {ok, workers}, queue: {pending}, audio_engine: {configured, ok, detail}}` | `GET /health` |

**`alfred_projects`** counts the clips on each project's timeline: 0 for a
project built from a form rather than on the timeline. `duration` is the one
saved with the timeline, null when none was.

**`alfred_render_enqueue` with `project_id`** renders the project's saved
timeline exactly as alfred returns it: no clip is placed, filled in or dropped,
and a null stays a null (`fadeIn: null` is no fade, where a missing `fadeIn`
is the engine's default). `width`, `height`, `fps` and `quality` come from the
input, else the project's saved `render_settings`, else alfred's defaults; a
null in the result means alfred's default. `quality` is one of alfred's
encoding presets: `draft`, `standard` or `high`. A project with no clips is
refused before anything is queued. **With `edl`**, the EDL and the options
given are sent, and nothing else.

**Captions in a render.** A project whose caption style is enabled has its
captions burned in: its `captions` and `caption_style` go with the render as
alfred stores them, and alfred turns each caption into a text clip over the
timeline -- qmcp maps nothing. `captions: true` burns them in with the style
switched on, whatever the project saved, and is refused before anything is
queued for a project with no captions; `captions: false` sends none. A style
that does not say `enabled: true` counts as switched off. `captions` in the
result is how many went with the render; with `edl` there are none, and
`captions: true` is refused.

**`alfred_render_status`**: `state` is alfred's -- `queued`, `rendering`,
`rendered` or `failed` -- and `error` says why a render failed. Once a render is
rendered, an alfred that issues signed links answers a `download_url`; the tool
makes it absolute against `QMCP_ALFRED_URL`, and `download_expires_at` is when
it stops working, read from the link. The link needs no token until then, and
each call issues a fresh one. An alfred that issues no links answers null.

**`alfred_render_deliver`** sends a finished render to a destination alfred
offers, as the `destination` query parameter; unset, alfred chooses -- where
the render was last sent, else its local store. `location` is where it went,
made absolute against `QMCP_ALFRED_URL` when alfred answers a path: a local
delivery is the render's download route, which asks for the owner's session,
not a signed link. alfred refuses a render that has not finished (409) and a
destination it cannot reach (502). alfred answers only once it has copied the
render or uploaded it, which takes longer than any other call for a long one,
so the tool waits `QMCP_ALFRED_DELIVER_TIMEOUT` rather than
`QMCP_ALFRED_TIMEOUT`. A delivery that outlasts that is an error saying so;
alfred carries on with it and records it, and asking again delivers it again.

**`alfred_captions`**: `captions` is a list of `{start, end, text}` in timeline
seconds, empty for a project never captioned; `style` is the caption style,
every field of it for every project -- `{enabled, fontsize, color, bg,
position, align, font}`, null where it is not set -- with `enabled` false until
someone switches the captions on; `job` is the last transcription, `{state,
error, engine, segments, updated_at}`, its `state` one of `queued`,
`transcribing`, `done` or `failed` -- or null when the project was never
transcribed. A transcription that finishes stores a style that says nothing of
`enabled` as switched off.

**`alfred_captions_transcribe`** asks alfred to transcribe the project's
saved timeline and answers with the job alfred started; alfred stores the
captions on the project when it is done, and `alfred_captions` follows it.
alfred answers 503 for two reasons: it has no audio engine it can ask -- none
configured, or a setting it cannot use -- or its broker would not take the job,
which alfred then records as failed. It answers 409 while a transcription of
the project runs. alfred names each with a `code` beside its `detail`:
`audio_engine_missing`, `transcription_not_queued` and `transcription_running`,
and the code is what tells them apart (`TRANSCRIPTION_REFUSALS` in
`qmcp.tools.alfred`). An alfred from before codes sends the detail alone, and
its wording tells them apart instead: a detail naming the engine or an
`ALFRED_AUDIO_ENGINE_*` setting, or one opening `The transcription could not be
queued`. The error opens with `alfred has no audio engine`, `alfred could not
queue the transcription` or `a transcription is already running`
(`NO_AUDIO_ENGINE`, `TRANSCRIPTION_NOT_QUEUED` and `TRANSCRIPTION_RUNNING`),
then alfred's own refusal. A refusal with any other code, or with no code and
wording that says neither -- such as a proxy in front of alfred answers while
alfred is down -- is passed on as it came.

**`alfred_captions_set`** is a person's correction. alfred checks each caption
it is sent -- it starts at 0 or later, ends after it starts, and has text, at
most 500 characters of it -- refuses more than 1000, and stores the captions in
order of when they start, however they were sent, with the style and nothing
else. A caption it refuses is named by alfred's validation error, counting from
0, as in `body.captions.1.__root__: a caption must end after it starts (start 4.0, end 3.0)`.
alfred checks the style's fields too, a colour as `#rrggbb` for one. Give the
captions, the style, or both; the one left out is read from alfred and sent
back as alfred answered it, so a route that stores both does not reset it.
alfred answers both for every project, the style whole and switched off until
it is set. A style sent without `enabled` keeps the `enabled` alfred holds. The
result is what alfred then holds, read back with `GET` when alfred's answer
does not carry it.

**`alfred_health`** is alfred's report as alfred answers it: the tool adds,
drops and fills in nothing. alfred answers it to a session, as it does every
route the tools call, and an alfred without the report answers 404, which the
tool raises as an error. `db` is `ok` or `down`; `renderer.workers` is how many
render workers answered alfred's ping, and `renderer.ok` whether any did;
`queue.pending` is how many renders wait for one, null when alfred could not ask
its broker. `audio_engine` says whether an audio engine is configured --
alfred's `ALFRED_AUDIO_ENGINE_URL` -- and whether it answers, with `detail`
saying why not, null when it does; transcribing needs both, and a render worker
to mix the timeline. An alfred whose report predates its audio engine answers
without `audio_engine`, and the tool passes that on as it is.

## Errors

A tool that cannot do what it was asked raises. The server then answers the
invocation with `result: null` and `error: "Tool execution failed: <message>"`,
and records it as failed.

A refusal alfred's own code made carries a `code` beside its `detail`, saying
what it means in a word that stays the same when the sentence is reworded --
alfred's `docs/api/errors.md` lists them. The message carries the sentence; the
`AlfredError` the tool raises carries both, as `detail` and `code`, for a
caller in Python to switch on. `code` is None for a refusal without one: from an
alfred that predates codes, FastAPI's validation list, a route alfred does not
have, or a proxy's page.

| Cause | The message |
|---|---|
| No `QMCP_ALFRED_URL` | `alfred is not configured: ...`, naming the settings |
| An email without a password, or the reverse | `alfred's account is half configured: QMCP_ALFRED_PASSWORD is unset` |
| alfred refused a request | `alfred refused <method> <path> (<status>): <alfred's detail>`; a validation error list is given as `body.width: value is not a valid integer; ...` |
| alfred refused with 401 | the same, followed by what to change: the account was refused after a fresh login, the token was refused, or alfred asks for an account |
| An id alfred cannot read, or a project or render it does not hold | `alfred refused ... (422): Not a valid identifier` for an id that is not an ObjectId's 24 hex digits, before alfred looks anything up; `(404): Item not found` for one it does not hold, or that is not the account's |
| The login was refused | `alfred refused the login for <email> (<status>): <detail>` |
| A route the server does not have | `... (404): Not Found (the server at <url> calls itself '<title>')`, the title read from the server's `/openapi.json`; an alfred that does not caption answers the caption routes this way |
| No audio engine, on `alfred_captions_transcribe` | `alfred has no audio engine: alfred refused POST /project/<id>/captions/transcribe (503): <alfred's detail>` |
| A transcription alfred could not queue | `alfred could not queue the transcription: alfred refused POST ... (503): The transcription could not be queued: <what its broker raised>` |
| A transcription of the project running | `a transcription is already running: alfred refused POST ... (409): <alfred's detail>` |
| No answer | `alfred at <url> could not be reached ...`, or `did not answer ... in time (<error>, <seconds> s)`, followed for a delivery by `QMCP_ALFRED_DELIVER_TIMEOUT` and that alfred carries on with it |
| An answer that is not JSON | `alfred answered <method> <path> with <content type>, not JSON`, and the server's title when it gives one |

The title names what is actually listening at `QMCP_ALFRED_URL`, which differs
from alfred when another service holds the port.

## The approval flow

`examples/flows/alfred_render.py` renders a project and asks a person to
approve the result. Its steps are `qmcp.cookbook.alfred_render`, and every call
to alfred goes through the tools above, under the correlation id
`alfred-render-<run id>`.

```bash
uv run qmcp serve                    # with the QMCP_ALFRED_* settings
uv run python examples/flows/alfred_render.py run --project-id <id>
uv run qmcp human respond <request id> approve      # or reject
```

It is also the `alfred-render` recipe: `uv run qmcp cookbook run alfred-render
--project-id <id>`, or `run_recipe_local` from the MCP server in `qmcp_mcp.py`.
`run --help` lists the parameters and their defaults.

| Step | What it does |
|---|---|
| `start` | checks the qmcp server answers |
| `enqueue` | `alfred_render_enqueue` for the project, with `--width`, `--height` and `--quality` when given |
| `follow` | `alfred_render_status` every `--poll-interval` seconds, until the render is rendered or failed |
| `ask_person` | for a rendered render: an approval with the options `approve` and `reject` that expires after `--approval-timeout`, then the answer |
| `end` | prints how the run ended |

**How `follow` ends.** `rendered`; `failed`, with alfred's reason; `no
renderer`, for a render still queued after `--queued-timeout` when
`alfred_health` reports no worker running, or for one that never left the queue
within `--render-timeout` on an alfred without the report; and `timed out`
otherwise, which a queue behind running workers is. Only a rendered render is
put to a person.

**What the person is asked.** The request is
`alfred-render-<render id>-<run id>`. Its prompt names the project and carries
the download link with the moment it stops working; its context holds the
render's id, project, size and quality, the link, and the run. alfred's links
are signed for minutes while an approval may wait far longer, so the prompt
also says that `alfred_render_status` for the render issues a fresh link.

**How the answer is waited for.** The flow watches the pending listing until
the request leaves it, then reads the request once: reading a request is what
marks an expired one expired (`AGENTS.md`, *One read in this API is a write*).
A deadline two polls past the expiry covers a server whose listing keeps it.

**What approval does.** It ends the run, approved; nothing is delivered or
published by the flow. A rejection and an approval nobody gave end it too, each
said in the last line. The caption flow below can deliver an approved render,
when it is asked to.

## The caption flow

`examples/flows/alfred_caption_render.py` captions a project and renders it
with the captions burned in, asking a person twice: about the transcript before
it is rendered, and about the render before it goes anywhere. Its steps are
`qmcp.cookbook.alfred_captions`, beside `qmcp.cookbook.alfred_render`'s, and
every call to alfred goes through the tools above, under the correlation id
`alfred-caption-render-<run id>`.

```bash
uv run qmcp serve                    # with the QMCP_ALFRED_* settings
uv run python examples/flows/alfred_caption_render.py run --project-id <id>
uv run python examples/flows/alfred_caption_render.py run --project-id <id> --deliver True
uv run qmcp human respond <request id> approve      # or reject, for each request
```

It is also the `alfred-caption-render` recipe: `uv run qmcp cookbook run
alfred-caption-render --project-id <id>`, or `run_recipe_local` from the MCP
server in `qmcp_mcp.py`. `run --help` lists the parameters and their defaults.

| Step | What it does |
|---|---|
| `start` | checks the qmcp server answers |
| `transcribe` | `alfred_captions_transcribe`, then `alfred_captions` every `--poll-interval` seconds until the job is done or failed |
| `review` | for a transcript with captions: an approval carrying it, that expires after `--review-timeout`, then the answer |
| `render` | for an approved transcript: `alfred_render_enqueue` with `captions: true`, with `--width`, `--height` and `--quality` when given |
| `follow` | `alfred_render_status` until the render settles, as `alfred-render` follows one |
| `ask_person` | for a rendered render: an approval with its link that expires after `--approval-timeout`, then the answer |
| `delivery` | for an approved render, only when `--deliver True`: `alfred_render_deliver`, to `--destination` or alfred's choice |
| `end` | prints how the run ended |

**How `transcribe` ends.** `done`, with the captions; `failed`, with alfred's
reason -- a timeline in which the audio engine heard no speech ends here, in
alfred's words, and the project keeps the captions it had; `no captions`, a job
done while the project holds none, as when they are cleared after it finishes,
since alfred stores a job's captions with it; `no audio engine`, at once, when
alfred refuses for want of one; `not queued`, at once, when alfred's broker
would not take the job; `no renderer`, for a job still queued after
`--queued-timeout` when `alfred_health` reports no worker running -- a
transcription is one of alfred's worker tasks, as a render is -- or for one
that never left the queue within `--transcribe-timeout` on an alfred without
the report; and `timed out` otherwise. A transcription of the project already
running is followed to its end, since it makes the same captions. Only `done`
is put to a person.

**How long `transcribe` waits.** `--transcribe-timeout` is 2700 s unless given
(`TRANSCRIBE_TIMEOUT` in `qmcp.cookbook.alfred_captions`): as long as alfred
lets a transcription run before it lets another start in its place, which is
its audio engine's timeout -- `ALFRED_AUDIO_ENGINE_TIMEOUT`, 1800 s unless set
-- and 900 s to mix the timeline. Against an alfred given a longer engine
timeout, give a longer wait.

**What the person reads first.** The request is
`alfred-captions-<project id>-<run id>`. Its prompt lists the transcript, a
line per caption, `m:ss.s-m:ss.s  text`, at most `REVIEW_LINES`
(`qmcp.cookbook.alfred_captions`) of them, and its context holds every
caption, the engine and the run. Approving renders the
project with the captions alfred holds when the render is queued, so a
correction saved in alfred meanwhile -- in alfred's editor, or with
`alfred_captions_set` -- is in the render; rejecting ends the run with nothing
rendered.

**What the person reads second.** The request is
`alfred-caption-render-<render id>-<run id>`, `alfred-render`'s approval with
the link and when it stops working, and it says what approving does: delivers
the render, when the run was started with `--deliver True`, and otherwise
nothing more.

**Delivery is an opt-in.** Without `--deliver True` an approval ends the run,
as it does in `alfred-render`. With it, the approved render goes to
`--destination`, or where alfred chooses when that is unset. A delivery alfred
refuses is said in the last line rather than raised: the render is fine, and
`alfred_render_deliver` can be asked again.

Both answers are waited for as `alfred-render` waits: the pending listing is
watched until the request leaves it, and the request read once.

## What is checked

`tests/test_alfred_tools.py`, `tests/test_alfred_caption_tools.py`,
`tests/test_alfred_render_flow.py` and `tests/test_alfred_caption_flow.py` drive
a stand-in alfred through `httpx.MockTransport` (`tests/alfred_stub.py`), which
also holds projects' captions, a transcription job and a render that each move
one state per look. One run of `alfred-render`'s steps goes through a qmcp
server on an ephemeral port and its queue; the caption flow's file is run step
by step there, over a stand-in for Metaflow's classes and decorators, through
each way a run can end. Each guard names the mutation that turns it red. A
running alfred is not part of the suite, and neither is Metaflow itself where it
cannot be imported (`import fcntl`); `tests/test_examples_smoke.py` imports
both flows where it can.
