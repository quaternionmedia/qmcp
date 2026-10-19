# alfred Integration

qmcp drives alfred, a self-hosted video editor that renders on its server. Its
tools list alfred's projects and templates; queue a render, follow it to a link
a person can open, and deliver it; and transcribe, read and correct a project's
captions for a render to burn in. Two flows put the work to a person:
`alfred-render` asks about a finished render before anything else happens to
it, and `alfred-caption-render` asks about the transcript, then the render.

The tools use alfred's HTTP API and import no alfred or render-engine code.
Every call is a tool invocation, so qmcp records it.
`walkthrough/10-a-render-waits-for-a-person.md` and
`walkthrough/11-captions-are-read-before-they-are-burned-in.md` run the tools
and the flows' steps against a stand-in alfred.

## Configuration

| Setting | Meaning |
|---|---|
| `QMCP_ALFRED_URL` | Where alfred's API answers, such as `http://localhost:8000`. A path is kept: `https://host/alfred` sends `/project` to `https://host/alfred/project`. Unset, every tool says alfred is not configured |
| `QMCP_ALFRED_EMAIL`, `QMCP_ALFRED_PASSWORD` | An alfred account qmcp logs in as. Both or neither |
| `QMCP_ALFRED_TOKEN` | A token alfred issued, sent before any login |
| `QMCP_ALFRED_TIMEOUT` | Seconds alfred has to answer one request; the default is in `qmcp/config.py`. A connection must open within `CONNECT_TIMEOUT` (`qmcp.tools.alfred`) |
| `QMCP_ALFRED_DELIVER_TIMEOUT` | Seconds alfred has to answer a delivery, which takes the longest (`alfred_render_deliver`, below); the default, longer, is in `qmcp/config.py` |

qmcp reads them as it reads every setting: once per server process, from the
environment or a `.env` file in the directory the server starts in. git
ignores `.env`. The password and token are kept as secrets, out of every
`repr`.

**A local alfred.** In alfred's checkout, `./al family up` starts alfred with
joe as its audio engine and prints the `QMCP_ALFRED_URL` to use. `./al family
smoke` then runs qmcp's `alfred_health` tool against it, from the qmcp checkout
beside alfred's or the one `ALFRED_FAMILY_QMCP` names. alfred's
`docs/family.md` has the rest.

**How qmcp signs in.**

- **An account.** qmcp logs in on the first tool call (`POST /auth/jwt/login`)
  and reuses the token. A token with less than `RENEW_MARGIN` seconds left is
  renewed with `POST /auth/jwt/refresh` before it is sent. The time left is
  measured by the token's own `exp - iat`, so the two machines' clocks need not
  agree. A request refused with 401 is sent once more after a fresh login; a
  second 401 is an error.
- **A token alone.** It is sent, and renewed the same way while in use,
  measured against this machine's clock. Once alfred refuses it, every tool
  says so. alfred's tokens expire, so an account is the setting that lasts.
- **Neither.** Requests carry no token. That suits an alfred that asks for
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

**`alfred_projects`**: `clips` counts the clips on each project's timeline, 0
for a project built from a form. `duration` is the one saved with the timeline,
null when none was.

**`alfred_render_enqueue` with `project_id`** renders the project's saved
timeline exactly as alfred returns it. No clip is placed, filled in or dropped,
and a null stays a null: `fadeIn: null` is no fade, while a missing `fadeIn` is
the engine's default. `width`, `height`, `fps` and `quality` come from the
input, else the project's saved `render_settings`, else alfred's defaults; a
null in the result means alfred's default. `quality` is one of alfred's
encoding presets: `draft`, `standard` or `high`. A project with no clips is
refused before anything is queued. **With `edl`**, only the EDL and the
options given are sent.

**Captions in a render.** A project whose caption style is enabled has its
captions burned in. Its `captions` and `caption_style` go with the render as
alfred stores them, and alfred turns each caption into a text clip over the
timeline; qmcp maps nothing. A style that does not say `enabled: true` counts
as switched off. The `captions` input:

- unset: the project's style decides;
- `true`: burned in with the style switched on, whatever the project saved.
  Refused before anything is queued for a project with no captions, and with
  `edl`;
- `false`: none are sent.

`captions` in the result is how many went with the render; with `edl`, none.

**`alfred_render_status`**: `state` is alfred's: `queued`, `rendering`,
`rendered` or `failed`. `error` says why a render failed. Once a render is
rendered, an alfred that issues signed links answers a `download_url`, and one
that issues none answers null. The tool makes the link absolute against
`QMCP_ALFRED_URL`. `download_expires_at`, read from the link, is when it stops
working; until then it needs no token. Each call issues a fresh link.

**`alfred_render_deliver`** sends a finished render to a destination alfred
offers, as the `destination` query parameter. Unset, alfred chooses: where the
render was last sent, else its local store. `location` is where it went, made
absolute against `QMCP_ALFRED_URL` when alfred answers a path. A local
delivery's location is the render's download route, which asks for the owner's
session, not a signed link. alfred refuses a render that has not finished (409)
and a destination it cannot reach (502).

alfred answers a delivery only once it has copied or uploaded the render, so
the tool waits `QMCP_ALFRED_DELIVER_TIMEOUT`, not `QMCP_ALFRED_TIMEOUT`. A
delivery that outlasts it is an error saying so; alfred carries on and records
it, and asking again delivers it again.

**`alfred_captions`** answers:

- `captions`: a list of `{start, end, text}` in timeline seconds, empty for a
  project never captioned.
- `style`: the caption style, every field of it for every project, `{enabled,
  fontsize, color, bg, position, align, font}`, null where not set. `enabled` is
  false until someone switches the captions on. A transcription that finishes
  stores a style that says nothing of `enabled` as switched off.
- `job`: the last transcription, `{state, error, engine, segments,
  updated_at}`, or null when the project was never transcribed. Its `state` is
  `queued`, `transcribing`, `done` or `failed`.

**`alfred_captions_transcribe`** asks alfred to transcribe the project's saved
timeline and answers with the job alfred started. alfred stores the captions on
the project when the job is done; `alfred_captions` follows it.

alfred refuses a transcription three ways, each named by a `code` beside its
`detail`, and the code tells them apart (`TRANSCRIPTION_REFUSALS` in
`qmcp.tools.alfred`). The tool's error opens with what the refusal means, then
alfred's own:

| alfred's refusal | `code` | The error, and the constant it opens with |
|---|---|---|
| 503: no audio engine it can ask, none configured or a setting it cannot use | `audio_engine_missing` | `alfred has no audio engine: alfred refused POST /project/<id>/captions/transcribe (503): <alfred's detail>` (`NO_AUDIO_ENGINE`) |
| 503: its broker would not take the job, which alfred records as failed | `transcription_not_queued` | `alfred could not queue the transcription: alfred refused POST ... (503): The transcription could not be queued: <what its broker raised>` (`TRANSCRIPTION_NOT_QUEUED`) |
| 409: a transcription of the project is running | `transcription_running` | `a transcription is already running: alfred refused POST ... (409): <alfred's detail>` (`TRANSCRIPTION_RUNNING`) |

An alfred that sends no codes sends the detail alone, and its wording tells
them apart: a detail naming the engine or an `ALFRED_AUDIO_ENGINE_*` setting,
or one opening `The transcription could not be queued`. A refusal with any
other code, or with no code and neither wording, is passed on as it came, such
as a proxy's answer while alfred is down.

**`alfred_captions_set`** stores a person's correction. Give the captions, the
style, or both. The one left out is read from alfred and sent back as alfred
answered it, so a route that stores both does not reset it. alfred answers both
for every project, the style whole and switched off until it is set. A style
sent without `enabled` keeps the `enabled` alfred holds. The result is what
alfred then holds, read back with `GET` when alfred's answer does not carry it.

alfred checks what it is sent:

- each caption starts at 0 or later, ends after it starts, and has text, at
  most 500 characters;
- at most 1000 captions;
- the style's fields, such as a colour as `#rrggbb`.

It stores the captions in order of when they start, however they were sent,
with the style and nothing else. A refused caption is named by alfred's
validation error, counting from 0, as in
`body.captions.1.__root__: a caption must end after it starts (start 4.0, end 3.0)`.

**`alfred_health`** passes on alfred's report as alfred answers it: the tool
adds, drops and fills in nothing. alfred answers it to a session, as it does
every route the tools call. An alfred without the report answers 404, which the
tool raises as an error.

- `db`: `ok` or `down`.
- `renderer.workers`: how many render workers answered alfred's ping;
  `renderer.ok`: whether any did.
- `queue.pending`: how many renders wait for a worker, null when alfred could
  not ask its broker.
- `audio_engine`: whether an audio engine is configured (alfred's
  `ALFRED_AUDIO_ENGINE_URL`) and whether it answers, with `detail` saying why
  not, null when it does. An older alfred's report has no `audio_engine`; the
  tool passes it on as it is.

Transcribing needs an audio engine configured and answering, and a render
worker to mix the timeline.

## Errors

A tool that cannot do what it was asked raises. The server answers the
invocation with `result: null` and `error: "Tool execution failed: <message>"`,
and records it as failed.

A refusal made by alfred's own code carries a `code` beside its `detail`: a
word for what it means that stays the same when the sentence is reworded.
alfred's `docs/api/errors.md` lists them. The message carries the sentence. The
`AlfredError` the tool raises carries both, as `detail` and `code`, for a
caller in Python to switch on. `code` is None for a refusal without one: from
an alfred that sends no codes, FastAPI's validation list, a route alfred does
not have, or a proxy's page.

| Cause | The message |
|---|---|
| No `QMCP_ALFRED_URL` | `alfred is not configured: ...`, naming the settings |
| An email without a password, or the reverse | `alfred's account is half configured: QMCP_ALFRED_PASSWORD is unset` |
| alfred refused a request | `alfred refused <method> <path> (<status>): <alfred's detail>`; a validation error list is given as `body.width: value is not a valid integer; ...` |
| alfred refused with 401 | the same, then what to change: the account was refused after a fresh login, the token was refused, or alfred asks for an account |
| An id that is not an ObjectId's 24 hex digits | `alfred refused ... (422): Not a valid identifier`, before alfred looks anything up |
| A project or render alfred does not hold, or that is not the account's | `alfred refused ... (404): Item not found` |
| The login was refused | `alfred refused the login for <email> (<status>): <detail>` |
| A route the server does not have | `... (404): Not Found (the server at <url> calls itself '<title>')`, the title read from the server's `/openapi.json`; an alfred that does not caption answers the caption routes this way |
| A transcription alfred refuses | the three under `alfred_captions_transcribe`, above |
| No answer | `alfred at <url> could not be reached ...`, or `did not answer ... in time (<error>, <seconds> s)`; for a delivery, followed by `QMCP_ALFRED_DELIVER_TIMEOUT` and that alfred carries on with it |
| An answer that is not JSON | `alfred answered <method> <path> with <content type>, not JSON`, and the server's title when it gives one |

The title names what actually listens at `QMCP_ALFRED_URL`: another service,
when one holds the port.

## The approval flow

`examples/flows/alfred_render.py` renders a project and asks a person to
approve the result.

```bash
uv run qmcp serve                    # with the QMCP_ALFRED_* settings
uv run python examples/flows/alfred_render.py run --project-id <id>
uv run qmcp human respond <request id> approve      # or reject
```

It is also the `alfred-render` recipe: `uv run qmcp cookbook run alfred-render
--project-id <id>`, or `run_recipe_local` from the MCP server in `qmcp_mcp.py`.
`run --help` lists the parameters and their defaults. Its steps are in
`qmcp.cookbook.alfred_render`. Every call to alfred goes through the tools
above, under the correlation id `alfred-render-<run id>`.

| Step | What it does |
|---|---|
| `start` | checks the qmcp server answers |
| `enqueue` | `alfred_render_enqueue` for the project, with `--width`, `--height` and `--quality` when given |
| `follow` | `alfred_render_status` every `--poll-interval` seconds, until the render is rendered or failed |
| `ask_person` | for a rendered render: an approval with the options `approve` and `reject` that expires after `--approval-timeout`, then the answer |
| `end` | prints how the run ended |

**How `follow` ends.** Only a rendered render is put to a person.

- `rendered`.
- `failed`, with alfred's reason.
- `no renderer`: still queued after `--queued-timeout` while `alfred_health`
  reports no worker running; or, on an alfred without the report, never left
  the queue within `--render-timeout`.
- `timed out` otherwise, as for a queue behind running workers.

**What the person is asked.** The request is
`alfred-render-<render id>-<run id>`. Its prompt names the project and carries
the download link and the moment it stops working. alfred's links are signed
for minutes, and an approval may wait far longer, so the prompt also says that
`alfred_render_status` for the render issues a fresh link. Its context holds the
render's id, project, size and quality, the link, and the run.

**How the answer is waited for.** The flow watches the pending listing until
the request leaves it, then reads the request once. Reading a request is what
marks an expired one expired (`AGENTS.md`, *One read in this API is a write*).
A deadline two polls past the expiry covers a server whose listing keeps it.

**What approval does.** It ends the run, approved; the flow delivers and
publishes nothing. A rejection, or an approval nobody gave, ends it too, and
the last line says which. The caption flow below can deliver an approved
render, when asked to.

## The caption flow

`examples/flows/alfred_caption_render.py` captions a project and renders it
with the captions burned in. It asks a person twice: about the transcript
before it is rendered, and about the render before it goes anywhere.

```bash
uv run qmcp serve                    # with the QMCP_ALFRED_* settings
uv run python examples/flows/alfred_caption_render.py run --project-id <id>
uv run python examples/flows/alfred_caption_render.py run --project-id <id> --deliver True
uv run qmcp human respond <request id> approve      # or reject, for each request
```

It is also the `alfred-caption-render` recipe: `uv run qmcp cookbook run
alfred-caption-render --project-id <id>`, or `run_recipe_local` from the MCP
server in `qmcp_mcp.py`. Its steps are in `qmcp.cookbook.alfred_captions`, beside
`qmcp.cookbook.alfred_render`'s. Every call to alfred goes through the tools
above, under the correlation id `alfred-caption-render-<run id>`.

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

**How `transcribe` ends.** Only `done` is put to a person.

- `done`, with the captions.
- `failed`, with alfred's reason. A timeline in which the audio engine heard no
  speech ends here, in alfred's words, and the project keeps the captions it
  had.
- `no captions`: the job is done but the project holds none, as when they are
  cleared after it finishes, since alfred stores a job's captions with it.
- `no audio engine`, at once, when alfred refuses for want of one.
- `not queued`, at once, when alfred's broker would not take the job.
- `no renderer`: still queued after `--queued-timeout` while `alfred_health`
  reports no worker running, since a transcription is one of alfred's worker
  tasks, as a render is; or, on an alfred without the report, never left the
  queue within `--transcribe-timeout`.
- `timed out` otherwise.

A transcription of the project already running is followed to its end, since it
makes the same captions.

**How long `transcribe` waits.** `--transcribe-timeout` is 2700 s unless given
(`TRANSCRIBE_TIMEOUT` in `qmcp.cookbook.alfred_captions`). That is as long as
alfred lets a transcription run before it lets another start in its place: its
audio engine's timeout, `ALFRED_AUDIO_ENGINE_TIMEOUT` (1800 s unless set), and
900 s to mix the timeline. Against an alfred given a longer engine timeout,
give a longer wait.

**What the person reads first.** The request is
`alfred-captions-<project id>-<run id>`. Its prompt lists the transcript, one
`m:ss.s-m:ss.s  text` line per caption, at most `REVIEW_LINES`
(`qmcp.cookbook.alfred_captions`) of them. Its context holds every caption, the
engine and the run. Approving renders the project with the captions alfred
holds when the render is queued, so a correction saved in alfred meanwhile, in
alfred's editor or with `alfred_captions_set`, is in the render. Rejecting ends
the run with nothing rendered.

**What the person reads second.** The request is
`alfred-caption-render-<render id>-<run id>`: `alfred-render`'s approval, with
the link and when it stops working. It says what approving does: delivers the
render when the run was started with `--deliver True`, and otherwise nothing
more.

**Delivery is opt-in.** Without `--deliver True`, an approval ends the run, as
in `alfred-render`. With it, the approved render goes to `--destination`, or
where alfred chooses when that is unset. A delivery alfred refuses is said in
the last line, not raised: the render is fine, and `alfred_render_deliver` can
be asked again.

Both answers are waited for as `alfred-render` waits for its one.

## What is checked

`tests/test_alfred_tools.py`, `tests/test_alfred_caption_tools.py`,
`tests/test_alfred_render_flow.py` and `tests/test_alfred_caption_flow.py` drive
a stand-in alfred through `httpx.MockTransport` (`tests/alfred_stub.py`). It
also holds projects' captions, a transcription job and a render that each move
one state per look. One run of `alfred-render`'s steps goes through a qmcp
server on an ephemeral port and its queue. The caption flow's file is run step
by step there, over a stand-in for Metaflow's classes and decorators, through
each way a run can end. Each guard names the mutation that turns it red.

Not in the suite: a running alfred, and Metaflow where it cannot be imported
(`import fcntl`). `tests/test_examples_smoke.py` imports both flows where it
can.
