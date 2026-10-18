# alfred Integration

qmcp can drive alfred, a self-hosted video editor whose renders are made on its
server. Five tools let an agent see alfred's projects and templates, queue a
render, and follow it to a link a person can open; an approval flow puts a
finished render in front of a person before anything else happens to it. Every
call to alfred is a tool invocation, so it is in qmcp's record.
`walkthrough/10-a-render-waits-for-a-person.md` runs the tools and the flow's
steps against an alfred stood in for.

The tools talk to alfred over its HTTP API and nothing else: no alfred or render
engine code is imported.

## Configuration

| Setting | Meaning |
|---|---|
| `QMCP_ALFRED_URL` | Where alfred's API answers, such as `http://localhost:8000`. Unset, every tool answers that alfred is not configured. A path is kept, so `https://host/alfred` sends `/project` to `https://host/alfred/project` |
| `QMCP_ALFRED_EMAIL`, `QMCP_ALFRED_PASSWORD` | An alfred account qmcp logs in as. Both or neither |
| `QMCP_ALFRED_TOKEN` | A token alfred issued, sent before any login |
| `QMCP_ALFRED_TIMEOUT` | Seconds alfred has to answer one request; the default is in `qmcp/config.py`. Opening a connection may take at most `CONNECT_TIMEOUT` (`qmcp.tools.alfred`) |

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
| `alfred_render_enqueue` | `project_id` or `edl`, exactly one; `name`, `description`, `width`, `height`, `fps`, `quality` (optional) | `{render_id, project_id, name, width, height, fps, quality}` | `GET /project/{id}`, `POST /render` |
| `alfred_render_status` | `render_id` | `{render_id, state, progress, error, download_url, download_expires_at}` | `GET /render/{id}/status` |
| `alfred_health` | none | alfred's report, `{api, db, renderer: {ok, workers}, queue: {pending}}` | `GET /health` |

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

**`alfred_render_status`**: `state` is alfred's -- `queued`, `rendering`,
`rendered` or `failed` -- and `error` says why a render failed. Once a render is
rendered, an alfred that issues signed links answers a `download_url`; the tool
makes it absolute against `QMCP_ALFRED_URL`, and `download_expires_at` is when
it stops working, read from the link. The link needs no token until then, and
each call issues a fresh one. An alfred that issues no links answers null.

**`alfred_health`**: an alfred without the report answers 404, which the tool
raises as an error.

## Errors

A tool that cannot do what it was asked raises. The server then answers the
invocation with `result: null` and `error: "Tool execution failed: <message>"`,
and records it as failed.

| Cause | The message |
|---|---|
| No `QMCP_ALFRED_URL` | `alfred is not configured: ...`, naming the settings |
| An email without a password, or the reverse | `alfred's account is half configured: QMCP_ALFRED_PASSWORD is unset` |
| alfred refused a request | `alfred refused <method> <path> (<status>): <alfred's detail>`; a validation error list is given as `body.width: value is not a valid integer; ...` |
| alfred refused with 401 | the same, followed by what to change: the account was refused after a fresh login, the token was refused, or alfred asks for an account |
| The login was refused | `alfred refused the login for <email> (<status>): <detail>` |
| A route the server does not have | `... (404): Not Found (the server at <url> calls itself '<title>')`, the title read from the server's `/openapi.json` |
| No answer | `alfred at <url> could not be reached ...`, or `did not answer ... in time` |
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
said in the last line.

## What is checked

`tests/test_alfred_tools.py` and `tests/test_alfred_render_flow.py` drive a
stand-in alfred through `httpx.MockTransport`, and one run of the flow's steps
goes through a qmcp server on an ephemeral port and its queue. Each guard names
the mutation that turns it red. A running alfred is not part of the suite, and
neither is the Metaflow flow where Metaflow cannot be imported (`import
fcntl`); `tests/test_examples_smoke.py` imports it where it can.
