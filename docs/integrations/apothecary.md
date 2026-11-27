# apothecary Integration

qmcp reads apothecary, the workshop's parts library, its sites and the jobs
its machines run. Its tools list the parts and read one, list the sites and
read one, and list the jobs and read one. Every tool reads; none starts a
machine, changes a part or moves a structure.

The tools use apothecary's HTTP API and import no apothecary code. Every call
is a tool invocation, so qmcp records it.

## Configuration

| Setting | Meaning |
|---|---|
| `QMCP_APOTHECARY_URL` | Where apothecary's server answers, such as `http://127.0.0.1:8000`. Unset, every tool says apothecary is not configured |
| `QMCP_APOTHECARY_TIMEOUT` | Seconds apothecary has to answer one request; the default is in `qmcp/config.py`. A connection must open within `CONNECT_TIMEOUT` (`qmcp.tools.apothecary`) |

qmcp reads them as it reads every setting: once per server process, from the
environment or a `.env` file in the directory the server starts in.

apothecary answers the machine it runs on and asks for no account, so the
tools send none and qmcp runs beside it.

## Tools

| Tool | Input | Output |
|---|---|---|
| `apothecary_parts` | `search`, optional: words, any of which a part carries in its name, category, tags or description | `parts`: each part's `name`, `description`, `category` and `tags`. `not_loaded`: the names apothecary holds and could not load. `categories`, only when a search finds no part: the categories there are |
| `apothecary_part` | `name` | The part's row, with `size_mm` (the size of its bounds, null when apothecary measured none), `params` at their own values, `bought` (what it needs that is not made), `has_params`, `kinematics`, `motion`, `can_generate_stl` with `stl_note`, `readme` and `source_file` |
| `apothecary_sites` | none | The sites' names |
| `apothecary_site` | `name` | `is_valid`, `violations`, and `structures`: each one's `name`, `material`, `status`, `build_volume` and `systems` |
| `apothecary_jobs` | `site`, `kind`: each optional, sent to apothecary | Each job's `id`, `kind`, `machine`, `site`, `part`, `input`, `started_at`, `finished_at`, `outcome`, `reason` and `progress`, the running ones first |
| `apothecary_job` | `id` | The job's row, and `log`: the last `LOG_LINES` lines of what its machine said |
| `apothecary_health` | none | apothecary's own answer: its `status` and `version` |

A list is cut down for a caller with a small budget for reading: a part's row
leaves out its file links, its bounds and its colour, and a site leaves out
its tree and its generated OpenSCAD. `apothecary_part` and apothecary's own
page hold the rest.

## Errors

A tool that cannot answer raises, and the server records the message as the
invocation's error.

| What happened | The error says |
|---|---|
| `QMCP_APOTHECARY_URL` is unset | apothecary is not configured, and which setting names it |
| Nothing answers at the address | apothecary at that address could not be reached, with the request |
| It did not answer in time | which request, and the seconds it had |
| apothecary refused the request | its status and its own `detail`: `Part 'x' not found`, `no such job` |
| The answer is not JSON, or not the shape apothecary serves | what was answered instead; the address may name another server |

## Testing

`tests/test_apothecary_tools.py` runs every tool against a stand-in behind
`httpx.MockTransport`. `tests/apothecary_answers.json` holds what it answers:
an apothecary's own answers, with the commit that served them in its `commit`
key. A job's shape follows apothecary's `Job` model, because the apothecary
asked had run none. A running apothecary is checked by hand:

```bash
QMCP_APOTHECARY_URL=http://127.0.0.1:8000 uv run python -c "from qmcp.tools.apothecary import apothecary_parts as f; print(len(f({})['parts']))"
```
