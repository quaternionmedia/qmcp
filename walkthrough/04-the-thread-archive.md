# 04 — The thread archive, end to end

Everything on this page runs. It is executed by the ordinary test command, so an
example that stops being true fails the build rather than sitting here
misleading somebody.

Three sources, one archive, one index, and four read-only routes. Nothing here
spends anything or reaches the network.

## What the archive is for

A conversation is mostly steps. What is worth keeping is what it *settled*, and
what it settled is only useful if you can get back to the work — the branch, the
pull request, the repository. So the archive keeps three things and refuses to
collapse them: the conversations, what they produced, and how each changed.

## The three sources

    >>> from qmcp.threads.claude import ClaudeThreads
    >>> from qmcp.threads.chatgpt import ChatGPTThreads
    >>> from qmcp.threads.claudecode import ClaudeCodeThreads
    >>> [s.name for s in (ClaudeThreads(), ChatGPTThreads(), ClaudeCodeThreads())]
    ['claude', 'chatgpt', 'claude-code']

They are not interchangeable, and the difference is what each knows:

    >>> [s.perspective for s in (ClaudeThreads(), ChatGPTThreads(), ClaudeCodeThreads())]
    ['claude/thread', 'chatgpt/thread', 'claude-code/session']

Two assistants discussing the same work produce two sets of deltas and neither
is the other's duplicate. `same-as` is how somebody says they are one strand
after reading both.

## Setting one up

Everything below runs against a scratch cache, so the page cannot touch a real
archive:

    >>> import json, tempfile
    >>> from pathlib import Path
    >>> from qmcp.spend import Budget
    >>> root = Path(tempfile.mkdtemp())

    >>> def write(source, name, document):
    ...     folder = root / source
    ...     folder.mkdir(parents=True, exist_ok=True)
    ...     (folder / name).write_text(json.dumps(document), encoding="utf-8")

A Claude Code session, because it is the one that carries the joins:

    >>> def session(turns, session_id="s-1", agent=None, branch=None, pr=None):
    ...     records = []
    ...     for uid, text in turns:
    ...         record = {"type": "assistant", "uuid": uid,
    ...                   "sessionId": session_id,
    ...                   "message": {"content": [{"type": "text", "text": text}]}}
    ...         if agent: record["agentId"] = agent
    ...         if branch: record["gitBranch"] = branch
    ...         records.append(record)
    ...     if pr:
    ...         records.append({"type": "pr-link", "sessionId": session_id,
    ...                         "prRepository": pr[0], "prNumber": pr[1]})
    ...     return "\n".join(json.dumps(r) for r in records) + "\n"

    >>> sessions = root / "sessions"
    >>> (sessions / "proj").mkdir(parents=True)
    >>> _ = (sessions / "proj" / "main.jsonl").write_text(session(
    ...     [("u-1", "should the ask kind be its own address?"),
    ...      ("u-2", "DECISION: add ask as an address kind")],
    ...     branch="evolve/ask-kind", pr=("quaternionmedia/qm", 91)),
    ...     encoding="utf-8")

## 1. Read it — free, and it says so

    >>> code = ClaudeCodeThreads(root=sessions)
    >>> surveyed = code.survey()
    >>> surveyed.available, surveyed.would_need
    (1, 0)

`would_need` is a real zero: reading a local file costs no calls, so the paid
work genuinely is none. It is not the sentinel that would mean nobody counted —
that would be `{"unknown": ...}` with a reason.

    >>> thread = code.fetch([], Budget())[0]
    >>> thread.id, len(thread.turns)
    ('s-1', 2)

## 2. The joins a web export cannot give you

    >>> code.project_of(thread)
    'quaternionmedia/qm'

**Read from the session, not defaulted.** A conversation belongs to no
repository; a session says which one it was in. The thread delta carries the
branch and the pull request as addresses:

    >>> payloads = code.deltas(thread, Budget())
    >>> sorted({l["link_type"] for l in payloads[0]["links"]})
    ['address', 'branch', 'pr', 'thread']

    >>> [l["target_name"] for l in payloads[0]["links"] if l["link_type"] == "pr"]
    ['quaternionmedia/qm/pr/91']

A branch keeps its slashes, because the grammar takes everything after the kind
verbatim:

    >>> [l["target_name"] for l in payloads[0]["links"] if l["link_type"] == "branch"]
    ['quaternionmedia/qm/branch/evolve/ask-kind']

## 3. A thread is a delta; what it settled is part of it

    >>> [(p["delta"]["delta_type"], p["delta"]["name"]) for p in payloads]
    [('thread', 'thread-s-1'), ('chore', 's-1-add-ask-as-an-address-kind-0')]

    >>> [(r["relation"], r["target"].split("/delta/")[-1])
    ...  for r in code.relations(thread, Budget())]
    [('part-of', 'thread-s-1')]

Both open at `brainstorm`, and nothing here moves them further:

    >>> {p["delta"]["phase"] for p in payloads}
    {'brainstorm'}

A source recognising a decision has *noticed* something. It has not established
that anybody acted, and a row opened at `planning` would assert they had.

Extraction finds only what a conversation **marked** — `DECISION:` or
`DECIDED:`. Reading an unmarked conversation and working out what it settled
needs a model, which is the paid path:

    >>> unmarked = code.decisions(
    ...     type(thread)(id="x", turns=(thread.turns[0],)), Budget())
    >>> unmarked
    []

## 4. A subagent is its own thread

A subagent file carries its parent's session id. Keying on that alone collapsed
many files into one row, each overwriting the last — and the index read every
overwrite as the thread diverging.

    >>> _ = (sessions / "proj" / "agent.jsonl").write_text(session(
    ...     [("u-9", "looked something up")], agent="a-7"), encoding="utf-8")
    >>> sorted(t.id for t in ClaudeCodeThreads(root=sessions).fetch([], Budget()))
    ['s-1', 's-1/agent-a-7']

And the subagent is `part-of` the session that launched it, stated rather than
inferred from the shared prefix:

    >>> code = ClaudeCodeThreads(root=sessions)
    >>> side = [t for t in code.fetch([], Budget()) if "/agent-" in t.id][0]
    >>> [(r["relation"], r["target"].split("/delta/")[-1])
    ...  for r in code.relations(side, Budget())]
    [('part-of', 'thread-s-1')]

## 5. Index it, and keep every version

    >>> from qmcp.threads import index as index_module
    >>> entries = index_module.build([code])
    >>> doc = index_module.document(entries)
    >>> doc["totals"]["threads"], doc["totals"]["diverged"]
    (2, 0)

Index again after the session grows, and the archive keeps what it knew:

    >>> _ = (sessions / "proj" / "main.jsonl").write_text(session(
    ...     [("u-1", "should the ask kind be its own address?"),
    ...      ("u-2", "DECISION: add ask as an address kind"),
    ...      ("u-3", "and the vectors are written")],
    ...     branch="evolve/ask-kind", pr=("quaternionmedia/qm", 91)),
    ...     encoding="utf-8")

    >>> before = {e.key: e for e in entries}
    >>> merged, changed = index_module.merge(
    ...     before, index_module.build([ClaudeCodeThreads(root=sessions)]))
    >>> grown = [e for e in merged if e.id == "s-1"][0]
    >>> grown.history[0].kind, grown.history[0].detail
    ('grew', '1 turn(s) added')

Growth is the boring case. A turn that says something *else* now is not:

    >>> _ = (sessions / "proj" / "main.jsonl").write_text(session(
    ...     [("u-1", "rewritten after the fact")]), encoding="utf-8")
    >>> after, _ = index_module.merge(
    ...     {e.key: e for e in merged},
    ...     index_module.build([ClaudeCodeThreads(root=sessions)]))
    >>> edited = [e for e in after if e.id == "s-1"][0]
    >>> edited.history[-1].kind
    'diverged'
    >>> edited.diverged
    True

**Nothing is repaired.** The prior digest stays, because a divergence somebody
deletes to make the index tidy is the one fact nobody can recover.

## 6. Serve it, to this machine only

    >>> from fastapi import FastAPI
    >>> from fastapi.testclient import TestClient
    >>> from qmcp.threads.service import register

    >>> index_path = sessions / index_module.INDEX_NAME
    >>> _ = index_path.write_text(
    ...     json.dumps(index_module.document(after)), encoding="utf-8")

    >>> app = FastAPI()
    >>> register(app, sessions, sessions)
    >>> client = TestClient(app)

    >>> listing = client.get("/v1/threads").json()
    >>> listing["totals"]["threads"]
    2

    >>> client.get("/v1/threads/diverged").json()["diverged"][0]["id"]
    's-1'

    >>> client.get("/v1/threads/claude-code/s-1").json()["turns"][0]["role"]
    'assistant'

A source nobody declared is a 404 rather than an attempt, because `source`
reaches a filesystem path:

    >>> client.get("/v1/threads/madeup/x").status_code
    404

An absent index is an absent answer, not an empty archive:

    >>> empty = FastAPI()
    >>> register(empty, Path(tempfile.mkdtemp()))
    >>> TestClient(empty).get("/v1/threads").status_code
    404

### The routes are not mounted off loopback

`qmcp cookbook serve` offers `--host 0.0.0.0` so a container can reach the
server. The archive is somebody's conversations, so it is served only to this
machine — and the routes are **not registered at all** rather than registered
and refusing, because a 403 still tells a caller the archive is there:

    >>> from qmcp.server import is_loopback
    >>> [is_loopback(h) for h in ("127.0.0.1", "::1", "localhost")]
    [True, True, True]
    >>> [is_loopback(h) for h in ("0.0.0.0", "10.0.0.5")]
    [False, False]

Anything unrecognised is treated as remote. A guard that fails open on an
unfamiliar string stops guarding the first time somebody names an interface.

## 7. Look at it

**In the control panel, and only there.**

    uv run python -m qmcp serve        # here, on loopback
    uv run dossier dashboard           # there, the Threads tab

This project rendered a second view of the archive — a self-contained HTML page
— and it was removed rather than kept. Two views of one dataset are two
definitions of what a figure means, and they drift the first time one is fixed.

The panel reads this harness over HTTP and imports nothing from it. When the
harness is not running it says so, with the command that starts it, rather than
showing an empty table — because an empty table would say the archive is empty
when the truth is that nobody answered.

The commands beside this one stay: a command line is for machines and for
debugging, which is what `qmcp threads list` and `--check` are.

## The whole thing, as commands

    uv run qmcp threads import ~/Downloads/claude-export.zip --dry-run
    uv run qmcp threads import ~/Downloads/claude-export.zip
    uv run qmcp threads index --write
    uv run qmcp threads list --diverged
    uv run python -m qmcp serve            # serves /v1/threads on loopback
    uv run dossier dashboard               # the Threads tab reads it

Requesting the export is the one step nothing automates:
`governance/qm/records/DRAFT-acts-that-are-a-persons-by-constitution.md` is why,
and the service enforces the same rule independently.
