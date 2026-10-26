# Harness

A repository's gates are only useful if somebody sees what they found. In this
role qmcp runs a repository's own gates as real subprocesses, records each run
the way the server records a tool call, turns each failure into a unit of work
and a question for a person, and hands both to dossier, the control panel that
shows them. qmcp and dossier import nothing from each other: what crosses is a
schema, and an address names the same row on both sides.

It needs the shared setup in [`README.md`](README.md); the panel needs a
dossier checkout of its own.

## Run the gates, recorded

```bash
uv run qmcp selfcheck --database run.db
uv run qmcp human list --database run.db
uv run qmcp dashboard --database run.db
```

`selfcheck` runs this repository's gates in the order a release meets them and
records each as an invocation. A failing check raises a question in the human
queue; a passing one raises nothing. `--database` keeps the run out of the
clone's own queue; without it, the run lands in `QMCP_DATABASE_URL`.

Answering a raised question is a person's act, and is recorded as one:

```bash
uv run qmcp human respond <request-id> defer --database run.db --by "<your name>"
```

`walkthrough/02-a-run-that-found-something.md` executes all of this with real
subprocesses.

## Hand it to the panel

```bash
uv run qmcp selfcheck --database run.db --deltas > deltas.json
uv run qmcp dashboard --database run.db --json > harness.json
```

dossier ingests them with `dossier deltas ingest` and `dossier harness
ingest`, in its own checkout. dossier reads a live harness at
`127.0.0.1:3141`; `DOSSIER_HARNESS_PORT` moves it.

## Run the workflows before a pull request

`selfcheck` records the gates; `preflight` simulates the workflows under
`.github/workflows/` the way the hosted runner would, and records nothing:

```bash
uv run qmcp preflight --event pull_request --base-ref main
```

A pass is evidence and not proof: the runner does not reproduce `uses:` steps,
the runner image or secrets. On Windows, after a change to `pyproject.toml`,
run the same runner by path, outside `uv run`, because its sync must replace
the `qmcp.exe` it would be running from:

```bash
.venv/Scripts/python governance/qm/project-seed/ci/run_workflows_locally.py --event pull_request --base-ref main
```

## Reference

- [`../self-checks.md`](../self-checks.md): the dashboard, audit, self-check, the feedback loop and intake, and what each cannot see
- [`../threads.md`](../threads.md): conversations read as units of work
