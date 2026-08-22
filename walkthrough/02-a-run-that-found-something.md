# 02 — A run that found something

Everything on this page runs. The subprocesses are real, the invocation rows are
real, and the delta at the end is the one this repository actually has.

`qmcp selfcheck` points the harness at its own repository. The checks are the
gates this project is already held to, so nothing is invented for the occasion —
and one of them does not pass.

## A check is a real subprocess, recorded like any other invocation

    >>> from pathlib import Path
    >>> from qmcp.selfcheck import Check, run_check

    >>> passing = Check(name="demo-ok", what="a command that succeeds",
    ...                 argv=("python", "-c", "print('all good')"))
    >>> finding, invocation = run_check(passing, Path("."), "quaternionmedia/qmcp")
    >>> finding.ok
    True
    >>> finding.detail
    'all good'

The row that records it is the same row the server writes when a tool is invoked
over HTTP, and the same row `qmcp dashboard` reads back:

    >>> invocation.tool_name
    'demo-ok'
    >>> invocation.status.name
    'SUCCESS'
    >>> invocation.duration_ms >= 0
    True

Its address is what lets a second system point at the same row:

    >>> finding.address == f"quaternionmedia/qmcp/invocation/{invocation.id}"
    True

## A failing check is a unit of work

    >>> failing = Check(name="demo-gate", what="whether the thing holds",
    ...                 argv=("python", "-c", "raise SystemExit('the gate refused')"))
    >>> finding, invocation = run_check(failing, Path("."), "quaternionmedia/qmcp")
    >>> finding.ok
    False
    >>> invocation.status.name
    'FAILED'

The exit status comes from the process, not from reading its output. A check
read by grepping its text is one a wording change silently disables:

    >>> invocation.result
    {'exit_code': 1}

## It opens at `brainstorm`, because noticing is not deciding

    >>> from qmcp.selfcheck import to_delta
    >>> payload = to_delta(finding, "quaternionmedia/qmcp")
    >>> payload["delta"]["phase"]
    'brainstorm'
    >>> payload["delta"]["name"]
    'demo-gate-does-not-pass'

The name is the check, not the run. Running it again finds the same delta rather
than opening a second one — a queue that grows by one row per run is a queue
nobody reads:

    >>> other_run, _ = run_check(failing, Path("."), "quaternionmedia/qmcp")
    >>> to_delta(other_run, "quaternionmedia/qmcp")["delta"]["name"] == payload["delta"]["name"]
    True
    >>> other_run.invocation_id == finding.invocation_id
    False

## A green check is not work

    >>> from qmcp.selfcheck import render
    >>> "does-not-pass" in render([run_check(passing, Path("."), "q/r")[0]])
    False

## The question it raises, and the phase that follows the answer

A failing gate is not a decision. Whether to fix it, accept it or defer it is a
person's, so the run asks:

    >>> from qmcp.selfcheck import ask
    >>> request = ask(finding, "quaternionmedia/qmcp")
    >>> request.options
    ['fix', 'accept', 'defer']

Once somebody has answered, the delta moves — and only that far:

    >>> to_delta(finding, "quaternionmedia/qmcp", answered=True)["delta"]["phase"]
    'planning'

Nothing here reaches `implementation` or `complete`. Re-running a check cannot
establish that anybody acted on it, which is the same reason
`governance/qm/records/DRAFT-a-disagreement-is-a-delta.md` 4 has a detector
report convergence and never close a delta.

## The whole run, against this repository

    uv run qmcp selfcheck --database run.db          # the report
    uv run qmcp human list --database run.db         # what is waiting on a person
    uv run qmcp human respond selfcheck-tag-claims defer --database run.db
    uv run qmcp selfcheck --database run.db --deltas > deltas.json
    uv run qmcp dashboard --database run.db --json   > harness.json

Then, in the control panel:

    DOSSIER_DATABASE_URL=sqlite:///run-dossier.db uv run dossier db upgrade
    DOSSIER_DATABASE_URL=sqlite:///run-dossier.db uv run dossier projects add quaternionmedia/qmcp
    DOSSIER_DATABASE_URL=sqlite:///run-dossier.db uv run dossier harness ingest harness.json --write
    DOSSIER_DATABASE_URL=sqlite:///run-dossier.db uv run dossier deltas ingest deltas.json --write

**Name a database.** Both sides open one relative to the working directory
otherwise, and a demo that forgets writes into whatever the operator was using.
That happened while this page was being written.

## What the run finds here, today

`tag-claims` does not pass. This project's suite skips tests needing optional
dependencies, and `governance/qm/records/DRAFT-version-tags-are-claims.md` 3 says
a skipped test contributes nothing to the automated-validation claim — so the
gate refuses, and the delta is real rather than staged.

It sits at `planning` once somebody has answered, and it does not move further,
because the blocker has not gone anywhere. A demo that ended with the delta
`complete` would be demonstrating the one thing this design refuses to do.
