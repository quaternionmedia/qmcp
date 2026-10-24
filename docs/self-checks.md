# How qmcp reports on itself

qmcp records what it runs, and several commands read that record back or run
the project's own gates through it. This page describes each and what it
cannot see. `walkthrough/02-a-run-that-found-something.md` runs the self-check
end to end.

| Entry point | Module | What it answers |
|---|---|---|
| `qmcp dashboard` | `qmcp/dashboard.py` | what the harness has run, and what waits on a person |
| `audit.models_run`, `audit.in_flight` | `qmcp/audit.py` | which models ran in a window; which deltas are moving, through what |
| `qmcp selfcheck` | `qmcp/selfcheck.py` | whether this repository passes its own gates |
| `feedback.once` | `qmcp/feedback.py` | the gates and the panel's diagnostics, together |

`audit` and `feedback` are Python entry points with no command: their modules
name `uv run qmcp audit ...` and `uv run qmcp feedback run`, and
`tests/test_declared_commands.py` lists both as claimed and not routed.

## Unknown is not zero

Every one of these follows the corpus's convention: a figure that could not be
established is `{"unknown": "<reason>"}` (or `None` in Python), never `0`. A
database missing a table reports its counts as unknown; a stage that could not
run is absent from the run, never assumed green. A consumer that reads unknown
as zero is making a claim the harness declined to make.

## The dashboard

    qmcp dashboard              # the view, rendered for a terminal
    qmcp dashboard --json       # the same data, for something else to render

The dashboard reads the database directly and read-only, not the HTTP API, so
it answers when the server is down. The topologies it lists come from the
module, not from `/v1/topology`, for the same reason; they are what this build
can serve, and `qm dashboard` -- the governance corpus's command, run from its
submodule as `uv run --project governance/qm qm dashboard` -- is the view that
knows whether anything is listening.

**Every row is addressed.** An invocation is a bare UUID inside the database;
as `<owner>/<repo>/invocation/<id>` it is a row dossier can point at, a delta
can link to and the corpus's `qm divergence` can compare. Each question in the human queue
crosses the same way, as `<owner>/<repo>/ask/<id>` with its prompt, its
options and its answer if it has one. Answered questions are kept: what was
asked and what was said are the audit trail.

**Caps are stated.** The payload carries a window of recent invocations and a
longer list of queue rows. Truncating a log loses history; truncating a work
list loses work, so the payload states `queue_total` beside `queue_shown`.

**Schema.** The payload's `schema` moves only for a breaking change. Additive
keys do not move it, because `dossier.harness.check_schema` requires an exact
match.

What the dashboard cannot see: anything the database does not record (a tool
that failed before its row was written leaves nothing, so the count of what ran
is a floor); whether a result was any good; and dossier's half of any row. It
addresses its own data so the two can be joined, and does not judge between
them (`records/DRAFT-a-disagreement-is-a-delta.md`).

## Audit

`audit.models_run(rows, window)` and `audit.in_flight(rows, at)` take rows
rather than a session, so one function answers for the harness's tables and
for what crosses the seam.

An audit answer is deterministic: two runs over the same rows give the same
answer. Three things would break that, and each is ruled out:

- **The clock.** "Today" is a fact about a timezone. `created_at` is written in
  UTC, so a window is a UTC half-open interval `[start, end)` stated in the
  answer; nothing reads `date.today()`.
- **The default.** A window filled in from the wall clock returns different
  rows on two runs. `Window` has no default; `Window.day()` makes naming one
  easy.
- **The silent omission.** An invocation that recorded no model is `unknown`,
  counted and shown beside the models that are named.

A model is read from the invocation's `input_params` (`model`, `model_name` or
`model_id`), where `record_model` puts it; nothing is inferred from a tool's
name. An absent model stays absent rather than being written as the string
"unknown", so it can be recorded truthfully if it is learned later.

**The bridge test.** The harness and the control panel can each compute the
same figures, the harness from its tables and the panel from what crosses the
seam. `compare()` reports every figure they read differently, including a
figure only one side reports, and picks no winner: deciding which side is right
needs somebody who can look at both. Pass only the figures the route claims to
carry, since anything else one side does not send reads as a loss.

## Self-check

    qmcp selfcheck                 # run the checks, show what they found
    qmcp selfcheck --deltas        # the failures, as units of work
    qmcp selfcheck --json          # the run, as data

The checks are the gates this project is held to, run in the order a release
meets them. Each is a real subprocess against the working tree, recorded as a
`ToolInvocation`: the row the server writes for a tool called over HTTP, which
the dashboard reads back. A check's verdict is its exit status, never its text.
`tag-claims` reads the output `suite` captured rather than running the suite
itself, so it cannot be satisfied by a run nobody kept.

A failing check becomes a delta; a passing one becomes nothing. The delta's
name is the check's, so re-running finds the same delta, and the question it
raises in the human queue has the same id every run. Its phase comes from
facts: `brainstorm` until a person is asked and answers, then `planning`, and
never further, because re-running a check cannot establish that anybody acted.

What self-check cannot see: whether a check is the right check, or whether a
passing gate enforces anything. A check that reports success while enforcing
nothing reports success here too.

## The feedback loop

`feedback.once(harness, panel)` runs one pass; `FeedbackTopology.run` wraps
it. `FeedbackTopology` is a pipeline topology that implements `run` for one shape.
It is not registered: `TopologyRegistry` is keyed by `TopologyType` and
replaces silently, so a second class claiming `PIPELINE` would win or lose by
import order.

Its two stages look different ways, and are fixed rather than configured so the
loop cannot be reduced to the stage that always passes:

| Stage | Runs | Answers |
|---|---|---|
| `gates` | the harness's test suite | does this pass |
| `diagnostics` | `dossier.diagnostics`, in the panel's own environment | is a passing gate enforcing anything |

The stages are independent, so the diagnostics run when the suite is red. The
diagnostics run as a subprocess rather than an import, so the harness does not
depend on the panel.

**The ratchet.** Running checks repeatedly finds recurrences. A failure that no
check covers is a hole, and closing it means writing the check as well as the
fix; `unratcheted` lists those failures, the queue that makes the next run
stronger than this one.

The loop does not spend, schedule itself or close anything. Nothing writes a
cron entry (`DRAFT-no-unattended-spending.md`), and each failure becomes a
delta at `brainstorm` and stops.

## Intake

`intake` is one pipeline of four stages, each feeding the next: **unpack** an
export into one file per conversation, **index** each with an identity and a
digest, **read** the turns (naming a conversation that will not parse), and
**relate** each thread to the projects it is about. The step before it is a
person's: requesting the export and downloading it. `docs/threads.md`
describes each stage.
