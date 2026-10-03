# AGENTS.md

This project is governed by the Quaternion Media constitution, vendored at
`governance/qm` (a submodule pinned to this project's `project/qmcp`
branch of that repo). If you are an AI coding agent opening this repo with
no other briefing, read this file fully before your first commit or edit.

## Before you do anything

**Establish four facts about this session before you write anything**, because
each has been got wrong here by inheriting a previous session's belief instead of
asking the repository:

1. **The commit you are working against**, and the branch.
2. **Which pull requests you already hold open, and how they stack** — small,
   atomic pull requests, a dependent one stacked on its parent as a draft:
   `python governance/qm/project-seed/ci/check_one_pr.py --repo <owner/name>`.
3. **What else is in flight in this clone** — a dirty tree you did not dirty, a
   sibling branch, an unpushed commit. Other sessions are likely running right
   now, in other repositories, for the same reviewer;
   `governance/qm/handbook/async-contract.md` is the set of rules that exist only
   because of that, and it is short.
4. **Which gates exist**, and what each cannot see:
   `python governance/qm/project-seed/ci/run_workflows_locally.py`.

Those are the invariants. **How** you gather them is yours to choose — read the
repository, run the scripts above, or use an adapter if one exists for your
tooling. `governance/qm/adapters/` holds any that do, each named for the product
it targets and none of them required. This file names no vendor, and neither
should anything you add to it.

**Read the corpus's committed status documents before re-deriving what they
hold.** `governance/qm/harness-status.json` carries its own refresh command and
staleness budget in a `reading:` block inside the file.
`governance/qm/governance-status.yaml` does **not** — for that one,
`governance/qm/handbook/generated-documents.md` is the only statement of its
refresh command and its 168-hour budget. Check the age before quoting a figure.

1. Read `governance/qm/PRINCIPLES.md` in full and the three invariants in `governance/qm/README.md`. For namespaces and precedence, see `governance/qm/docs/ref/namespaces.md` and `governance/qm/docs/ref/precedence.md`.
2. This project's own decision records live in `governance/qm/adr/` — inside
   the submodule, on this project's own branch, not at this repo's root — as
   `ADR-NNNN` (numbered locally, at ratification) or `DRAFT-*.md` before
   ratification. A human ratifies; you draft.
3. **Everything you produce arrives as a pull request, and the pull request is
   an audit record rather than a request for anyone's attention.** Work on a
   branch and open a PR — in this repo, and in the `governance/qm` submodule
   when you touch this project's records there — then **merge it yourself once
   every gate is green.** Your job is a default branch that is clean and
   working, entered through a pull request so the gates ran and the diff stays
   readable afterwards. **Never push a shared branch directly**: that is the
   one act that destroys the audit record.
   **The default branch is not a claim, so merging into it is not a release.**
   Per `governance/qm/records/DRAFT-version-tags-are-claims.md` §4, the default
   branch, a pull request and a local build are all drafts — they may be
   perfectly good and they assert nothing. **The two human gates are
   ratification, for what a record says, and the version tag, for what this
   project ships.** A `v*` tag asserts a human reviewed the change set, a human
   manually tested it against its real runtime, and deterministic automated
   validation passed. Keeping the default branch clean is what makes cutting
   one cheap.
   **Never request a review**, and add the person who asked for the work as
   **assignee**. Reviewers are named at the tag, by the human cutting it. A
   review request pulls a second person into work that asserts nothing yet, and
   against a branch carrying a live `CODEOWNERS` it fires the moment the PR
   opens — you name no one, and the notification cannot be recalled.
   **Draft means unfinished, or stacked on an open PR, and nothing else.** It
   is not a holding pen for finished work: a green PR left in draft against
   the default branch is a change that never landed.
   **One change per PR; parallel when independent, stacked when dependent.**
   Two changes that could land alone are two PRs, each ready against its base.
   A PR that needs another's work is cut from that branch, targets it as its
   base, and stays a draft until the one beneath it merges, however finished
   it is. When it does, retarget the next PR onto the target *before*
   deleting the merged branch — deleting it first closes every PR based on
   it. Never close a PR in favour of one that contains it
   (`governance/qm/handbook/async-contract.md` §1).
   `.github/workflows/one-pr-check.yml` refuses a ready stacked PR; run
   `governance/qm/project-seed/ci/check_one_pr.py` before you open anything.
   **The pull request body speaks as the contributor, to the world.** It is
   posted under a human's account and addresses nobody — a question or a
   handling instruction for the person who asked belongs in the session,
   never in the body (`governance/qm/handbook/async-contract.md` §3;
   `check_pr_voice.py` refuses the second person).
4. **Human-only contributorship applies to every commit you make here** (see
   `governance/qm/records/DRAFT-human-only-contributorship.md`): do not add
   yourself, your model name, or any co-author trailer naming an unmonitored
   address (e.g. a vendor `noreply@` address) to any commit. If your default
   tooling normally appends a `Co-Authored-By:` trailer, suppress it for
   this repo. Tool involvement is disclosed as a `Tools:` note where the
   artifact calls for one, never as a byline.
5. Follow the drafting-session handoff contract in
   `governance/qm/adr/README.md` before writing or amending any record.
6. A QM record may be tightened by this project's own records, never
   relaxed — see `governance/qm/docs/ref/precedence.md`.
7. **Put explanation in one place**, per
   `governance/qm/handbook/style-guide.md`: inline comments carry clarifying
   facts about the code, `README.md` is a shallow onramp to the docs, `docs/`
   is reference, and **every why goes to a retrospective in
   `governance/qm/perspectives/`**. A record's Context and Alternatives are
   the one exception, answering *why this decision* rather than *why it went
   that way*.
8. Banned in any pre-ratification `DRAFT-*.md` record: "previously",
   "originally", "earlier draft", "re-review", "renumber", "retroactive",
   "supersedes the ... (stance|finding)", "corrected". Drafts are rewritten
   in place, not narrated. The ADR lint enforces this over prose only, so
   quoting the list in a code span is fine.
9. **Establish a fact before asserting it, and check a signal before reading
   it.** A claim that something is broken, unsupported or behaves a certain way
   carries the command you ran and what it returned. Before reporting what a
   result means, name one other thing that would produce the same output — a
   tool version, a flag's semantics, stale local state, the working directory,
   a substring matching prose. An unexpected uniform result is a tooling fault
   until shown otherwise, and a check that has only ever been seen green has
   not been tested: break the thing it names and watch it go red.
10. **A claim about what facts *mean* names what else could produce them.**
    This is the sibling of the rule above and catches a different failure: the
    facts are all true and the sentence built from them is wrong. Name the
    ordinary cause before the interesting one — same author, same source, same
    tooling, same period — and state direction and date, because "A resembles
    B" is symmetric and the useful version rarely is. **A correction carries
    the same burden as the claim it replaces**: an overclaim gets caught by a
    reader who knows better, while a deflation reads as rigour, closes the
    topic, and can quietly delete something real. See
    `governance/qm/records/DRAFT-decision-record-discipline.md` §7 and §8.
11. **The scaffolding you measure with is part of the measurement.** Item 9 is
    the tool answering a different question than you asked. This is the tool
    being fine and the setup not — nothing errors, and the result describes your
    own scaffolding rather than the subject. Real instances: a diff run against
    files a redirect never wrote, reported as a hundred lines of drift when the
    truth was none; a working tree read after a merge that exited non-zero; file
    copies written through a text API that converted every line ending, so the
    diff was entirely encoding; a mutation test whose baseline was already
    failing, so it proved nothing in either direction. **Prefer the artefact you
    did not create** — read a document's own answer instead of recomputing one —
    and assert the intermediate: non-empty, exit zero, baseline green.
12. **A guard is not finished until someone has tried to route around it.**
    Breaking it and watching it go red proves it fires on the case you thought
    of; it cannot find the case you did not. Ask for a pass whose brief is to
    satisfy the check while doing the thing it forbids. A guard with a hole is
    worse than no guard — it is a green check standing exactly where a reader
    believes something is enforced. See the same record's §9 and §10.

13. **Show it by running it** — `show-it-by-running-it` of the charter, with
    `governance/qm/records/DRAFT-one-executable-walkthrough.md` as the record.
    This project's `walkthrough/` is one ordered set of pages that the ordinary
    test command executes: `walkthrough/NN-<slug>.md`, run by pytest with
    `--doctest-glob=*.md`. The example a reader reads is the example that ran.
    Do not write a second copy of a behaviour beside the code — no prose example
    that is not executed, no screenshot that is not a byproduct of a test
    asserting what the code did. What text cannot hold is emitted by that test
    and **recorded, never compared**: a test that diffs images fails on a font
    and gets switched off. Regeneration rides the command you already run before
    a pull request, so drift shows up as an uncommitted diff rather than as
    staleness nobody sees. A skip is not a pass, and a page that always skips is
    deleted.
14. **The workstation, the agent and the conversation are not the
    organisation** — record
    `governance/qm/records/DRAFT-what-is-not-the-organisation.md`, with
    `governance/qm/handbook/what-is-not-the-organisation.md` as the thing to
    do. A committed file states what is true of this project: not where a
    clone sits on one disk, which editor held it, what the tool driving a
    session did with its shell, or what anybody said. A decision enters as a
    decision, never as reported speech; a commit message describes the change
    and not the conversation behind it. The seed's `leak-check.yml` runs
    `check_leaks.py` on the mechanical part — a home path, a personal folder,
    a scratch path, a shared link; the rest is yours to read for before you
    push.

## One-time setup on a fresh clone (Windows)

`CLAUDE.md` and `.github/copilot-instructions.md` are real symlinks to this
file, not copies — POSIX checkouts resolve them with no setup. On Windows,
enable Developer Mode (Settings → For developers) and run `git config
core.symlinks true` once per clone, then `git checkout -- .` if the files
were already checked out before that. Skipping this doesn't break
anything — the files degrade to one-line pointers containing just the
target path — but it isn't the intended, tested experience; see the
IDE-integrated governance discovery record in `governance/qm/records/` for
what was actually verified.

<!-- Project-specific setup commands, test commands, and conventions belong
     below this line. -->

## Setting up, and running the suite

```
uv sync --all-extras
uv run pytest -q
```

**Install through the lock.** `uv sync`, never `uv pip install <package>`. An
unpinned install resolves a different stack: this repository has a recorded
instance where it dragged `starlette` forward and broke 52 tests, which was a
property of installing outside the lock rather than of the dependencies.

**The suite runs on a runner now**, in `.github/workflows/tests.yml`. Until it
did, five checks reported on every pull request and not one of them executed a
test -- so a green pull request meant the governance checks passed, and a
reader reasonably took it to mean the code worked. **The runner sees things
this platform cannot**: `import metaflow` fails on Windows at `import fcntl`,
so every flow test skips here and runs there. The first CI run found a broken
import in `examples/flows/approved_deploy.py` that no local run could reach.

**Every command is `uv run qmcp <command>`**, the server included
(`uv run qmcp serve`); `uv run qmcp --help` lists them. `python -m qmcp` and
`python qmcp` start the same CLI and are not documented as alternatives. On
Windows a running server holds `Scripts/qmcp.exe`, so a sync that reinstalls
qmcp fails with os error 32 until the server is stopped -- which a change to
qmcp's own dependencies needs anyway.

**Every gate runs locally, and `uv run qmcp preflight` is the command.** It
routes to `governance/qm/project-seed/ci/run_workflows_locally.py` with its
arguments unchanged (`--event`, `--ref`, `--base-ref`, `--head-ref`,
`--workflows`, and `--help`, which is the script's own) and exits with that
script's status, so a pull request is run through the workflows' actual steps
before anybody claims they are green. The hosted runs under
`.github/workflows/` mirror this and are not the only place the gates run. A
pass is evidence and not proof: `uses:` steps are not run and the working tree
stands in for them, and the runner image is not reproduced, so a step can pass
here and fail there. One step is red through this route on every uv-managed
checkout and green under a system interpreter: `reuse-lint.yml`'s `Install
REUSE` runs `python -m pip install reuse`, `uv run` puts the project's virtual
environment first on `PATH`, and a uv-created environment ships no `pip`. That
is the workflow depending on whichever interpreter is first on `PATH`, and the
repair belongs to the workflow -- a seed file, so to its copy under
`project-seed/ci/` in the governance repository and then here -- not to this
command, which decides nothing. The script's own docstring is the full list
of what it cannot see.

## The tag is the human gate, and nothing else is

**There are exactly two human gates in this organisation.** Ratification, for
what the constitution says, and **the version tag, for what this project
ships**. A pull request is neither. Per
`governance/qm/records/DRAFT-version-tags-are-claims.md`:

- **A version tag is a human act, never an automated or an assistant one.**
  Assistants prepare releases; a human cuts the tag.
- **A `v*` tag asserts three things**, all of which must hold at the tagged
  commit: a human **reviewed** the change set; a human **manually tested** it
  against its real runtime; and its **deterministic automated validation
  passed**.
- **Only deterministic tests count as that validation.** A test that retries,
  depends on timing, or skips when a fixture is absent contributes nothing.
  **A skipped test is an absent test that has announced itself** -- better
  than silence, and still not evidence. This matters here more than in most
  repositories: the flow tests skip on Windows and the shared address vectors
  skip until the governance pin carries them, and neither absence may be
  counted toward a tag.
- **Everything untagged carries no release claim.** `main`, a pull request and
  a local build are drafts. They may be perfectly good; they assert nothing,
  and nobody outside this project may read them as a release.

`.github/workflows/tag-claims.yml` checks what a pushed tag *says* -- that it is
annotated and carries `Reviewed-by`, `Manually-tested`, `Automated-gate` and
`Not-covered`. **It cannot check that any of it happened.** It reads an
annotation a human wrote, after the tag already exists. The gate is the person.

## Two commands that damage another session's work

Six sessions share one workstation here, and `governance/qm/handbook/async-contract.md`
is the contract that exists because of it. Two defaults in this repository
break it, both recorded as conflicts rather than fixed, so read this before
running either.

**`qmcp test` deletes the human gate queue.** `--clean` defaults to *true* and
unlinks `./qmcp.db` before the run and again after it, with no prompt. That
file holds pending human-in-the-loop requests -- somebody's unanswered
approvals. Afterwards an empty queue is indistinguishable from nobody having
asked for anything. Pass `--clean=False`, or point `QMCP_DATABASE_URL` at a
path of your own, before you run the suite.

**The server binds a default port.** `port` defaults to 3333 and
`database_url` to `sqlite+aiosqlite:///./qmcp.db`, which resolves against
whatever directory the process started in. So two clones have two different
queues that both answer `/health` identically, and `cookbook dev` prints
"MCP server already running" and *uses* whatever answered -- silently
borrowing another session's server and reporting it as success. Choose a port,
pass it explicitly, and ask the server what it is before believing anything
you measure against it.

## One read in this API is a write

`GET /v1/human/requests/{id}` assigns `status = EXPIRED` to a pending request
whose `expires_at` has passed, and the session commits on context exit, so the
transition is persisted *by the read*. It is the only thing that ever produces
`expired`, the list endpoint applies no expiry so the two endpoints disagree
about the same row, and expiry is terminal -- the answer POST then returns 410
and nothing un-expires a request.

Do not poll it. A loop that reads detail URLs to see who answered expires the
gates it is watching, and each expiry is a decision a human can no longer make.
