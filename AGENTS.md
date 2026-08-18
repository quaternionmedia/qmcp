# AGENTS.md

This project is governed by the Quaternion Media constitution, vendored at
`governance/qm` (a submodule pinned to this project's `project/qmcp`
branch of that repo). If you are an AI coding agent opening this repo with
no other briefing, read this file fully before your first commit or edit.

## Before you do anything

**Run `/cowork` first.** It builds this session's brief from the repository —
the commit you are on, whether your pull request slot is free, what else is in
flight in this clone, which gates exist — instead of letting you inherit a
previous session's beliefs. Other sessions are likely running right now, in
other repositories, for the same reviewer;
`governance/qm/handbook/async-contract.md` is the set of rules that exist only
because of that, and it is short. `/preflight`, `/handoff` and `/status` close the same
loop at the other end.

**Read the corpus's committed status documents before re-deriving what they
hold.** `governance/qm/governance-status.yaml` and
`governance/qm/harness-status.json` each carry their own refresh command and
staleness budget inside the file; `governance/qm/handbook/generated-documents.md`
indexes them. Check the age before quoting a figure.

1. Read `governance/qm/README.md` and `governance/qm/PRINCIPLES.md` in full
   — the namespaces/precedence rules and the charter. Both are short.
2. This project's own decision records live in `governance/qm/adr/` — inside
   the submodule, on this project's own branch, not at this repo's root — as
   `ADR-NNNN` (numbered locally, at ratification) or `DRAFT-*.md` before
   ratification. A human ratifies; you draft.
3. **Everything you produce arrives as a pull request, opened as a draft.**
   Work on a branch and open a PR with `gh pr create --draft` — in this repo,
   and in the `governance/qm` submodule when you touch this project's records
   there. Never commit to, merge into, or push a shared branch directly, and
   never merge your own work, however small or mechanical the change looks.
   If you cannot open a PR, hand the branch back rather than merging it.
   **Draft is load-bearing, and never request a review.** A ready PR against a
   branch carrying `CODEOWNERS` requests review from those owners the moment it
   opens — you name no one, and the notification cannot be recalled. So "open a
   PR for human review", read literally, is the act of pulling a second person
   into work nobody has tested. A draft PR fires none of it. Add the person who
   asked for the work as **assignee**, which is also how you reach them when
   they authored the branch and GitHub therefore refuses a review request on
   it. Leaving draft is their call, made after their own testing.
   **Keep it to one open PR per repository, per contributor.** Not one per
   task. Two PRs that must merge in a given order are a sequencing puzzle
   handed to your reviewer. Land the upstream change first and let propagation
   carry it. `.github/workflows/one-pr-check.yml` enforces this; run
   `governance/qm/project-seed/ci/check_one_pr.py` before you open anything.
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
   relaxed — see `governance/qm/README.md`'s "Namespaces and precedence."
7. **Put explanation in one place**, per
   `governance/qm/handbook/style-guide.md`: inline comments carry clarifying
   facts about the code, `README.md` is a shallow onramp to what follows it,
   `docs/` is reference, and **every why goes to a retrospective in
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

**For a dev server you leave running, start it as a module:**

```
uv run python -m qmcp serve
```

not `uv run qmcp serve`. The console script is `Scripts/qmcp.exe`, and Windows
locks a running executable, so any `uv sync` that reinstalls the package fails
until the server is stopped.

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
