# The thread archive

qmcp reads conversations with assistants and turns what they settled into
units of work. A conversation is a *thread*; what it settled are *decisions*;
both become delta payloads a control panel can ingest. The code is in
`qmcp/threads/`, the commands are under `qmcp threads`, and the walkthroughs
`walkthrough/03-archiving-a-conversation-export.md` and
`walkthrough/04-the-thread-archive.md` run it end to end.

## Sources and their contract

A source implements three methods of `ThreadSource` (`qmcp/threads/base.py`)
and inherits the rest:

| Method | Spends | Returns |
|---|---|---|
| `survey()` | never; takes no budget | how many threads exist (`available`) and how many paid calls pulling them would take (`would_need`) |
| `fetch(ids, budget)` | against `budget`, one `spend()` per call | the named threads, or every one when `ids` is empty |
| `decisions(thread, budget)` | against `budget`, when recognition needs a model | what the thread settled |
| `deltas`, `relations` | inherited | payloads built by `to_delta`, the same shape for every source |

The rules the contract holds a source to:

- **Free first.** `survey` is the free pass of
  `DRAFT-no-unattended-spending.md` 3: it establishes what the paid work would
  cost without spending. A figure it cannot establish for free is
  `unknown(reason)`, never a guess.
- **Zero is a count.** `available=0` means nothing is there;
  `available=unknown(...)` means nothing was counted. The two never render
  alike.
- **Nothing is scheduled.** Every method that spends takes a `Budget` a
  person issued for one command. There is no retry and no backfill. A budget
  that runs out stops the pull, and a thread cut short has `partial` set, so
  its turn count is read as an excerpt's.
- **Every payload names its perspective.** A turn is a step and a decision is
  a unit of work, and which is which depends on the level a source speaks at
  (`DRAFT-granularity-is-a-perspective.md`). A source has no default
  perspective, and `to_delta` refuses an empty one.
- **A decision cites its turns.** `Decision.from_turns` is what a person
  checks the extraction against; `to_delta` refuses a decision that cites
  none. The contract can hold a source to the shape of an answer, and cannot
  tell whether the extracted decisions are the ones the thread made.

### A thread is itself a delta

A conversation is a unit of work: started for a reason, run, and at some point
finished with. So a thread is a delta of type `thread`, addressed
`<owner>/<repo>/delta/thread-<id>`, rather than an address kind of its own.
Each decision it settled is `part-of` it, a relation the composition
vocabulary already has, stated explicitly because a consumer must not infer
containment from a shared prefix (`DRAFT-deltas-compose.md` 5). The `thread-`
prefix keeps thread ids and decision names from colliding in one namespace.

The thread's payload is emitted first and always, so a conversation that
settled nothing still appears. Every payload opens at phase `brainstorm`:
recognising something does not establish that anybody acted on it, and a later
phase is a person's to set.

## The local sources

| Source | Reads | Perspective |
|---|---|---|
| `claude` | `~/.qmcp/threads/claude/*.json`, one conversation per file | `claude/thread` |
| `chatgpt` | `~/.qmcp/threads/chatgpt/*.json`, one conversation per file | `chatgpt/thread` |
| `claude-code` | the Claude Code session store, one `.jsonl` session per file | `claude-code/session` |

None of them spends or reaches the network, so `survey` reports real numbers
and `would_need: 0`. The cache lives under the user's home rather than in a
repository, so a conversation export is never committed as source.

**What the cache class does.** `LocalCacheSource` finds the files, parses
every one, and records each file it cannot read as an `Unreadable`, whose
count and names travel with the survey. The survey parses rather than counts
files, so it reports how many are threads and agrees with what `fetch`
returns.

**Decisions are marked, not inferred.** A local source finds a decision only
where a conversation marked one, on a line opening `DECISION:` or `DECIDED:`.
Reading an unmarked conversation for what it settled needs a model, which
would spend against the budget `decisions` takes.

**Exports are snapshots.** A web export is as current as the day it was
downloaded, and the survey's note says so. The session store is live, and its
note says that instead.

**ChatGPT's tree is flattened.** A ChatGPT export stores messages as a tree of
nodes; the conversation a person read is one path through it. The reader takes
every node with text in timestamp order, so regenerated or branched replies
appear as successive turns. Two assistants discussing the same work produce
two sets of deltas under two perspectives; `same-as` states that they are one
strand, and neither address is retired (`DRAFT-deltas-compose.md` 4).

**Claude Code sessions carry their work.** Each session records the branch
(`gitBranch`), the checkout (`cwd`) and the pull requests it produced
(`pr-link`). The reader addresses a session's deltas to the repository its
first `pr-link` names, falling back to the source's own project, and links the
thread to each branch and pull request by address. A subagent's file (a
sidechain) carries its parent's `sessionId`; it is read as its own thread,
addressed `<session>/agent-<id>` and `part-of` the session that launched it.
The format is private and may change: a malformed line is skipped, and a
session file with no turns is reported as unreadable.

## Importing an export

A model API does not expose the conversation history of `claude.ai` or
`chatgpt.com`. The route is each service's data export, requested by the
account holder in the web interface; that request is the person's act, and
what follows may be automated
(`DRAFT-acts-that-are-a-persons-by-constitution.md` clause 3).

    uv run qmcp threads import <export.zip>
    uv run qmcp threads import <conversations.json> --source claude

An export is a ZIP holding one `conversations.json`, an array of
conversations. `qmcp threads import` splits it into one file per conversation.
The service is detected by structure (`mapping` for ChatGPT, `chat_messages`
for Claude); a conversation carrying neither, or contradicting `--source`, is
reported and not imported. A conversation already cached is overwritten, and
the run reports each one that changed. A conversation with no id is named by
its position in the array; positional names shift when an earlier conversation
is deleted, and the run lists them.

## The index and the archive

    uv run qmcp threads index            # what is cached, and what changed
    uv run qmcp threads index --write    # write it
    uv run qmcp threads index --check    # re-derive and report drift

The index answers from one file what a survey answers by reading every file.
It has two layers:

| Layer | Holds | Derived from |
|---|---|---|
| cache | id, title, turn count, digest | the files on disk alone |
| archive | `first_seen`, `last_seen`, and every change | every index before this one |

The digest is over turn ids and text, not file bytes, so an exporter that
reformats its JSON does not register as a change. When a thread's digest
changes, the change is classified:

- **grew** -- the earlier turn ids are a proper prefix of the new ones. The
  ordinary case.
- **diverged** -- an earlier turn is gone, reordered or edited. An export
  disagreeing with an earlier record of itself is a tool changing its format,
  somebody editing history, or an id being reused.

The archive never overwrites and never drops: a divergence is kept with the
prior digest, and a thread the cache no longer holds keeps its entry. `--check`
compares only the cache layer, because the archive cannot be re-derived from
the files. The index describes one machine's cache and is not committed. It
counts what is cached, not what exists.

## The HTTP routes

`qmcp/threads/service.py` serves the archive from the same process as the
tools and the human queue, on loopback:

    GET  /v1/threads                       what is indexed
    GET  /v1/threads/diverged              exports disagreeing with an earlier record
    GET  /v1/threads/{source}/{id}         one thread, read from its store
    GET  /v1/threads/{source}/{id}/deltas  what it settled, as payloads and relations
    POST /v1/threads/reindex               rebuild the index, import nothing
    POST /v1/threads/import                unpack an export, then re-index

A caller reads what qmcp archived without importing qmcp or parsing a session
format itself. The listing answers from the index; with no index it is a 404
that says so, not an empty archive. Each row carries the thread's delta
address, built by the same naming rule the payloads use. The two `POST` routes
write only into the cache and its index; the caller names a file, and the
harness does the writing.

## Consolidation: which projects a thread is about

    uv run qmcp threads consolidate

Consolidation reads each thread against the corpus's roster of repositories
(`governance/qm/ci/workspace.yaml`) and states what it is about as relations.

**Evidence and claim are separate.** `mentions` is measured: where each
repository's name appears, as a whole word, in which turns. `about` is a claim
produced by a rule, and the rule travels with the answer and into each
relation's `stated_by`. Changing the rule changes the claim and leaves the
evidence as it was.

**The rule.** A repository counts when it is named in the title, or in at
least `min_turns` turns (two by default, a chosen starting point), and, when
`min_share` is given, in at least that share of the thread's turns. A fixed
turn count is strong evidence in a short thread and weak in a long one; the
share accounts for length and is off by default.

**Unknown is a value.** A thread that meets the rule for no project is
unknown, never assigned a default home. Most conversations in a personal
archive are about none of the roster.

**One project or several.** A thread about one project is `part-of` that
project's work. A thread about several `crosses` each: both must happen and
neither contains the other. A thread whose chosen projects are at least half
of the roster, and at least four, is a survey of the workspace -- a status
sweep, a roster review, a handoff -- and gets no relation; it is reported as
its own kind.

**Strength.** Each relation carries a `weight` and the evidence it was read
from. The weight is the thread's share of turns naming the project, at least
0.5 when the project is in the title, capped at 1.0, and None when nothing can
be measured. None is not zero: an unmeasured thread is not a weak correlation.
The owner in each address is derived from the checkout's identity.
