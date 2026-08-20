"""Claude Code sessions, which are already local and carry more than a web export.

    ~/.claude/projects/<project>/*.jsonl

**THIS IS THE ANSWER TO "WHAT CAN THE API ADD", AND THE ANSWER IS THAT THE API
IS NOT WHERE THIS LIVES EITHER.** Neither Anthropic's API nor OpenAI's exposes
conversation history. What does exist, on this machine, with no credential and
no network, is Claude Code's own session store -- and for the purpose of
tracking work it is better than either web export, because of what it already
knows:

    gitBranch       which branch the session was working on
    cwd             which checkout
    pr-link         which pull requests the session produced, by number and
                    repository

A web export has none of that. It has the conversation and nothing about the
work. **These sessions already carry the joins**, which is what makes their
deltas addressable to the right project instead of to a default somebody chose.

WHAT THAT SETTLES. `plans/qmpm-standardisations.md` 1 asks whether a delta may
span owners, and until now a thread's deltas went to a `project` a person set,
because a conversation belongs to no repository. A Claude Code session does not
have that problem: it says which repository it was working in. So this source
derives the project per session rather than defaulting it, and falls back only
when a session says nothing.

FORMAT, AND WHAT IS NOT VERIFIED. One JSON object per line. The records this
reads are `user` and `assistant` for turns, `ai-title` for a name, and
`pr-link` for the join. Other record types exist -- file snapshots, queue
operations, attachments -- and are skipped rather than guessed at. **This is a
private on-disk format that may change without notice**, which is an argument
for reading it defensively and not for avoiding it: a malformed line is skipped
and counted, and a session that yields no turns is reported rather than dropped.

NOTHING HERE SPENDS AND NOTHING REACHES THE NETWORK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmcp.spend import Budget
from qmcp.threads.base import THREAD_LINK, Thread, Turn
from qmcp.threads.cache import LocalCacheSource

# Where Claude Code keeps sessions. Not under the thread cache: this reads a
# store somebody else owns and writes nothing into it.
SESSION_ROOT = Path.home() / ".claude" / "projects"

# The record types that carry a turn. Everything else in a session file is
# machinery -- file snapshots, queue operations, attachments -- and is skipped
# rather than rendered into prose that reads like conversation and is not.
TURN_TYPES = ("user", "assistant")

# `link_type` values for the joins a session already knows. These are addresses
# on the other side, which is what makes them worth carrying.
BRANCH_LINK = "branch"
PR_LINK = "pr"

# A SUBAGENT FILE CARRIES ITS PARENT'S `sessionId`, AND THAT IS THE TRAP.
# Most files under this root are sidechains -- a subagent's own turns, written
# beside the session that launched it. Keying a thread on `sessionId` alone
# collapsed them: many files became one id, each overwriting the last, and the
# index read every overwrite as the thread diverging. Seven "divergences" on a
# first index, none of them real.
#
# A false divergence is worse than a missed one. The archive keeps divergences
# precisely so somebody looks at them, and a reader who finds the first seven
# are noise stops looking at the eighth.
#
# So a sidechain is its own thread, addressed by both ids, and related `part-of`
# the session that launched it. Which is also the honest reading: from the
# session's perspective a subagent run is a step; from the subagent's own
# perspective it is a conversation. `DRAFT-granularity-is-a-perspective.md`
# says that question was always missing its subject.
AGENT_PREFIX = "agent"


class ClaudeCodeThreads(LocalCacheSource):
    """Sessions from Claude Code, read from their own store.

    A `LocalCacheSource` because the shape is the same -- a directory of files,
    free to read, one thread each. What differs is the format and, more
    usefully, that each session knows what work it was doing.
    """

    name = "claude-code"
    perspective = "claude-code/session"

    # A fallback only. Most sessions name their own repository, and `project_of`
    # prefers what the session says.
    project = "quaternionmedia/qmcp"

    # Not an export. These are written as the work happens, so the snapshot
    # caveat the cache class carries would be false here.
    caveat = ("A live store, written as sessions run. The newest is whatever "
              "was happening most recently.")

    def __init__(self, root: Path | None = None, **kw) -> None:
        super().__init__(root=root if root is not None else SESSION_ROOT, **kw)

    @property
    def directory(self) -> Path:
        return self.root

    def files(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob("**/*.jsonl"))

    # --- reading ------------------------------------------------------------

    def parse(self, document: Any, path: Path) -> Thread:
        """Never called: a session is lines, not one object.

        `LocalCacheSource.fetch` reads whole JSON files. This overrides `fetch`
        instead, and `parse` stays refusing so that a subclass which forgot to
        cannot fall through to a shape that does not apply.
        """
        raise NotImplementedError(
            "a Claude Code session is line-delimited; this source overrides "
            "fetch rather than parsing one object")

    def fetch(self, ids: list[str], budget: Budget) -> list[Thread]:
        """Read every session file. Spends nothing, and there is nothing to spend."""
        from qmcp.threads.cache import Unreadable

        wanted = set(ids)
        threads: list[Thread] = []
        self.unreadable = []
        self.skipped: list[str] = []
        self.context: dict[str, dict[str, Any]] = {}

        for path in self.files():
            try:
                records = _records(path)
            except OSError as exc:
                self.unreadable.append(Unreadable(path.name, str(exc)))
                continue

            thread, context = _session(records, path)
            if thread is None:
                # A `.jsonl` under this root is not necessarily a session --
                # workflow journals live here too. A file carrying no session
                # marker at all is a different kind of file, not a session that
                # failed to read, and calling it unreadable puts a scary count
                # on the page for something entirely fine.
                if any(record.get("sessionId") or record.get("type") in TURN_TYPES
                       for record in records):
                    self.unreadable.append(Unreadable(
                        path.name, "carries session records and no turn"))
                else:
                    self.skipped.append(path.name)
                continue
            if wanted and thread.id not in wanted:
                continue
            threads.append(thread)
            self.context[thread.id] = context
        return threads

    # --- what a session knows about its own work ----------------------------

    def project_of(self, thread: Thread) -> str:
        """Which repository this session's deltas belong to.

        Read from the session rather than defaulted. A `pr-link` names a
        repository outright and is preferred; without one the fallback is this
        source's own project, and that is a guess rather than a finding.
        """
        context = getattr(self, "context", {}).get(thread.id, {})
        repositories = context.get("repositories") or []
        return repositories[0] if repositories else self.project

    def deltas(self, thread: Thread, budget: Budget) -> list[dict[str, Any]]:
        """The session and what it settled, addressed to the repository it was in.

        Overridden so the project comes from the session. Everything else is
        the inherited shape -- a source that reshaped its own payload is one
        whose rows a consumer special-cases.
        """
        from qmcp.threads.base import to_delta, to_thread_delta

        project = self.project_of(thread)
        payloads = [to_thread_delta(thread, project=project,
                                    perspective=self.perspective)]
        payloads[0]["links"].extend(self._joins(thread, project))
        payloads += [
            to_delta(decision, thread, project=project,
                     perspective=self.perspective)
            for decision in self.decisions(thread, budget)
        ]
        return payloads

    def relations(self, thread: Thread, budget: Budget) -> list[dict[str, str]]:
        from qmcp.threads.base import PART_OF, relations_for, thread_name

        project = self.project_of(thread)
        found = relations_for(thread, self.decisions(thread, budget),
                              project=project)

        # A subagent's conversation is `part-of` the session that launched it.
        # Stated rather than derived from the shared id prefix -- a consumer
        # must not infer containment from two rows looking alike.
        context = getattr(self, "context", {}).get(thread.id, {})
        parent = context.get("parent")
        if parent:
            found.append({
                "source": f"{project}/delta/{thread_name(thread)}",
                "relation": PART_OF,
                "target": (f"{project}/delta/"
                           f"{thread_name(Thread(id=parent))}"),
            })
        return found

    def _joins(self, thread: Thread, project: str) -> list[dict[str, Any]]:
        """Branches and pull requests this session touched, as addresses.

        THE PART A WEB EXPORT CANNOT DO. A conversation that produced a pull
        request is linked to it by address, so the same row is reachable from
        the control panel's own view of that pull request.
        """
        context = getattr(self, "context", {}).get(thread.id, {})
        links: list[dict[str, Any]] = []
        for branch in sorted(context.get("branches") or []):
            links.append({"link_type": BRANCH_LINK, "target_id": None,
                          "target_name": f"{project}/branch/{branch}"})
        for repository, number in sorted(context.get("pulls") or []):
            links.append({"link_type": PR_LINK, "target_id": None,
                          "target_name": f"{repository}/pr/{number}"})
        return links


def _records(path: Path) -> list[dict]:
    """Every readable line. A malformed one is skipped, not fatal.

    A session file is appended to while a session runs, so the last line may be
    half-written at the moment this reads it. Refusing the whole file for that
    would make an in-progress session unreadable, which is the one it is most
    interesting to read.
    """
    found = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                found.append(record)
    return found


def thread_id(session_id: str | None, agent_id: str | None,
              fallback: str) -> str:
    """What identifies one file's conversation.

    A sidechain's id carries both, because its `sessionId` is its parent's and
    on its own would collide with every other subagent of the same session.
    """
    base = session_id or fallback
    return f"{base}/{AGENT_PREFIX}-{agent_id}" if agent_id else base


def _session(records: list[dict], path: Path) -> tuple[Thread | None, dict]:
    """One session file as a thread, plus what it knows about its work."""
    turns: list[Turn] = []
    branches: set[str] = set()
    repositories: list[str] = []
    pulls: set[tuple[str, int]] = set()
    title: str | None = None
    identifier: str | None = None
    agent: str | None = None
    sidechain = False
    started: str | None = None

    for record in records:
        kind = record.get("type")
        identifier = identifier or record.get("sessionId")
        agent = agent or record.get("agentId")
        sidechain = sidechain or bool(record.get("isSidechain"))

        if record.get("gitBranch"):
            branches.add(str(record["gitBranch"]))

        if kind == "ai-title" and record.get("aiTitle"):
            title = str(record["aiTitle"])

        elif kind == "pr-link":
            repository = record.get("prRepository")
            number = record.get("prNumber")
            if repository and number is not None:
                pulls.add((str(repository), int(number)))
                if str(repository) not in repositories:
                    repositories.append(str(repository))

        elif kind in TURN_TYPES:
            text = _text_of(record.get("message"))
            started = started or record.get("timestamp")
            turns.append(Turn(
                id=str(record.get("uuid") or record.get("requestId")
                       or f"{path.stem}-{len(turns)}"),
                role=str(kind),
                at=_maybe_str(record.get("timestamp")),
                text=text,
            ))

    if not turns:
        return None, {}

    return Thread(
        id=thread_id(identifier, agent, path.stem),
        title=title,
        started_at=_maybe_str(started),
        url=None,
        turns=tuple(turns),
    ), {
        "branches": branches,
        "repositories": repositories,
        "pulls": pulls,
        # What this file was a sidechain of, so the subagent's thread can be
        # related to the session that launched it rather than floating loose.
        "parent": str(identifier) if agent and identifier else None,
        "sidechain": sidechain or bool(agent),
    }


def _maybe_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _text_of(message: Any) -> str:
    """The text of a turn, from whichever shape the record uses.

    A block that is not text -- a tool call, an image -- is skipped rather than
    stringified. A tool invocation rendered as its repr becomes searchable prose
    that says nothing, and would then be scanned for decision markers it cannot
    contain.
    """
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""
