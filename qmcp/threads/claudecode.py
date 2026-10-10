"""Claude Code sessions, which are already local and carry more than a web export.

    ~/.claude/projects/<project>/*.jsonl    (leaks: allow the fixed location this module reads, not one machine's path)

The session store is local and needs no credential. Besides the conversation,
each session records the work it was doing:

    gitBranch       which branch the session was working on
    cwd             which checkout
    pr-link         which pull requests the session produced, by number and
                    repository

So this source derives each session's project from the session, and falls
back to its own `project` only when a session names no repository.

The format is one JSON object per line. The records read are `user` and
`assistant` for turns, `ai-title` for a name and `pr-link` for the join; other
record types are skipped. It is a private on-disk format that may change
without notice: a malformed line is skipped, and a session file that yields
no turns is reported as unreadable.

Nothing here spends and nothing reaches the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qmcp import identity
from qmcp.spend import Budget
from qmcp.threads.base import THREAD_LINK, Thread, Turn
from qmcp.threads.cache import LocalCacheSource

# Where Claude Code keeps sessions. Not under the thread cache: this reads a
# store another tool owns and writes nothing into it.
SESSION_ROOT = Path.home() / ".claude" / "projects"

# The record types that carry a turn. Everything else in a session file is
# machinery -- file snapshots, queue operations, attachments -- and is skipped
# rather than rendered into prose that reads like conversation and is not.
TURN_TYPES = ("user", "assistant")

# `link_type` values for the joins a session already knows. These are addresses
# on the other side, which is what makes them worth carrying.
BRANCH_LINK = "branch"
PR_LINK = "pr"

# A subagent file (a sidechain) carries its parent's `sessionId`. Keyed on that
# alone, every sidechain of a session would share one id, and the index would
# read each as the thread diverging. So a sidechain is its own thread,
# addressed by both ids (`<session>/agent-<id>`) and `part-of` the session that
# launched it.
AGENT_PREFIX = "agent"


class ClaudeCodeThreads(LocalCacheSource):
    """Sessions from Claude Code, read from their own store.

    A `LocalCacheSource`: a directory of files, free to read, one thread
    each. It overrides the format and derives each session's project.
    """

    name = "claude-code"
    perspective = "claude-code/session"

    # A fallback only: `project_of` prefers the repository a session names.
    # Derived from the checkout; see `qmcp.identity`.
    project = identity.this_project()

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

        `LocalCacheSource.fetch` reads whole JSON files; this source overrides
        `fetch`, and `parse` raises so nothing falls through to that shape.
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
                # A `.jsonl` under this root is not necessarily a session;
                # workflow journals live here too. A file with no session
                # marker is skipped, not counted as unreadable.
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

        The first repository a `pr-link` names; without one, this source's own
        `project`, which is a fallback rather than a finding.
        """
        context = getattr(self, "context", {}).get(thread.id, {})
        repositories = context.get("repositories") or []
        return repositories[0] if repositories else self.project

    def deltas(self, thread: Thread, budget: Budget) -> list[dict[str, Any]]:
        """The session and what it settled, addressed to the repository it was in.

        Overridden so the project comes from the session and the thread's
        payload carries the branch and pull request links. The payload shape is
        the inherited one.
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

        # A subagent's conversation is `part-of` the session that launched it,
        # stated because a consumer must not infer it from the shared prefix.
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

        A pull request link is `<owner>/<repo>/pr/<number>`, so the thread is
        reachable from the control panel's view of that pull request.
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

    A session file is appended to while a session runs, so its last line may
    be half-written; skipping it keeps a running session readable.
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

    A sidechain's id carries both, because its `sessionId` is its parent's
    and would collide with every other subagent of that session.
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

    A block that is not text, such as a tool call or an image, is skipped
    rather than rendered as its repr.
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
