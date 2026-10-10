"""Which projects a conversation is about, and on what evidence.

    uv run qmcp threads consolidate

Each thread is read against the corpus's roster of repositories, and what it
is about is stated as relations in the vocabulary of
`governance/qm/records/DRAFT-deltas-compose.md`.

- `mentions` is evidence: where each repository's name appears, in which
  turns. `about` is a claim produced by a rule, and the rule is returned
  beside the answer. Changing the rule changes the claim and leaves the
  evidence as it is.
- A thread that meets the rule for no project is unknown, never assigned a
  default.
- A thread about one project is `part-of` that project's work; a thread about
  several is `crosses` each of them. A thread naming most of a substantial
  roster is a survey of the workspace and gets no relation.

`docs/threads.md` describes the rule and its thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from qmcp import identity
from qmcp.threads.base import Thread, thread_name

# The relation names, from the corpus. Restated as constants rather than typed
# at each use, and checked against the corpus in the tests.
PART_OF = "part-of"
CROSSES = "crosses"

# How many turns must name a repository before a thread counts as being about
# it. A chosen starting point, not a measured one: `about` takes it as a
# parameter and reports it in the rule.
DEFAULT_MIN_TURNS = 2

# A thread whose chosen projects reach this share of the roster is a survey of
# the workspace -- a status sweep, a roster review, a handoff -- rather than
# work crossing part of it. Surveys are reported as their own kind.
ROSTER_SHARE = 0.5

# A survey also names at least this many projects, so "two of four" is a
# crossing. Both conditions must hold.
MIN_SURVEY_PROJECTS = 4


@dataclass(frozen=True)
class Mention:
    """Where one repository's name appears in one thread. Measured."""

    project: str
    turns: tuple[str, ...]
    """Ids of the turns naming it -- so a reader can go and look."""

    total: int
    """How many times, across those turns."""

    in_title: bool = False

    of_turns: int = 0
    """How many turns the thread has. Carried so `share` is computable from
    the evidence alone, without the reader needing the thread beside it."""

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def share(self) -> float | None:
        """The fraction of the conversation that names this project.

        A fixed turn count is strong evidence in a short thread and weak in a
        long one; the share accounts for length.

        `None` when the thread's length is unknown.
        """
        if not self.of_turns:
            return None
        return self.turn_count / self.of_turns

    @property
    def strength(self) -> float | None:
        """How strongly this thread is about this project, 0 to 1.

        A claim derived from the measured `share`, `total` and `in_title`,
        which stay on the `Mention` beside it. The rule:

          * `share` is the base;
          * a project in the title is at least 0.5, because a title is a
            person's own summary of the conversation;
          * the result is capped at 1.0.

        `None` when nothing can be measured, which is not zero: an unmeasured
        thread is not a weak correlation.
        """
        share = self.share
        if share is None:
            return 0.5 if self.in_title else None
        if self.in_title:
            return max(0.5, min(1.0, share))
        return min(1.0, share)

    @property
    def basis(self) -> str:
        """What the strength was read from, in words a person can argue with."""
        if self.share is None:
            return "named in the title; the thread's length is unknown"                 if self.in_title else "nothing measurable"
        parts = [f"{self.turn_count} of {self.of_turns} turns "
                 f"({self.share:.1%})", f"{self.total} mention(s)"]
        if self.in_title:
            parts.append("and in the title")
        return ", ".join(parts)


@dataclass(frozen=True)
class Reading:
    """What a thread is about, the evidence for it, and the rule used.

    The three travel together, so a reading always carries the rule that
    produced it.
    """

    thread: str
    evidence: tuple[Mention, ...] = ()
    projects: tuple[str, ...] = ()
    rule: str = ""
    surveys_the_roster: bool = False
    """Names most of the roster: an inventory of the workspace, not a crossing."""

    @property
    def is_unknown(self) -> bool:
        """No project met the rule. Not the same as no evidence."""
        return not self.projects

    @property
    def relation(self) -> str | None:
        """How this thread joins what it is about, or None when nothing does.

        A roster survey gets none: being named in a list of everything is not
        a relation between two pieces of work.
        """
        if not self.projects or self.surveys_the_roster:
            return None
        return PART_OF if len(self.projects) == 1 else CROSSES


def roster(corpus: Path, owner: str | None = None) -> dict[str, str]:
    """Repository short name -> `<owner>/<name>`, from the corpus's own roster.

    Read from `governance/qm/ci/workspace.yaml`, which this repository embeds,
    so the harness does not depend on the control panel for the list.
    """
    import yaml

    document = yaml.safe_load(
        (corpus / "ci" / "workspace.yaml").read_text(encoding="utf-8"))
    # `workspace.yaml` declares bare repository names, so the owner is derived
    # from this checkout's identity unless one is given.
    account = owner or identity.owner()
    found: dict[str, str] = {}
    for entry in document.get("repositories") or []:
        name = entry.get("name")
        if not name:
            continue
        found[name] = f"{account}/{name}"
    return found


def _pattern(name: str) -> re.Pattern[str]:
    """A repository name as a whole word.

    Word-bounded, so `rad` does not match inside `gradient` or `radius`.
    """
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", re.IGNORECASE)


def mentions(thread: Thread, names: Iterable[str]) -> tuple[Mention, ...]:
    """Where each repository is named in this thread. Pure measurement."""
    found: list[Mention] = []
    title = thread.title or ""
    for name in names:
        pattern = _pattern(name)
        turns: list[str] = []
        total = 0
        for turn in thread.turns:
            hits = len(pattern.findall(turn.text or ""))
            if hits:
                turns.append(turn.id)
                total += hits
        in_title = bool(pattern.search(title))
        if total or in_title:
            found.append(Mention(project=name, turns=tuple(turns),
                                 total=total, in_title=in_title,
                                 of_turns=len(thread.turns)))
    return tuple(sorted(found, key=lambda m: (-m.turn_count, m.project)))


def about(thread: Thread, names: Iterable[str],
          min_turns: int = DEFAULT_MIN_TURNS,
          min_share: float | None = None) -> Reading:
    """Which projects this thread is about, and why.

    A repository named in the title counts on that alone, because the title
    is a person's own summary of the conversation. Otherwise it must be named
    in at least `min_turns` turns and, when `min_share` is given, in at least
    that share of them.
    """
    known = list(names)
    evidence = mentions(thread, known)

    def counts(m: Mention) -> bool:
        if m.in_title:
            return True
        if m.turn_count < min_turns:
            return False
        # `min_share` is off by default; `Mention.share` says what it adds.
        if min_share is not None and (m.share or 0) < min_share:
            return False
        return True

    chosen = tuple(m.project for m in evidence if counts(m))
    surveys = (len(chosen) >= MIN_SURVEY_PROJECTS
               and len(chosen) >= len(known) * ROSTER_SHARE)
    rule = f"named in the title, or in at least {min_turns} turns"
    if min_share is not None:
        rule += f", and in at least {min_share:.0%} of them"
    if surveys:
        rule += (f"; and it named {len(chosen)} of {len(known)} repositories, "
                 f"which reads as a survey of the workspace rather than work "
                 f"that crosses them")
    return Reading(thread=thread.id, evidence=evidence,
                   projects=chosen, rule=rule, surveys_the_roster=surveys)


def relations_for(thread: Thread, reading: Reading, *,
                  project_of: dict[str, str]) -> list[dict[str, Any]]:
    """The relation payloads this reading implies.

    A relation joins addresses rather than rows, so its target may name a
    delta that does not exist yet. The source is the harness's delta for the
    thread; the target is the repository's own work.

    `stated_by` carries the rule that produced the relation.
    """
    # `relation`, not `is_unknown`: a survey has projects and no relation.
    relation = reading.relation
    if relation is None:
        return []

    # The harness that pulled the thread owns the delta, and which harness that
    # is comes from the checkout rather than from a literal.
    source = f"{identity.this_project()}/delta/{thread_name(thread)}"
    out: list[dict[str, Any]] = []
    for project in reading.projects:
        owner_repo = project_of.get(project)
        if not owner_repo:
            continue
        out.append({
            "schema": 1,
            "source": source,
            "relation": relation,
            "target": f"{owner_repo}/delta/the-work",
            "stated_by": f"qmcp threads consolidate: {reading.rule}",
            # The weight is a reading; the evidence it was read from travels
            # with it.
            "weight": next((m.strength for m in reading.evidence
                            if m.project == project), None),
            "evidence": [
                {"project": m.project, "turns": m.turn_count,
                 "of_turns": m.of_turns, "share": m.share,
                 "mentions": m.total, "in_title": m.in_title,
                 "strength": m.strength, "basis": m.basis}
                for m in reading.evidence if m.project == project
            ],
        })
    return out


@dataclass
class Consolidation:
    """Every thread's reading, and what the set of them looks like."""

    readings: list[Reading] = field(default_factory=list)

    @property
    def unknown(self) -> list[Reading]:
        return [r for r in self.readings if r.is_unknown]

    @property
    def crossing(self) -> list[Reading]:
        """Threads that genuinely cross projects, surveys excluded."""
        return [r for r in self.readings
                if len(r.projects) > 1 and not r.surveys_the_roster]

    @property
    def surveys(self) -> list[Reading]:
        """Threads that took stock of the workspace rather than working in it."""
        return [r for r in self.readings if r.surveys_the_roster]

    def by_project(self) -> dict[str, list[str]]:
        """Which threads each project has, which is the overlap made visible."""
        found: dict[str, list[str]] = {}
        for reading in self.readings:
            for project in reading.projects:
                found.setdefault(project, []).append(reading.thread)
        return found

    def summary(self) -> str:
        """One paragraph, counting what is known and what is not.

        The count of threads about no project is always reported.
        """
        placed = len(self.readings) - len(self.unknown)
        lines = [
            f"{len(self.readings)} thread(s) read: {placed} about a project, "
            f"{len(self.unknown)} about none of them.",
            f"{len(self.crossing)} cross more than one project, and "
            f"{len(self.surveys)} survey the workspace rather than crossing it.",
        ]
        for project, threads in sorted(self.by_project().items(),
                                       key=lambda kv: -len(kv[1])):
            lines.append(f"  {project:<22} {len(threads)}")
        return "\n".join(lines)


def consolidate(threads: Iterable[Thread], names: dict[str, str],
                min_turns: int = DEFAULT_MIN_TURNS) -> Consolidation:
    """Read every thread against the roster."""
    return Consolidation(
        readings=[about(thread, names, min_turns) for thread in threads])
