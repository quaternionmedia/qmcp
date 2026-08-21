"""Which projects a conversation is about, and on what evidence.

    uv run qmcp threads consolidate

**THE OVERLAP IS THE POINT.** Two hundred threads and sixteen repositories, and
almost every thread is about at least one of them -- but nothing said so, so a
project's board showed its branches and its pull requests and none of the
conversations that produced them. This works out the overlap and states it as
relations, which is the vocabulary
`governance/qm/records/DRAFT-deltas-compose.md` already settled.

**EVIDENCE AND CLAIM ARE SEPARATE, AND NEITHER IS DERIVED FROM THE OTHER.**
`mentions` counts where a repository's name appears and in which turns: that is
measured, and it is what a person checks. `about` decides which projects a
thread is *about*: that is a claim, produced by a rule, and the rule is named in
the output beside its answer. Changing the rule changes the claim and leaves
every piece of evidence exactly as it was, which is the property that makes the
claim arguable rather than authoritative.

**A THREAD ABOUT NOTHING IS `unknown`, NOT UNASSIGNED TO A DEFAULT.** Most
conversations in a personal archive are not about this organisation at all. A
consolidator that guessed a home for them would fill sixteen boards with
somebody's holiday planning, and the guess would be indistinguishable from a
finding. Unknown is a value here as it is everywhere else in this corpus.

**PART-OF FOR ONE, CROSSES FOR SEVERAL.** A thread about one project is
`part-of` that project's work: closing the work requires closing what the thread
settled. A thread about two is `crosses` -- both must happen, they interact, and
neither contains the other. Reaching for `part-of` twice would claim a
conversation belongs wholly to each of two repositories, which is the shape that
made a hierarchy the wrong model in the first place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from qmcp.threads.base import Thread, thread_name

# The relation names, from the corpus. Restated as constants rather than typed
# at each use, and checked against the corpus in the tests.
PART_OF = "part-of"
CROSSES = "crosses"

# How many turns must name a repository before a thread counts as being about
# it. One passing mention is a citation; this is the difference between a
# conversation that referenced a repository and a conversation about it.
#
# A THRESHOLD IS A CLAIM AND IT IS STATED, NOT HIDDEN. Two is a starting point
# somebody chose, not a discovery -- which is why `about` reports the rule it
# used beside the answer, and why the number is a parameter rather than a
# literal buried in a comparison.
DEFAULT_MIN_TURNS = 2

# Above this share of the roster, a thread is describing the workspace rather
# than crossing part of it.
#
# **AN INVENTORY IS NOT A CROSSING, AND THEY MEASURE THE SAME.** `crosses` means
# both must happen, they interact at one point, and neither contains the other.
# A session that lists every repository -- a status sweep, a roster review, a
# handoff -- names most of them in most of its turns, and comes out of the rule
# above looking like a conversation that crossed eleven projects at once. It did
# not; it enumerated them.
#
# Measured on the real archive: 33 threads crossed more than one project and the
# largest named 11 of 13, which is what prompted this. Reported as its own kind
# rather than dropped, because a workspace sweep is a real thing a person did
# and the boards should be able to find it -- just not as eleven crossings.
ROSTER_SHARE = 0.5

# And never fewer than this many, whatever the share works out to. A share
# alone reads "two of four" as a survey, which it plainly is not -- a survey is
# most of a *substantial* roster, and two projects is the smallest possible
# genuine crossing. Both conditions have to hold.
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

        **THE MEASURE THE ABSOLUTE THRESHOLD WAS MISSING, FOUND BY ASKING A
        REAL QUESTION.** Pointed at `codecartographer`, the rule "named in at
        least 2 turns" admitted a 5,153-turn session that named it in 18 (0.3%)
        and a 1,885-turn session that named it twice (0.1%) -- alongside a
        63-turn session that named it in 8 (12.7%), which is plainly about it.
        Two turns is a lot of evidence in a short thread and none at all in a
        long one.

        `None` when the thread's length is unknown: a share computed against a
        zero would be a number this made up.
        """
        if not self.of_turns:
            return None
        return self.turn_count / self.of_turns

    @property
    def strength(self) -> float | None:
        """How strongly this thread is about this project, 0 to 1.

        **A CLAIM, DERIVED FROM MEASURED PARTS, AND THE PARTS TRAVEL WITH IT.**
        `share`, `total` and `in_title` are counted; this is a reading of them,
        and a renderer drawing a thick line is drawing this reading rather than
        a fact. Every component stays on the `Mention` so somebody can disagree
        with the combination without losing the evidence.

        The rule, stated rather than tuned:

          * `share` is the base -- the fraction of the conversation that names
            the project, which is the measure the absolute turn threshold was
            missing.
          * a project in the **title** is at least 0.5, because a title is a
            person's own summary of what the conversation was about and
            outranks any proportion.
          * `share` is capped at 1.0 rather than allowed to exceed it.

        `None` when nothing can be measured -- and **`None` is not zero**. A
        thread whose length is unknown is not a weak correlation; it is an
        unmeasured one, and a window that drew it as a hairline would be
        asserting weakness nobody established.
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

    The three are carried together on purpose. A reading passed around without
    its rule is a verdict, and a verdict is what a person cannot argue with.
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

        A roster survey gets none. Every relation this vocabulary has says
        something specific about two pieces of work, and "was mentioned in a
        list of everything" is not one of them.
        """
        if not self.projects or self.surveys_the_roster:
            return None
        return PART_OF if len(self.projects) == 1 else CROSSES


def roster(corpus: Path) -> dict[str, str]:
    """Repository short name -> `<owner>/<name>`, from the corpus's own roster.

    Read from `governance/qm`, which this repository already embeds. Asking the
    control panel would make the harness depend on the panel for a list the
    corpus publishes, and the two would then disagree the moment one was
    fetched and the other was not.
    """
    import yaml

    document = yaml.safe_load(
        (corpus / "ci" / "workspace.yaml").read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for entry in document.get("repositories") or []:
        name = entry.get("name")
        if not name:
            continue
        found[name] = f"quaternionmedia/{name}"
    return found


def _pattern(name: str) -> re.Pattern[str]:
    """A repository name as a whole word.

    Word-bounded because `rad` is three letters that live inside `gradient`,
    `radius` and `radical`, and a substring match would report a conversation
    about colour as a conversation about the menu library.
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

    THE TITLE COUNTS FOR MORE THAN A MENTION. A repository named in the title is
    what the conversation was called, which is a person's own summary of what it
    was about -- stronger evidence than any number of passing references, and
    the one case where a single occurrence is enough.
    """
    known = list(names)
    evidence = mentions(thread, known)

    def counts(m: Mention) -> bool:
        if m.in_title:
            return True
        if m.turn_count < min_turns:
            return False
        # `min_share` is off by default, so existing readings keep their
        # meaning. Turning it on is a decision somebody makes, and the reason
        # is in `Mention.share`: two turns is strong evidence in an eighteen
        # turn thread and negligible in an eighteen hundred turn one.
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

    Addresses on both sides, because a relation joins addresses rather than
    rows -- so one of these may name a delta that does not exist yet, and that
    is allowed. The thread's own address is the harness's; the project's is the
    repository's own work.

    `stated_by` carries the rule. A relation somebody finds later and disagrees
    with should say what produced it without anybody having to guess.
    """
    # `relation`, not `is_unknown`: a survey has projects and no relation, so
    # a check on emptiness alone would emit eleven relations for a thread that
    # crossed nothing. One condition, and it is the one that decides.
    relation = reading.relation
    if relation is None:
        return []

    source = f"quaternionmedia/qmcp/delta/{thread_name(thread)}"
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
            # THE WEIGHT AND ITS PARTS TRAVEL TOGETHER. A consumer drawing a
            # thick line is drawing a reading; sending the number without what
            # it was read from would make that reading unarguable.
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

        The unknown count is reported first and never omitted: a consolidator
        that printed only its hits would read as though it had placed
        everything.
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
