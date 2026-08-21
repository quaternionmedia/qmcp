"""Running one sweep: a worker per shape of work, chosen by the work.

    uv run qmcp sweep run <plan.json>

**THE TOPOLOGY IS NOT A SETTING.** Nothing here picks "one agent per repository"
or "one agent for all of them" in advance. Each share of the work arrives
carrying its shape, a worker is registered for each shape, and the topology is
whatever falls out -- nine parsers and six questions today, a different mix
tomorrow, without a line changing. A dispatcher that chose the topology first
would be deciding the answer before reading the question.

**A MODEL IS THE WRONG TOOL FOR A KNOWN EDIT.** Rewriting `>=0.115.0` to
`>=0.116.0` is a thing a parser does correctly every time; a model does it
slower, sometimes wrong, and -- on a paid endpoint -- for money. So `mechanical`
gets a parser. The GPU is for the shares a parser genuinely cannot read, and
those are the minority: measured on this organisation's real archive, nine of
twenty-four.

**A WORKER THAT IS NOT THERE IS REPORTED, NEVER SKIPPED.** With no model served,
`judgement` shares come back `needs a worker` and wait for a person. They do not
silently vanish from the sweep, which would leave fifteen of twenty-four
repositories looking finished because nothing tried them.

**NOTHING HERE MERGES OR DEPLOYS.** A worker prepares a change on a branch. The
approval and the tag are a person's by constitution --
`governance/qm/ci/attested-registry.yaml` -- and a dispatcher that could do them
would make the registry a description of what it chose not to do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# The shapes, as `dossier.sweep` states them. Copied across the seam rather than
# imported: the two repositories do not depend on each other, and the vocabulary
# is small enough to check. `tests/test_sweep.py` asserts they still agree.
MECHANICAL = "mechanical"
JUDGEMENT = "judgement"
HUMAN = "human"
UNKNOWN = "unknown"

DONE = "done"
NEEDS_WORKER = "needs a worker"
REFUSED = "refused"
FAILED = "failed"

_CONSTRAINT = re.compile(r"^\s*(?P<op>[<>=!~^]*)\s*(?P<version>[0-9][0-9A-Za-z.\-+]*)\s*$")


@dataclass(frozen=True)
class Outcome:
    """What happened to one share, and what it would take to finish it."""

    project: str
    state: str
    detail: str = ""
    edit: str | None = None
    """The change a worker prepared, if it prepared one. Never applied here."""

    @property
    def is_done(self) -> bool:
        return self.state == DONE


@dataclass
class Run:
    """One pass over a sweep's shares."""

    outcomes: list[Outcome] = field(default_factory=list)

    def by_state(self) -> dict[str, list[Outcome]]:
        found: dict[str, list[Outcome]] = {}
        for outcome in self.outcomes:
            found.setdefault(outcome.state, []).append(outcome)
        return found

    @property
    def ready(self) -> list[Outcome]:
        """Shares a person could approve. The batch."""
        return [o for o in self.outcomes if o.is_done]

    @property
    def waiting(self) -> list[Outcome]:
        """Shares nothing could do, which is a queue and not a silence."""
        return [o for o in self.outcomes if o.state == NEEDS_WORKER]

    def summary(self) -> str:
        counts = {state: len(rows) for state, rows in self.by_state().items()}
        parts = ", ".join(f"{n} {state}" for state, n in sorted(counts.items()))
        return f"{len(self.outcomes)} share(s): {parts}"


def mechanical_worker(share: dict[str, Any], to_version: str) -> Outcome:
    """Rewrite one constraint. Deterministic, and it costs nothing.

    Prepares the edit and does not apply it. The sweep is approved as a whole,
    so a worker that wrote to a repository before anybody had seen the batch
    would be committing to the part before the whole was decided.
    """
    declared = share.get("declared")
    project = share.get("project", "?")
    if not declared or "," in str(declared):
        return Outcome(project, REFUSED,
                       f"{declared!r} is not a single constraint")
    found = _CONSTRAINT.match(str(declared))
    if not found:
        return Outcome(project, REFUSED, f"{declared!r} did not parse")
    operator = found.group("op") or ">="
    return Outcome(project, DONE,
                   f"{declared} -> {operator}{to_version}",
                   edit=f"{operator}{to_version}")


def judgement_worker(share: dict[str, Any], to_version: str) -> Outcome:
    """The share a parser cannot read.

    **THIS IS DELIBERATELY NOT A MODEL CALL.** Whether a model runs here is a
    deployment decision: register a different worker for `JUDGEMENT` and the
    dispatcher does not change. With none registered, the honest answer is that
    nobody looked -- which is what this returns, and it is why these shares
    appear in a person's queue rather than in a total.
    """
    return Outcome(share.get("project", "?"), NEEDS_WORKER,
                   share.get("why") or "needs judgement")


def human_worker(share: dict[str, Any], to_version: str) -> Outcome:
    """Refused on purpose. Some acts change meaning when a machine does them."""
    return Outcome(share.get("project", "?"), REFUSED,
                   "a person's by constitution; never dispatched")


DEFAULT_WORKERS: dict[str, Callable[..., Outcome]] = {
    MECHANICAL: mechanical_worker,
    JUDGEMENT: judgement_worker,
    HUMAN: human_worker,
}


def run(shares: Iterable[dict[str, Any]], to_version: str,
        workers: dict[str, Callable[..., Outcome]] | None = None) -> Run:
    """Dispatch every share to the worker registered for its shape.

    A shape with no worker is `needs a worker`, named. That includes `unknown`,
    which has none on purpose: a share nothing could classify is not a share
    something should guess at.
    """
    registry = DEFAULT_WORKERS if workers is None else workers
    outcomes = []
    for share in shares:
        shape = share.get("shape", UNKNOWN)
        worker = registry.get(shape)
        if worker is None:
            outcomes.append(Outcome(
                share.get("project", "?"), NEEDS_WORKER,
                f"no worker registered for {shape!r}"))
            continue
        try:
            outcomes.append(worker(share, to_version))
        except Exception as exc:                  # noqa: BLE001
            # A worker that threw took one share with it, not the sweep. The
            # other twenty-three are still worth preparing, and the failure is
            # reported against the share it belongs to.
            outcomes.append(Outcome(share.get("project", "?"), FAILED,
                                    f"{type(exc).__name__}: {exc}"))
    return Run(outcomes=outcomes)


def branch_for(package: str, to_version: str) -> str:
    """The branch each repository's share is prepared on.

    `evolve/` because a sweep is org-level work arriving in a project, and the
    corpus's branch namespaces name that one -- `docs/ref/namespaces.md`. The
    same name in every repository on purpose: one sweep, one branch name, so a
    person checking twenty-four repositories is checking one thing.
    """
    return f"evolve/sweep-{package}-{to_version}"
