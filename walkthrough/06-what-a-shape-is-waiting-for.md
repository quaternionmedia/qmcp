# 06 — What a shape is waiting for, and the journey through it

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

**The problem.** `orchestration plane` said what each topology *would do* —
whether it spends, whether it decides, whether it is refused here — and never
said what it was short of. A reader asking "can I run this now" got a status
word and had to work the rest out from prose. `dossier` had the same gap in its
views the same week, and the fix is the same one: a shape declares what it
needs, and every need names what supplies it.

## A status and a need answer different questions

    >>> from qmcp.orchestration import (NEEDS, PLANE, by_type, runnable_now,
    ...                                 unmet)
    >>> from qmcp.agentframework.models.enums import TopologyType

`status` says whether anybody built it. `needs` says what a built one still
wants from the caller. Both are asked, because reporting only the first would
tell somebody to write code they already have:

    >>> ensemble = by_type()[TopologyType.ENSEMBLE]
    >>> ensemble.status
    'brainstorm'
    >>> [need.key for need in ensemble.needs]
    ['build', 'budget']

Supply the build, and what remains is the real answer:

    >>> [need.key for need in unmet(ensemble, built=True)]
    ['budget']
    >>> unmet(ensemble, built=True, budget=5)
    ()

## The journey: nothing supplied, then enough

With an empty hand, nothing runs:

    >>> runnable_now()
    []

Two workers is what a router and a consensus each need — one worker is not a
consensus:

    >>> [t.value for t in runnable_now(workers=2)]
    ['delegation', 'crosscheck']

And a budget adds nothing to those two, because neither spends:

    >>> [t.value for t in runnable_now(workers=2, budget=5)]
    ['delegation', 'crosscheck']

## The one a caller cannot supply

`council` is refused here: its arbiter takes the final decision when consensus
fails, and that is an act `ci/attested-registry.yaml` reserves for a person.
**Give it everything and it is still short**, which is the property worth
having — a refusal that an argument could lift would not be one:

    >>> council = by_type()[TopologyType.COUNCIL]
    >>> [n.key for n in unmet(council, built=True, budget=99, workers=9, model=True)]
    ['person']

The need says so rather than leaving it to be inferred:

    >>> "a person decides" in council.needs[0].supplied_by
    True

## Every need names what supplies it

A shape that cannot run and does not say what would make it run leaves a reader
where the silence did. So each one carries its remedy:

    >>> every = [(c.topology.value, n.supplied_by) for c in PLANE for n in c.needs]
    >>> all(remedy.strip() for _, remedy in every)
    True

Every shape declares at least one, and none is left without a remedy. The
number of needs is not asserted here: it is whatever the plane declares today,
and pinning it would fail the moment somebody adds a shape while nothing was
wrong.

    >>> all(c.needs for c in PLANE)
    True

And no need is declared that the vocabulary does not hold — a key outside it
would be one no resolver has a branch for, so it would silently never block:

    >>> sorted({n.key for c in PLANE for n in c.needs} - set(NEEDS))
    []

## What this page does not claim

That a shape whose needs are met will *work*. `unmet` reads declarations; it
runs nothing. A `RUNS` topology with every need supplied can still fail on the
work it is given, and that is a different report — the one the invocation
record carries.
