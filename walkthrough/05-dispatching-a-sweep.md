# 05 — Dispatching a sweep

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

The other half is the control panel's:
`../dossier/walkthrough/06-a-sweep-across-the-org.md` finds who declares a
dependency and works out what shape each repository's share is. This page takes
those shares and runs them.

## The topology is not a setting

Nothing here chooses "one agent per repository" or "one agent for all of them".
A worker is registered per *shape of work*, and the topology is whatever falls
out of the shares that arrive.

    >>> from qmcp.sweep import run
    >>> shares = (
    ...     [{"project": f"org/m{i}", "shape": "mechanical",
    ...       "declared": ">=0.100.0"} for i in range(9)]
    ...     + [{"project": f"org/j{i}", "shape": "judgement",
    ...         "why": "no version declared"} for i in range(6)])
    >>> done = run(shares, "0.116.0")
    >>> len(done.ready), len(done.waiting)
    (9, 6)

Nine parsers and six questions today. A different mix tomorrow, without a line
changing here. A dispatcher that chose the topology in advance would be deciding
the answer before reading the question.

## A model is the wrong tool for a known edit

Rewriting `>=0.115.0` to `>=0.116.0` is something a parser does correctly every
time. A model does it slower, sometimes wrong, and on a paid endpoint for money.

    >>> from qmcp.sweep import mechanical_worker
    >>> outcome = mechanical_worker({"project": "org/a", "declared": "~=0.95"},
    ...                             "0.116.0")
    >>> outcome.state, outcome.edit
    ('done', '~=0.116.0')

**It prepares the edit and does not apply it.** The sweep is approved as a
whole, so a worker that wrote to a repository before anybody had seen the batch
would be committing to the part before the whole was decided.

It refuses what it cannot read, rather than guessing:

    >>> mechanical_worker({"project": "org/b",
    ...                    "declared": "<1.0.0,>=0.92.0"}, "0.116.0").state
    'refused'

That is the failure a mechanical tool is supposed to be incapable of: a ceiling
somebody put there on purpose, flattened to one number by something that did not
understand it.

## A worker that is not there is reported

On the machine this was written on there is a GPU and no model served. So
`judgement` shares come back unanswered — by name, in a queue:

    >>> waiting = run([{"project": "org/c", "shape": "judgement",
    ...                 "why": "no version declared"}], "0.116.0")
    >>> waiting.outcomes[0].state
    'needs a worker'
    >>> waiting.outcomes[0].detail
    'no version declared'

They do not silently vanish from the sweep. Fifteen of twenty-four repositories
disappearing would leave the other nine looking like the whole job.

**Registering a worker changes what runs, not the dispatcher:**

    >>> from qmcp.sweep import Outcome, DONE
    >>> answered = run([{"project": "org/c", "shape": "judgement"}], "0.116.0",
    ...     workers={"judgement":
    ...              lambda share, to: Outcome(share["project"], DONE, "read it")})
    >>> len(answered.ready)
    1

Whether that worker is a local model, a remote one, or a person reading a diff
is a deployment decision. It is not a design one, and nothing above knows which
it got.

## A shape nobody registered is not guessed at

    >>> run([{"project": "org/x", "shape": "unknown"}], "0.116.0").outcomes[0].state
    'needs a worker'

`unknown` has no worker on purpose. A share nothing could classify is not a
share something should have a go at.

## What is never dispatched

    >>> run([{"project": "org/y", "shape": "human"}], "0.116.0").outcomes[0].state
    'refused'

Approving the batch, merging, and cutting the tag are a person's by
constitution — `governance/qm/ci/attested-registry.yaml`. Some acts change what
they assert when a machine performs them, and a dispatcher able to do them would
make that registry a description of what it chose not to do.

## One bad share does not take the sweep

    >>> def explodes(share, to_version):
    ...     raise RuntimeError("the manifest was a directory")
    >>> mixed = run([{"project": "org/a", "shape": "mechanical",
    ...               "declared": ">=0.1.0"},
    ...              {"project": "org/bad", "shape": "judgement"}], "0.116.0",
    ...             workers={"mechanical": mechanical_worker,
    ...                      "judgement": explodes})
    >>> sorted((o.project, o.state) for o in mixed.outcomes)
    [('org/a', 'done'), ('org/bad', 'failed')]

The failure is reported against the share it belongs to. The other twenty-three
are still worth preparing.

## One branch name everywhere

    >>> from qmcp.sweep import branch_for
    >>> branch_for("fastapi", "0.116.0")
    'evolve/sweep-fastapi-0.116.0'

`evolve/` because a sweep is org-level work arriving in a project —
`governance/qm/docs/ref/namespaces.md`. The same name in every repository on
purpose: a person checking twenty-four repositories is checking one thing, and
one open pull request per repository per contributor is the corpus rule that
makes that the only workable shape.
