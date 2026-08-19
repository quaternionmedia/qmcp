# 01 — The harness

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

qmcp is the harness: it invokes tools, runs pipelines, and holds the
human-in-the-loop queue. It addresses everything it does, which is what lets a
second application point at the same rows.

## Addresses

Every row qmcp records is named the same way, and the grammar is the corpus's
rather than this project's:

    >>> from qmcp.addresses import parse
    >>> address = parse("quaternionmedia/qmcp/invocation/4ea1e830-1963-4578")
    >>> address.owner, address.repo, address.kind
    ('quaternionmedia', 'qmcp', 'invocation')

Everything after the kind is the identifier, verbatim. It is split off last and
never re-parsed, because an identifier may contain slashes:

    >>> parse("quaternionmedia/qmcp/delta/feature/nested").id
    'feature/nested'

## What this harness has run

`qmcp dashboard` is qmcp's own view of itself. The JSON form is what crosses to
a control panel:

    >>> import sqlite3, tempfile
    >>> from pathlib import Path
    >>> from sqlmodel import SQLModel, create_engine
    >>> from qmcp.db.models import ToolInvocation
    >>> from qmcp.dashboard import build, to_dict

    >>> root = Path(tempfile.mkdtemp())
    >>> engine = create_engine(f"sqlite:///{(root / 'a.db').as_posix()}")
    >>> SQLModel.metadata.create_all(engine)
    >>> sorted(to_dict(build(root / "a.db")))
    ['by_status', 'by_tool', 'database', 'missing_tables', 'project', 'recent', 'schema', 'totals']

That list came out of the emitter. It used to be a sorted copy of the keys
written into this page, which is a sentence about itself: the example passed
whatever `to_dict` returned.

`totals` are counts over this harness's whole history. `recent` is an excerpt.
They are different claims and a consumer must not derive one from the other —
which is why the payload carries both rather than leaving it to be worked out.

## A count nobody took is not a count of zero

When a table this reads is absent, the count is `unknown` with a reason:

    >>> broken = root / "b.db"
    >>> connection = sqlite3.connect(str(broken))
    >>> _ = connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    >>> connection.commit(); connection.close()

    >>> to_dict(build(broken))["totals"]["invocations"]
    {'unknown': 'no tool_invocations table in this database'}

Zero would say somebody counted. The table count does not carry the difference
either — this database reports one like any other — so a consumer reading zero
here would record a harness with nothing wrong:

    >>> to_dict(build(broken))["totals"]["tables"]
    1

## Units of work cross as deltas

A step in a pipeline and a unit of work on a board are the same thing seen from
two ends. `qmcp deltas` emits the correspondence:

    >>> from qmcp.cookbook.delta import SCHEMA
    >>> SCHEMA
    1

The keys inside `delta` are *dossier's* column names, deliberately, so the
consumer writes `ProjectDelta(**payload["delta"])` and nothing translates in
between. `project_id` is the one column deliberately absent: it is an integer
primary key in the consumer's database, and this side cannot know it. The
`project` key beside the row carries `owner/repo`, which is what the consumer
resolves.

## Nothing here imports the other side

    >>> from pathlib import Path
    >>> sources = list(Path("qmcp").rglob("*.py"))
    >>> [path.name for path in sources
    ...  if "import dossier" in path.read_text(encoding="utf-8")]
    []

The seam is a schema. Importing across it would mean neither application ships
without the other, which is the opposite of a pair.

## Running the pair

    # here
    uv run qmcp dashboard --json > harness.json
    uv run qmcp deltas > deltas.json

    # in dossier
    uv run dossier harness ingest harness.json --write
    uv run dossier deltas ingest deltas.json --write

Ingesting reports a field that differs rather than overwriting it. Neither side
is authoritative: two independent observers of a moving system will differ, and
`governance/qm/records/DRAFT-a-disagreement-is-a-delta.md` is why picking a
winner by fiat discards the more interesting fact.
