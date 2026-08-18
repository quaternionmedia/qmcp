"""What shape a database actually has, and whether the code agrees with it.

Two questions this answers that nothing else did, and the second is the one
that cost a day:

    what is in the database   `describe(path)`
    does it match the models  `drift(path)`

WHY `drift` EXISTS. A server started against a database whose schema had moved
on returned HTTP 500 on every read of one endpoint, with the reason --
`no such column: tool_invocations.execution_id` -- visible only in a traceback
in the log. Nothing asked the question before the first request did. An intact
database and a current one are different facts: `PRAGMA integrity_check` passes
happily on a schema the code cannot use.

WHAT IT COMPARES. Table names, column names per table, and nothing else. Not
types, not constraints, not indexes. That is deliberate: SQLite stores a
declared type it does not enforce, SQLModel and SQLAlchemy spell the same type
several ways, and a comparison that flagged `VARCHAR` against `TEXT` on every
run would be one nobody reads. Missing tables and missing columns are the
failures that actually take an endpoint down.

WHAT IT CANNOT SEE. A column that exists with the right name and the wrong
type, or a constraint that was dropped. It also cannot see whether the *data* is
right -- a migration that ran and mangled every row leaves a schema this reports
as perfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Alembic's own bookkeeping. It is in a migrated database and in no model, and
# reporting it as an extra table on every run would train a reader to ignore
# the extra-table line -- which is where a real orphan would then hide.
IGNORED_TABLES = frozenset({"alembic_version"})


@dataclass(frozen=True)
class Drift:
    """Where the database and the models disagree about shape."""

    missing_tables: list[str] = field(default_factory=list)
    extra_tables: list[str] = field(default_factory=list)
    missing_columns: dict[str, list[str]] = field(default_factory=dict)
    extra_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (self.missing_tables or self.extra_tables
                    or self.missing_columns or self.extra_columns)

    def lines(self) -> list[str]:
        out: list[str] = []
        for name in self.missing_tables:
            out.append(f"table `{name}` is in the models and not the database")
        for name in self.extra_tables:
            out.append(f"table `{name}` is in the database and not the models")
        for table, columns in sorted(self.missing_columns.items()):
            out.append(f"`{table}` is missing {', '.join(columns)} "
                       f"-- a read of that table will fail")
        for table, columns in sorted(self.extra_columns.items()):
            out.append(f"`{table}` has {', '.join(columns)}, which no model declares")
        return out


def describe(path: Path) -> dict[str, list[str]]:
    """Column names per table, read from the database itself."""
    import sqlite3

    if not path.is_file():
        raise FileNotFoundError(f"{path}: no database there.")
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        names = [
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            name: sorted(row[1] for row in connection.execute(f'PRAGMA table_info("{name}")'))
            for name in names
        }
    finally:
        connection.close()


def model_schema() -> dict[str, list[str]]:
    """Column names per table, for the *server's* models only.

    Scoped through `qmcp.db.registry`, not read off `SQLModel.metadata`. That
    metadata is global and also holds `qmcp.cookbook.persistence`'s tables,
    which belong to a per-run flow database and were never meant to be in this
    one -- so an unscoped read reports five tables as missing the moment
    anything has imported the cookbook.
    """
    from sqlmodel import SQLModel

    from qmcp.db.registry import server_table_names

    wanted = server_table_names()
    return {
        name: sorted(column.name for column in table.columns)
        for name, table in SQLModel.metadata.tables.items()
        if name in wanted
    }


def compare_schemas(models: dict[str, list[str]],
                    database: dict[str, list[str]]) -> Drift:
    """Shape difference, with alembic's bookkeeping table left out."""
    left = {k: v for k, v in models.items() if k not in IGNORED_TABLES}
    right = {k: v for k, v in database.items() if k not in IGNORED_TABLES}

    missing_columns: dict[str, list[str]] = {}
    extra_columns: dict[str, list[str]] = {}
    for name in sorted(set(left) & set(right)):
        absent = sorted(set(left[name]) - set(right[name]))
        surplus = sorted(set(right[name]) - set(left[name]))
        if absent:
            missing_columns[name] = absent
        if surplus:
            extra_columns[name] = surplus

    return Drift(
        missing_tables=sorted(set(left) - set(right)),
        extra_tables=sorted(set(right) - set(left)),
        missing_columns=missing_columns,
        extra_columns=extra_columns,
    )


def drift(path: Path) -> Drift:
    """Does this database have the shape the code expects?"""
    return compare_schemas(model_schema(), describe(path))
