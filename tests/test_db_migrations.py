"""The migration chain, and the claim a stamped baseline has to earn.

STAMPING IS AN ASSERTION. `alembic stamp` writes a revision into a database
without running anything -- it says "this database already has that shape". The
existing database was stamped that way, and the only thing that makes it true is
this: a database built from nothing by `alembic upgrade head` has the same shape
as the one the models describe. If those two ever differ, the stamp was a lie
and every later migration is being applied to a shape nobody verified.

That is what `test_a_fresh_upgrade_reproduces_the_models_schema` checks, and it
is the reason this file exists rather than a note in a handoff.

NOTHING HERE TOUCHES THE CONFIGURED DATABASE. Every test migrates a fresh file
in a temporary directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from qmcp.db.schema import IGNORED_TABLES, compare_schemas, describe, model_schema

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "alembic" / "versions"


def alembic(*args: str, database: Path) -> subprocess.CompletedProcess:
    """Run alembic against a named database, through its real entry point.

    The URL is passed by environment rather than by editing `alembic.ini`,
    because `env.py` derives it from the application's settings -- which is the
    property that stops a migration and the server pointing at different files.
    """
    environment = {
        **os.environ,
        "QMCP_DATABASE_URL": f"sqlite+aiosqlite:///{database.as_posix()}",
    }
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=str(ROOT), env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


@pytest.fixture(scope="module")
def upgraded(tmp_path_factory) -> Path:
    """A database built from nothing by the migration chain."""
    database = tmp_path_factory.mktemp("migrated") / "fresh.db"
    result = alembic("upgrade", "head", database=database)
    assert result.returncode == 0, result.stdout + result.stderr
    return database


# --- the chain runs ----------------------------------------------------------


def test_the_chain_upgrades_from_nothing(upgraded):
    assert upgraded.is_file()
    assert describe(upgraded)


def test_the_chain_has_exactly_one_head():
    """Two heads is a merge nobody made, and `upgrade head` then refuses to
    pick. dossier has hit this; qmcp has one chain and should keep it."""
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "heads"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    heads = [line for line in result.stdout.splitlines() if "(head)" in line]
    assert len(heads) == 1, f"expected one head, got: {heads}"


def test_every_revision_file_is_importable():
    """A revision that does not import is one `upgrade` discovers at the worst
    possible moment -- part-way through a chain, against a real database."""
    files = sorted(VERSIONS.glob("*.py"))
    assert files, "no revisions; the chain would be vacuously fine"
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


# --- the assertion the stamp rests on ----------------------------------------


def test_a_fresh_upgrade_reproduces_the_models_schema(upgraded):
    """The stamped baseline is honest only if this holds.

    A database migrated from nothing must have the shape the models describe.
    Where it does not, either a migration is missing or the baseline described
    something the models no longer say -- and the existing database, which was
    stamped rather than migrated, is then a shape nobody checked.
    """
    found = compare_schemas(model_schema(), describe(upgraded))
    assert found.clean, "\n".join(found.lines())


def test_the_migrated_database_records_its_revision(upgraded):
    result = alembic("current", database=upgraded)
    assert result.returncode == 0
    assert "(head)" in result.stdout + result.stderr


def test_alembics_bookkeeping_table_is_not_reported_as_drift(upgraded):
    """`alembic_version` is in every migrated database and in no model.
    Reported as an extra table it would train a reader to ignore that line,
    which is where a real orphan would then hide."""
    assert "alembic_version" in describe(upgraded)
    assert compare_schemas(model_schema(), describe(upgraded)).clean
    assert "alembic_version" in IGNORED_TABLES


# --- migrating twice, and back -----------------------------------------------


def test_upgrading_an_already_current_database_is_a_no_op(upgraded):
    before = describe(upgraded)
    result = alembic("upgrade", "head", database=upgraded)
    assert result.returncode == 0, result.stdout + result.stderr
    assert describe(upgraded) == before


def test_the_chain_downgrades_to_the_baseline(tmp_path):
    """A downgrade nobody has run is a downgrade that does not work. This is
    the cheapest possible proof that the reverse of each step was written."""
    database = tmp_path / "down.db"
    assert alembic("upgrade", "head", database=database).returncode == 0
    at_head = describe(database)

    result = alembic("downgrade", "-1", database=database)
    assert result.returncode == 0, result.stdout + result.stderr
    stepped_back = describe(database)
    assert stepped_back != at_head, "downgrade changed nothing"

    result = alembic("upgrade", "head", database=database)
    assert result.returncode == 0, result.stdout + result.stderr
    assert describe(database) == at_head, "the round trip did not return"


# --- the failure that actually happened --------------------------------------


def test_every_foreign_key_in_a_revision_is_named():
    """SQLite migrations run in batch mode: alembic rebuilds the table around
    the change, and to rebuild a constraint it must be able to name it.

    `create_foreign_key(None, ...)` raises `ValueError: Constraint must have a
    name` *after* earlier statements in the same migration have committed --
    leaving a half-migrated database with its revision unchanged. That happened
    here, and the recovery was a restore from backup.
    """
    for path in sorted(VERSIONS.glob("*.py")):
        # Code lines only. The first version of this scanned the whole file and
        # matched the comment in the migration that explains the rule -- the
        # documented trap of a check finding the text that forbids the thing.
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "create_foreign_key(None" not in stripped, (
                f"{path.name}: unnamed foreign key. Batch mode cannot rebuild "
                f"it, and the failure lands mid-migration: {stripped}"
            )


def test_no_revision_adds_a_not_null_column_without_a_default():
    """Adding a NOT NULL column with no default succeeds against empty tables
    and fails with "Cannot add a NOT NULL column with default value NULL" for
    anyone whose tables have rows. Autogenerate emits exactly that, which is
    what its own "please adjust" comment is about."""
    for path in sorted(VERSIONS.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if "add_column" not in stripped or "nullable=False" not in stripped:
                continue
            assert "server_default" in stripped, (
                f"{path.name}: adds a NOT NULL column with no server_default:\n"
                f"  {stripped}"
            )


# --- the two databases this repository has -----------------------------------


def test_the_flow_databases_tables_are_not_the_servers():
    """`SQLModel.metadata` is global and holds two unrelated databases.

    `qmcp.cookbook.persistence` creates `flowrun`, `agentrun`, `mcpinvocation`,
    `artifact` and `checklistitem` in whatever file `FlowPersistence` is handed
    -- one per run. Read unscoped, the server's schema appears to be missing
    five tables, and an autogenerated migration would try to *add* them to
    `qmcp.db`. This only shows up once something has imported the cookbook,
    which is why it passed alone and failed in the full suite.
    """
    import qmcp.cookbook.persistence  # noqa: F401 -- registering them is the point

    flow_tables = {"flowrun", "agentrun", "mcpinvocation", "artifact", "checklistitem"}
    assert flow_tables & set(model_schema()) == set()


def test_the_server_schema_still_holds_the_servers_own_tables():
    """The other half: scoping must not have excluded the real ones."""
    import qmcp.cookbook.persistence  # noqa: F401

    found = set(model_schema())
    assert {"tool_invocations", "human_requests", "agent_tool_invocations"} <= found
