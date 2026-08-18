"""The backup mechanism, which is relied on exactly when the original is gone.

Every test builds its own database in a temporary directory. Nothing here
touches the configured one.

THE TESTS THAT MATTER ARE THE ONES ABOUT FAILURE. A backup tool is easy to write
so that it always reports success -- the happy path is a file copy. What has to
hold is that it refuses: a truncated copy is deleted rather than left to be
found and trusted, an overwrite is refused rather than silently taken, and a
restore backs up what it is about to destroy with no flag to skip it.

The live-writer test is the reason the online backup API is used at all, and it
is written to fail against `shutil.copy`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qmcp.db.backup import (
    BACKUP_DIR,
    backup_path,
    compare,
    listing,
    restore,
    table_counts,
    take,
    verify,
)


def make_db(path: Path, rows: int = 3, table: str = "items") -> Path:
    connection = sqlite3.connect(str(path))
    connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, value TEXT)")
    connection.executemany(
        f"INSERT INTO {table} (value) VALUES (?)", [(f"row-{i}",) for i in range(rows)]
    )
    connection.commit()
    connection.close()
    return path


# --- verify establishes rather than assumes ----------------------------------


def test_a_good_database_verifies(tmp_path):
    checked = verify(make_db(tmp_path / "a.db"))
    assert checked.ok
    assert checked.integrity == "ok"
    assert checked.tables == {"items": 3}


def test_a_missing_file_is_absent_rather_than_created(tmp_path):
    """`sqlite3.connect` on a missing path makes an empty database. Verifying a
    backup that does not exist must not bring one into being and call it fine."""
    missing = tmp_path / "gone.db"
    checked = verify(missing)
    assert not checked.ok
    assert checked.integrity == "absent"
    assert not missing.exists()


def test_a_file_that_is_not_a_database_does_not_verify(tmp_path):
    path = tmp_path / "notadb.db"
    path.write_bytes(b"this is not a sqlite file, it is a sentence")
    checked = verify(path)
    assert not checked.ok
    assert checked.reason


def test_a_truncated_database_does_not_verify(tmp_path):
    """The failure a backup tool actually has: a copy that stopped early."""
    path = make_db(tmp_path / "a.db", rows=200)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])
    assert not verify(path).ok


def test_row_counts_are_read_per_table(tmp_path):
    path = tmp_path / "two.db"
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE a (id INTEGER PRIMARY KEY)")
    connection.execute("CREATE TABLE b (id INTEGER PRIMARY KEY)")
    connection.execute("INSERT INTO a DEFAULT VALUES")
    connection.commit()
    counts = table_counts(connection)
    connection.close()
    assert counts == {"a": 1, "b": 0}


def test_sqlite_internal_tables_are_not_counted(tmp_path):
    path = tmp_path / "seq.db"
    connection = sqlite3.connect(str(path))
    connection.execute("CREATE TABLE a (id INTEGER PRIMARY KEY AUTOINCREMENT)")
    connection.execute("INSERT INTO a DEFAULT VALUES")
    connection.commit()
    counts = table_counts(connection)
    connection.close()
    assert not any(name.startswith("sqlite_") for name in counts)


# --- taking a backup ---------------------------------------------------------


def test_a_backup_is_taken_and_verified(tmp_path):
    source = make_db(tmp_path / "qmcp.db")
    target, checked = take(source)
    assert target.is_file()
    assert checked.ok
    assert checked.tables == {"items": 3}


def test_the_backup_lands_in_its_own_directory(tmp_path):
    source = make_db(tmp_path / "qmcp.db")
    target, _ = take(source)
    assert target.parent.name == BACKUP_DIR
    assert target.parent.parent == tmp_path


def test_the_backup_name_carries_the_stamp_and_the_source_name(tmp_path):
    source = make_db(tmp_path / "qmcp.db")
    target, _ = take(source, stamp="20260818T000000Z")
    assert target.name == "qmcp-20260818T000000Z.db"


def test_a_backup_never_overwrites_another(tmp_path):
    """History of copies must not depend on how fast somebody typed."""
    source = make_db(tmp_path / "qmcp.db")
    take(source, stamp="fixed")
    with pytest.raises(FileExistsError):
        take(source, stamp="fixed")


def test_backing_up_a_database_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        take(tmp_path / "absent.db")


def test_the_copy_holds_the_same_rows(tmp_path):
    source = make_db(tmp_path / "qmcp.db", rows=17)
    target, _ = take(source)
    assert compare(source, target) == []


def test_a_backup_survives_a_writer_holding_the_database_open(tmp_path):
    """The reason for the online backup API rather than a file copy.

    A connection with uncommitted work is open across the backup. The snapshot
    is of committed state: the three committed rows, not the fourth.
    """
    source = make_db(tmp_path / "live.db", rows=3)
    writer = sqlite3.connect(str(source))
    try:
        writer.execute("BEGIN")
        writer.execute("INSERT INTO items (value) VALUES ('uncommitted')")
        target, checked = take(source)
    finally:
        writer.rollback()
        writer.close()
    assert checked.ok
    assert checked.tables == {"items": 3}


def test_a_backup_that_fails_verification_is_removed(tmp_path, monkeypatch):
    """A bad copy left on disk is one somebody finds later and trusts."""
    import qmcp.db.backup as module

    source = make_db(tmp_path / "qmcp.db")
    target = backup_path(source, stamp="fixed")

    def corrupt(path):
        if path == target:
            return module.Verification(path, False, "unreadable", {}, "planted")
        return module.verify(path)

    monkeypatch.setattr(module, "verify", corrupt)
    with pytest.raises(RuntimeError, match="Removed"):
        module.take(source, stamp="fixed")
    assert not target.exists()


# --- compare -----------------------------------------------------------------


def test_a_differing_row_count_is_reported(tmp_path):
    source = make_db(tmp_path / "a.db", rows=3)
    other = make_db(tmp_path / "b.db", rows=4)
    assert any("source 3 rows, copy 4" in p for p in compare(source, other))


def test_a_table_missing_from_the_copy_is_reported(tmp_path):
    source = make_db(tmp_path / "a.db")
    other = make_db(tmp_path / "b.db", table="other")
    problems = compare(source, other)
    assert any("in the source and not the copy" in p for p in problems)
    assert any("in the copy and not the source" in p for p in problems)


def test_comparing_against_an_unreadable_file_says_so(tmp_path):
    source = make_db(tmp_path / "a.db")
    assert compare(source, tmp_path / "absent.db")


# --- listing -----------------------------------------------------------------


def test_backups_are_listed_newest_first(tmp_path):
    source = make_db(tmp_path / "qmcp.db")
    for stamp in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
        take(source, stamp=stamp)
    assert [p.name for p in listing(source)] == [
        "qmcp-20260301T000000Z.db",
        "qmcp-20260201T000000Z.db",
        "qmcp-20260101T000000Z.db",
    ]


def test_listing_with_no_backups_is_empty_rather_than_an_error(tmp_path):
    assert listing(make_db(tmp_path / "qmcp.db")) == []


def test_another_databases_backups_are_not_listed(tmp_path):
    source = make_db(tmp_path / "qmcp.db")
    other = make_db(tmp_path / "other.db")
    take(source, stamp="s")
    take(other, stamp="s")
    assert [p.name for p in listing(source)] == ["qmcp-s.db"]


# --- restore, the one operation that destroys -------------------------------


def test_restoring_puts_the_backup_back(tmp_path):
    source = make_db(tmp_path / "qmcp.db", rows=3)
    target, _ = take(source, stamp="s")
    make_db(tmp_path / "qmcp.db".replace("qmcp", "tmp"))  # unrelated file
    source.unlink()
    make_db(source, rows=99)

    displaced, checked = restore(target, source, stamp="r")
    assert checked.ok
    assert checked.tables == {"items": 3}
    assert displaced is not None


def test_restoring_backs_up_what_it_displaces(tmp_path):
    """The state restore destroys is the state somebody is about to need."""
    source = make_db(tmp_path / "qmcp.db", rows=3)
    target, _ = take(source, stamp="s")
    source.unlink()
    make_db(source, rows=42)

    displaced, _ = restore(target, source, stamp="r")
    assert displaced is not None and displaced.is_file()
    assert verify(displaced).tables == {"items": 42}


def test_restoring_over_nothing_displaces_nothing(tmp_path):
    source = make_db(tmp_path / "qmcp.db")
    target, _ = take(source, stamp="s")
    source.unlink()
    displaced, checked = restore(target, source)
    assert displaced is None
    assert checked.ok


def test_a_backup_that_does_not_verify_is_never_restored(tmp_path):
    """Restoring a corrupt file over a good one destroys the good one."""
    source = make_db(tmp_path / "qmcp.db")
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database")
    with pytest.raises(RuntimeError, match="will not restore"):
        restore(bad, source)
    assert verify(source).ok


def test_a_failed_restore_leaves_the_destination_untouched(tmp_path):
    source = make_db(tmp_path / "qmcp.db", rows=7)
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database")
    with pytest.raises(RuntimeError):
        restore(bad, source)
    assert verify(source).tables == {"items": 7}
