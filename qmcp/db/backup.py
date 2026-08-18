"""Take a copy of a live SQLite database, and prove the copy is good.

    qmcp db backup            # timestamped copy of the configured database
    qmcp db backups           # what has been taken, newest first
    qmcp db verify <path>     # is this file a readable, intact database?
    qmcp db restore <path>    # put one back, after backing up what is there

WHY NOT `shutil.copy`. A running server holds the database open. Copying the
file byte-for-byte while a writer is mid-transaction produces a file that opens
cleanly and is wrong -- and with WAL journalling the `-wal` sidecar carries
committed data the `.db` file does not, so a copy of one file without the other
loses whatever had not been checkpointed. `sqlite3.Connection.backup()` is the
online backup API: it reads through the same locking the writer uses and yields
a consistent snapshot of committed state with the server still running.

WHY IT VERIFIES BEFORE REPORTING SUCCESS. A backup nobody has read is a belief.
Every copy is reopened, `PRAGMA integrity_check` is run against it, and its
per-table row counts are compared with the source. A backup mechanism that
reported success on a truncated file would be worse than none, because it is
relied on precisely when the original is already gone.

WHAT IT CANNOT DO.

  * Capture a transaction that has not committed. The snapshot is of committed
    state, which is the only state with a defined answer.
  * Notice that the source itself is already corrupt in a way `integrity_check`
    passes -- a schema that drifted from the code reads as a perfectly intact
    database, which is exactly the condition this repository is in.
  * Make a backup a migration. Restoring an old file restores an old schema.
  * Protect against the disk it is written to. Backups land beside the database
    by default, and a second copy on the same volume is one failure away from
    the first.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Backups live here, beside the database, and `*.db` is already gitignored --
# but the directory is named so a reader sees a deliberate location rather than
# files scattered next to the original.
BACKUP_DIR = "db-backups"
STAMP = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class Verification:
    """What was established about a file, and never inferred."""

    path: Path
    readable: bool
    integrity: str
    tables: dict[str, int]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.readable and self.integrity == "ok"


def table_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Row count per table, which is the cheapest claim worth checking.

    Not a checksum: two databases with equal counts can still differ. It is the
    check that catches the failure that actually happens -- a copy that stopped
    early.
    """
    counts: dict[str, int] = {}
    names = [
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for name in names:
        counts[name] = connection.execute(
            f'SELECT count(*) FROM "{name}"'  # noqa: S608 -- name from sqlite_master
        ).fetchone()[0]
    return counts


def verify(path: Path) -> Verification:
    """Open the file, check it, and say what was found.

    Read-only, through a URI, so verifying never creates the file it was asked
    about -- `sqlite3.connect` on a missing path happily makes an empty database
    and would report a backup that does not exist as a fine empty one.
    """
    if not path.is_file():
        return Verification(path, False, "absent", {}, "no such file")
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return Verification(path, False, "unreadable", {}, str(exc))
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        counts = table_counts(connection)
    except sqlite3.DatabaseError as exc:
        return Verification(path, False, "unreadable", {}, str(exc))
    finally:
        connection.close()
    return Verification(path, True, integrity, counts)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime(STAMP)


def backup_path(source: Path, stamp: str | None = None,
                directory: Path | None = None) -> Path:
    folder = directory or source.parent / BACKUP_DIR
    return folder / f"{source.stem}-{stamp or now_stamp()}{source.suffix or '.db'}"


def take(source: Path, destination: Path | None = None,
         stamp: str | None = None) -> tuple[Path, Verification]:
    """Copy `source` to a new file and verify the result.

    Refuses to overwrite. A backup that silently replaced an earlier one would
    make the history of copies depend on how fast somebody ran the command.
    """
    if not source.is_file():
        raise FileNotFoundError(f"{source}: no database to back up.")

    target = destination or backup_path(source, stamp)
    if target.exists():
        raise FileExistsError(
            f"{target}: already there. Backups are never overwritten -- "
            f"pass a different destination."
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    # The online backup API, not a file copy. See the module docstring.
    origin = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        copy = sqlite3.connect(str(target))
        try:
            origin.backup(copy)
        finally:
            copy.close()
    finally:
        origin.close()

    checked = verify(target)
    if not checked.ok:
        # A bad copy is removed rather than left to be found later and trusted.
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"{target}: backup failed verification ({checked.integrity}"
            f"{': ' + checked.reason if checked.reason else ''}). Removed."
        )
    return target, checked


def compare(source: Path, copy: Path) -> list[str]:
    """Differences between two databases' tables and row counts.

    The assertion a backup is for: same tables, same number of rows in each.
    """
    left, right = verify(source), verify(copy)
    if not left.ok:
        return [f"{source}: {left.integrity} {left.reason}".strip()]
    if not right.ok:
        return [f"{copy}: {right.integrity} {right.reason}".strip()]

    problems: list[str] = []
    for name in sorted(set(left.tables) | set(right.tables)):
        before, after = left.tables.get(name), right.tables.get(name)
        if before is None:
            problems.append(f"table `{name}` is in the copy and not the source")
        elif after is None:
            problems.append(f"table `{name}` is in the source and not the copy")
        elif before != after:
            problems.append(f"table `{name}`: source {before} rows, copy {after}")
    return problems


def listing(source: Path, directory: Path | None = None) -> list[Path]:
    """Backups of this database, newest first, by filename.

    Sorted by name rather than mtime: the stamp is in the name, and an mtime is
    changed by anything that touches the file.
    """
    folder = directory or source.parent / BACKUP_DIR
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{source.stem}-*{source.suffix or '.db'}"), reverse=True)


def restore(backup: Path, destination: Path,
            stamp: str | None = None) -> tuple[Path | None, Verification]:
    """Put a backup back, having first backed up what is currently there.

    Returns the safety copy's path -- or None when there was nothing to
    displace -- and the verification of the restored file.

    THE SAFETY COPY IS NOT OPTIONAL. Restore is the one operation here that
    destroys data, and the state it destroys is the state somebody is about to
    discover they needed. There is no flag to skip it.
    """
    checked = verify(backup)
    if not checked.ok:
        raise RuntimeError(
            f"{backup}: will not restore a file that does not verify "
            f"({checked.integrity}{': ' + checked.reason if checked.reason else ''})."
        )

    displaced: Path | None = None
    if destination.is_file():
        displaced, _ = take(destination, stamp=(stamp or now_stamp()) + "-replaced")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(backup, destination)
    return displaced, verify(destination)
