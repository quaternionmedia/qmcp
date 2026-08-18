"""Where the configured database is, which is not what the setting says.

`database_url` is a SQLAlchemy URL, not a path. Reading it as one yields
`sqlite+aiosqlite:/qmcp.db`, and `sqlite3.connect` creates that as a new empty
database beside the real one -- a backup of nothing, reported as a success.
"""

from __future__ import annotations

from pathlib import Path

from qmcp.db.paths import database_file


def test_the_configured_default_resolves_to_a_file(tmp_path):
    """The shipped default, which is what a backup will actually be handed."""
    found = database_file("sqlite+aiosqlite:///./qmcp.db", root=tmp_path)
    assert found == (tmp_path / "qmcp.db").resolve()


def test_a_plain_sqlite_url_resolves(tmp_path):
    assert database_file("sqlite:///qmcp.db", root=tmp_path) == (tmp_path / "qmcp.db").resolve()


def test_the_driver_suffix_is_not_taken_as_part_of_the_path(tmp_path):
    """The defect this exists for: the whole URL read as a filename."""
    found = database_file("sqlite+aiosqlite:///./qmcp.db", root=tmp_path)
    assert "aiosqlite" not in str(found)
    assert found.name == "qmcp.db"


def test_an_absolute_url_is_taken_as_absolute():
    found = database_file("sqlite:////var/data/qmcp.db")
    assert found is not None and found.is_absolute()
    assert found.name == "qmcp.db"


def test_a_relative_path_is_resolved_against_the_given_root(tmp_path):
    """Relative to the working directory is the real semantics, and a caller
    that cannot say which directory would get a different database per shell."""
    assert database_file("sqlite:///x.db", root=tmp_path).parent == tmp_path.resolve()


def test_an_in_memory_database_has_no_file():
    """None, not a path. A backup handed an invented path writes an empty file
    and reports success."""
    assert database_file("sqlite+aiosqlite:///:memory:") is None


def test_a_non_sqlite_url_has_no_file():
    assert database_file("postgresql://user@host/db") is None


def test_an_empty_url_has_no_file():
    assert database_file("") is None
