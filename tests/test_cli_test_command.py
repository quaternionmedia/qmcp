"""`qmcp test` runs the suite and leaves the configured database alone.

The configured database holds the human queue: requests somebody has not
answered yet. The suite builds its own databases, so the command has no reason
to touch that file, and deleting it is silent and permanent -- an empty queue
afterwards reads as nobody having asked for anything. `--clean` is the one way
to delete it, and it deletes the file the settings name, not a `./qmcp.db` the
command happens to start beside.

pytest is a subprocess the suite must not start from inside a test of the
suite, so `subprocess.run` is replaced with a fake that answers with a chosen
status. Every test moves into a temporary directory first, so the `./qmcp.db`
there is one the test made, and the configured database sits outside it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from click.testing import CliRunner

import qmcp.cli as cli

QUEUE = b"somebody's unanswered requests"


def _fake_run(calls: list[list[str]], returncode: int, watch=None):
    def run(argv, **kwargs):
        calls.append(list(argv))
        if watch is not None:
            watch()
        return subprocess.CompletedProcess(argv, returncode)
    return run


@pytest.fixture
def databases(monkeypatch, tmp_path):
    """(the configured database, a `./qmcp.db` in the working directory), both holding a queue."""
    work = tmp_path / "work"
    work.mkdir()
    configured = tmp_path / "data" / "queue.db"
    configured.parent.mkdir()
    configured.write_bytes(QUEUE)
    beside = work / "qmcp.db"
    beside.write_bytes(QUEUE)
    monkeypatch.chdir(work)
    monkeypatch.setenv("QMCP_DATABASE_URL", f"sqlite+aiosqlite:///{configured.as_posix()}")
    cli.get_settings.cache_clear()
    yield configured, beside
    cli.get_settings.cache_clear()


def test_the_suite_runs_and_both_databases_are_left_alone(monkeypatch, databases) -> None:
    # Mutation seen red: `default=False` on `--clean/--no-clean` set back to
    # `default=True` -- the configured database was deleted before the run and
    # the first byte comparison failed on a missing file.
    configured, beside = databases
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0))

    result = CliRunner().invoke(cli.cli, ["test"])

    assert result.exit_code == 0, result.output
    assert calls == [[sys.executable, "-m", "pytest"]]
    assert configured.read_bytes() == QUEUE
    assert beside.read_bytes() == QUEUE


def test_no_clean_is_an_option(monkeypatch, databases) -> None:
    # Mutation seen red: the option declared as `--clean`, `is_flag=True` again
    # -- click answered "No such option: --no-clean" with exit status 2 and the
    # suite never ran.
    configured, beside = databases
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0))

    result = CliRunner().invoke(cli.cli, ["test", "--no-clean"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert configured.read_bytes() == QUEUE
    assert beside.read_bytes() == QUEUE


def test_clean_deletes_the_configured_database_before_the_run(monkeypatch, databases) -> None:
    # Mutation seen red: `database_file(get_settings().database_url)` replaced
    # with `Path("qmcp.db")` -- the configured database was still there while
    # the suite ran, because the file deleted was the one beside the command.
    configured, beside = databases
    calls: list[list[str]] = []
    present_during_run: list[bool] = []
    watch = lambda: present_during_run.append(configured.exists())  # noqa: E731
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0, watch))

    result = CliRunner().invoke(cli.cli, ["test", "--clean"])

    assert result.exit_code == 0, result.output
    assert present_during_run == [False]
    assert not configured.exists()
    assert beside.read_bytes() == QUEUE
    assert str(configured) in result.output


def test_clean_deletes_what_the_run_left_behind(monkeypatch, databases) -> None:
    # Mutation seen red: the `_clean_database("after")` call removed -- the
    # file the fake run wrote survived the command.
    configured, _ = databases
    calls: list[list[str]] = []
    rewrite = lambda: configured.write_bytes(b"written by the run")  # noqa: E731
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0, rewrite))

    result = CliRunner().invoke(cli.cli, ["test", "--clean"])

    assert result.exit_code == 0, result.output
    assert not configured.exists()


def test_clean_says_what_it_deleted_on_a_console_that_is_not_utf8(monkeypatch, databases) -> None:
    # Mutation seen red: the message given back its leading check mark -- the
    # cp1252 console raised UnicodeEncodeError after the database was gone and
    # before pytest ran, and the command did not exit 0.
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0))

    result = CliRunner(charset="cp1252").invoke(cli.cli, ["test", "--clean"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_the_exit_status_is_pytests(monkeypatch, databases) -> None:
    # Mutation seen red: `sys.exit(result.returncode)` replaced with
    # `sys.exit(0)` -- the exit code read 0.
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 5))

    result = CliRunner().invoke(cli.cli, ["test", "tests/test_hitl.py"])

    assert result.exit_code == 5
    assert calls == [[sys.executable, "-m", "pytest", "tests/test_hitl.py"]]
