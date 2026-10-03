"""`qmcp preflight` is a route to the seed runner, and decides nothing itself.

The runner is a real subprocess the suite must not start -- it would run this
repository's workflows inside a test of this repository -- so `subprocess.run`
is replaced with a fake that records what it was asked to run and answers with
whatever exit status the test chose. What is asserted is the exact argv, the
working directory, and that the status came back through the command unchanged.

Every test that asserts the working directory first moves the process into a
temporary directory. Without that move the assertion holds whether the command
runs from the repository root or from the cwd, because the suite is launched
from the repository root: the test's answer would be a property of where pytest
was started, not of the code.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

import qmcp.cli as cli

REPO = Path(cli.__file__).resolve().parent.parent
RUNNER = REPO / "governance" / "qm" / "project-seed" / "ci" / "run_workflows_locally.py"


def _fake_run(calls: list[dict], returncode: int):
    def run(argv, **kwargs):
        calls.append({"argv": list(argv), **kwargs})
        return subprocess.CompletedProcess(argv, returncode)
    return run


def test_preflight_passes_every_argument_through_unchanged(monkeypatch, tmp_path) -> None:
    # Mutations seen red, with pytest launched from the repository root:
    # dropping `*args` from the argv (the second assertion fails on the missing
    # options); `cwd=repo` replaced with `cwd=Path.cwd()` (the cwd assertion
    # reads tmp_path, because of the chdir below -- without the chdir that
    # mutation stayed green from the repository root).
    calls: list[dict] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0))
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli.cli, [
        "preflight", "--event", "pull_request", "--base-ref", "origin/main",
        "--head-ref", "feat/x", "--workflows", ".github/workflows", "--ref", "main",
    ])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["argv"] == [
        sys.executable, str(RUNNER),
        "--event", "pull_request", "--base-ref", "origin/main",
        "--head-ref", "feat/x", "--workflows", ".github/workflows", "--ref", "main",
    ]
    assert calls[0]["cwd"] == REPO
    # Nothing of the command's own reaches the output: the runner speaks.
    assert result.output == ""


def test_preflight_forwards_help_to_the_runner(monkeypatch) -> None:
    # Mutation seen red: `help_option_names=[]` removed from the command's
    # context settings -- click answered `--help` with the command's own page,
    # the fake runner was never called, and the length assertion failed on an
    # empty `calls`. The runner's options are this command's options, so its
    # help is the page a reader needs.
    calls: list[dict] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0))

    result = CliRunner().invoke(cli.cli, ["preflight", "--help"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["argv"] == [sys.executable, str(RUNNER), "--help"]
    assert result.output == ""


def test_preflight_exits_with_the_runner_status(monkeypatch) -> None:
    # Mutation seen red: `ctx.exit(result.returncode)` replaced with
    # `ctx.exit(0)` -- the exit code read 0 and the assertion failed.
    calls: list[dict] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 3))

    result = CliRunner().invoke(cli.cli, ["preflight"])

    assert result.exit_code == 3
    assert calls[0]["argv"] == [sys.executable, str(RUNNER)]


def test_preflight_refuses_without_the_governance_submodule(monkeypatch, tmp_path) -> None:
    # Mutations seen red: the `script.exists()` guard removed -- the fake runner
    # was called and the exit code read 0 against a tree with no submodule;
    # `err=True` removed from the refusal's echo -- the refusal moved to stdout
    # and the stderr assertion found it empty. `result.output` merges the two
    # streams and would not have noticed the second.
    calls: list[dict] = []
    monkeypatch.setattr(cli.subprocess, "run", _fake_run(calls, 0))
    monkeypatch.setattr(cli, "_package_repo_root", lambda: tmp_path)

    result = CliRunner().invoke(cli.cli, ["preflight"])

    assert result.exit_code == 2
    assert calls == []
    assert result.stdout == ""
    assert "governance submodule is not checked out" in result.stderr
    assert "git submodule update --init governance/qm" in result.stderr
