"""Which repository this is, decided in one place.

An address is what says two readings are about one thing
(`records/DRAFT-a-route-is-an-address.md`), so a fork's deltas and invocations
must carry the fork's owner. The last section checks that no module names the
organisation in code.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from qmcp import identity

PACKAGE = Path(identity.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _fresh():
    """The answer is cached, because a remote cannot usefully change mid-run.
    Every test here changes the world, so every test clears it."""
    identity.forget()
    yield
    identity.forget()


# --- where the answer comes from ----------------------------------------------


def test_it_is_derived_from_the_repository(monkeypatch):
    """**DERIVED, NOT CONFIGURED.** A setting can disagree with reality and
    nothing notices; a git remote is the thing that is actually true.

    Mutation: return a literal and this fails in any checkout but this one.
    """
    monkeypatch.delenv("QMCP_PROJECT", raising=False)
    found = identity.this_project()

    remote = subprocess.run(["git", "remote", "get-url", "origin"],
                            cwd=str(PACKAGE.parent), capture_output=True,
                            text=True).stdout.strip()
    if not remote:
        pytest.skip("this checkout has no origin remote")

    owner, repo = found.split("/", 1)
    assert owner in remote and repo in remote, (found, remote)


def test_the_environment_wins_over_the_checkout(monkeypatch):
    """For a caller that knows better than the checkout: a container, a test, a
    repository vendored somewhere odd.

    Mutation: ignore the variable and this fails.
    """
    monkeypatch.setenv("QMCP_PROJECT", "acme/other")
    assert identity.this_project() == "acme/other"
    assert identity.owner() == "acme"


def test_a_checkout_with_no_remote_says_it_does_not_know(monkeypatch):
    """A fresh `git init`, or a tarball, is a real state, and its identity is
    `UNKNOWN` rather than a guessed owner.

    Mutation: fall back to a literal owner and this fails.
    """
    monkeypatch.delenv("QMCP_PROJECT", raising=False)
    monkeypatch.setattr(identity, "_from_remote", lambda *a, **k: None)

    assert identity.this_project() == identity.UNKNOWN
    assert identity.is_known() is False


def test_unknown_is_a_value_a_caller_can_test(monkeypatch):
    """A caller that must not emit a guessed address checks this. Nothing here
    refuses on its behalf — whether an unknown owner is fatal depends on what is
    being written, and this module is not the place that knows."""
    monkeypatch.setenv("QMCP_PROJECT", "acme/other")
    assert identity.is_known() is True
    assert identity.is_known(identity.UNKNOWN) is False


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/acme/thing.git", "acme/thing"),
    ("https://github.com/acme/thing", "acme/thing"),
    ("git@github.com:acme/thing.git", "acme/thing"),
    ("ssh://git@example.com:22/acme/thing.git", "acme/thing"),
    ("https://gitlab.example.com/acme/thing/", "acme/thing"),
])
def test_it_reads_the_shapes_a_remote_actually_takes(url, expected, monkeypatch):
    """SSH, HTTPS, with and without `.git`, with and without a trailing slash.

    **Loose about the host on purpose**: an address's owner is the account, and
    which forge it is on is not part of it.

    Mutation: anchor the pattern to one forge and this fails.
    """
    found = identity._REMOTE.search(url)
    assert found, url
    assert f"{found.group('owner')}/{found.group('repo')}" == expected


def test_the_answer_is_cached(monkeypatch):
    """A remote cannot usefully change mid-run, and shelling out to git on every
    address would put a subprocess inside a formatting call."""
    monkeypatch.delenv("QMCP_PROJECT", raising=False)
    calls = []
    monkeypatch.setattr(identity, "_from_remote",
                        lambda *a, **k: calls.append(1) or "acme/thing")

    assert identity.this_project() == "acme/thing"
    assert identity.this_project() == "acme/thing"
    assert len(calls) == 1, f"asked git {len(calls)} times"


# --- nothing names the organisation itself ------------------------------------


def test_no_module_hardcodes_the_organisation():
    """No module outside `identity.py` names the organisation in code.
    Docstrings and comments may name it as an example.

    Mutation: put the literal back in any module and this fails, naming the
    file.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "identity.py":
            continue
        source = path.read_text(encoding="utf-8")
        # Strip docstrings and comments: this is about what the code emits.
        without_docs = re.sub(r'"""..*?"""', "", source, flags=re.DOTALL)
        for number, line in enumerate(without_docs.splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if "quaternionmedia" in line:
                offenders.append(f"{path.relative_to(PACKAGE)}:{number}: {line.strip()}")

    assert not offenders, (
        "these name the organisation in code rather than deriving it:\n  "
        + "\n  ".join(offenders))


def test_the_modules_that_used_to_hardcode_it_now_agree_with_identity(monkeypatch):
    """One decision, several readers. If they disagreed the addresses would too.

    Each module is reloaded back afterwards: `project` is evaluated at import,
    and `monkeypatch` restores the environment while leaving a reloaded module
    carrying `acme/other` for every test that runs after it.

    Mutation: give any reader its own default and this fails.
    """
    import importlib

    from qmcp.threads import chatgpt, claude

    try:
        monkeypatch.setenv("QMCP_PROJECT", "acme/other")
        identity.forget()
        importlib.reload(claude)
        importlib.reload(chatgpt)
        assert claude.ClaudeThreads.project == "acme/other"
        assert chatgpt.ChatGPTThreads.project == "acme/other"
    finally:
        monkeypatch.delenv("QMCP_PROJECT", raising=False)
        identity.forget()
        importlib.reload(claude)
        importlib.reload(chatgpt)
