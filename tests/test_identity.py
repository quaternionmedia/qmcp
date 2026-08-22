"""Which repository this is, and why four modules used to answer separately.

**THE DEFECT THIS CLOSES.** `quaternionmedia/qmcp` was typed into the dashboard,
the cookbook and both thread sources. `qmcp.addresses` already had the
vocabulary — `Address`, `owner`, `parse` — and nothing decided what the owner
*was*, so every caller decided for itself. A fork of this harness emitted deltas
and invocations claiming to belong to this organisation.

`records/DRAFT-a-route-is-an-address.md` is why that is a defect rather than
untidiness: an address is what says two readings are about one thing. A wrong
owner joins a fork's work to this org's, in the one field whose whole job is
identity.

THE TEST WORTH READING IS THE LAST SECTION: no module may hold the literal any
more, which is what stops the fifth one appearing.
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
    """THE ONE THAT MATTERS.

    A fresh `git init`, or a tarball, is a real state. An address built on a
    guessed owner would be worse than one that says it does not know, because a
    guess is indistinguishable from a fact once it is written down.

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


# --- and nothing names the organisation itself any more -----------------------


def test_no_module_hardcodes_the_organisation():
    """**THE GUARD THAT STOPS THE FIFTH ONE.**

    Four modules held this literal. Each was reasonable alone; together they
    meant a fork could not change its own identity. Docstrings may name it as
    an example — code may not.

    Mutation: put any of the four literals back and this fails, naming the file.
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

    **RELOADED BACK, AND THAT IS THE WHOLE CARE HERE.** `project` is evaluated
    at import, so reading it under a different identity means re-importing the
    module -- and `monkeypatch` restores the environment while leaving the
    reloaded module carrying `acme/other` for everything that runs afterwards.
    The first version of this test did exactly that and turned an unrelated test
    red in the full run while passing on its own, which is the hardest shape of
    failure to read.

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
