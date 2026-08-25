"""Every `uv run qmcp ...` this package tells somebody to run must resolve.

**SIX OF TEN DID NOT.** `qmcp.topology_view` opened with `uv run qmcp topology
gallery`, `qmcp.orchestration` with `uv run qmcp orchestration plane`, and
neither command existed -- `qmcp.cli` had no mention of either word. The modules
imported, their functions worked and their tests passed, so nothing was red. The
contract was tested and the seam was not deployed, which is the failure
`qmcp.topology_service` documents about itself in its own opening paragraph, in
a different register.

**WHY THIS IS A CHECK AND NOT A TIDY-UP.** `governance/qm` P17 asks that the
black box be reached through a deterministic loop somebody can issue by name. A
module docstring naming a command that does not exist is that loop being
described rather than being there, and a reader cannot tell those apart without
typing it.

WHAT IT SCANS, AND WHAT IT DELIBERATELY DOES NOT. `qmcp/`, `docs/`, the README
and the quickstart -- a command named on a page a reader reaches before the
source is read by more people than one named in a docstring, and until `SCANNED`
had a second entry that half was unchecked. **`tests/` is excluded on purpose.**
This file names several commands that do not resolve, and a scan that read the
tests would match its own exemption list and report the thing it was written to
permit -- the trap `governance/qm` item 10 records as "a text scan matching the
docstring that forbade it". The cost is that a claim made in a test is
unchecked, and that is the trade being made rather than an oversight.

THE MUTATIONS, per P16, because a check nobody has seen fail is a check nobody
has evidence for. Three were run, and each is quoted as it printed rather than
as it was expected to.

Renaming the `topology` group in `qmcp/cli.py` so it no longer resolves:

    AssertionError: qmcp/governed.py claims `uv run qmcp topology show`, and
    `topology` is not a command
      qmcp/topology_view.py claims `uv run qmcp topology gallery`, and
      `topology` is not a command
      qmcp/topology_view.py claims `uv run qmcp topology show`, and
      `topology` is not a command

Deleting `sweep` from `ONLY_CLAIMED` while it is still unrouted:

    AssertionError: qmcp/sweep.py claims `uv run qmcp sweep run`, and `sweep`
    is not a command

Adding `threads`, which does resolve, to `ONLY_CLAIMED`:

    AssertionError: `threads` resolves now. Delete it from ONLY_CLAIMED -- an
    exemption for a command that works reads as a gap that is still open.

Each restoration goes green. The first run of this file, before any mutation,
found a seventh broken claim nobody had looked for: `threads consolidate` was
named in `qmcp/threads/consolidate.py` and the group it belonged to existed, so
checking the groups by hand had missed it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from qmcp.cli import cli

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "qmcp"

# Where a claim can be made. The package, and the pages a reader reaches before
# the package -- a command named in a README is read by more people than one
# named in a docstring, and until this list had a second entry it was the
# unchecked half.
SCANNED: tuple[Path, ...] = (PACKAGE, ROOT / "docs", ROOT / "README.md",
                             ROOT / "quickstart.md")

# A claim looks like `uv run qmcp <group> [<subcommand>]`. Everything from the
# first option, backtick, quote or redirect onward is prose or arguments, and
# neither is a command name.
CLAIM = re.compile(r"uv run qmcp ((?:[a-z][a-z-]*)(?:\s+[a-z][a-z-]*)?)")
STOP = re.compile(r"^[a-z][a-z-]*$")

# Commands this package tells people to run and this package does not provide.
# **Named rather than skipped**: each needs a decision about where its input
# comes from, and that decision is a change with its own reasoning rather than
# a line of wiring. A gate that quietly excluded them would be a green check
# standing where a reader believes every command works.
#
# `test_nothing_is_exempted_that_already_works` keeps this list shrinking: an
# entry that starts resolving fails until somebody deletes it.
ONLY_CLAIMED = {
    # `audit.models_run` and `audit.in_flight` read invocation rows. Which
    # store, and what a window means when the run is still open, are both
    # unsettled.
    "audit",
    # `feedback.once(harness, panel)` wants two generated documents by path.
    # Where those come from when nobody passes them is the open question.
    "feedback",
    # `sweep.run(shares, to_version)` wants the shares of a package across the
    # estate. That survey is `dossier`'s, and how it arrives here is a seam
    # nobody has drawn yet.
    "sweep",
}


def _pages() -> list[Path]:
    """Every file a claim can be made in."""
    found: list[Path] = []
    for root in SCANNED:
        if root.is_dir():
            found.extend(sorted(root.rglob("*.py")))
            found.extend(sorted(root.rglob("*.md")))
        elif root.is_file():
            found.append(root)
    return found


def declared() -> list[tuple[Path, str]]:
    """Every command this repository names, with the file that named it."""
    found = []
    for path in _pages():
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in CLAIM.finditer(text):
            parts = [p for p in match.group(1).split() if STOP.match(p)]
            if parts:
                found.append((path, " ".join(parts)))
    return found


def resolves(claim: str) -> str | None:
    """`None` if the claim resolves, otherwise what is missing.

    **A SECOND TOKEN AFTER A PLAIN COMMAND IS AN ARGUMENT, NOT A SUBCOMMAND.**
    An earlier version of this function did not make that distinction and
    reported `docs/contributing.md` as claiming a command that does not exist,
    on the strength of `uv run qmcp test tests/test_hitl.py` -- where `tests`
    begins the `TEST_PATH` argument `qmcp test` declares. The document was
    right and the check was answering a different question, which is the
    reading `governance/qm` item 10 asks for before a result is believed.

    What that costs: `uv run qmcp serve nonsense` resolves here. Whether a
    command accepts its arguments is that command's own business, and a check
    that took it on would be re-implementing click.
    """
    parts = claim.split()
    group = cli.commands.get(parts[0])
    if group is None:
        return f"`{parts[0]}` is not a command"
    if len(parts) == 1:
        return None
    subcommands = getattr(group, "commands", None)
    if subcommands is None:
        return None
    if parts[1] not in subcommands:
        return f"`{parts[0]}` has no `{parts[1]}`"
    return None


def test_the_scan_finds_something() -> None:
    """A scan that matched nothing would pass every test below.

    The empty assertion this organisation keeps catching: a guard whose input
    is empty reports green and has checked nothing.
    """
    claims = declared()
    assert len(claims) > 5, f"only {len(claims)} claims found; the scan is broken"
    assert any("topology" in claim for _, claim in claims)


def test_every_declared_command_resolves() -> None:
    """No module tells somebody to run a command that is not there."""
    broken = []
    for path, claim in declared():
        if claim.split()[0] in ONLY_CLAIMED:
            continue
        problem = resolves(claim)
        if problem:
            broken.append(
                f"{path.relative_to(ROOT).as_posix()} claims "
                f"`uv run qmcp {claim}`, and {problem}")
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize("name", sorted(ONLY_CLAIMED))
def test_nothing_is_exempted_that_already_works(name: str) -> None:
    """An exemption that stopped being true is a hole nobody can see.

    This is the direction a list like `ONLY_CLAIMED` usually rots in: somebody
    wires the command, nothing tells them the exemption is stale, and the entry
    stays for years describing a gap that closed.
    """
    assert name not in cli.commands, (
        f"`{name}` resolves now. Delete it from ONLY_CLAIMED -- an exemption "
        f"for a command that works reads as a gap that is still open.")


def test_the_governed_seam_is_reachable_by_name() -> None:
    """P17's loop is a command somebody types, or it is a description of one."""
    assert resolves("topology show") is None
    assert resolves("topology gallery") is None
