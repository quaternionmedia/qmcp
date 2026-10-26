"""Every `uv run qmcp ...` this package tells somebody to run must resolve.

A module that imports, works and passes its tests can still name a command
`qmcp.cli` does not have; this check is what makes that red. `governance/qm`
P17 asks that the black box be reached through a loop somebody can issue by
name, and a docstring naming a missing command describes that loop without it
being there.

It scans `qmcp/`, `docs/` and the README for `uv run qmcp
...` anywhere and, in a page, for a command given bare in code -- a fenced
line or an inline span that starts `qmcp ...`. A page's Python blocks are
read too: every `from qmcp... import ...` they show must import. `tests/` is
excluded: this file names commands that do not resolve, and a scan that read
it would match its own exemption list (`governance/qm` item 10, "a text scan
matching the docstring that forbade it"). A claim made in a test is therefore
unchecked.

The mutations, per P16, each quoted as it printed:

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

Each restoration goes green.
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
# named in a docstring.
SCANNED: tuple[Path, ...] = (PACKAGE, ROOT / "docs", ROOT / "README.md")

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


# A page also gives a command bare -- `qmcp cookbook dev simple-plan`, the
# `uv run` left to the reader -- in a fenced block or an inline span. Prose
# names the package qmcp too, so only code is read: a fenced line that starts
# with the command, or an inline span that does.
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.M | re.S)
BARE_LINE = re.compile(r"^[ \t]*qmcp ((?:[a-z][a-z-]*)(?:[ \t]+[a-z][a-z-]*)?)", re.M)
BARE_SPAN = re.compile(r"`qmcp ((?:[a-z][a-z-]*)(?:[ \t]+[a-z][a-z-]*)?)")


def _bare(text: str):
    for block in FENCE.findall(text):
        yield from BARE_LINE.finditer(block)
    yield from BARE_SPAN.finditer(text)


def declared() -> list[tuple[Path, str]]:
    """Every command this repository names, with the file that named it."""
    found = []
    for path in _pages():
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(CLAIM.finditer(text))
        if path.suffix == ".md":
            matches += list(_bare(text))
        for match in matches:
            parts = [p for p in match.group(1).split() if STOP.match(p)]
            if parts:
                found.append((path, " ".join(parts)))
    return found


def resolves(claim: str) -> str | None:
    """`None` if the claim resolves, otherwise what is missing.

    A second token after a plain command is an argument, not a subcommand:
    in `uv run qmcp test tests/test_hitl.py`, `tests/...` is the `TEST_PATH`
    argument `qmcp test` declares.

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

    A guard whose input is empty reports green and has checked nothing.
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


# --- the imports a page shows ----------------------------------------------------
#
# A page that shows `from qmcp.x import y` is read as "this works". Every such
# line in a Python block of a page a reader reaches is imported here, and every
# name it lists must be there. Design records under docs/adrs/ hold code that
# was planned rather than built, and are not read.

PYTHON_FENCE = re.compile(r"^```python\n(.*?)^```", re.M | re.S)


def _import_statements(block: str):
    """Each `from qmcp... import ...` statement in a block, joined across lines."""
    lines = iter(block.splitlines())
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(("from qmcp", "import qmcp")):
            continue
        statement = stripped.split("#")[0]
        while "(" in statement and ")" not in statement:
            statement += " " + next(lines, ")").split("#")[0].strip()
        yield statement


def shown_imports() -> list[tuple[Path, str]]:
    found = []
    for path in _pages():
        if path.suffix != ".md" or "adrs" in path.parts:
            continue
        for block in PYTHON_FENCE.findall(path.read_text(encoding="utf-8", errors="replace")):
            found.extend((path, statement) for statement in _import_statements(block))
    return found


def test_the_import_scan_finds_something() -> None:
    assert len(shown_imports()) > 5


def test_every_import_a_page_shows_imports() -> None:
    """Mutation seen red: overview.md's model types imported from
    `qmcp.agentframework` again, as the page had them until the reference
    pages were checked against the code -- the four names were reported
    missing."""
    import ast
    import importlib

    broken = []
    for path, statement in shown_imports():
        where = path.relative_to(ROOT).as_posix()
        node = ast.parse(statement).body[0]
        try:
            if isinstance(node, ast.ImportFrom):
                module = importlib.import_module(node.module)
                missing = [a.name for a in node.names if not hasattr(module, a.name)]
                if missing:
                    broken.append(f"{where}: {statement} -- no {', '.join(missing)}")
            else:
                for alias in node.names:
                    importlib.import_module(alias.name)
        except ImportError as exc:
            broken.append(f"{where}: {statement} -- {exc}")
    assert not broken, "\n".join(broken)
