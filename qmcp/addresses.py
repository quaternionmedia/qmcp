"""qmcp's implementation of the org's data-point address grammar.

    <owner>/<repo>/<kind>/<id>

The first three segments are owner, repo and kind. **Everything after is the
id, verbatim, slashes included.** A bare `<owner>/<repo>` denotes the repository.

qmcp implements the grammar rather than importing the corpus's reference
implementation, so this server and the governance repository can ship
separately, as `qmcp/cookbook/delta.py` does for the delta schema. What is
shared is the cases: `governance/qm/project-seed/address-vectors.json`
reaches every fork through the submodule, and `tests/test_addresses.py` holds
this implementation to it.

qmcp records tool invocations with bare UUIDs. Addressed as
`<owner>/<repo>/invocation/<id>`, the same row can be pointed at from dossier,
from a delta and from this server's dashboard.

What this cannot do: say whether the thing addressed exists. It is a grammar;
resolution belongs to whichever system holds the row.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA = 1

KINDS: frozenset[str] = frozenset({
    "branch", "pr", "issue", "ver", "doc", "delta", "invocation", "ask",
    # A saved topology design, by its name. **Ahead of the corpus**: the kind
    # table in `governance/qm/docs/ref/addresses.md` and the reference
    # implementation do not hold it yet, and the shared vectors carry no case
    # for it either way, so this implementation accepts an address the
    # reference one refuses until the kind is proposed there and lands.
    "topology",
})

# Reserved by dossier for entities that are not repository-scoped.
GLOBAL_PREFIXES = ("github/user/", "lang/", "pkg/")

REPO = "repo"


@dataclass(frozen=True)
class Address:
    owner: str
    repo: str
    kind: str
    id: str = ""

    @property
    def project(self) -> str:
        return f"{self.owner}/{self.repo}"

    def format(self) -> str:
        if self.kind == REPO:
            return self.project
        return f"{self.project}/{self.kind}/{self.id}"


def is_global(text: str) -> bool:
    return text.startswith(GLOBAL_PREFIXES)


def parse(text: str) -> Address | None:
    """The address this string denotes, or None when it denotes none.

    None rather than an exception: callers sweep mixed lists where most names
    are not addresses, and raising would make the ordinary case exceptional.
    """
    if not text or is_global(text):
        return None
    parts = text.split("/")
    if len(parts) < 2 or not all(parts[:2]):
        return None
    owner, repo = parts[0], parts[1]
    if len(parts) == 2:
        return Address(owner, repo, REPO)
    kind = parts[2]
    if kind not in KINDS:
        return None
    identifier = "/".join(parts[3:])
    if not identifier:
        return None
    return Address(owner, repo, kind, identifier)


def format_address(owner: str, repo: str, kind: str, identifier: str = "") -> str:
    if kind != REPO and kind not in KINDS:
        raise ValueError(f"{kind!r} is not a kind. Known: {', '.join(sorted(KINDS))}")
    if kind != REPO and not identifier:
        raise ValueError(f"a {kind} address needs an id")
    return Address(owner, repo, kind, identifier).format()


def ask_address(request_id: str, project: str) -> str:
    """`quaternionmedia/qmcp/ask/<id>` for one question put to a person.

    The id is the request's own and is stable across runs, so a control panel
    that already showed this question finds the same row rather than a second
    one.
    """
    owner, _, repo = project.partition("/")
    if not owner or not repo:
        raise ValueError(f"{project!r}: expected owner/repo")
    return format_address(owner, repo, "ask", request_id)


def invocation_address(invocation_id: str, project: str) -> str:
    """`quaternionmedia/qmcp/invocation/<id>` for one recorded tool call."""
    owner, _, repo = project.partition("/")
    if not owner or not repo:
        raise ValueError(f"{project!r}: expected owner/repo")
    return format_address(owner, repo, "invocation", str(invocation_id))


def topology_address(name: str, project: str) -> str:
    """`quaternionmedia/qmcp/topology/<name>` for one saved topology design.

    The name rather than the row's integer id, because the name is the
    identity a designer chose and the id is the database's -- a second store
    holding the same design would give it a different id and the same name.
    """
    owner, _, repo = project.partition("/")
    if not owner or not repo:
        raise ValueError(f"{project!r}: expected owner/repo")
    return format_address(owner, repo, "topology", name)
