"""qmcp's implementation of the org's data-point address grammar.

    <owner>/<repo>/<kind>/<id>

The first three segments are owner, repo and kind. **Everything after is the
id, verbatim, slashes included.** A bare `<owner>/<repo>` denotes the repository.

WHY qmcp IMPLEMENTS THIS RATHER THAN IMPORTING IT. The corpus holds a reference
implementation and, more importantly, a set of conformance vectors. Importing
the corpus's Python would couple this server to the governance repository, so
neither could ship without the other -- the same trade `qmcp/cookbook/delta.py`
refuses for the delta schema. What is shared is the *cases*, not the code:
`governance/qm/project-seed/address-vectors.json` reaches every fork through the
submodule, and both implementations are held to it.

THOSE VECTORS ARE NOT REACHABLE YET. The submodule pins corpus `d4479cd`, which
predates them -- they are on an unmerged corpus branch. Until the pin moves,
this implementation is verified by its own tests and *not* against the shared
contract, and `tests/test_addresses.py` says so out loud rather than passing
quietly.

WHAT THIS IS FOR HERE. qmcp records tool invocations with bare UUIDs, which no
other system can name. Addressed as
`quaternionmedia/qmcp/invocation/<id>`, the same row can be pointed at from
dossier, from a delta, and from this server's own dashboard.

WHAT IT CANNOT DO. Say whether the thing addressed exists -- it is a grammar,
and resolution belongs to whichever system holds the row.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA = 1

KINDS: frozenset[str] = frozenset({
    "branch", "pr", "issue", "ver", "doc", "delta", "invocation",
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


def invocation_address(invocation_id: str, project: str) -> str:
    """`quaternionmedia/qmcp/invocation/<id>` for one recorded tool call."""
    owner, _, repo = project.partition("/")
    if not owner or not repo:
        raise ValueError(f"{project!r}: expected owner/repo")
    return format_address(owner, repo, "invocation", str(invocation_id))
