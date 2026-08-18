"""qmcp's implementation of the org address grammar.

TWO IMPLEMENTATIONS, ONE SET OF CASES. The corpus holds a reference
implementation and a file of conformance vectors. qmcp does not import either --
that would couple the server to the governance repository and neither could ship
without the other. What is shared is the cases.

THE SHARED CASES ARE NOT REACHABLE YET, AND THIS FILE SAYS SO. The governance
submodule pins corpus `d4479cd`, which predates
`project-seed/address-vectors.json` -- the vectors are on an unmerged corpus
branch. `test_the_shared_conformance_vectors` skips with that reason rather than
passing quietly, and starts running the moment the pin moves. Until then this
implementation is verified against its own tests and *not* against the contract,
which is a weaker claim and is stated as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qmcp.addresses import (
    GLOBAL_PREFIXES,
    KINDS,
    REPO,
    Address,
    format_address,
    invocation_address,
    is_global,
    parse,
)

ROOT = Path(__file__).resolve().parent.parent
VECTORS = ROOT / "governance" / "qm" / "project-seed" / "address-vectors.json"


# --- the shared contract -----------------------------------------------------


def test_the_shared_conformance_vectors():
    """Every case the corpus and dossier are also held to.

    Skipped while the submodule pin predates the file. That is a real gap, not
    a passing test: two implementations of one grammar are only kept honest by
    the same cases, and right now only one of them has run them.
    """
    if not VECTORS.is_file():
        pytest.skip(
            f"{VECTORS.relative_to(ROOT).as_posix()} is not in the pinned "
            f"governance submodule (corpus d4479cd predates it). The grammar "
            f"is unverified against the shared contract until the pin moves."
        )
    cases = json.loads(VECTORS.read_text(encoding="utf-8"))["cases"]
    assert cases, "an empty vector file verifies nothing"
    problems: list[str] = []
    for case in cases:
        text = case["address"]
        found = parse(text)
        if not case.get("valid", True):
            if found is not None:
                problems.append(f"{text!r}: parsed, and the vector says it must not")
            continue
        if found is None:
            problems.append(f"{text!r}: did not parse, and the vector says it must")
            continue
        for attribute in ("owner", "repo", "kind", "id"):
            if attribute in case and getattr(found, attribute) != case[attribute]:
                problems.append(
                    f"{text!r}: {attribute} is {getattr(found, attribute)!r}, "
                    f"vector says {case[attribute]!r}")
        if found.format() != text:
            problems.append(f"{text!r}: formats back as {found.format()!r}")
    assert not problems, "\n".join(problems)


# --- the rule that makes an address reversible -------------------------------


def test_a_branch_keeps_the_slashes_git_gave_it():
    """Everything after the kind is the id. Slugging the slash makes the
    address unable to name the ref it came from."""
    assert parse("quaternionmedia/qm/branch/evolve/protect-main").id == "evolve/protect-main"


def test_a_slashed_and_a_hyphenated_branch_stay_distinct():
    assert (parse("quaternionmedia/qm/branch/evolve/protect-main").id
            != parse("quaternionmedia/qm/branch/evolve-protect-main").id)


def test_a_kind_name_inside_an_id_does_not_retrigger_a_match():
    found = parse("quaternionmedia/qm/branch/feature/pr/nested")
    assert (found.kind, found.id) == ("branch", "feature/pr/nested")


def test_every_kind_round_trips_with_a_slashed_id():
    for kind in KINDS:
        text = f"o/r/{kind}/a/b/c"
        found = parse(text)
        assert found is not None and found.format() == text


# --- what is and is not an address -------------------------------------------


def test_a_bare_owner_repo_is_the_repository():
    found = parse("quaternionmedia/qmcp")
    assert found.kind == REPO and found.format() == "quaternionmedia/qmcp"


def test_an_unknown_third_segment_is_not_an_address():
    assert parse("quaternionmedia/qm/tools/build.sh") is None


def test_a_kind_with_no_id_is_not_an_address():
    assert parse("quaternionmedia/qm/branch/") is None


def test_a_non_address_returns_none_rather_than_raising():
    for text in ("", "x", "not an address", "quaternionmedia"):
        assert parse(text) is None


def test_the_global_buckets_are_reserved():
    for prefix in GLOBAL_PREFIXES:
        assert is_global(prefix + "x") and parse(prefix + "x") is None


# --- formatting --------------------------------------------------------------


def test_formatting_and_parsing_are_inverses():
    text = format_address("quaternionmedia", "qmcp", "invocation", "abc-123")
    assert parse(text) == Address("quaternionmedia", "qmcp", "invocation", "abc-123")


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="not a kind"):
        format_address("o", "r", "sprocket", "x")


def test_a_kind_with_no_id_is_refused():
    with pytest.raises(ValueError, match="needs an id"):
        format_address("o", "r", "invocation", "")


# --- the address this server actually mints ----------------------------------


def test_an_invocation_gets_an_address_other_systems_can_name():
    """A bare UUID names nothing outside this database."""
    text = invocation_address("b9532e20-d725-4589", "quaternionmedia/qmcp")
    assert text == "quaternionmedia/qmcp/invocation/b9532e20-d725-4589"
    assert parse(text).kind == "invocation"


def test_a_non_string_invocation_id_is_still_addressable():
    """SQLite hands back whatever it stored; a UUID column may not be a str."""
    from uuid import UUID

    value = UUID("b9532e20-d725-4589-a055-477d4e947b8d")
    assert invocation_address(value, "quaternionmedia/qmcp").endswith(str(value))


def test_a_project_that_is_not_owner_repo_is_refused():
    """Guessing an owner would mint addresses that collide across orgs."""
    with pytest.raises(ValueError, match="owner/repo"):
        invocation_address("abc", "qmcp")
