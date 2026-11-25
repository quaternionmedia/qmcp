"""`qmcp tour`: a roadmap read against the tour its application serves.

Three parts are held here: the readers (a roadmap's lines, a clone's pages),
the review built from them and the application's two documents, and the
command, run through click with `httpx.get` replaced by a fake that answers
from a dict and keeps what it was asked. No test needs a running application.

Each test names the mutation of `qmcp/tour.py` or `qmcp/cli.py` it goes red
under.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

import qmcp.cli as cli
from qmcp import tour
from qmcp.tour import Line, Page

ROADMAP = """\
# Roadmap

## Designing a part

- [x] A part is chosen and its numbers are staged. `journey:design`
- [x] An assembly is navigated by its pieces. [09](../../../walkthrough/09-an-assembly.md)

## Making a thing

- [x] A part is sliced and printed on a machine. `journey:make`
- [ ] A print is weighed against its estimate.
"""

AT = "http://127.0.0.1:6283"


def stage(journey: str, name: str, tries: tuple[str, ...] = (), feedback=None) -> dict:
    """One stage, shaped as the application's `GET /tour` serves it."""
    return {"id": f"{journey}/{name}", "journey": journey, "stage": name,
            "title": name.capitalize(), "words": "",
            "tries": [{"page": page, "index": None, "heading": page, "words": "",
                       "picture": None, "shown": None} for page in tries],
            "feedback": feedback}


def said(of: str, verdict: str, note: str = "", at: str = "2026-11-15T12:00:00+00:00",
         commit: str = "f0bc8430", changed: bool = False, by: str | None = None) -> dict:
    """One feedback record, shaped as `GET /tour/feedback` lists it. A record
    kept before words were named has no `by`."""
    record = {"stage": of, "verdict": verdict, "note": note, "at": at,
              "commit": commit, "changed": changed}
    return record if by is None else {**record, "by": by}


# `design` is on the roadmap and served; `make` is on the roadmap and not
# served; `calibrate` is served and not on the roadmap. The design stages are
# out of alphabetical order, so an order that is not the journey's shows.
SERVED = {"commit": "f0bc8430", "changed": False, "journeys": [
    {"id": "design", "name": "Design a part", "stages": [
        stage("design", "choose", ("04-designing-a-part.md",)),
        stage("design", "numbers"),
        stage("design", "apply"),
        stage("design", "ready")]},
    {"id": "calibrate", "name": "Calibrate a machine", "stages": [
        stage("calibrate", "level")]},
]}

# The application runs f0bc8430. Of the design stages, `choose` was said off
# of an earlier build and works in this one, `numbers` is off in this one,
# `apply` was last said off of an earlier build, and nobody has answered
# `ready`.
RECORDS = [
    said("design/choose", "off", "the list is empty", at="2026-11-14T09:00:00+00:00",
         commit="5caeea86"),
    said("design/apply", "off", "nothing is applied", at="2026-11-14T09:05:00+00:00",
         commit="5caeea86", by="Lin"),
    said("design/numbers", "off", "a staged number is lost on reload",
         at="2026-11-15T12:10:00+00:00", changed=True),
    said("design/choose", "works", by="Ada"),
]

PAGES = [Page("04-designing-a-part.md", "Designing a part"),
         Page("09-an-assembly.md", "An assembly"),
         Page("10-a-drawing.md", "A drawing")]


def lines() -> list[Line]:
    return tour.roadmap_lines(ROADMAP)


def states(row: dict) -> list[tuple[str, str]]:
    return [(stop.get("id") or stop.get("page") or stop["journey"], stop["state"])
            for stop in row["stops"]]


# --- the roadmap ----------------------------------------------------------------


def test_a_line_carries_its_group_its_tick_and_what_shows_it() -> None:
    """Mutation: leave the names in a line's text -- red on the first text.
    Mutation: never open a group at a `## ` heading -- red on the groups."""
    found = lines()

    assert [line.group for line in found] == [
        "Designing a part", "Designing a part", "Making a thing", "Making a thing"]
    assert [line.ticked for line in found] == [True, True, True, False]
    assert found[0].text == "A part is chosen and its numbers are staged."
    assert found[0].journeys == ("design",) and found[0].pages == ()
    assert found[1].text == "An assembly is navigated by its pieces."
    assert found[1].journeys == () and found[1].pages == ("09-an-assembly.md",)


def test_a_line_that_names_nothing_is_kept_and_shown_by_nothing() -> None:
    """Mutation: drop a line that names neither a journey nor a page -- red."""
    found = lines()

    assert found[3].text == "A print is weighed against its estimate."
    assert found[3].shown_by_nothing
    assert [line.shown_by_nothing for line in found[:3]] == [False, False, False]


@pytest.mark.parametrize("target", [
    "walkthrough/09-an-assembly.md",
    "./walkthrough/09-an-assembly.md",
    "../../../walkthrough/09-an-assembly.md",
    "../../../walkthrough/09-an-assembly.md#by-its-pieces",
])
def test_a_page_is_named_by_a_link_under_any_prefix(target: str) -> None:
    """Mutation: match the link only with no prefix before `walkthrough/` --
    red on the relative targets. Mutation: refuse a fragment -- red on the last."""
    found = tour.roadmap_lines(f"- [x] An assembly is navigated. [09]({target})\n")

    assert found[0].pages == ("09-an-assembly.md",)
    assert found[0].text == "An assembly is navigated."


@pytest.mark.parametrize("words", [
    "Read [09](../docs/09-an-assembly.md).",
    "Read [the walkthrough](../walkthrough/README.md).",
    "Read `walkthrough/09-an-assembly.md`.",
    "Shown by journey:design, and by `journey:Design`.",
])
def test_a_name_written_another_way_is_prose(words: str) -> None:
    """Mutation: take any link ending `.md` as a page -- red on the first two.
    Mutation: read `journey:` outside a code span, or in any case -- red on the last."""
    found = tour.roadmap_lines(f"- [x] {words}\n")

    assert found[0].shown_by_nothing
    assert found[0].text == words


def test_a_line_names_a_journey_and_pages_together() -> None:
    """Mutation: stop at the first page link -- red on the second page."""
    found = tour.roadmap_lines(
        "- [x] Cut. `journey:cut` [20](walkthrough/20-a-cut.md) [21](walkthrough/21-a-fit.md)\n")

    assert found[0].journeys == ("cut",)
    assert found[0].pages == ("20-a-cut.md", "21-a-fit.md")
    assert found[0].text == "Cut."


def test_the_commas_between_and_after_the_names_are_not_the_claim() -> None:
    """A line that names two pages, with a comma between them.

    Mutation: take out the names and leave what stands between them -- red:
    the text ends ` ,`."""
    found = tour.roadmap_lines(
        "- [x] A camera's picture becomes pieces, and a piece a print."
        " [01](../../../walkthrough/01-photographs-into-pieces.md),"
        " [03](../../../walkthrough/03-a-picture-to-a-print.md)\n")

    assert found[0].text == "A camera's picture becomes pieces, and a piece a print."
    assert found[0].pages == ("01-photographs-into-pieces.md", "03-a-picture-to-a-print.md")


@pytest.mark.parametrize("words, text", [
    ("Cut. `journey:cut`, [20](walkthrough/20-a-cut.md), [21](walkthrough/21-a-fit.md),", "Cut."),
    ("Cut, as `journey:cut` walks it, and fitted.", "Cut, as walks it, and fitted."),
    ("Cut, fitted, and glued. [20](walkthrough/20-a-cut.md)", "Cut, fitted, and glued."),
])
def test_only_the_commas_beside_a_name_go_with_it(words: str, text: str) -> None:
    """Mutation: take every comma out of a line that names something -- red
    on the claim's own commas."""
    assert tour.roadmap_lines(f"- [x] {words}\n")[0].text == text


def test_a_wrapped_line_is_one_line() -> None:
    """A checkbox item's indented continuation is part of its claim; a nested
    item and the paragraph after it are not.

    Mutation: read the first line of an item only -- red, the journey is on
    the second. Mutation: take a nested item as a continuation -- red on the text."""
    found = tour.roadmap_lines(
        "## Cutting\n"
        "\n"
        "- [x] A sheet is placed, its axes set,\n"
        "  and it is cut. `journey:cut`\n"
        "  - the spindle is a person's to start\n"
        "\n"
        "A paragraph.\n")

    assert len(found) == 1
    assert found[0].text == "A sheet is placed, its axes set, and it is cut."
    assert found[0].journeys == ("cut",)


def test_a_fenced_example_holds_no_lines() -> None:
    """Mutation: read inside a fence -- red, the example becomes a second line."""
    found = tour.roadmap_lines(
        "## Format\n"
        "\n"
        "```markdown\n"
        "## An example group\n"
        "- [x] An example. `journey:example`\n"
        "```\n"
        "\n"
        "- [ ] A real line.\n")

    assert [(line.group, line.text) for line in found] == [("Format", "A real line.")]


def test_only_a_second_level_heading_opens_a_group() -> None:
    """Mutation: let any heading open a group -- red on the third line's group."""
    found = tour.roadmap_lines(
        "# Roadmap\n"
        "- [ ] Above every group.\n"
        "## Printing\n"
        "- [ ] In the group.\n"
        "### A note\n"
        "- [ ] Still in the group.\n")

    assert [line.group for line in found] == [None, "Printing", "Printing"]


def test_a_roadmap_with_a_byte_that_is_not_utf8_is_still_read(tmp_path: Path) -> None:
    """Mutation: decode the file strictly -- red on a `UnicodeDecodeError`."""
    path = tmp_path / "ROADMAP.md"
    path.write_bytes("## Café\n\n- [x] A cup is served. `journey:serve`\n".encode("latin-1"))

    found = tour.read_roadmap(path)

    assert [(line.text, line.journeys) for line in found] == [
        ("A cup is served.", ("serve",))]


# --- the pages ------------------------------------------------------------------


def test_pages_are_the_numbered_files_in_order_with_their_titles(tmp_path: Path) -> None:
    """Mutation: keep the number in a title -- red on the first title.
    Mutation: list every `.md` -- red on the README.
    Mutation: order by name as text -- red, `100` sorts before `11`."""
    folder = tmp_path / "walkthrough"
    folder.mkdir()
    (folder / "09-an-assembly.md").write_text("# 09 — An assembly\n\nWords.\n", encoding="utf-8")
    (folder / "11-fitting.md").write_text("Words first.\n\n# Fitting\n", encoding="utf-8")
    (folder / "100-a-late-page.md").write_text("No heading.\n", encoding="utf-8")
    (folder / "README.md").write_text("# The walkthrough\n", encoding="utf-8")

    assert tour.read_pages(tmp_path) == [
        Page("09-an-assembly.md", "An assembly"),
        Page("11-fitting.md", "Fitting"),
        Page("100-a-late-page.md", None),
    ]


def test_a_clone_without_a_walkthrough_has_no_pages(tmp_path: Path) -> None:
    """Mutation: list the folder with `iterdir` -- red, there is none to list."""
    assert tour.read_pages(tmp_path) == []


# --- the review -----------------------------------------------------------------


def test_a_line_stops_at_its_journeys_stages_in_order_then_its_pages() -> None:
    """Mutation: sort the stages -- red, `apply` moves first.
    Mutation: put the pages before the stages -- red."""
    line = Line("Designing a part", "Designed and read.", True,
                journeys=("design",), pages=("09-an-assembly.md",))

    row = tour.review([line], PAGES, SERVED, RECORDS)["lines"][0]

    assert states(row) == [
        ("design/choose", "works"),
        ("design/numbers", "off"),
        ("design/apply", "said of an earlier build"),
        ("design/ready", "not yet"),
        ("09-an-assembly.md", "read whole"),
    ]
    assert [stop["kind"] for stop in row["stops"]] == [
        "stage", "stage", "stage", "stage", "page"]
    assert row["stops"][0]["tries"] == ["04-designing-a-part.md"]
    assert row["stops"][4]["title"] == "An assembly"
    assert (row["group"], row["text"], row["ticked"], row["shown_by_nothing"]) == (
        "Designing a part", "Designed and read.", True, False)


def test_the_latest_record_is_a_stages_feedback() -> None:
    """`design/choose` was said `off` and later `works`.

    Mutation: keep the first record of a stage -- red."""
    row = tour.review(lines(), PAGES, SERVED, RECORDS)["lines"][0]

    assert row["stops"][0]["state"] == "works"
    assert row["stops"][0]["feedback"] == RECORDS[3]
    assert row["stops"][3]["feedback"] is None


def test_a_word_counts_for_the_build_it_was_said_of() -> None:
    """`design/apply` was last said off while the application ran 5caeea86,
    and it runs f0bc8430. `design/numbers` was said off of f0bc8430 with
    changes beside it, which is this build.

    Mutation: count a stage's latest word whatever build it was said of --
    red, `apply` is `off`. Mutation: drop a word of an earlier build -- red,
    `apply` is `not yet` and its record is gone."""
    row = tour.review(lines(), PAGES, SERVED, RECORDS)["lines"][0]

    assert states(row) == [
        ("design/choose", "works"),
        ("design/numbers", "off"),
        ("design/apply", "said of an earlier build"),
        ("design/ready", "not yet"),
    ]
    assert row["stops"][2]["feedback"] == RECORDS[1]
    assert tour.counts(row["stops"]) == {
        "works": 1, "off": 1, "said of an earlier build": 1, "not yet": 1}


def test_an_application_that_does_not_say_its_build_counts_every_latest_word() -> None:
    """Mutation: hold every word to a build nobody named -- red, each is of
    an earlier build. Mutation: say nothing of it in the header -- red."""
    unsaid = {**SERVED, "commit": None, "changed": None}

    document = tour.review(lines(), PAGES, unsaid, RECORDS, at=AT)
    said_of = tour.render(document)

    assert states(document["lines"][0]) == [
        ("design/choose", "works"),
        ("design/numbers", "off"),
        ("design/apply", "off"),
        ("design/ready", "not yet"),
    ]
    assert said_of.splitlines()[0] == (
        f"application  {AT}  did not say its build: every latest word counts as said of it")
    assert said_of.count("did not say its build") == 1
    # No word is marked as of an earlier build: there is no build to be earlier than.
    assert ("        ✗ design/apply  Lin  5caeea86  2026-11-14T09:05:00+00:00"
            "  nothing is applied\n") in said_of
    assert "  of an earlier build," not in said_of


def test_the_records_are_read_and_the_embedded_copy_is_not() -> None:
    """The copy a stage embeds says of itself whether it is current.

    Mutation: fall back to the `feedback` a stage embeds -- red."""
    embedded = {**SERVED, "journeys": [{"id": "design", "name": "Design a part", "stages": [
        stage("design", "choose",
              feedback={**said("design/choose", "works"), "current": True})]}]}

    row = tour.review(lines(), PAGES, embedded, [])["lines"][0]

    assert states(row) == [("design/choose", "not yet")]
    assert row["stops"][0]["feedback"] is None


def four_lines() -> dict:
    """The review of four lines: `B` ticked with two stages unanswered, `A`
    ticked with a stage off and one unanswered, `C` ticked with every stage
    working, and `D` unticked with a stage off."""
    served = {"commit": "f0bc8430", "changed": False, "journeys": [
        {"id": "a", "name": "A", "stages": [stage("a", "1"), stage("a", "2"), stage("a", "3")]},
        {"id": "b", "name": "B", "stages": [stage("b", "1"), stage("b", "2"), stage("b", "3")]},
        {"id": "c", "name": "C", "stages": [stage("c", "1")]},
        {"id": "d", "name": "D", "stages": [stage("d", "1")]},
    ]}
    records = [said("a/1", "works"), said("a/2", "off", "no"), said("b/1", "works"),
               said("c/1", "works"), said("d/1", "off", "no")]
    roadmap = [Line("G", "B is built.", True, journeys=("b",)),
               Line("G", "A is built.", True, journeys=("a",)),
               Line("G", "C is built.", True, journeys=("c",)),
               Line("G", "D is planned.", False, journeys=("d",))]
    return tour.review(roadmap, [], served, records)


def test_a_ticked_line_with_a_stage_off_or_unanswered_is_a_disagreement() -> None:
    """Mutation: count only `off` -- red, `B` is lost.
    Mutation: ignore the tick -- red, `D` appears.
    Mutation: call any ticked journey line a disagreement -- red on `C`."""
    document = four_lines()

    assert document["disagreements"] == [
        {"group": "G", "text": "B is built.", "roadmap": "ticked",
         "off": [], "earlier": [], "not_yet": ["b/2", "b/3"]},
        {"group": "G", "text": "A is built.", "roadmap": "ticked",
         "off": ["a/2"], "earlier": [], "not_yet": ["a/3"]},
    ]
    # Neither side is taken: the line stays ticked and the stage stays off.
    assert document["lines"][1]["ticked"] is True
    assert states(document["lines"][1])[1] == ("a/2", "off")


def test_a_ticked_line_with_no_word_of_this_build_is_waiting_on_a_word() -> None:
    """Three ticked lines: `A`, whose only `off` was said of an earlier
    build; `B`, whose one stage worked in an earlier build; `C`, whose stage
    works in this one.

    Mutation: read a word of an earlier build as said of this one -- red:
    `A` has a stage off and `B` is no disagreement.
    Mutation: give a waiting line its unanswered count alone -- red.
    Mutation: leave an earlier word out from under its line -- red.
    Mutation: print an earlier word like one of this build -- red."""
    served = {"commit": "f0bc8430", "changed": False, "journeys": [
        {"id": "a", "name": "A", "stages": [stage("a", "1"), stage("a", "2")]},
        {"id": "b", "name": "B", "stages": [stage("b", "1")]},
        {"id": "c", "name": "C", "stages": [stage("c", "1")]},
    ]}
    records = [said("a/1", "off", "no", commit="5caeea86"),
               said("b/1", "works", commit="5caeea86"),
               said("c/1", "works")]
    roadmap = [Line("G", "A is built.", True, journeys=("a",)),
               Line("G", "B is built.", True, journeys=("b",)),
               Line("G", "C is built.", True, journeys=("c",))]

    document = tour.review(roadmap, [], served, records)
    said_of = tour.render(document)

    assert document["disagreements"] == [
        {"group": "G", "text": "A is built.", "roadmap": "ticked",
         "off": [], "earlier": ["a/1"], "not_yet": ["a/2"]},
        {"group": "G", "text": "B is built.", "roadmap": "ticked",
         "off": [], "earlier": ["b/1"], "not_yet": []},
    ]
    assert ("  only waiting on a word:\n"
            "    [x] A is built.  1 said of an earlier build, 1 not yet\n"
            "    [x] B is built.  1 said of an earlier build, 0 not yet\n") in said_of
    assert "a stage is off:" not in said_of
    # The earlier `off` can still be read under its line, marked as of its commit.
    assert ("      journey:a  0 works, 0 off, 1 said of an earlier build, 1 not yet\n"
            "        ✗ a/1  of an earlier build, 5caeea86  2026-11-15T12:00:00+00:00  no\n"
            ) in said_of


def test_a_page_no_line_names_and_no_stage_tries_is_not_on_the_roadmap() -> None:
    """Page 04 is tried by `design/choose`, page 09 is linked by a line.

    Mutation: ignore what the stages try -- red, 04 appears.
    Mutation: ignore what the lines link -- red, 09 appears."""
    document = tour.review(lines(), PAGES, SERVED, RECORDS)

    assert document["pages_not_on_the_roadmap"] == [
        {"page": "10-a-drawing.md", "title": "A drawing"}]
    assert [entry["page"] for entry in document["pages_no_line_names"]] == [
        "04-designing-a-part.md", "10-a-drawing.md"]


def test_a_journey_on_one_side_only_is_listed_on_that_side() -> None:
    """Mutation: list no journey as unnamed -- red on `calibrate`.
    Mutation: give an unserved journey no stop -- red on the `make` line."""
    document = tour.review(lines(), PAGES, SERVED, RECORDS)

    unnamed = document["journeys_not_on_the_roadmap"]
    assert [(journey["id"], journey["name"]) for journey in unnamed] == [
        ("calibrate", "Calibrate a machine")]
    assert [(stop["id"], stop["state"]) for stop in unnamed[0]["stops"]] == [
        ("calibrate/level", "not yet")]
    assert document["journeys_not_served"] == [
        {"journey": "make", "group": "Making a thing",
         "text": "A part is sliced and printed on a machine."}]
    assert states(document["lines"][2]) == [("make", "not served")]
    # An unserved journey is its own finding, and no stage of it is counted.
    assert [found["text"] for found in document["disagreements"]] == [
        "A part is chosen and its numbers are staged."]


def test_without_the_application_nothing_of_it_is_claimed() -> None:
    """Mutation: review an unread application as one that serves nothing --
    red: every list of the application's is empty where it is unknown."""
    document = tour.review(lines(), PAGES, unread="no --at was given")

    assert document["application"] == {"at": None, "read": False, "why": "no --at was given"}
    for theirs in ("disagreements", "pages_not_on_the_roadmap", "journeys_not_on_the_roadmap",
                   "journeys_not_served", "feedback_on_no_stage"):
        assert document[theirs] == {"unknown": "no --at was given"}, theirs
    assert states(document["lines"][0]) == [("design", "the application's to say")]
    # What the clone and the roadmap hold alone is still said.
    assert states(document["lines"][1]) == [("09-an-assembly.md", "read whole")]
    assert [entry["page"] for entry in document["pages_no_line_names"]] == [
        "04-designing-a-part.md", "10-a-drawing.md"]


def test_the_application_read_is_named_with_its_commit() -> None:
    """Mutation: leave the commit out of the document -- red."""
    document = tour.review(lines(), PAGES, {**SERVED, "changed": True}, RECORDS, at=AT)

    assert document["application"] == {
        "at": AT, "read": True, "commit": "f0bc8430", "changed": True}


def test_a_linked_page_the_clone_does_not_hold_is_not_found() -> None:
    """Mutation: call every linked page `read whole` -- red."""
    line = Line("G", "Read.", True, pages=("12-nowhere.md",))

    document = tour.review([line], PAGES)

    assert document["lines"][0]["stops"] == [
        {"kind": "page", "page": "12-nowhere.md", "title": None, "state": "not found"}]
    assert document["pages_not_found"] == [{"page": "12-nowhere.md", "group": "G", "text": "Read."}]


def test_feedback_on_a_stage_nothing_serves_is_kept() -> None:
    """Mutation: list only what a served stage was told -- red."""
    orphan = said("design/sketch", "off", "the sketch is gone")

    document = tour.review(lines(), PAGES, SERVED, [*RECORDS, orphan])

    assert document["feedback_on_no_stage"] == [orphan]


def test_only_off_keeps_the_lines_with_a_stage_off() -> None:
    """Mutation: keep every line -- red."""
    document = tour.review(lines(), PAGES, SERVED, RECORDS)

    narrowed = tour.only_off(document)

    assert [row["text"] for row in narrowed["lines"]] == [
        "A part is chosen and its numbers are staged."]
    assert narrowed["narrowed_to"] == "off" and document["narrowed_to"] is None
    assert len(document["lines"]) == 4


def test_only_off_keeps_a_line_whose_latest_off_was_said_of_an_earlier_build() -> None:
    """An `off` nobody has taken back stays in what is off when the
    application's commit moves; its line is then waiting on a word.

    Mutation: keep only what is off of this build -- red."""
    records = [said("design/apply", "off", "nothing is applied", commit="5caeea86")]

    narrowed = tour.only_off(tour.review(lines(), PAGES, SERVED, records))
    said_of = tour.render(narrowed)

    assert [row["text"] for row in narrowed["lines"]] == [
        "A part is chosen and its numbers are staged."]
    assert narrowed["disagreements"] == [
        {"group": "Designing a part", "text": "A part is chosen and its numbers are staged.",
         "roadmap": "ticked", "off": [], "earlier": ["design/apply"],
         "not_yet": ["design/choose", "design/numbers", "design/ready"]}]
    assert "        ✗ design/apply  of an earlier build, 5caeea86" in said_of
    assert said_of.endswith(
        "  only waiting on a word:\n"
        "    [x] A part is chosen and its numbers are staged."
        "  1 said of an earlier build, 3 not yet")


def test_only_off_drops_a_disagreement_that_is_only_unanswered() -> None:
    """Mutation: keep every disagreement -- red."""
    quiet = tour.review(lines(), PAGES, SERVED, [said("design/choose", "works")])
    assert [found["off"] for found in quiet["disagreements"]] == [[]]

    assert tour.only_off(quiet)["disagreements"] == []
    assert tour.only_off(quiet)["lines"] == []


def test_only_off_keeps_a_journey_no_line_names_when_a_stage_of_it_is_off() -> None:
    """`calibrate` is served and on no line, so no line can show its stage.

    Mutation: keep every unnamed journey -- red on the first assertion.
    Mutation: keep none -- red on the second."""
    level = said("calibrate/level", "off", "the bed is out by a turn")

    def unnamed(records: list[dict]) -> list[str]:
        narrowed = tour.only_off(tour.review(lines(), PAGES, SERVED, records))
        return [journey["id"] for journey in narrowed["journeys_not_on_the_roadmap"]]

    assert unnamed(RECORDS) == []
    assert unnamed([*RECORDS, level]) == ["calibrate"]


# --- for a terminal -------------------------------------------------------------

REVIEW = """\
roadmap      ROADMAP.md
application  http://127.0.0.1:6283  f0bc8430

Designing a part
  [x] A part is chosen and its numbers are staged.
      journey:design  1 works, 1 off, 1 said of an earlier build, 1 not yet
        ✓ design/choose  Ada  f0bc8430  2026-11-15T12:00:00+00:00
        ✗ design/numbers  f0bc8430 (changed)  2026-11-15T12:10:00+00:00  a staged number is lost on reload
        ✗ design/apply  Lin  of an earlier build, 5caeea86  2026-11-14T09:05:00+00:00  nothing is applied
  [x] An assembly is navigated by its pieces.
      09-an-assembly.md  An assembly  read whole

Making a thing
  [x] A part is sliced and printed on a machine.
      journey:make  the application does not serve it
  [ ] A print is weighed against its estimate.
      shown by nothing

Disagreements
  The roadmap ticks each line below, and the application holds a stage of
  it off, or with no word said of this build. Both stand; nothing here
  writes to a roadmap.
  a stage is off:
    [x] A part is chosen and its numbers are staged.
        off: design/numbers
        1 said of an earlier build, 1 not yet

Not on the roadmap
  pages no line names and no stage tries:
    10-a-drawing.md  A drawing
  journeys the application serves and no line names:
    journey:calibrate  Calibrate a machine  0 works, 0 off, 0 said of an earlier build, 1 not yet

Named by the roadmap, and not there
  journey:make  the application does not serve it  named by: A part is sliced and printed on a machine.
"""  # noqa: E501


def test_the_text_is_one_entry_a_line_with_its_counts_and_its_answered_stages() -> None:
    """The whole review, as a terminal shows it.

    Mutation: leave the note off an answered stage -- red.
    Mutation: leave out the commit and the time a record carries -- red.
    Mutation: list the stages nobody answered as well -- red.
    Mutation: leave the disagreements out -- red."""
    document = tour.review(lines(), PAGES, SERVED, RECORDS, roadmap="ROADMAP.md", at=AT)

    assert tour.render(document) + "\n" == REVIEW


def test_the_text_puts_the_lines_with_a_stage_off_before_those_only_waiting() -> None:
    """`B` stands first in the roadmap and is only waiting; `A` has a stage off.

    Mutation: list the disagreements in the roadmap's order, as one kind --
    red. Mutation: name every unanswered stage of a waiting line -- red."""
    said_of = tour.render(four_lines())

    assert (
        "Disagreements\n"
        "  The roadmap ticks each line below, and the application holds a stage of\n"
        "  it off, or with no word said of this build. Both stand; nothing here\n"
        "  writes to a roadmap.\n"
        "  a stage is off:\n"
        "    [x] A is built.\n"
        "        off: a/2\n"
        "        0 said of an earlier build, 1 not yet\n"
        "  only waiting on a word:\n"
        "    [x] B is built.  0 said of an earlier build, 2 not yet\n"
        "\n"
        "Not on the roadmap\n") in said_of


def test_the_text_of_an_application_nobody_has_answered_is_one_line_a_claim() -> None:
    """Every ticked journey line is a disagreement while nobody has said
    anything, and each takes one line.

    Mutation: head a block that holds nothing -- red on `a stage is off:`."""
    said_of = tour.render(tour.review(lines(), PAGES, SERVED, []))

    assert (
        "  writes to a roadmap.\n"
        "  only waiting on a word:\n"
        "    [x] A part is chosen and its numbers are staged."
        "  0 said of an earlier build, 4 not yet\n"
        "\n"
        "Not on the roadmap\n") in said_of
    assert "a stage is off:" not in said_of


def test_the_text_names_who_said_a_word_and_nothing_where_nobody_is_named() -> None:
    """One word with a name, one with an empty name, one kept before names.

    Mutation: leave the name out -- red on the first line.
    Mutation: print `by` whatever it holds -- red, `None` stands for a name."""
    records = [said("design/choose", "works", by="Ada"),
               said("design/numbers", "off", "a staged number is lost", by=""),
               said("design/apply", "works")]

    document = tour.review(lines(), PAGES, SERVED, records)
    said_of = tour.render(document)

    assert ("        ✓ design/choose  Ada  f0bc8430  2026-11-15T12:00:00+00:00\n"
            "        ✗ design/numbers  f0bc8430  2026-11-15T12:00:00+00:00"
            "  a staged number is lost\n"
            "        ✓ design/apply  f0bc8430  2026-11-15T12:00:00+00:00\n") in said_of
    assert "None" not in said_of
    # The document keeps the record as the application gave it.
    assert document["lines"][0]["stops"][0]["feedback"]["by"] == "Ada"
    assert "by" not in document["lines"][0]["stops"][2]["feedback"]


def test_the_text_says_shown_by_nothing_in_those_words() -> None:
    """Mutation: print nothing under a line that names nothing -- red."""
    said_of = tour.render(tour.review(lines(), PAGES))

    assert ("  [ ] A print is weighed against its estimate.\n"
            "      shown by nothing\n") in said_of


def test_the_text_counts_nothing_of_an_application_it_did_not_read() -> None:
    """Mutation: print a journey's counts whether or not it was read -- red:
    three zeros stand where nothing is known."""
    said_of = tour.render(tour.review(lines(), PAGES, unread="no --at was given"))

    assert "application  not read: no --at was given" in said_of
    assert ("      journey:design  its stages are the application's to say:"
            " pass --at URL, the address it is running at") in said_of
    assert "0 works" not in said_of
    assert "Disagreements\n  The application's to say." in said_of
    assert ("  pages no line names; whether a stage tries one is the application's to say:\n"
            "    04-designing-a-part.md  Designing a part\n"
            "    10-a-drawing.md  A drawing\n"
            "  journeys no line names: the application's to say.") in said_of


def test_the_text_lists_what_was_said_of_a_stage_nothing_serves() -> None:
    """Mutation: leave the section out -- red."""
    orphan = said("design/sketch", "off", "the sketch is gone", by="Lin")
    older = said("design/trace", "works", commit="5caeea86")

    said_of = tour.render(tour.review(lines(), PAGES, SERVED, [*RECORDS, orphan, older]))

    assert said_of.endswith(
        "Said of a stage the application does not serve\n"
        "  ✗ design/sketch  Lin  f0bc8430  2026-11-15T12:00:00+00:00  the sketch is gone\n"
        "  ✓ design/trace  of an earlier build, 5caeea86  2026-11-15T12:00:00+00:00")


def test_the_text_of_a_review_with_no_disagreement_says_none() -> None:
    """Mutation: print the disagreement preamble over an empty list -- red."""
    served = {"commit": "f0bc8430", "changed": False, "journeys": [
        {"id": "design", "name": "Design a part", "stages": [stage("design", "choose")]}]}
    roadmap = [Line("G", "Designed.", True, journeys=("design",))]

    said_of = tour.render(tour.review(roadmap, [], served, [said("design/choose", "works")]))

    assert said_of.endswith(
        "Disagreements\n"
        "  None: no ticked line has a stage off, or waiting on a word of this build.\n"
        "\n"
        "Not on the roadmap\n"
        "  Nothing: a line names every journey served, and a line or a stage every page.")


def test_the_narrowed_text_shows_a_stage_off_in_a_journey_no_line_names() -> None:
    """Mutation: print only the lines under `--off` -- red: the text says no
    line has a stage off, and the stage that is off is nowhere in it."""
    level = said("calibrate/level", "off", "the bed is out by a turn")
    document = tour.review(lines(), PAGES, SERVED, [said("design/choose", "works"), level])

    said_of = tour.render(tour.only_off(document))

    assert said_of.endswith(
        "No line has a stage off.\n"
        "\n"
        "Off, in a journey no line names\n"
        "    journey:calibrate  Calibrate a machine"
        "  0 works, 1 off, 0 said of an earlier build, 0 not yet\n"
        "      ✗ calibrate/level  f0bc8430  2026-11-15T12:00:00+00:00  the bed is out by a turn")


def test_the_narrowed_text_says_none_only_of_what_is_off() -> None:
    """An unticked line with a stage off, and a ticked line with a stage
    nobody answered: narrowed to what is off, no disagreement is left, and
    the ticked line still has its unanswered stage.

    Mutation: say of the narrowed list what is said of the whole -- red."""
    served = {"commit": "f0bc8430", "changed": False, "journeys": [
        {"id": "b", "name": "B", "stages": [stage("b", "1")]},
        {"id": "d", "name": "D", "stages": [stage("d", "1")]}]}
    roadmap = [Line("G", "B is built.", True, journeys=("b",)),
               Line("G", "D is planned.", False, journeys=("d",))]
    document = tour.review(roadmap, [], served, [said("d/1", "off", "no")])
    assert [found["text"] for found in document["disagreements"]] == ["B is built."]

    said_of = tour.render(tour.only_off(document))

    assert said_of.endswith("Disagreements\n  None: no ticked line has a stage off.")
    assert "B is built." not in said_of


def test_the_narrowed_text_claims_nothing_of_an_application_it_did_not_read() -> None:
    """Mutation: narrow an unread review like a read one -- red: it says no
    line has a stage off, of stages it never read."""
    said_of = tour.render(tour.only_off(tour.review(lines(), PAGES, unread="no --at was given")))

    assert said_of == ("application  not read: no --at was given\n"
                       "\n"
                       "Whether a stage is off is the application's to say.")


def test_the_narrowed_text_with_nothing_off_says_so_and_stops() -> None:
    """Mutation: print the disagreements under a narrowed review with no
    line -- red: `None` stands under a line with an unanswered stage."""
    document = tour.review(lines(), PAGES, SERVED, [said("design/choose", "works")])

    said_of = tour.render(tour.only_off(document))

    assert said_of.endswith("f0bc8430\n\nNo line has a stage off.")


# --- the command ----------------------------------------------------------------


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """A clone with a roadmap where a project keeps it, and its three pages."""
    root = tmp_path / "clone"
    (root / "governance" / "qm" / "project").mkdir(parents=True)
    (root / "governance" / "qm" / "project" / "ROADMAP.md").write_text(ROADMAP, encoding="utf-8")
    (root / "walkthrough").mkdir()
    for page in PAGES:
        number = page.name.split("-")[0]
        (root / "walkthrough" / page.name).write_text(
            f"# {number} — {page.title}\n", encoding="utf-8")
    return root


def answering(monkeypatch, answers: dict) -> list[str]:
    """Replace `httpx.get` with one that answers from `answers`, by address.

    An address it has no answer for refuses the connection. A dict is served
    as JSON, a string as text, an int as that status. Returns the addresses
    asked, in order.
    """
    asked: list[str] = []

    def fake_get(url, timeout=None):
        asked.append(url)
        request = httpx.Request("GET", url)
        if url not in answers:
            raise httpx.ConnectError("All connection attempts failed", request=request)
        answer = answers[url]
        if isinstance(answer, int):
            return httpx.Response(answer, request=request)
        if isinstance(answer, str):
            return httpx.Response(200, text=answer, request=request)
        return httpx.Response(200, json=answer, request=request)

    monkeypatch.setattr(httpx, "get", fake_get)
    return asked


def application() -> dict:
    return {f"{AT}/tour": SERVED, f"{AT}/tour/feedback": {"feedback": RECORDS}}


def test_the_command_reads_the_roadmap_the_pages_and_both_documents(monkeypatch, clone) -> None:
    """Mutation: read `/tour` alone and take the feedback it embeds -- red on
    what was asked. Mutation: look for the roadmap at the clone's root -- red."""
    asked = answering(monkeypatch, application())

    result = CliRunner().invoke(cli.cli, ["tour", str(clone), "--at", AT + "/"])

    assert result.exit_code == 0, result.output
    assert asked == [f"{AT}/tour", f"{AT}/tour/feedback"]
    roadmap = clone / "governance" / "qm" / "project" / "ROADMAP.md"
    assert result.output == REVIEW.replace("ROADMAP.md", str(roadmap), 1).replace(
        f"application  {AT}  ", f"application  {AT}/  ")


def test_a_missing_roadmap_is_named_with_the_path_looked_at(monkeypatch, tmp_path) -> None:
    """Mutation: read the roadmap without asking whether it is there -- red:
    the exit is an unhandled `FileNotFoundError`, and nothing names the path."""
    answering(monkeypatch, {})
    (tmp_path / "walkthrough").mkdir()

    result = CliRunner().invoke(cli.cli, ["tour", str(tmp_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit)
    looked_at = tmp_path / "governance" / "qm" / "project" / "ROADMAP.md"
    assert f"no roadmap at {looked_at}" in result.output


def test_roadmap_names_a_roadmap_that_stands_elsewhere(monkeypatch, clone, tmp_path) -> None:
    """Mutation: ignore `--roadmap` -- red, the clone's own roadmap is read."""
    answering(monkeypatch, {})
    elsewhere = tmp_path / "another.md"
    elsewhere.write_text("## Elsewhere\n\n- [ ] A line of another roadmap.\n", encoding="utf-8")

    result = CliRunner().invoke(cli.cli, ["tour", str(clone), "--roadmap", str(elsewhere)])

    assert result.exit_code == 0, result.output
    assert "Elsewhere\n  [ ] A line of another roadmap.\n      shown by nothing" in result.output
    assert "A part is chosen" not in result.output


def test_without_at_nothing_is_asked_and_the_line_says_how_to_pass_it(monkeypatch, clone) -> None:
    """Mutation: ask a default address when `--at` is not given -- red on
    what was asked."""
    asked = answering(monkeypatch, application())

    result = CliRunner().invoke(cli.cli, ["tour", str(clone)])

    assert result.exit_code == 0, result.output
    assert asked == []
    assert "application  not read: no --at was given" in result.output
    assert ("journey:design  its stages are the application's to say:"
            " pass --at URL, the address it is running at") in result.output
    assert "09-an-assembly.md  An assembly  read whole" in result.output


@pytest.mark.parametrize("answers, why", [
    ({}, f"{AT}/tour did not answer (All connection attempts failed)"),
    ({f"{AT}/tour": SERVED, f"{AT}/tour/feedback": 404}, f"{AT}/tour/feedback answered 404"),
    ({f"{AT}/tour": "<html>a page</html>"}, f"{AT}/tour did not answer in JSON"),
    ({f"{AT}/tour": {"detail": "Not Found"}}, f"{AT}/tour answered without a `journeys` list"),
    ({f"{AT}/tour": SERVED, f"{AT}/tour/feedback": {"feedback": None}},
     f"{AT}/tour/feedback answered without a `feedback` list"),
])
def test_an_application_that_does_not_answer_leaves_the_review_printed(
        monkeypatch, clone, answers, why) -> None:
    """Each way an address can fail to hand over its document.

    Mutation: let a refused connection, a refusal status or a body that is
    not JSON raise -- red, the command ends on the exception. Mutation: take
    whatever JSON came back as the document -- red on the last two."""
    answering(monkeypatch, answers)

    result = CliRunner().invoke(cli.cli, ["tour", str(clone), "--at", AT])

    assert result.exception is None, result.exception
    assert result.exit_code == 0
    assert f"application  not read: {why}" in result.output
    assert (f"journey:design  its stages are the application's to say,"
            f" and it did not answer at {AT}") in result.output
    assert "09-an-assembly.md  An assembly  read whole" in result.output
    assert "0 works" not in result.output


def test_json_emits_the_review_document(monkeypatch, clone) -> None:
    """Mutation: print the text whatever `--json` says -- red, it does not parse."""
    answering(monkeypatch, application())

    result = CliRunner().invoke(cli.cli, ["tour", str(clone), "--at", AT, "--json"])

    assert result.exit_code == 0, result.output
    roadmap = clone / "governance" / "qm" / "project" / "ROADMAP.md"
    assert json.loads(result.output) == tour.review(
        lines(), PAGES, SERVED, RECORDS, roadmap=str(roadmap), at=AT)


def test_off_narrows_the_text_and_the_document(monkeypatch, clone) -> None:
    """Mutation: ignore `--off` -- red, every line prints."""
    answering(monkeypatch, application())

    text = CliRunner().invoke(cli.cli, ["tour", str(clone), "--at", AT, "--off"])
    data = CliRunner().invoke(cli.cli, ["tour", str(clone), "--at", AT, "--off", "--json"])

    assert text.exit_code == 0, text.output
    assert "✗ design/numbers" in text.output
    assert "An assembly is navigated" not in text.output
    assert "Not on the roadmap" not in text.output
    assert "journey:calibrate" not in text.output
    assert ("        off: design/numbers\n"
            "        1 said of an earlier build, 1 not yet") in text.output
    assert "only waiting on a word" not in text.output
    assert [row["text"] for row in json.loads(data.output)["lines"]] == [
        "A part is chosen and its numbers are staged."]


def test_off_without_at_asks_nothing_and_claims_nothing(monkeypatch, clone) -> None:
    """Mutation: ask a default address under `--off` -- red on what was asked."""
    asked = answering(monkeypatch, application())

    result = CliRunner().invoke(cli.cli, ["tour", str(clone), "--off"])

    assert result.exit_code == 0, result.output
    assert asked == []
    assert "Whether a stage is off is the application's to say." in result.output
    assert "No line has a stage off" not in result.output
