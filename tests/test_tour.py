"""`qmcp.tour`: a roadmap read against the tour its application serves.

Two parts are held here: the readers (a roadmap's lines, a clone's pages) and
the review built from them and the application's two documents. Those
documents are written out here, so no test needs an application running.

Each test names the mutation of `qmcp/tour.py` it goes red under.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
         commit: str = "f0bc8430", changed: bool = False) -> dict:
    """One feedback record, shaped as `GET /tour/feedback` lists it."""
    return {"stage": of, "verdict": verdict, "note": note, "at": at,
            "commit": commit, "changed": changed}


# `design` is on the roadmap and served; `make` is on the roadmap and not
# served; `calibrate` is served and not on the roadmap. The design stages are
# out of alphabetical order, so an order that is not the journey's shows.
SERVED = {"commit": "f0bc8430", "changed": False, "journeys": [
    {"id": "design", "name": "Design a part", "stages": [
        stage("design", "choose", ("04-designing-a-part.md",)),
        stage("design", "numbers"),
        stage("design", "apply")]},
    {"id": "calibrate", "name": "Calibrate a machine", "stages": [
        stage("calibrate", "level")]},
]}

RECORDS = [
    said("design/choose", "off", "the list is empty", at="2026-11-14T09:00:00+00:00",
         commit="5caeea86"),
    said("design/numbers", "off", "a staged number is lost on reload",
         at="2026-11-15T12:10:00+00:00", commit="0a1b2c3d", changed=True),
    said("design/choose", "works"),
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
        ("design/apply", "not yet"),
        ("09-an-assembly.md", "read whole"),
    ]
    assert [stop["kind"] for stop in row["stops"]] == ["stage", "stage", "stage", "page"]
    assert row["stops"][0]["tries"] == ["04-designing-a-part.md"]
    assert row["stops"][3]["title"] == "An assembly"
    assert (row["group"], row["text"], row["ticked"], row["shown_by_nothing"]) == (
        "Designing a part", "Designed and read.", True, False)


def test_the_latest_record_is_a_stages_feedback() -> None:
    """`design/choose` was said `off` and later `works`.

    Mutation: keep the first record of a stage -- red."""
    row = tour.review(lines(), PAGES, SERVED, RECORDS)["lines"][0]

    assert row["stops"][0]["state"] == "works"
    assert row["stops"][0]["feedback"] == RECORDS[2]
    assert row["stops"][2]["feedback"] is None


def test_the_records_are_read_and_the_embedded_copy_is_not() -> None:
    """Mutation: fall back to the `feedback` a stage embeds -- red."""
    embedded = {**SERVED, "journeys": [{"id": "design", "name": "Design a part", "stages": [
        stage("design", "choose", feedback=said("design/choose", "works"))]}]}

    row = tour.review(lines(), PAGES, embedded, [])["lines"][0]

    assert states(row) == [("design/choose", "not yet")]
    assert row["stops"][0]["feedback"] is None


def test_a_ticked_line_with_a_stage_off_or_unanswered_is_a_disagreement() -> None:
    """Four lines: off and unanswered, unanswered only, every stage works, and
    an unticked line with a stage off.

    Mutation: count only `off` -- red, the second line is lost.
    Mutation: ignore the tick -- red, the fourth line appears.
    Mutation: call any ticked journey line a disagreement -- red on the third."""
    served = {"commit": "f0bc8430", "changed": False, "journeys": [
        {"id": "a", "name": "A", "stages": [stage("a", "1"), stage("a", "2"), stage("a", "3")]},
        {"id": "b", "name": "B", "stages": [stage("b", "1"), stage("b", "2")]},
        {"id": "c", "name": "C", "stages": [stage("c", "1")]},
        {"id": "d", "name": "D", "stages": [stage("d", "1")]},
    ]}
    records = [said("a/1", "works"), said("a/2", "off", "no"), said("b/1", "works"),
               said("c/1", "works"), said("d/1", "off", "no")]
    roadmap = [Line("G", "A is built.", True, journeys=("a",)),
               Line("G", "B is built.", True, journeys=("b",)),
               Line("G", "C is built.", True, journeys=("c",)),
               Line("G", "D is planned.", False, journeys=("d",))]

    document = tour.review(roadmap, [], served, records)

    assert document["disagreements"] == [
        {"group": "G", "text": "A is built.", "roadmap": "ticked",
         "off": ["a/2"], "not_yet": ["a/3"]},
        {"group": "G", "text": "B is built.", "roadmap": "ticked",
         "off": [], "not_yet": ["b/2"]},
    ]
    # Neither side is taken: the line stays ticked and the stage stays off.
    assert document["lines"][0]["ticked"] is True
    assert states(document["lines"][0])[1] == ("a/2", "off")


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
      journey:design  1 works, 1 off, 1 not yet
        ✓ design/choose  f0bc8430  2026-11-15T12:00:00+00:00
        ✗ design/numbers  0a1b2c3d (changed)  2026-11-15T12:10:00+00:00  a staged number is lost on reload
  [x] An assembly is navigated by its pieces.
      09-an-assembly.md  An assembly  read whole

Making a thing
  [x] A part is sliced and printed on a machine.
      journey:make  the application does not serve it
  [ ] A print is weighed against its estimate.
      shown by nothing

Disagreements
  The roadmap ticks each line below, and the application holds a stage of
  it off or not yet answered. Both stand; nothing here writes to a roadmap.
  [x] A part is chosen and its numbers are staged.
      off: design/numbers
      not yet: design/apply

Not on the roadmap
  pages no line names and no stage tries:
    10-a-drawing.md  A drawing
  journeys the application serves and no line names:
    journey:calibrate  Calibrate a machine  0 works, 0 off, 1 not yet

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
    orphan = said("design/sketch", "off", "the sketch is gone")

    said_of = tour.render(tour.review(lines(), PAGES, SERVED, [*RECORDS, orphan]))

    assert said_of.endswith(
        "Said of a stage the application does not serve\n"
        "  ✗ design/sketch  f0bc8430  2026-11-15T12:00:00+00:00  the sketch is gone")


def test_the_text_of_a_review_with_no_disagreement_says_none() -> None:
    """Mutation: print the disagreement preamble over an empty list -- red."""
    served = {"commit": "f0bc8430", "changed": False, "journeys": [
        {"id": "design", "name": "Design a part", "stages": [stage("design", "choose")]}]}
    roadmap = [Line("G", "Designed.", True, journeys=("design",))]

    said_of = tour.render(tour.review(roadmap, [], served, [said("design/choose", "works")]))

    assert said_of.endswith(
        "Disagreements\n"
        "  None: no ticked line has a stage off or not yet answered.\n"
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
        "    journey:calibrate  Calibrate a machine  0 works, 1 off, 0 not yet\n"
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
