"""A project's roadmap, read against the tour its application serves.

    uv run qmcp tour <clone>                              # the roadmap and the pages
    uv run qmcp tour <clone> --at http://127.0.0.1:6283   # and what people said

A roadmap is a Markdown document in the project's branch of the governance
corpus. Each checkbox item in it is a claim, ticked when the project says it
is built, and it names what shows it: a journey, as `journey:<id>` in a code
span, or walkthrough pages, by link. A journey is an ordered set of stages the
application's own guide walks a person through, and the application keeps what
each person said of a stage: `works`, or `off` with a note.

The review lays those side by side: each line of the roadmap, the stages and
pages that show it, and the latest thing said of each stage. Nothing here
writes. A ticked line with a stage `off`, or with a stage nobody has answered,
is a disagreement and is reported as one, with what each side says
(`records/DRAFT-a-disagreement-is-a-delta.md`); the roadmap is a person's to
change.

What only the application holds, when it was not asked or did not answer, is
`{"unknown": "<reason>"}` in the document, never an empty list.

What this cannot see:

  * whether feedback is about the build the roadmap describes: a record
    carries the commit it was said of, and nothing here compares commits;
  * whether a page shows what its line claims: a page is read whole by a
    person, and its state here says only that;
  * a name written any other way: a journey outside a code span or a page
    named without a link is prose, and its line is `shown by nothing`.

`docs/tour.md` is the reference for the command and the documents it reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = 1
"""The review document's version. It moves only for a breaking change."""

# What a stage's latest record says, or that it has none.
WORKS = "works"
OFF = "off"
NOT_YET = "not yet"

# A page carries no feedback: a person reads it whole, or the clone has no
# such file.
READ_WHOLE = "read whole"
NOT_FOUND = "not found"

# A journey whose stages cannot be listed: no application was read, or the one
# read does not serve it.
THEIRS = "the application's to say"
NOT_SERVED = "not served"

SHOWN_BY_NOTHING = "shown by nothing"

MARK = {WORKS: "✓", OFF: "✗"}

_GROUP = re.compile(r"^##\s+(?P<name>.+?)\s*$")
_ITEM = re.compile(r"^\s*[-*+]\s+\[(?P<tick>[ xX])\]\s+(?P<words>.*)$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_JOURNEY = re.compile(r"`journey:([a-z0-9-]+)`")
# The target ends in `walkthrough/NN-<slug>.md`, after any relative prefix and
# before an optional fragment. The page is named by its file.
_PAGE_LINK = re.compile(
    r"\[[^\]]*\]\((?:[^)\s]*/)?walkthrough/(\d{2,}-[^/)\s#]+\.md)(?:#[^)\s]*)?\)")
_PAGE_FILE = re.compile(r"^(?P<number>\d{2,})-.+\.md$")
_TITLE = re.compile(r"^# (?:\d+ — )?(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class Line:
    """One acceptance line of a roadmap: a claim, and what it names as showing it."""

    group: str | None
    """The `## ` heading the line stands under, or None above the first one."""
    text: str
    ticked: bool
    journeys: tuple[str, ...] = ()
    pages: tuple[str, ...] = ()
    """Walkthrough pages, each by its file name."""

    @property
    def shown_by_nothing(self) -> bool:
        return not self.journeys and not self.pages


@dataclass(frozen=True)
class Page:
    """One walkthrough page of a clone."""

    name: str
    title: str | None
    """Its first `# ` heading without the page's number, or None without one."""


# --- the roadmap ----------------------------------------------------------------


def _line(group: str | None, ticked: bool, words: str) -> Line:
    # A name leaves the claim with the commas and spaces that follow it:
    # those between it and the next name, and those after the last.
    marked = _PAGE_LINK.sub("\0", _JOURNEY.sub("\0", words))
    claim = re.sub(r"\0[\s,]*", " ", marked)
    return Line(
        group=group,
        text=" ".join(claim.split()),
        ticked=ticked,
        journeys=tuple(dict.fromkeys(_JOURNEY.findall(words))),
        pages=tuple(dict.fromkeys(_PAGE_LINK.findall(words))),
    )


def roadmap_lines(text: str) -> list[Line]:
    """Every acceptance line of a roadmap, in the roadmap's order.

    A `## ` heading opens a group. A checkbox item is a line, with the indented
    lines that continue it; a nested list item is not part of it. A fenced
    block is an example and holds no lines.
    """
    found: list[Line] = []
    group: str | None = None
    ticked, words = False, []
    fenced = False

    def close() -> None:
        if words:
            found.append(_line(group, ticked, " ".join(words)))
            words.clear()

    for raw in text.splitlines():
        if raw.lstrip().startswith(("```", "~~~")):
            close()
            fenced = not fenced
            continue
        if fenced:
            continue
        item = _ITEM.match(raw)
        if item:
            close()
            ticked = item["tick"] != " "
            words.append(item["words"])
            continue
        if words and raw[:1] in (" ", "\t") and raw.strip() and not _BULLET.match(raw):
            words.append(raw.strip())
            continue
        close()
        heading = _GROUP.match(raw)
        if heading:
            group = heading["name"]
    close()
    return found


def read_roadmap(path: Path) -> list[Line]:
    return roadmap_lines(Path(path).read_text(encoding="utf-8", errors="replace"))


# --- the pages ------------------------------------------------------------------


def page_title(text: str) -> str | None:
    """A page's first `# ` heading, written `# NN — Title` or `# Title`."""
    for line in text.splitlines():
        heading = _TITLE.match(line)
        if heading:
            return heading["title"]
    return None


def read_pages(clone: Path) -> list[Page]:
    """The clone's `walkthrough/NN-*.md`, in the pages' order."""
    folder = Path(clone) / "walkthrough"
    numbered = []
    for path in folder.glob("*.md"):
        named = _PAGE_FILE.match(path.name)
        if named:
            numbered.append((int(named["number"]), path.name, path))
    return [Page(name, page_title(path.read_text(encoding="utf-8", errors="replace")))
            for _, name, path in sorted(numbered)]


# --- the review -----------------------------------------------------------------


def unknown(reason: str) -> dict[str, str]:
    """What only the application holds, when it was not read."""
    return {"unknown": reason}


def latest(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each stage's feedback: the latest record that names it.

    Records are kept oldest first and never rewritten, so the order they are
    listed in is the order they were said in.
    """
    said: dict[str, dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and record.get("stage"):
            said[record["stage"]] = record
    return said


def _stage_stop(journey: dict[str, Any], stage: dict[str, Any],
                said: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = said.get(stage.get("id"))
    verdict = record.get("verdict") if record else None
    return {
        "kind": "stage",
        "id": stage.get("id"),
        "journey": journey.get("id"),
        "stage": stage.get("stage"),
        "title": stage.get("title"),
        "tries": [t["page"] for t in stage.get("tries") or [] if t.get("page")],
        "state": verdict if verdict in (WORKS, OFF) else NOT_YET,
        "feedback": record,
    }


def _stops(journey: dict[str, Any], said: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [_stage_stop(journey, stage, said) for stage in journey.get("stages") or []]


def review(lines: list[Line], pages: list[Page],
           served: dict[str, Any] | None = None,
           records: list[dict[str, Any]] | None = None, *,
           roadmap: str | None = None, at: str | None = None,
           unread: str = "the application was not asked") -> dict[str, Any]:
    """The review document: every line of the roadmap, and what shows it.

    `served` is the application's `GET /tour` document and `records` the list
    in its `GET /tour/feedback`; a stage's feedback is read from the records,
    never from the copy `served` embeds. With `served` None the application
    was not read, `unread` says why, and everything only it holds is
    `unknown(unread)`.

    A line's stops are its journeys' stages in each journey's order, then its
    pages. `roadmap` and `at` are carried into the document as given.
    """
    read = served is not None
    journeys = {j.get("id"): j for j in (served or {}).get("journeys") or []
                if isinstance(j, dict)}
    said = latest(records or [])
    titles = {page.name: page.title for page in pages}

    rows, not_served, not_found = [], [], []
    for line in lines:
        stops: list[dict[str, Any]] = []
        for name in line.journeys:
            if not read:
                stops.append({"kind": "journey", "journey": name, "state": THEIRS})
            elif name not in journeys:
                stops.append({"kind": "journey", "journey": name, "state": NOT_SERVED})
                not_served.append({"journey": name, "group": line.group, "text": line.text})
            else:
                stops += _stops(journeys[name], said)
        for name in line.pages:
            held = name in titles
            stops.append({"kind": "page", "page": name, "title": titles.get(name),
                          "state": READ_WHOLE if held else NOT_FOUND})
            if not held:
                not_found.append({"page": name, "group": line.group, "text": line.text})
        rows.append({
            "group": line.group,
            "text": line.text,
            "ticked": line.ticked,
            "journeys": list(line.journeys),
            "pages": list(line.pages),
            "shown_by_nothing": line.shown_by_nothing,
            "stops": stops,
        })

    linked = {name for line in lines for name in line.pages}
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "roadmap": roadmap,
        "application": {"at": at, "read": read},
        "narrowed_to": None,
        "lines": rows,
        "pages_no_line_names": [{"page": page.name, "title": page.title}
                                for page in pages if page.name not in linked],
        "pages_not_found": not_found,
    }
    if not read:
        document["application"]["why"] = unread
        for theirs in ("disagreements", "pages_not_on_the_roadmap",
                       "journeys_not_on_the_roadmap", "journeys_not_served",
                       "feedback_on_no_stage"):
            document[theirs] = unknown(unread)
        return document

    document["application"].update(commit=served.get("commit"), changed=served.get("changed"))

    disagreements = []
    for row in rows:
        states = {state: [stop["id"] for stop in row["stops"]
                          if stop["kind"] == "stage" and stop["state"] == state]
                  for state in (OFF, NOT_YET)}
        if row["ticked"] and (states[OFF] or states[NOT_YET]):
            disagreements.append({"group": row["group"], "text": row["text"],
                                  "roadmap": "ticked",
                                  "off": states[OFF], "not_yet": states[NOT_YET]})
    document["disagreements"] = disagreements

    stages = [stage for journey in journeys.values() for stage in journey.get("stages") or []]
    tried = {t.get("page") for stage in stages for t in stage.get("tries") or []}
    document["pages_not_on_the_roadmap"] = [
        entry for entry in document["pages_no_line_names"] if entry["page"] not in tried]

    named = {name for line in lines for name in line.journeys}
    document["journeys_not_on_the_roadmap"] = [
        {"id": name, "name": journey.get("name"), "stops": _stops(journey, said)}
        for name, journey in journeys.items() if name not in named]
    document["journeys_not_served"] = not_served

    served_stages = {stage.get("id") for stage in stages}
    document["feedback_on_no_stage"] = [
        record for stage, record in said.items() if stage not in served_stages]
    return document


def _has_off(stops: list[dict[str, Any]]) -> bool:
    return any(stop["state"] == OFF for stop in stops)


def only_off(document: dict[str, Any]) -> dict[str, Any]:
    """The review narrowed to what somebody said is off.

    The lines with a stage off, the disagreements among them, and the journeys
    no line names that have a stage off. The rest of the document stands.
    """
    narrowed = dict(document, narrowed_to=OFF)
    narrowed["lines"] = [row for row in document["lines"] if _has_off(row["stops"])]
    if document["application"]["read"]:
        narrowed["disagreements"] = [d for d in document["disagreements"] if d["off"]]
        narrowed["journeys_not_on_the_roadmap"] = [
            journey for journey in document["journeys_not_on_the_roadmap"]
            if _has_off(journey["stops"])]
    return narrowed


# --- for a terminal -------------------------------------------------------------


def counts(stops: list[dict[str, Any]]) -> dict[str, int]:
    """How many of these stops are stages in each state."""
    states = [stop["state"] for stop in stops if stop["kind"] == "stage"]
    return {state: states.count(state) for state in (WORKS, OFF, NOT_YET)}


def _tally(stops: list[dict[str, Any]]) -> str:
    return ", ".join(f"{n} {state}" for state, n in counts(stops).items())


def _commit(carrier: dict[str, Any]) -> str:
    commit = carrier.get("commit") or "no commit"
    return f"{commit} (changed)" if carrier.get("changed") else commit


def _answered(record: dict[str, Any]) -> str:
    """One record: its mark and stage, the commit and time it carries, its note."""
    parts = [f"{MARK.get(record.get('verdict'), '?')} {record.get('stage')}", _commit(record),
             record.get("at") or "no time", record.get("note") or ""]
    return "  ".join(part for part in parts if part)


def _stage_lines(stops: list[dict[str, Any]], indent: str) -> list[str]:
    return [indent + _answered(stop["feedback"]) for stop in stops
            if stop["kind"] == "stage" and stop["state"] in MARK]


def _box(ticked: bool) -> str:
    return "[x]" if ticked else "[ ]"


def _row_lines(row: dict[str, Any], application: dict[str, Any]) -> list[str]:
    out = [f"  {_box(row['ticked'])} {row['text']}"]
    if row["shown_by_nothing"]:
        out.append(f"      {SHOWN_BY_NOTHING}")
    for name in row["journeys"]:
        mine = [stop for stop in row["stops"] if stop.get("journey") == name]
        if mine and mine[0]["kind"] == "journey":
            if mine[0]["state"] == NOT_SERVED:
                out.append(f"      journey:{name}  the application does not serve it")
            elif application["at"]:
                out.append(f"      journey:{name}  its stages are the application's to say,"
                           f" and it did not answer at {application['at']}")
            else:
                out.append(f"      journey:{name}  its stages are the application's to say:"
                           " pass --at URL, the address it is running at")
            continue
        out.append(f"      journey:{name}  {_tally(mine)}")
        out += _stage_lines(mine, "        ")
    for stop in row["stops"]:
        if stop["kind"] == "page":
            parts = [stop["page"], stop["title"] or "",
                     READ_WHOLE if stop["state"] == READ_WHOLE
                     else "not found in the clone's walkthrough"]
            out.append("      " + "  ".join(part for part in parts if part))
    return out


def _journey_lines(journeys: list[dict[str, Any]]) -> list[str]:
    out = []
    for journey in journeys:
        parts = [f"journey:{journey['id']}", journey["name"] or "", _tally(journey["stops"])]
        out.append("    " + "  ".join(part for part in parts if part))
        out += _stage_lines(journey["stops"], "      ")
    return out


def _page_lines(pages: list[dict[str, Any]]) -> list[str]:
    return ["    " + "  ".join(part for part in (page["page"], page["title"]) if part)
            for page in pages]


def _named(entry: dict[str, Any], what: str) -> str:
    return f"  {what}  named by: {entry['text']}"


def _disagreement_lines(document: dict[str, Any]) -> list[str]:
    disagreements = document["disagreements"]
    if not document["application"]["read"]:
        return ["  The application's to say."]
    if not disagreements and document["narrowed_to"] == OFF:
        return ["  None: no ticked line has a stage off."]
    if not disagreements:
        return ["  None: no ticked line has a stage off or not yet answered."]
    out = ["  The roadmap ticks each line below, and the application holds a stage of",
           "  it off or not yet answered. Both stand; nothing here writes to a roadmap."]
    # The lines a person has something to read on come first; a line nobody
    # has answered takes one line, with a count where the document has the ids.
    with_off = [found for found in disagreements if found["off"]]
    waiting = [found for found in disagreements if not found["off"]]
    if with_off:
        out.append("  a stage is off:")
    for found in with_off:
        out += [f"    [x] {found['text']}", "        off: " + ", ".join(found["off"])]
        if found["not_yet"]:
            out.append(f"        {len(found['not_yet'])} {NOT_YET}")
    if waiting:
        out.append("  only waiting on a word:")
    out += [f"    [x] {found['text']}  {len(found['not_yet'])} {NOT_YET}" for found in waiting]
    return out


def _rest_lines(document: dict[str, Any]) -> list[str]:
    """What is on one side only: not on the roadmap, or named by it and not there."""
    read = document["application"]["read"]
    out = ["", "Not on the roadmap"]
    if read:
        pages = document["pages_not_on_the_roadmap"]
        unnamed = document["journeys_not_on_the_roadmap"]
        if not pages and not unnamed:
            out.append("  Nothing: a line names every journey served, and a line or a"
                       " stage every page.")
        if pages:
            out.append("  pages no line names and no stage tries:")
        out += _page_lines(pages)
        if unnamed:
            out.append("  journeys the application serves and no line names:")
        out += _journey_lines(unnamed)
    else:
        pages = document["pages_no_line_names"]
        if pages:
            out.append("  pages no line names; whether a stage tries one is the"
                       " application's to say:")
        out += _page_lines(pages)
        out.append("  journeys no line names: the application's to say.")

    not_served = document["journeys_not_served"] if read else []
    if not_served or document["pages_not_found"]:
        out += ["", "Named by the roadmap, and not there"]
        out += [_named(entry, f"journey:{entry['journey']}  the application does not serve it")
                for entry in not_served]
        out += [_named(entry, f"{entry['page']}  not found in the clone's walkthrough")
                for entry in document["pages_not_found"]]

    orphans = document["feedback_on_no_stage"] if read else []
    if orphans:
        out += ["", "Said of a stage the application does not serve"]
        out += ["  " + _answered(record) for record in orphans]
    return out


def render(document: dict[str, Any]) -> str:
    """The review, for a terminal. Reads the document and nothing else."""
    application = document["application"]
    read = application["read"]
    narrowed = document["narrowed_to"] == OFF
    out = []
    if document["roadmap"]:
        out.append(f"roadmap      {document['roadmap']}")
    if read:
        parts = [application["at"] or "", _commit(application)]
        out.append("application  " + "  ".join(part for part in parts if part))
    else:
        out.append(f"application  not read: {application['why']}")
    if narrowed and not read:
        out += ["", "Whether a stage is off is the application's to say."]
        return "\n".join(out)

    rows = document["lines"]
    if narrowed and not rows:
        out += ["", "No line has a stage off."]
    elif not rows:
        out += ["", "The roadmap has no acceptance lines: a line is a `- [x]` or `- [ ]` item."]
    group = object()
    for row in rows:
        if row["group"] != group:
            group = row["group"]
            out += ["", group or "(under no heading)"]
        out += _row_lines(row, application)

    if not narrowed:
        out += ["", "Disagreements", *_disagreement_lines(document), *_rest_lines(document)]
        return "\n".join(out)
    if rows:
        out += ["", "Disagreements", *_disagreement_lines(document)]
    if document["journeys_not_on_the_roadmap"]:
        out += ["", "Off, in a journey no line names",
                *_journey_lines(document["journeys_not_on_the_roadmap"])]
    return "\n".join(out)
