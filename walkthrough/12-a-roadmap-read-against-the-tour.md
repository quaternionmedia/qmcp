# 12 — A roadmap read against the tour

Everything on this page runs. It is executed by the ordinary test command, so
an example that stops being true fails the build rather than sitting here
misleading somebody.

A project's roadmap says what the project claims is built. An application
that carries a guide of its own says something else: the guide walks a person
through the application's *journeys*, stage by stage, and keeps what they said
of each stage, `works`, or `off` with a note. `qmcp tour` reads the roadmap,
the clone's walkthrough pages and the application's two documents, and lays
them side by side. `docs/tour.md` is the reference. This page reads a small
roadmap, with the application's documents written out by hand.

## A line names what shows it

A roadmap is Markdown. A `## ` heading opens a group, and each checkbox item
under it is a claim, ticked when the project says it is built. A line names a
journey as `journey:<id>` in a code span, and a walkthrough page by a link:

    >>> from qmcp import tour
    >>> roadmap = """\
    ... # Roadmap
    ...
    ... ## Designing a part
    ...
    ... - [x] A part is chosen and its numbers are staged. `journey:design`
    ... - [x] An assembly is navigated by its pieces. [09](../../../walkthrough/09-an-assembly.md)
    ... - [ ] A part is priced.
    ... """
    >>> lines = tour.roadmap_lines(roadmap)
    >>> for line in lines:
    ...     print(line.ticked, line.journeys, line.pages, "|", line.text)
    True ('design',) () | A part is chosen and its numbers are staged.
    True () ('09-an-assembly.md',) | An assembly is navigated by its pieces.
    False () () | A part is priced.

The names are taken out of the claim's text. The third line names neither a
journey nor a page, and it is kept:

    >>> [line.shown_by_nothing for line in lines]
    [False, False, True]

## The clone's pages

A page is `walkthrough/NN-<slug>.md` in the clone, and its title is its first
heading without its number:

    >>> import tempfile
    >>> from pathlib import Path
    >>> clone = Path(tempfile.mkdtemp())
    >>> (clone / "walkthrough").mkdir()
    >>> for name, heading in [("04-designing-a-part.md", "# 04 — Designing a part"),
    ...                       ("09-an-assembly.md", "# 09 — An assembly"),
    ...                       ("10-a-drawing.md", "# 10 — A drawing")]:
    ...     _ = (clone / "walkthrough" / name).write_text(heading + "\n", encoding="utf-8")
    >>> pages = tour.read_pages(clone)
    >>> [(page.name, page.title) for page in pages]
    [('04-designing-a-part.md', 'Designing a part'), ('09-an-assembly.md', 'An assembly'), ('10-a-drawing.md', 'A drawing')]

## Without the application, nothing of it is claimed

The roadmap and the pages are a review already. What only the application
holds is marked as its to say, with the reason, and is not an empty list:

    >>> alone = tour.review(lines, pages, unread="no --at was given")
    >>> alone["lines"][0]["stops"]
    [{'kind': 'journey', 'journey': 'design', 'state': "the application's to say"}]
    >>> alone["lines"][1]["stops"]
    [{'kind': 'page', 'page': '09-an-assembly.md', 'title': 'An assembly', 'state': 'read whole'}]
    >>> alone["disagreements"]
    {'unknown': 'no --at was given'}

A page carries no feedback. A person reads it whole, and `read whole` is the
state the review gives it.

## What the application serves

Two documents. `GET /tour` lists the journeys, each with its stages in order
and the pages a stage tries. Here they are cut down to the keys the review
reads:

    >>> def stage(journey, name, *pages):
    ...     return {"id": f"{journey}/{name}", "journey": journey, "stage": name,
    ...             "title": name.capitalize(), "tries": [{"page": page} for page in pages],
    ...             "feedback": None}
    >>> served = {"commit": "f0bc8430", "changed": False, "journeys": [
    ...     {"id": "design", "name": "Design a part", "stages": [
    ...         stage("design", "choose", "04-designing-a-part.md"),
    ...         stage("design", "numbers"),
    ...         stage("design", "ready")]},
    ...     {"id": "calibrate", "name": "Calibrate a machine", "stages": [
    ...         stage("calibrate", "level")]}]}

`GET /tour/feedback` lists every record the application keeps, oldest first.
Choosing a part was said to be off, and on a later commit to work; the staged
numbers were said to be off; nobody has answered the third stage:

    >>> def said(of, verdict, note, at, commit):
    ...     return {"stage": of, "verdict": verdict, "note": note, "at": at,
    ...             "commit": commit, "changed": False}
    >>> records = [
    ...     said("design/choose", "off", "the list is empty",
    ...          "2026-11-14T09:00:00+00:00", "5caeea86"),
    ...     said("design/choose", "works", "", "2026-11-15T12:00:00+00:00", "f0bc8430"),
    ...     said("design/numbers", "off", "a staged number is lost on reload",
    ...          "2026-11-15T12:10:00+00:00", "f0bc8430")]

## Each line, with its stops

A line's stops are its journey's stages in the journey's order, then its
pages. A stage's feedback is the latest record that names it:

    >>> document = tour.review(lines, pages, served, records, at="http://application.test")
    >>> for stop in document["lines"][0]["stops"]:
    ...     print(stop["id"], "|", stop["state"], "|", (stop["feedback"] or {}).get("commit"))
    design/choose | works | f0bc8430
    design/numbers | off | f0bc8430
    design/ready | not yet | None

## A ticked line with a stage off is a disagreement

The roadmap ticks the first line. The application holds one of its stages off
and one unanswered. The review reports that as a disagreement, with what each
side says, and changes neither:

    >>> document["disagreements"]
    [{'group': 'Designing a part', 'text': 'A part is chosen and its numbers are staged.', 'roadmap': 'ticked', 'off': ['design/numbers'], 'not_yet': ['design/ready']}]
    >>> document["lines"][0]["ticked"]
    True

The document holds the ids of both kinds of stage. The text gives a line with
a stage off its ids, and a line that is only waiting on a word one line with
a count. Before anybody has answered, every ticked journey line is one of
those:

    >>> unanswered = tour.render(tour.review(lines, pages, served, []))
    >>> print(unanswered[unanswered.index("Disagreements"):unanswered.index("\n\nNot on")])
    Disagreements
      The roadmap ticks each line below, and the application holds a stage of
      it off or not yet answered. Both stand; nothing here writes to a roadmap.
      only waiting on a word:
        [x] A part is chosen and its numbers are staged.  3 not yet

## What is not on the roadmap

Page 04 is tried by a stage and page 09 is linked by a line, so page 10 is
the one nothing names. The application also serves a journey no line names:

    >>> document["pages_not_on_the_roadmap"]
    [{'page': '10-a-drawing.md', 'title': 'A drawing'}]
    >>> [journey["id"] for journey in document["journeys_not_on_the_roadmap"]]
    ['calibrate']

## The command

`qmcp tour CLONE --at URL` reads the roadmap from the clone's
`governance/qm/project/ROADMAP.md` and asks the application at `URL` for the
two documents. Here the application is stood in for: `httpx.get` answers from
the documents above.

    >>> import httpx
    >>> from click.testing import CliRunner
    >>> from qmcp.cli import cli
    >>> (clone / "governance" / "qm" / "project").mkdir(parents=True)
    >>> _ = (clone / "governance" / "qm" / "project" / "ROADMAP.md").write_text(
    ...     roadmap, encoding="utf-8")
    >>> answers = {"http://application.test/tour": served,
    ...            "http://application.test/tour/feedback": {"feedback": records}}
    >>> def answer(url, timeout=None):
    ...     return httpx.Response(200, json=answers[url], request=httpx.Request("GET", url))
    >>> real_get, httpx.get = httpx.get, answer

The first line of the output names the roadmap that was read, by its path; the
rest is the review:

    >>> result = CliRunner().invoke(cli, ["tour", str(clone), "--at", "http://application.test"])
    >>> print(result.output.split("\n", 1)[1].rstrip())
    application  http://application.test  f0bc8430
    <BLANKLINE>
    Designing a part
      [x] A part is chosen and its numbers are staged.
          journey:design  1 works, 1 off, 1 not yet
            ✓ design/choose  f0bc8430  2026-11-15T12:00:00+00:00
            ✗ design/numbers  f0bc8430  2026-11-15T12:10:00+00:00  a staged number is lost on reload
      [x] An assembly is navigated by its pieces.
          09-an-assembly.md  An assembly  read whole
      [ ] A part is priced.
          shown by nothing
    <BLANKLINE>
    Disagreements
      The roadmap ticks each line below, and the application holds a stage of
      it off or not yet answered. Both stand; nothing here writes to a roadmap.
      a stage is off:
        [x] A part is chosen and its numbers are staged.
            off: design/numbers
            1 not yet
    <BLANKLINE>
    Not on the roadmap
      pages no line names and no stage tries:
        10-a-drawing.md  A drawing
      journeys the application serves and no line names:
        journey:calibrate  Calibrate a machine  0 works, 0 off, 1 not yet

`--off` keeps the lines with a stage somebody said is off, and `--json` emits
the document:

    >>> import json
    >>> narrowed = CliRunner().invoke(
    ...     cli, ["tour", str(clone), "--at", "http://application.test", "--off", "--json"])
    >>> [line["text"] for line in json.loads(narrowed.output)["lines"]]
    ['A part is chosen and its numbers are staged.']

    >>> httpx.get = real_get

Without `--at` nothing is asked, and the journey line says whose its stages
are to say:

    >>> result = CliRunner().invoke(cli, ["tour", str(clone)])
    >>> print(result.output.splitlines()[5])
          journey:design  its stages are the application's to say: pass --at URL, the address it is running at

A clone with no roadmap is refused, with the path that was looked at:

    >>> bare = Path(tempfile.mkdtemp())
    >>> result = CliRunner().invoke(cli, ["tour", str(bare)])
    >>> result.exit_code
    1
    >>> expected = bare / "governance" / "qm" / "project" / "ROADMAP.md"
    >>> result.output.startswith(f"Error: no roadmap at {expected}.")
    True

## What this page does not claim

That an application serves these documents: the journeys and the records
above are written on the page, in the shape `docs/tour.md` gives, and an
application answering at an address is a person's check. Nor that the feedback
is about the build the roadmap describes. A record carries the commit it was
said of, the review prints it, and nothing compares it with any other.
