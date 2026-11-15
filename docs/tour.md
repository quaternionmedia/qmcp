# The tour review

A roadmap says what a project claims is built. `qmcp tour` shows what stands
behind each claim and what people said of it: the stages of the application's
own guide and the walkthrough pages a line names, with the latest feedback on
each stage. The code is `qmcp/tour.py`, and
`walkthrough/12-a-roadmap-read-against-the-tour.md` runs it end to end.

The guide itself is the application's. It walks a person through the
application's *journeys*, each an ordered set of *stages*, and keeps what they
said of a stage: `works`, or `off` with a note. qmcp reads that and the
roadmap, and writes to neither.

## The command

```bash
uv run qmcp tour <clone>
uv run qmcp tour <clone> --at http://127.0.0.1:6283
uv run qmcp tour <clone> --at http://127.0.0.1:6283 --off
uv run qmcp tour <clone> --json
```

| Argument | Meaning |
|---|---|
| `CLONE` | a clone of the repository under review; its `walkthrough/` holds the pages |
| `--roadmap PATH` | the roadmap to read; the default is `governance/qm/project/ROADMAP.md` in the clone |
| `--at URL` | where the application is running; `URL/tour` and `URL/tour/feedback` are read |
| `--off` | only what somebody said is off: the lines with a stage off, the disagreements among them, and a journey no line names that has a stage off |
| `--json` | the review document, as data |

A roadmap that is not there ends the command with the path it looked at and a
non-zero status. A disagreement does not change the exit status, and neither
does an application that does not answer: the review reports and is not a
gate.

Without `--at`, or when either address does not hand over its document, the
application is not read. The review still covers the roadmap and the pages;
its first lines say why the application was not read, and each journey line
says its stages are the application's to say.

## The roadmap

Markdown, in the project's branch of the governance corpus, which the clone
holds at `governance/qm`.

- A `## ` heading opens a **group**. Headings of any other level do not.
- A checkbox item, `- [x] text` or `- [ ] text`, is an **acceptance line**.
  Ticked means the project claims it is built. Indented lines under the item
  continue it; a nested list item does not.
- A line names a **journey** as `journey:<id>` in a code span. The id is
  lowercase letters, digits and hyphens.
- A line names a **walkthrough page** by a Markdown link whose target ends in
  `walkthrough/NN-<slug>.md`, after any relative prefix and before an optional
  fragment. The page is known by its file name.
- The line's text is the claim with those names taken out, each with the
  commas and spaces that follow it.
- A line that names neither is **shown by nothing**. It is kept and reported
  in those words.
- A fenced block holds no lines, so a roadmap can show its own format.

```markdown
## Designing a part

- [x] A part is chosen and its numbers are staged. `journey:design`
- [x] An assembly is navigated by its pieces. [09](../../../walkthrough/09-an-assembly.md)
```

## The pages

`walkthrough/NN-*.md` in the clone, in the order of their numbers. A page's
title is its first `# ` heading, written `# NN — Title` or `# Title`. A page
carries no feedback: a person reads it whole, and its state in the review is
`read whole`. A page a line links and the clone does not hold is `not found`.

## The application's two documents

Each is shown with the keys the review reads; an application serves others
beside them.

`GET /tour` lists the journeys the application serves:

```json
{"commit": "f0bc8430", "changed": false,
 "journeys": [
  {"id": "design", "name": "Design a part",
   "stages": [
    {"id": "design/choose", "journey": "design", "stage": "choose", "title": "Choose",
     "tries": [{"page": "04-designing-a-part.md"}],
     "feedback": null}
   ]}
 ]}
```

`GET /tour/feedback` lists every feedback record, oldest first:

```json
{"feedback": [
  {"stage": "design/choose", "verdict": "works", "note": "",
   "at": "2026-11-15T12:00:00+00:00", "commit": "f0bc8430", "changed": false}
]}
```

What the review reads of them:

| From | Key | Read as |
|---|---|---|
| `/tour` | `commit`, `changed` | the commit the application runs and whether its tree has changes beside it, printed in the review's first lines |
| `/tour` | `journeys[].id`, `name`, `stages` | a journey, and its stages in the order listed |
| `/tour` | `stages[].id`, `stage`, `title` | a stage; `id` is what a record names |
| `/tour` | `stages[].tries[].page` | a walkthrough page the stage tries, by file name |
| `/tour/feedback` | `feedback[].stage` | the stage a record is about |
| `/tour/feedback` | `verdict` | `works` or `off` |
| `/tour/feedback` | `note`, `at`, `commit`, `changed` | printed beside the stage, as the record carries them |

A stage's feedback is the **latest record that names it**, by the list's
order. The `feedback` a stage embeds in `/tour` is not read. Any other key is
passed over.

## The review document

`--json` emits it, and the text is printed from it.

| Key | Holds |
|---|---|
| `schema` | the document's version, which moves only for a breaking change |
| `roadmap` | the path of the roadmap read |
| `application` | `at` and `read`; with `commit` and `changed` when read, with `why` when not |
| `narrowed_to` | `"off"` under `--off`, where `lines`, `disagreements` and `journeys_not_on_the_roadmap` hold only what has a stage off; otherwise null |
| `lines` | every acceptance line, in the roadmap's order |
| `disagreements` | the ticked lines with a stage off or not yet answered |
| `pages_no_line_names` | the clone's pages no line links |
| `pages_not_on_the_roadmap` | those of them no stage tries either |
| `journeys_not_on_the_roadmap` | journeys the application serves and no line names, each with its stops |
| `journeys_not_served` | journeys a line names and the application does not serve, each with its line |
| `pages_not_found` | pages a line links and the clone does not hold, each with its line |
| `feedback_on_no_stage` | the latest record of each stage no served journey has |

A line carries `group`, `text`, `ticked`, `journeys`, `pages`,
`shown_by_nothing` and `stops`: its journeys' stages in each journey's order,
then its pages.

| Stop `kind` | Keys | `state` |
|---|---|---|
| `stage` | `id`, `journey`, `stage`, `title`, `tries`, `feedback` | `works`, `off`, or `not yet` with `feedback` null |
| `page` | `page`, `title` | `read whole`, or `not found` |
| `journey` | `journey` | `the application's to say` when it was not read, `not served` when it was |

A `journey` stop stands where a journey's stages cannot be listed.

**Unknown is not empty.** When the application was not read, `disagreements`,
`pages_not_on_the_roadmap`, `journeys_not_on_the_roadmap`,
`journeys_not_served` and `feedback_on_no_stage` are each
`{"unknown": "<reason>"}`, never an empty list.

## Disagreements

A ticked line with a stage `off`, or a stage nobody has answered, is a
disagreement between the roadmap and the application. The review lists it
with what each side says: `roadmap` is `ticked`, and `off` and `not_yet` hold
the stage ids. Neither side is taken. The line stays ticked in the review,
the stage keeps its state, and the roadmap is a person's to change
(`governance/qm/records/DRAFT-a-disagreement-is-a-delta.md`).

The text prints two kinds apart. The lines with a stage off come first, each
with the ids of its off stages and a count of those not yet answered. The
lines only waiting on a word follow, one line each with its count; while
nobody has answered an application, every ticked journey line is one of
these. The document holds both lists of ids for every disagreement.

A journey a line names and the application does not serve has no stages to
disagree about. It is listed on its own, under what the roadmap names and is
not there.

## The text

One entry for each line under its group's heading: the claim with its tick,
then, for each journey it names, the count of stages that work, are off and
are not yet answered. Each answered stage follows with `✓` or `✗`, the commit
and time its record carries, and its note. `(changed)` after a commit is the
record's own `changed`: the application's tree had changes beside that
commit. The entries are followed by the disagreements, then what is not on
the roadmap, what the roadmap names and is not there, and what was said of a
stage nothing serves. Under `--off` the entries are followed by their
disagreements and by any journey no line names that has a stage off.

## What it cannot see

- Whether feedback is about the build the roadmap describes. A record carries
  the commit it was said of, the review prints it, and nothing compares
  commits.
- Whether a page shows what its line claims.
- A name written any other way. `journey:design` outside a code span, or a
  page named in a code span with no link, is prose, and its line is shown by
  nothing.
- An application's stages when one of its two addresses fails. The review
  needs both documents, and reads neither on its own.
