# The tour review

A roadmap says what a project claims is built. `qmcp tour` shows what stands
behind each claim and what people said of it: the stages of the application's
own guide and the walkthrough pages a line names, with the latest feedback on
each stage, who gave it, and whether it was said of the build the application
runs. The code is `qmcp/tour.py`, and
`walkthrough/12-a-roadmap-read-against-the-tour.md` runs it end to end.

The guide itself is the application's. It walks a person through the
application's *journeys*, each an ordered set of *stages*, and keeps what they
said of a stage: `works`, or `off` with a note, with their name and the commit
the application ran. qmcp reads that and the roadmap, and writes to neither.

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
| `--off` | only what somebody said is off: the lines with a stage whose latest word is `off`, of this build or an earlier one, the disagreements of those lines, and a journey no line names that has such a stage |
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
  {"stage": "design/choose", "verdict": "works", "note": "", "by": "Ada",
   "at": "2026-11-15T12:00:00+00:00", "commit": "f0bc8430", "changed": false}
]}
```

What the review reads of them:

| From | Key | Read as |
|---|---|---|
| `/tour` | `commit`, `changed` | the build the application runs: the commit every record's own is held against, and whether its tree has changes beside it; printed in the review's first lines |
| `/tour` | `journeys[].id`, `name`, `stages` | a journey, and its stages in the order listed |
| `/tour` | `stages[].id`, `stage`, `title` | a stage; `id` is what a record names |
| `/tour` | `stages[].tries[].page` | a walkthrough page the stage tries, by file name |
| `/tour/feedback` | `feedback[].stage` | the stage a record is about |
| `/tour/feedback` | `verdict` | `works` or `off` |
| `/tour/feedback` | `by` | who said it, printed beside the stage; a record kept before words carried a name has no `by`, or an empty one, and nothing is printed in its place |
| `/tour/feedback` | `commit` | the build the word was said of |
| `/tour/feedback` | `note`, `at`, `changed` | printed beside the stage, as the record carries them |

A stage's feedback is the **latest record that names it**, by the list's
order. The `feedback` a stage embeds in `/tour`, and the `current` it carries,
are not read: the review works from the records and the commit at the top of
`/tour`. Any other key is passed over.

**A word counts for the build it was said of.** A stage's latest record is
current when its `commit` is the one the application runs. Otherwise it was
said of an earlier build: it is kept and printed, and it is not counted as
`works` or `off`. An application whose `commit` is null ties no word to a
build; every stage's latest record then counts, and the review's first lines
say that the application did not say its build.

## The review document

`--json` emits it, and the text is printed from it.

| Key | Holds |
|---|---|
| `schema` | the document's version, which moves only for a breaking change |
| `roadmap` | the path of the roadmap read |
| `application` | `at` and `read`; with `commit` and `changed` when read, with `why` when not |
| `narrowed_to` | `"off"` under `--off`, where `lines`, `disagreements` and `journeys_not_on_the_roadmap` hold only what has a stage whose latest word is `off`; otherwise null |
| `lines` | every acceptance line, in the roadmap's order |
| `disagreements` | the ticked lines with a stage off, or with a stage that has no word of the running build |
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
| `stage` | `id`, `journey`, `stage`, `title`, `tries`, `feedback` | `works` or `off`, of the running build; `said of an earlier build`, with the earlier record as `feedback`; or `not yet`, with `feedback` null |
| `page` | `page`, `title` | `read whole`, or `not found` |
| `journey` | `journey` | `the application's to say` when it was not read, `not served` when it was |

A `journey` stop stands where a journey's stages cannot be listed.

**Unknown is not empty.** When the application was not read, `disagreements`,
`pages_not_on_the_roadmap`, `journeys_not_on_the_roadmap`,
`journeys_not_served` and `feedback_on_no_stage` are each
`{"unknown": "<reason>"}`, never an empty list.

## Disagreements

The rule reads only the words said of the running build. A ticked line with a
stage `off` in that build, or with a stage that has no word of it, is a
disagreement between the roadmap and the application. The review lists it
with what each side says: `roadmap` is `ticked`, and three lists hold the
stage ids: `off`, the stages off in this build; `earlier`, those whose latest
word was said of an earlier build; and `not_yet`, those nobody has answered.
Neither side is taken. The line stays ticked in the review, the stage keeps
its state, and the roadmap is a person's to change
(`governance/qm/records/DRAFT-a-disagreement-is-a-delta.md`).

The text prints two kinds apart. The lines with a stage off come first, each
with the ids of its off stages and, beneath, how many of its stages were said
of an earlier build and how many are not yet answered. The lines only waiting
on a word follow, one line each with the same two counts; an `off` said of an
earlier build puts its line here. While nobody has answered an application,
or after its commit moves, every ticked journey line is one of these. The
document holds the three lists of ids for every disagreement.

A journey a line names and the application does not serve has no stages to
disagree about. It is listed on its own, under what the roadmap names and is
not there.

## The text

The first lines name the roadmap read and the application, with the commit it
runs, or that it did not say its build.

One entry for each line under its group's heading: the claim with its tick,
then, for each journey it names, four counts of its stages: those that work
and those that are off in the running build, those said of an earlier build,
and those not yet answered. Each stage with a word follows with `✓` or `✗`,
who said it, the commit and time its record carries, and its note. A word of
an earlier build has `of an earlier build` before its commit. `(changed)`
after a commit is the record's own `changed`: the application's tree had
changes beside that commit.

The entries are followed by the disagreements, then what is not on the
roadmap, what the roadmap names and is not there, and what was said of a
stage nothing serves. Under `--off` the entries are followed by their
disagreements and by any journey no line names that has a stage whose latest
word is `off`.

## What it cannot see

- Whether the build the application runs is the one the roadmap describes. A
  word is held to the running build, and a roadmap names none.
- Which of two builds is the later. A record whose commit is not the running
  one was said while the application ran another build, and that is all
  `earlier` means here: the two commits are not ordered.
- Whether a page shows what its line claims.
- A name written any other way. `journey:design` outside a code span, or a
  page named in a code span with no link, is prose, and its line is shown by
  nothing.
- An application's stages when one of its two addresses fails. The review
  needs both documents, and reads neither on its own.
