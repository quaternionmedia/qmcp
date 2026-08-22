# 03 — Archiving what is not stored locally

Everything on this page runs, except the one step that is a person's and cannot
be anything else.

## The API is not a route to this data

Worth stating before anything else, because it is the reasonable assumption and
it is wrong.

**Anthropic's API and OpenAI's API do not expose the conversation history of
`claude.ai` or `chatgpt.com`.** They are products for making new model calls,
against different storage, with no endpoint that lists your threads. A
credential would not help. Driving a browser session or replaying cookies would
be scraping a service against its terms, and fragile in the way that fails
quietly.

The sanctioned route is each service's **data export**, requested by the account
holder. That is a human step by construction — and one this organisation would
want to be a human step regardless, since it is somebody deciding to take a copy
of their own conversations.

## 1. Request the export — a person, in a browser

| service | where |
|---|---|
| Claude | Settings → Privacy → Export data |
| ChatGPT | Settings → Data controls → Export data |

Each emails a link. The download is a ZIP holding, among other files, one large
`conversations.json` — an array of conversations rather than a file each.

**Nothing automates this and nothing should.** It is free, it is occasional, and
it is the account holder proving they are the account holder.

## 2. Unpack it into the cache

    uv run qmcp threads import ~/Downloads/claude-export.zip
    uv run qmcp threads import ~/Downloads/chatgpt-export.zip

Which service wrote a conversation is read from its shape rather than its
filename — a filename is what somebody renamed:

    >>> from qmcp.threads.importer import detect
    >>> detect({"chat_messages": [], "uuid": "c-1"})
    'claude'
    >>> detect({"mapping": {}})
    'chatgpt'

Something carrying neither is not guessed at:

    >>> detect({"title": "who knows"}) is None
    True

An id from an export is somebody else's string, and one containing a separator
would write outside the folder it was meant for:

    >>> from qmcp.threads.importer import safe
    >>> safe("../../etc/passwd")
    'etc-passwd'

`--dry-run` reports and writes nothing. Run it first on an export you have not
seen before.

## 3. Index what arrived

    uv run qmcp threads index --write

The index answers from one file so that counting does not mean reading
everything, and `--check` re-derives from the files and says where the two
differ.

## 4. Import the next export, months later

This is where the archive earns itself. A conversation you kept talking in comes
back longer; one the exporter changed its mind about comes back different. They
are not the same finding:

    >>> from qmcp.threads.index import classify, entry_for, GREW, DIVERGED
    >>> from qmcp.threads.base import Thread, Turn
    >>> def at(*pairs):
    ...     return entry_for(
    ...         Thread(id="c-1", turns=tuple(
    ...             Turn(id=i, role="assistant", text=t) for i, t in pairs)),
    ...         "claude", "2026-08-20T00:00:00Z")

Turns appended to the end is growth, and it is the boring case:

    >>> classify(at(("m-1", "one")), at(("m-1", "one"), ("m-2", "two")))[0] == GREW
    True

A turn that says something else now is not:

    >>> kind, why = classify(at(("m-1", "as written")), at(("m-1", "rewritten")))
    >>> kind == DIVERGED
    True
    >>> "edited after" in why
    True

Nor is one that has gone:

    >>> classify(at(("m-1", "one"), ("m-2", "two")), at(("m-1", "one")))[0] == DIVERGED
    True

**Neither is repaired.** The prior digest stays, because a divergence somebody
deletes to make the index tidy is the one fact nobody can recover. An export is
supposed to be a record; one that disagrees with an earlier record of itself is
a tool changing its format, somebody editing history, or an id being reused —
and which of those it is, is a person's to say.

    uv run qmcp threads list --diverged

## 5. What the threads produced

A thread is itself a delta, and what it settled is `part-of` it:

    >>> from qmcp.threads.base import to_thread_delta, relations_for, Decision
    >>> thread = Thread(id="c-1", title="Naming the ask kind")
    >>> to_thread_delta(thread, project="quaternionmedia/qmcp",
    ...                 perspective="claude/thread")["delta"]["delta_type"]
    'thread'

    >>> decided = Decision("call-it-ask", "Call the address kind ask",
    ...                    from_turns=("m-2",))
    >>> relations_for(thread, [decided], project="q/r")[0]["relation"]
    'part-of'

Extraction finds what a conversation **marked** — a line opening `DECISION:` or
`DECIDED:`. It recognises nothing on its own, and says so rather than putting
confident rows on a board that a reader cannot tell apart from real ones.
Reading an unmarked conversation and working out what it settled needs a model,
and that is the paid path.

## What this costs

Nothing. Every command here reads a file the operator downloaded:

    >>> from qmcp.threads.claude import ClaudeThreads
    >>> import tempfile, pathlib
    >>> ClaudeThreads(root=pathlib.Path(tempfile.mkdtemp())).survey().would_need
    0

That is a real zero — there is no paid work to do — and not the sentinel that
would mean nobody counted. When an API source exists it will spend against a
budget a person issued for one command, and `qmcp/spend.py` is where that lives.
