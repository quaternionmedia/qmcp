"""The thread archive as one page, rendered locally and going nowhere.

    qmcp threads dashboard --out threads.html

**THIS PAGE IS SOMEBODY'S CONVERSATIONS AND MUST NOT BE PUBLISHED.** It carries
titles, session ids, branch names and repository names off one machine's
archive. It is written where the caller says, it is self-contained so it needs
no network to open, and the page says so in its own footer -- because the person
who finds it in six months will not have read this docstring.

IT READS THE INDEX AND RUNS NOTHING. Same rule as every other dashboard in this
organisation: the thing that measures is separate from the thing that renders,
so a view cannot quietly become a second definition of what a figure means. If a
number here is wrong, it is wrong in the index, and `qmcp threads index --check`
is what says so.

WHAT IT SHOWS, IN THE ORDER SOMEBODY WOULD ASK.

    what is archived      per source, because they are not the same kind of thing
    what disagrees        exports that contradict an earlier record of themselves
    what is here          every thread, newest reading first

The divergence section is first among the details deliberately. It is the only
part that is a finding rather than an inventory, and a page that buried it under
four hundred rows would be an inventory with a finding hidden in it.
"""

from __future__ import annotations

import html
from collections import Counter
from typing import Any

from qmcp.threads.index import DIVERGED

STYLE = """
:root { --ink:#1b1b1f; --dim:#6a6a75; --line:#e3e3ea; --bg:#fbfbfd;
        --warn:#8a3d00; --warnbg:#fff4ea; }
* { box-sizing:border-box; }
body { margin:0; padding:2.5rem 1.5rem 4rem; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:60rem; margin:0 auto; }
h1 { font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }
h2 { font-size:1rem; margin:2.5rem 0 .75rem; text-transform:uppercase;
     letter-spacing:.07em; color:var(--dim); font-weight:600; }
.sub { color:var(--dim); margin:0 0 2rem; }
.cards { display:flex; gap:.75rem; flex-wrap:wrap; }
.card { flex:1 1 10rem; background:#fff; border:1px solid var(--line);
        border-radius:.5rem; padding:.9rem 1rem; }
.card .n { font-size:1.7rem; font-weight:600; letter-spacing:-.02em; }
.card .l { color:var(--dim); font-size:.82rem; }
table { width:100%; border-collapse:collapse; background:#fff;
        border:1px solid var(--line); border-radius:.5rem; overflow:hidden; }
th,td { text-align:left; padding:.55rem .8rem; border-bottom:1px solid var(--line);
        font-size:.88rem; vertical-align:top; }
th { color:var(--dim); font-weight:600; font-size:.75rem; text-transform:uppercase;
     letter-spacing:.05em; }
tr:last-child td { border-bottom:none; }
td.num { text-align:right; font-variant-numeric:tabular-nums; color:var(--dim); }
code { font:12.5px ui-monospace,SFMono-Regular,Consolas,monospace; color:var(--dim); }
.warn { background:var(--warnbg); border:1px solid #f0d5bd; border-radius:.5rem;
        padding:1rem 1.15rem; margin:.5rem 0 0; }
.warn h3 { margin:0 0 .5rem; font-size:.95rem; color:var(--warn); }
.warn p { margin:.4rem 0 0; font-size:.87rem; }
.flag { color:var(--warn); font-weight:600; }
.note { color:var(--dim); font-size:.85rem; margin:.6rem 0 0; }
footer { margin-top:3rem; padding-top:1.25rem; border-top:1px solid var(--line);
         color:var(--dim); font-size:.82rem; }
footer strong { color:var(--warn); }
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def card(number: Any, label: str) -> str:
    return (f'<div class="card"><div class="n">{esc(number)}</div>'
            f'<div class="l">{esc(label)}</div></div>')


def render(document: dict[str, Any]) -> str:
    """The index as one self-contained page.

    Takes the index document rather than a path, so the renderer can be handed
    a document from anywhere and cannot go looking for one.
    """
    rows = document.get("threads") or []
    totals = document.get("totals") or {}
    per_source = Counter(row["source"] for row in rows)

    diverged_rows = [
        row for row in rows
        if any(change["kind"] == DIVERGED for change in row.get("history") or [])
    ]

    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "<meta name=\"robots\" content=\"noindex,nofollow\">",
        "<title>Thread archive</title>",
        f"<style>{STYLE}</style></head><body><main>",
        "<h1>Thread archive</h1>",
        f'<p class="sub">Indexed {esc(document.get("generated_at"))}. '
        "Counts what was exported and indexed &mdash; not what exists.</p>",
        '<div class="cards">',
        card(totals.get("threads", len(rows)), "threads archived"),
        card(len(diverged_rows), "disagree with an earlier record"),
        card(totals.get("unreadable", 0), "files unreadable"),
        card(len(per_source), "sources"),
        "</div>",
    ]

    # --- per source ---------------------------------------------------------
    parts.append("<h2>Where they came from</h2><table><tr><th>Source</th>"
                 "<th class=\"num\">Threads</th><th>What it knows</th></tr>")
    knows = {
        "claude": "the conversation",
        "chatgpt": "the conversation, as a tree flattened to a sequence",
        "claude-code": "the conversation and the work &mdash; branch, checkout, "
                       "and the pull requests it produced",
    }
    for source, count in sorted(per_source.items()):
        parts.append(f"<tr><td><code>{esc(source)}</code></td>"
                     f'<td class="num">{count}</td>'
                     f"<td>{knows.get(source, '')}</td></tr>")
    parts.append("</table>")

    # --- what disagrees -----------------------------------------------------
    if diverged_rows:
        parts.append("<h2>Disagrees with an earlier record</h2>")
        parts.append('<div class="warn"><h3>Nothing here has been repaired</h3>'
                     "<p>An export is a record. One that contradicts an earlier "
                     "record of itself is a tool changing its format, somebody "
                     "editing history, or an id being reused. The prior digest "
                     "is the only evidence it changed, so it is kept &mdash; "
                     "which of those it is, is a person&rsquo;s to say.</p></div>")
        parts.append("<table><tr><th>Thread</th><th>What changed</th>"
                     "<th>When</th></tr>")
        for row in diverged_rows:
            last = [c for c in row["history"] if c["kind"] == DIVERGED][-1]
            parts.append(
                f"<tr><td><code>{esc(row['source'])}/{esc(row['id'])}</code>"
                f"<br>{esc(row.get('title') or '')}</td>"
                f"<td>{esc(last['detail'])}<br>"
                f"<code>{esc(last['from_digest'])} &rarr; "
                f"{esc(last['to_digest'])}</code></td>"
                f"<td><code>{esc(last['at'])}</code></td></tr>")
        parts.append("</table>")

    # --- everything ---------------------------------------------------------
    parts.append("<h2>Everything archived</h2>")
    if not rows:
        parts.append('<p class="note">The index is empty. That is what it '
                     "holds, not a failure to look.</p>")
    else:
        parts.append("<table><tr><th>Thread</th><th>Source</th>"
                     "<th class=\"num\">Turns</th><th>First seen</th>"
                     "<th>Last seen</th><th class=\"num\">Changes</th></tr>")
        for row in sorted(rows, key=lambda r: r.get("last_seen") or "",
                          reverse=True):
            flagged = any(c["kind"] == DIVERGED for c in row.get("history") or [])
            title = esc(row.get("title") or row["id"])
            parts.append(
                f"<tr><td>{'<span class=\"flag\">!</span> ' if flagged else ''}"
                f"{title}<br><code>{esc(row['id'])}</code></td>"
                f"<td><code>{esc(row['source'])}</code></td>"
                f'<td class="num">{esc(row.get("turns", 0))}</td>'
                f"<td><code>{esc(row.get('first_seen'))}</code></td>"
                f"<td><code>{esc(row.get('last_seen'))}</code></td>"
                f'<td class="num">{len(row.get("history") or [])}</td></tr>')
        parts.append("</table>")

    parts += [
        '<p class="note">Newest reading first. A thread the cache no longer '
        "holds keeps its row: an export somebody deleted has not un-happened.</p>",
        "<footer>",
        "<p><strong>This page is local and must not be published.</strong> It "
        "carries conversation titles, session ids and repository names from one "
        "machine&rsquo;s archive. It is self-contained, so it needs no network "
        "to open &mdash; and it should not be given one.</p>",
        "<p>Rendered from the index and nothing else. If a figure here is "
        "wrong it is wrong in the index; "
        "<code>qmcp threads index --check</code> re-derives it from the "
        "files.</p>",
        "</footer></main></body></html>",
    ]
    return "\n".join(parts)
