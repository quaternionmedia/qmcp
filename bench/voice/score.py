"""Score each transcript the way qmcp's dialogs would read it. Run in qmcp's environment.

    uv run --project <qmcp> python score.py [transcripts.jsonl]
"""
import collections
import json
import pathlib
import sys

from qmcp.instructions.converse import DONE, MORE, STOP, plain
from qmcp.integrations.voice.adapter import choose_option, match_option, parse_yes_no

try:  # the fuzzy fallback, once it exists
    from qmcp.integrations.voice.adapter import near_option
except ImportError:
    near_option = None

HERE = pathlib.Path(__file__).parent
CONFIRM = ["record", "again"]
CONSENT = ["approve", "hold"]
CHOICE = ["qmcp", "vox"]
REPEAT = None


def taken(question: str, heard: str, fuzzy: bool) -> str | None:
    """What the dialog would do with `heard`, or None for a re-ask."""
    if question == "consent":
        decision = parse_yes_no(heard)
        if decision is not None:
            return choose_option(decision, CONSENT)
        named = match_option(heard, CONSENT)
        if named is None and fuzzy and near_option:
            named = near_option(heard, CONSENT)
        return named
    if question == "confirm":
        decision = parse_yes_no(heard)
        named = match_option(heard, CONFIRM)
        if decision is True or (decision is None and named == "record"):
            return "record"
        if decision is False or named == "again":
            return "again"
        if fuzzy and near_option:
            return near_option(heard, CONFIRM)
        return None
    if question == "choice":
        named = match_option(heard, CHOICE)
        if named is None and fuzzy and near_option:
            named = near_option(heard, CHOICE)
        return named
    words = plain(heard)
    if question == "more":
        return "yes" if words in MORE else "no" if words in DONE else None
    if question == "stop":
        return "stop" if words in STOP else None
    raise ValueError(question)


rows = [json.loads(line) for line in
        (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "transcripts.jsonl").open(encoding="utf-8")]
fuzzy_modes = (False, True) if near_option else (False,)
for fuzzy in fuzzy_modes:
    print(f"\n== scored {'with' if fuzzy else 'without'} the near-match fallback")
    table = collections.defaultdict(lambda: [0, 0])
    misses = collections.defaultdict(list)
    seconds = collections.defaultdict(list)
    for r in rows:
        ok = taken(r["question"], r["heard"], fuzzy) == r["expected"]
        for key in ((r["variant"], r["condition"]), (r["variant"], "all")):
            table[key][0] += ok
            table[key][1] += 1
        seconds[r["variant"]].append(r["seconds"])
        if not ok:
            misses[r["variant"]].append(f'{r["condition"]}:{r["text"]!r}->{r["heard"]!r}')
    variants = sorted({v for v, _ in table})
    conds = ["clean", "noise15", "clipped", "all"]
    print(f"{'variant':<14}" + "".join(f"{c:>12}" for c in conds) + f"{'s/clip':>9}")
    for v in variants:
        cells = "".join(f"{table[(v, c)][0]:>6}/{table[(v, c)][1]:<5}" for c in conds)
        print(f"{v:<14}{cells}{sum(seconds[v]) / len(seconds[v]):>9.2f}")
    for v in variants:
        print(f"  misses {v}: " + "; ".join(misses[v][:12]) + (" ..." if len(misses[v]) > 12 else ""))
