"""The stand-in alfred's arithmetic for the `excerpts` recipe: the transcript cut
into stretches at pauses and at the ends of sentences, within a length, and a
timeline and its captions cut to one stretch.

alfred's own arithmetic is in alfred (`alfred/core/utils/sculpt.py` and
`cut.py`); this is the stand-in's, enough for a stand-in to leave drafts that
look like alfred's, and no claim about alfred's cuts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_SENTENCE_END = (".", "!", "?", "…")
_CLOSERS = "\"')”’»]"
_FIRST_SENTENCE = re.compile(r"^(.*?[.!?…][\"')”’»\]]*)(?:\s|$)")


def _time(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _number(value: Any, default: float = 0.0) -> float:
    found = _time(value)
    return default if found is None else found


def spoken(captions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The captions that say something, each with a start and an end after it,
    in the order they start."""
    kept = []
    for caption in captions:
        start, end = _time(caption.get("start")), _time(caption.get("end"))
        text = str(caption.get("text") or "").strip()
        if start is None or end is None or end <= start or not text:
            continue
        kept.append({**caption, "start": start, "end": end, "text": text})
    return sorted(kept, key=lambda c: (c["start"], c["end"]))


def ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip(_CLOSERS).endswith(_SENTENCE_END)


def stretches(captions: Sequence[dict[str, Any]], gap: float) -> list[list[dict[str, Any]]]:
    """`captions`, already `spoken`, cut wherever `gap` seconds or more pass
    between one caption's end and the next's start."""
    runs: list[list[dict[str, Any]]] = []
    for caption in captions:
        if runs and caption["start"] - runs[-1][-1]["end"] < gap:
            runs[-1].append(caption)
        else:
            runs.append([caption])
    return runs


def _length(run: Sequence[dict[str, Any]]) -> float:
    return run[-1]["end"] - run[0]["start"]


def pieces(stretch: Sequence[dict[str, Any]], longest: float) -> list[list[dict[str, Any]]]:
    """One stretch cut into runs of at most `longest` seconds, each ending at
    the last caption within that length that ends a sentence, or at the last
    caption within it when none does."""
    cut: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(stretch):
        start = stretch[i]["start"]
        last_fit, last_sentence = i, None
        j = i
        while j < len(stretch) and stretch[j]["end"] - start <= longest:
            last_fit = j
            if ends_sentence(stretch[j]["text"]):
                last_sentence = j
            j += 1
        end = last_fit if last_sentence is None else last_sentence
        cut.append(list(stretch[i:end + 1]))
        i = end + 1
    return cut


def joined(pieces_: Sequence[Sequence[dict[str, Any]]], shortest: float,
           longest: float) -> list[list[dict[str, Any]]]:
    """The pieces of one stretch with each one shorter than `shortest` joined
    to the piece before it, else to the one after it, where the join stays
    within `longest`; what cannot be joined is left out."""
    kept: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    for piece in pieces_:
        piece = pending + list(piece)
        pending = []
        if _length(piece) >= shortest:
            kept.append(piece)
        elif kept and _length(kept[-1] + piece) <= longest:
            kept[-1] = kept[-1] + piece
        else:
            pending = piece
    return kept


def spans_of(captions: Sequence[dict[str, Any]], *, pause: float, chapter_pause: float,
             shortest: float, longest: float) -> list[tuple[float, float, int]]:
    """The stretches the `excerpts` recipe leaves, each `(start, end, chapter)`:
    a chapter between pauses of `chapter_pause` seconds, a stretch between
    pauses of `pause` within it, cut at sentence ends to `longest` and joined
    up to `shortest`."""
    found: list[tuple[float, float, int]] = []
    for chapter, part in enumerate(stretches(spoken(captions), chapter_pause), start=1):
        for stretch in stretches(part, pause):
            for run in joined(pieces(stretch, longest), shortest, longest):
                found.append((run[0]["start"], run[-1]["end"], chapter))
    return found


def cut_edl(edl: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    """The part of `edl` between `start` and `end`, as a timeline of its own
    that begins at 0: each clip that plays in it kept and re-timed, an edge the
    cut made with no fade."""
    duration = _time(edl.get("duration"))
    if duration is not None and end > duration:
        end = duration
    clips = []
    for clip in edl.get("clips") or []:
        offset = _number(clip.get("offset"))
        clip_start = _number(clip.get("start")) + max(offset, 0.0)
        inpoint = _number(clip.get("inpoint")) + max(-offset, 0.0)
        outpoint = _time(clip.get("outpoint"))
        length = outpoint - inpoint if outpoint is not None else _number(clip.get("duration"), 5.0)
        if length <= 0:
            continue
        clip_end = clip_start + length
        low, high = max(clip_start, start), min(clip_end, end)
        if high <= low:
            continue
        cut = {key: value for key, value in clip.items() if key != "offset"}
        cut["start"] = round(low - start, 3)
        cut["duration"] = round(high - low, 3)
        if clip.get("type") == "image":
            cut.pop("inpoint", None)
            cut.pop("outpoint", None)
        else:
            cut["inpoint"] = round(inpoint + (low - clip_start), 3)
            cut["outpoint"] = round(cut["inpoint"] + (high - low), 3)
        if low > clip_start:
            cut["fadeIn"] = 0
        if high < clip_end:
            cut["fadeOut"] = 0
        clips.append(cut)
    return {"clips": clips, "duration": round(end - start, 3)}


def cut_captions(captions: Sequence[dict[str, Any]], start: float,
                 end: float) -> list[dict[str, Any]]:
    """The captions that lie between `start` and `end`, in the cut's own
    seconds: moved by `start`, trimmed where the cut crosses one."""
    kept = []
    for caption in spoken(captions):
        low, high = max(caption["start"], start), min(caption["end"], end)
        if high <= low:
            continue
        kept.append({**caption, "start": round(low - start, 3), "end": round(high - start, 3)})
    return kept
