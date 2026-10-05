"""What the spoken loop listens for, declared once.

    uv run qmcp vocabulary          # what can be said, by group

Every phrase the loop acts on without a model -- the conversation's controls,
the answers to a closed question -- is declared in `vocabulary.toml` beside
this module, under a dotted key (`conversation.stop`, `answer.yes`) with a
`says` line for a person. Code reads its lists from here rather than spelling
them, so what `qmcp vocabulary` and joe's page show as what can be said is the
list the loop acts on, and a phrase added here is heard everywhere it applies.

How a list is matched stays with the code that reads it -- the whole
utterance, a word anywhere in it, or a phrase inside it -- because the same
word means different things in different places: "again" takes a read-back
again and is no request to repeat a question.
"""

from __future__ import annotations

import tomllib
from functools import cache
from importlib.resources import files
from typing import Any


@cache
def load() -> dict[str, Any]:
    """The declared vocabulary, read once."""
    text = files("qmcp.integrations.voice").joinpath("vocabulary.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def entry(key: str) -> dict[str, Any]:
    """The entry at a dotted key, such as `conversation.stop`."""
    node: Any = load()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"no vocabulary entry {key!r}")
        node = node[part]
    return node


def phrases(key: str) -> tuple[str, ...]:
    """An entry's phrases, in the order declared."""
    return tuple(entry(key).get("phrases", ()))


def words(key: str) -> tuple[str, ...]:
    """An entry's single words, matched anywhere in an utterance."""
    return tuple(entry(key).get("words", ()))


def says(key: str) -> str:
    """What an entry does, for a person."""
    return entry(key).get("says", "")


def projects() -> dict[str, dict[str, Any]]:
    """The core projects, each with what it is and the terms its instructions carry."""
    return {name: {"says": body.get("says", ""), "terms": list(body.get("terms", ()))}
            for name, body in load().get("projects", {}).items()}


def terms(names: Any = None) -> list[str]:
    """The terms of the projects named, in that order, or of every core
    project; without repeats."""
    known = projects()
    order = list(names) if names is not None else list(known)
    return list(dict.fromkeys(t for name in order for t in known.get(name, {}).get("terms", ())))


def entries() -> list[dict[str, Any]]:
    """Every entry with its key, what it does, and what to say: for a page or
    a command that shows what can be said."""
    shown: list[dict[str, Any]] = []
    for group, members in load().items():
        for name, body in members.items():
            if isinstance(body, dict) and ("phrases" in body or "words" in body):
                shown.append({"key": f"{group}.{name}", "says": body.get("says", ""),
                              "phrases": list(body.get("phrases", ())),
                              "words": list(body.get("words", ()))})
    return shown


__all__ = ["entries", "entry", "load", "phrases", "projects", "says", "terms", "words"]
