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

import json
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4


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


@dataclass(frozen=True)
class Check:
    """One of a project's own commands, run by voice behind consent."""

    project: str
    name: str
    says: str
    phrases: tuple[str, ...]
    argv: tuple[str, ...]
    minutes: float
    said: str | None = None
    """A regular expression for the line of the output that answers, where
    the last line printed is not it."""

    @property
    def command(self) -> str:
        return " ".join(self.argv)


def checks(project: str | None = None) -> list[Check]:
    """The declared checks of one project, or of every core project."""
    found = []
    for name, body in load().get("projects", {}).items():
        if project is not None and name != project:
            continue
        for item in body.get("checks", ()):
            found.append(Check(project=name, name=item["name"], says=item.get("says", ""),
                               phrases=tuple(item.get("phrases", ())),
                               argv=tuple(item["argv"]), minutes=float(item.get("minutes", 5)),
                               said=item.get("said")))
    return found


def match_check(words: str, project: str | None) -> Check | None:
    """The project's check whose phrase the words contain, as whole words, or
    None. Only the project's own checks are read: "run the tests" means the
    tests of the project the instruction was recorded for."""
    if not project:
        return None
    padded = f" {words} "
    for check in checks(project):
        if any(f" {phrase} " in padded for phrase in check.phrases):
            return check
    return None


def projects() -> dict[str, dict[str, Any]]:
    """The core projects, each with what it is, the terms its instructions
    carry, and the checks it declares."""
    return {name: {"says": body.get("says", ""), "terms": list(body.get("terms", ())),
                   "checks": [{"name": c.name, "says": c.says, "phrases": list(c.phrases),
                               "command": c.command} for c in checks(name)]}
            for name, body in load().get("projects", {}).items()}


def terms(names: Any = None) -> list[str]:
    """The terms of the projects named, in that order, or of every core
    project; without repeats."""
    known = projects()
    order = list(names) if names is not None else list(known)
    return list(dict.fromkeys(t for name in order for t in known.get(name, {}).get("terms", ())))


def _declared() -> list[dict[str, Any]]:
    shown: list[dict[str, Any]] = []
    for group, members in load().items():
        for name, body in members.items():
            if isinstance(body, dict) and ("phrases" in body or "words" in body):
                shown.append({"key": f"{group}.{name}", "says": body.get("says", ""),
                              "phrases": list(body.get("phrases", ())),
                              "words": list(body.get("words", ()))})
    return shown


def entries() -> list[dict[str, Any]]:
    """Every entry with its key, what it does, and what to say: for a page or
    a command that shows what can be said. An entry a person may add to lists
    their phrases after the declared ones, and again under `added`."""
    state = _read_overlay()
    shown = _declared()
    for item in shown:
        if item["key"] in EDITABLE:
            item["added"] = list(state["entries"].get(item["key"], ()))
            item["phrases"] += item["added"]
    return shown


# --- phrases a person adds, by voice -------------------------------------------
#
# The package vocabulary above is fixed. A person may add phrases of their own
# to the conversation's controls and to its iteration and diagnostic commands,
# in one file of their own outside any checkout, and every change is journalled
# so the last one can be undone. A phrase that already means something anywhere
# in the vocabulary is refused rather than given a second meaning.

EDITABLE = (
    "conversation.stop", "conversation.repeat", "conversation.done", "conversation.more",
    "iteration.try_again", "iteration.same_in", "iteration.never_mind",
    "diagnostic.what_heard", "diagnostic.how_did_it_go", "diagnostic.whats_waiting",
    "diagnostic.test_voice", "diagnostic.which_projects", "diagnostic.help",
)
"""The keys a person may add phrases to. No `answer.*`: consent is heard
through those, and a phrase of one's own there would change what agreeing
sounds like."""

SLOT = "{project}"
"""Where a project's name goes in a phrase such as "same in {project}"."""

MAX_PHRASE_CHARS = 80
MAX_PHRASE_WORDS = 8


def target(key: str) -> str:
    """The spoken name of an editable key: `iteration.try_again` is "try again"."""
    return key.partition(".")[2].replace("_", " ")


TARGETS = {target(key): key for key in EDITABLE}


def overlay_path() -> Path:
    """The per-user phrase file: `QMCP_VOICE_VOCABULARY_PATH`, or
    `~/.qmcp/voice-vocabulary.json`."""
    configured = os.getenv("QMCP_VOICE_VOCABULARY_PATH")
    return (Path(configured).expanduser() if configured
            else Path.home() / ".qmcp" / "voice-vocabulary.json")


def _read_overlay() -> dict[str, Any]:
    """The phrase file, or an empty one. A file that is not this shape is
    refused by name rather than read around."""
    path = overlay_path()
    if not path.exists():
        return {"schema": 1, "entries": {}, "history": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read vocabulary overlay {path}: {exc}") from exc
    if (not isinstance(state, dict) or state.get("schema") != 1
            or not isinstance(state.get("entries"), dict)
            or not isinstance(state.get("history"), list)):
        raise ValueError(f"{path}: unsupported or malformed vocabulary overlay")
    for key, added in state["entries"].items():
        if (key not in EDITABLE or not isinstance(added, list)
                or not all(isinstance(phrase, str) for phrase in added)):
            raise ValueError(f"{path}: malformed overlay entry {key!r}")
    for event in state["history"]:
        if (not isinstance(event, dict) or not isinstance(event.get("id"), str)
                or event.get("key") not in EDITABLE
                or not isinstance(event.get("before"), list)
                or not isinstance(event.get("after"), list)):
            raise ValueError(f"{path}: malformed vocabulary history event")
    return state


def overrides(key: str) -> tuple[str, ...]:
    """The phrases a person has added to an editable key, read as they stand,
    so an approved edit is heard on the next utterance."""
    _editable(key)
    return tuple(_read_overlay()["entries"].get(key, ()))


def _editable(key: str) -> None:
    if key not in EDITABLE:
        raise ValueError(f"{key!r} is not editable; the commands that take phrases are: "
                         f"{', '.join(TARGETS)}")


def _plain(phrase: str) -> str:
    return " ".join(re.sub(r"[^\w\s{}]", "", phrase.lower().replace("'", "")).split())


def _forms(term: str) -> set[str]:
    """Every plain form a term is heard as: itself, or one per core project
    where it carries the slot."""
    if SLOT in term:
        return {_plain(term.replace(SLOT, name)) for name in projects()}
    return {_plain(term)}


def _candidate(key: str, phrase: str) -> str:
    """`phrase` as it would be stored for `key`: plain words, with the project
    slot appended where the key's own phrases end in one."""
    candidate = _plain(phrase.replace(SLOT, ""))
    if not candidate:
        raise ValueError("a phrase must contain at least one word")
    if len(candidate) > MAX_PHRASE_CHARS or len(candidate.split()) > MAX_PHRASE_WORDS:
        raise ValueError(f"a phrase must be at most {MAX_PHRASE_CHARS} characters "
                         f"and {MAX_PHRASE_WORDS} words")
    if any(SLOT in declared for declared in phrases(key)):
        candidate = f"{candidate} {SLOT}"
    return candidate


def shown(phrase: str) -> str:
    """A stored phrase as a person would hear it read back."""
    return phrase.replace(SLOT, "a project")


def _meanings(state: dict[str, Any]) -> dict[str, str]:
    """Every plain form that already means something, and what it means: a
    declared entry, a phrase a person added, a check, or a project's name."""
    meant: dict[str, str] = {}
    for item in _declared():
        for term in (*item["phrases"], *item["words"]):
            for form in _forms(term):
                meant.setdefault(form, item["key"])
    for key, added in state["entries"].items():
        for term in added:
            for form in _forms(term):
                meant.setdefault(form, key)
    for check in checks():
        for term in check.phrases:
            meant.setdefault(_plain(term), f"the {check.name} check in {check.project}")
    for name, project in projects().items():
        for term in (name, *project["terms"]):
            meant.setdefault(_plain(term), f"project {name}")
    return meant


@dataclass(frozen=True)
class PhraseEdit:
    """A validated change to one key's added phrases, not yet saved."""

    action: str
    key: str
    phrase: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    target_event: str | None = None


def prepare_add(name: str, phrase: str) -> PhraseEdit:
    """Validate adding `phrase` to the command `name` (spoken, or a key),
    without changing the file."""
    key = TARGETS.get(name, name)
    _editable(key)
    candidate = _candidate(key, phrase)
    state = _read_overlay()
    meant = _meanings(state)
    for form in sorted(_forms(candidate)):
        if meant.get(form) == key:
            raise ValueError(f"{shown(candidate)!r} is already a phrase for {target(key)}")
        if form in meant:
            raise ValueError(f"{form!r} already means {meant[form]}")
    added = tuple(state["entries"].get(key, ()))
    return PhraseEdit("add", key, candidate, added, (*added, candidate))


def prepare_remove(name: str, phrase: str) -> PhraseEdit:
    """Validate removing a phrase a person added. The package's are fixed."""
    key = TARGETS.get(name, name)
    _editable(key)
    candidate = _candidate(key, phrase)
    added = tuple(_read_overlay()["entries"].get(key, ()))
    if candidate not in added:
        declared = {form for term in phrases(key) for form in _forms(term)}
        if _forms(candidate) & declared:
            raise ValueError("the package vocabulary is fixed; only a phrase a person "
                             "added can be removed")
        raise ValueError(f"{shown(candidate)!r} is not a phrase added to {target(key)}")
    return PhraseEdit("remove", key, candidate, added,
                      tuple(item for item in added if item != candidate))


def prepare_undo() -> PhraseEdit | None:
    """The reverse of the most recent add or remove not already undone."""
    history = _read_overlay()["history"]
    undone = {event.get("target_event") for event in history if event.get("action") == "undo"}
    latest = next((event for event in reversed(history)
                   if event.get("action") in ("add", "remove") and event["id"] not in undone),
                  None)
    if latest is None:
        return None
    return PhraseEdit("undo", latest["key"], latest["phrase"], tuple(latest["after"]),
                      tuple(latest["before"]), target_event=latest["id"])


def apply_edit(edit: PhraseEdit, *, actor: str = "voice") -> None:
    """Save an approved edit, journalled, by replacing the file whole.

    The edit is prepared again first and must come out the same: a file that
    changed while the edit waited for approval is not overwritten.
    """
    _editable(edit.key)
    if edit.action == "add":
        again = prepare_add(edit.key, edit.phrase)
    elif edit.action == "remove":
        again = prepare_remove(edit.key, edit.phrase)
    elif edit.action == "undo":
        again = prepare_undo()
    else:
        raise ValueError(f"unknown vocabulary edit {edit.action!r}")
    state = _read_overlay()
    if again != edit or tuple(state["entries"].get(edit.key, ())) != edit.before:
        raise RuntimeError("the vocabulary changed while this edit was awaiting approval")

    event = {"id": str(uuid4()), "action": edit.action, "key": edit.key,
             "phrase": edit.phrase, "before": list(edit.before), "after": list(edit.after),
             "actor": actor, "created_at": datetime.now(UTC).isoformat()}
    if edit.target_event:
        event["target_event"] = edit.target_event
    if edit.after:
        state["entries"][edit.key] = list(edit.after)
    else:
        state["entries"].pop(edit.key, None)
    state["history"].append(event)

    path = overlay_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


__all__ = ["EDITABLE", "SLOT", "TARGETS", "Check", "PhraseEdit", "apply_edit", "checks",
           "entries", "entry", "load", "match_check", "overlay_path", "overrides",
           "phrases", "prepare_add", "prepare_remove", "prepare_undo", "projects", "says",
           "shown", "target", "terms", "words"]
