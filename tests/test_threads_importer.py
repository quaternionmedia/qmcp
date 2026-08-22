"""Unpacking a data export, and what it refuses to guess.

The tests worth reading are the ones about names. An export's ids are somebody
else's strings, and this writes files from them.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from qmcp.threads.importer import (
    CONVERSATIONS,
    conversations_in,
    detect,
    identity,
    positional,
    render,
    safe,
    unpack,
)


def claude(identifier="c-1", text="hello"):
    return {"uuid": identifier, "name": "T",
            "chat_messages": [{"uuid": "m-1", "sender": "human", "text": text}]}


def chatgpt(identifier="g-1", text="hello"):
    return {"conversation_id": identifier, "title": "T", "mapping": {
        "n1": {"message": {"id": "t-1", "create_time": 1.0,
                           "author": {"role": "user"},
                           "content": {"parts": [text]}}}}}


def export(tmp_path, conversations, name="export.zip", member=CONVERSATIONS):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, json.dumps(conversations))
    return path


# --- which service wrote it -------------------------------------------------


def test_the_service_is_read_from_the_shape_not_the_filename():
    """A filename is what somebody renamed; the structure is what the exporter
    wrote."""
    assert detect(claude()) == "claude"
    assert detect(chatgpt()) == "chatgpt"


def test_something_carrying_neither_shape_is_not_guessed_at():
    """Mutation: fall back to the more likely service and this fails, which is
    a conversation filed under a source that did not write it."""
    assert detect({"title": "who knows"}) is None
    assert detect("not even an object") is None


def test_a_conversation_that_contradicts_the_stated_source_is_refused(tmp_path):
    """`--source claude` over a ChatGPT conversation files it wrong forever."""
    archive = export(tmp_path, [chatgpt()])
    report = unpack(archive, tmp_path / "cache", source="claude")
    assert report.written == []
    assert "looks like a chatgpt" in report.unreadable[0][1]


# --- names, which become filenames ------------------------------------------


def test_a_separator_in_an_id_cannot_escape_the_cache():
    """An id is somebody else's string. Obvious once written down, absent
    until then.

    Mutation: use the id unchanged and this fails.
    """
    assert "/" not in safe("../../etc/passwd")
    assert "\\" not in safe("..\\\\windows\\\\system32")
    assert safe("../../etc/passwd") == "etc-passwd"


def test_an_empty_or_hostile_id_still_yields_a_filename():
    assert safe("") == "unnamed"
    assert safe("///") == "unnamed"


def test_a_very_long_id_is_bounded():
    assert len(safe("x" * 500)) <= 120


def test_an_id_is_taken_from_whichever_key_the_export_used():
    assert identity({"uuid": "a"}, "claude", 0) == "a"
    assert identity({"conversation_id": "b"}, "chatgpt", 0) == "b"
    assert identity({"id": "c"}, "claude", 0) == "c"


def test_a_conversation_with_no_id_falls_back_to_its_position():
    """Weak, and better than dropping it. The run reports how many needed it,
    because a later export with one deleted shifts every name after it."""
    name = identity({"title": "no id"}, "chatgpt", 7)
    assert name == "chatgpt-position-00007"


def test_positional_names_are_reported_so_the_shift_is_not_a_surprise(tmp_path):
    archive = export(tmp_path, [claude(), {"chat_messages": [], "name": "no id"}])
    report = unpack(archive, tmp_path / "cache")
    assert len(list(positional(report))) == 1


# --- reading the archive ----------------------------------------------------


def test_a_zip_and_a_bare_json_are_both_accepted(tmp_path):
    """Both are things a person plausibly has."""
    zipped = export(tmp_path, [claude()])
    assert len(conversations_in(zipped)[0]) == 1

    plain = tmp_path / CONVERSATIONS
    plain.write_text(json.dumps([claude(), claude("c-2")]), encoding="utf-8")
    assert len(conversations_in(plain)[0]) == 2


def test_a_nested_conversations_file_is_found(tmp_path):
    """Claude's export nests everything under a dated folder."""
    archive = export(tmp_path, [claude()],
                     member=f"data-2026-08-20/{CONVERSATIONS}")
    assert len(conversations_in(archive)[0]) == 1


def test_a_zip_without_conversations_says_what_it_did_hold(tmp_path):
    """A reader who unpacked the wrong archive needs to know which one."""
    path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("users.json", "{}")
    with pytest.raises(ValueError) as raised:
        conversations_in(path)
    assert "users.json" in str(raised.value)


def test_an_export_that_is_not_a_list_is_refused(tmp_path):
    path = tmp_path / CONVERSATIONS
    path.write_text(json.dumps({"conversations": []}), encoding="utf-8")
    with pytest.raises(ValueError) as raised:
        unpack(path, tmp_path / "cache")
    assert "not a list" in str(raised.value)


# --- what the run reports ---------------------------------------------------


def test_a_new_conversation_is_written_once(tmp_path):
    cache = tmp_path / "cache"
    report = unpack(export(tmp_path, [claude()]), cache)
    assert report.written == ["c-1"]
    assert (cache / "claude" / "c-1.json").is_file()


def test_re_importing_the_same_export_writes_nothing_new(tmp_path):
    cache = tmp_path / "cache"
    archive = export(tmp_path, [claude()])
    unpack(archive, cache)
    again = unpack(archive, cache)
    assert again.written == []
    assert again.identical == ["c-1"]


def test_a_changed_conversation_is_reported_as_replaced(tmp_path):
    """It is not refused: re-importing a later export is the ordinary way this
    is used, and a conversation somebody kept talking in is supposed to change.
    What must not happen is the change passing unremarked."""
    cache = tmp_path / "cache"
    unpack(export(tmp_path, [claude(text="first")], name="a.zip"), cache)
    report = unpack(export(tmp_path, [claude(text="second")], name="b.zip"), cache)
    assert report.replaced == ["c-1"]
    assert "changed since the last export" in render(report, cache)


def test_a_dry_run_writes_nothing(tmp_path):
    cache = tmp_path / "cache"
    report = unpack(export(tmp_path, [claude()]), cache, dry_run=True)
    assert report.written == ["c-1"]
    assert not (cache / "claude" / "c-1.json").exists()
    assert "would write" in render(report, cache, dry_run=True)


def test_an_unrecognised_conversation_is_counted_and_named(tmp_path):
    """A source that silently dropped three of forty would report thirty-seven
    and look correct."""
    archive = export(tmp_path, [claude(), {"neither": "shape"}])
    report = unpack(archive, tmp_path / "cache")
    assert len(report.written) == 1
    assert len(report.unreadable) == 1
    assert "not recognised" in render(report, tmp_path / "cache")


def test_an_empty_export_says_that_is_what_it_held(tmp_path):
    """Not a failure to read it.

    Mutation: report an error on an empty array and this fails -- an account
    with no conversations is a real state.
    """
    report = unpack(export(tmp_path, []), tmp_path / "cache")
    assert report.total == 0
    assert "no conversations" in render(report, tmp_path / "cache")


def test_both_services_land_in_their_own_folders(tmp_path):
    cache = tmp_path / "cache"
    unpack(export(tmp_path, [claude(), chatgpt()]), cache)
    assert (cache / "claude" / "c-1.json").is_file()
    assert (cache / "chatgpt" / "g-1.json").is_file()


def test_what_was_imported_is_readable_by_the_source_that_reads_the_cache(tmp_path):
    """The end of the path: import writes what the reader reads.

    Two halves written from two guesses at a format would agree with each other
    and with nothing else -- which is a shape this organisation has already been
    caught by once, at a different seam.
    """
    from qmcp.spend import Budget
    from qmcp.threads.chatgpt import ChatGPTThreads
    from qmcp.threads.claude import ClaudeThreads

    cache = tmp_path / "cache"
    unpack(export(tmp_path, [claude(text="DECISION: ship it"), chatgpt()]), cache)

    pulled = ClaudeThreads(root=cache).fetch([], Budget())
    assert [t.id for t in pulled] == ["c-1"]
    assert ChatGPTThreads(root=cache).fetch([], Budget())[0].id == "g-1"

    source = ClaudeThreads(root=cache)
    deltas = source.deltas(pulled[0], Budget())
    assert [d["delta"]["delta_type"] for d in deltas] == ["thread", "chore"]


def test_the_importer_reaches_no_network():
    """It reads a file somebody downloaded. When something here fetches, this
    is the test that says the day it changed."""
    from pathlib import Path

    import qmcp.threads.importer as module

    text = Path(module.__file__).read_text(encoding="utf-8")
    for client in ("import requests", "import httpx", "urllib.request",
                   "import anthropic", "import openai"):
        assert client not in text
