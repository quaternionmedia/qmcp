"""The spoken loop's vocabulary, declared once and read by the code that acts on it.

`qmcp.integrations.voice.vocabulary` reads `vocabulary.toml`. These check that
each list the loop acts on is the declared one, that a phrase heard as a whole
utterance means one thing, and that every entry says what it does.
"""

from __future__ import annotations

import itertools

from qmcp.instructions import converse
from qmcp.instructions.dialog import AGREE_ALIASES, CONFIRM
from qmcp.integrations.voice import adapter, vocabulary

# The groups heard as a whole utterance, where one phrase in two would make the
# loop's answer depend on which it checked first.
WHOLE_UTTERANCE = ("conversation.stop", "conversation.repeat", "conversation.done",
                   "conversation.more")


def test_every_list_the_loop_acts_on_is_the_declared_one():
    """Mutation: spell one of the loop's lists in its module again -- red."""
    assert converse.STOP == vocabulary.phrases("conversation.stop")
    assert converse.DONE == vocabulary.phrases("conversation.done")
    assert converse.MORE == vocabulary.phrases("conversation.more")
    assert adapter.REPEAT == vocabulary.phrases("conversation.repeat")
    assert adapter._YES_WORDS == set(vocabulary.words("answer.yes"))
    assert adapter._NO_WORDS == set(vocabulary.words("answer.no"))
    assert adapter._YES_PHRASES == vocabulary.phrases("answer.yes")
    assert adapter._NO_PHRASES == vocabulary.phrases("answer.no")
    assert adapter._UNCLEAR_PHRASES == vocabulary.phrases("answer.unclear")
    agree, *aliases = vocabulary.phrases("answer.agree")
    assert CONFIRM == [agree, vocabulary.phrases("answer.again")[0]]
    assert AGREE_ALIASES == tuple(aliases)


def test_a_phrase_heard_whole_means_one_thing():
    for first, second in itertools.combinations(WHOLE_UTTERANCE, 2):
        shared = set(vocabulary.phrases(first)) & set(vocabulary.phrases(second))
        assert not shared, (first, second, shared)


def test_yes_and_no_share_no_word_or_phrase():
    assert not set(vocabulary.words("answer.yes")) & set(vocabulary.words("answer.no"))
    assert not set(vocabulary.phrases("answer.yes")) & set(vocabulary.phrases("answer.no"))


def test_every_entry_says_what_it_does_and_is_written_as_heard():
    """A phrase is compared lower-case with punctuation gone, so one written
    otherwise could never match."""
    shown = vocabulary.entries()
    assert {e["key"] for e in shown} >= set(WHOLE_UTTERANCE) | {"answer.yes", "answer.no"}
    for item in shown:
        assert item["says"], item["key"]
        for phrase in item["phrases"] + item["words"]:
            assert phrase == converse.plain(phrase), (item["key"], phrase)


def test_an_unknown_key_is_refused_by_name():
    import pytest

    with pytest.raises(KeyError, match="conversation.nothing"):
        vocabulary.phrases("conversation.nothing")
