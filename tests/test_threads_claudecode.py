"""Claude Code sessions, and the joins a web export does not have.

The tests worth reading are the ones about `project_of`. A conversation belongs
to no repository, which is why a web export's deltas go to a project somebody
chose -- and a Claude Code session does not have that problem, because it says
which repository it was working in.
"""

from __future__ import annotations

import json

from qmcp.spend import Budget
from qmcp.threads.claudecode import BRANCH_LINK, PR_LINK, ClaudeCodeThreads


def session(tmp_path, records, name="s1.jsonl", project="proj"):
    directory = tmp_path / project
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n",
                    encoding="utf-8")
    return path


def turn(role="user", text="hello", uuid="u-1", branch=None, session_id="s-1"):
    record = {"type": role, "uuid": uuid, "sessionId": session_id,
              "timestamp": "2026-08-20T00:00:00Z",
              "message": {"content": [{"type": "text", "text": text}]}}
    if branch:
        record["gitBranch"] = branch
    return record


# --- reading a session --------------------------------------------------------


def test_a_session_is_one_thread(tmp_path):
    session(tmp_path, [turn(uuid="u-1"), turn("assistant", "hi", "u-2")])
    threads = ClaudeCodeThreads(root=tmp_path).fetch([], Budget())
    assert len(threads) == 1
    assert [t.id for t in threads] == ["s-1"]
    assert len(threads[0].turns) == 2


def test_reading_a_session_spends_nothing(tmp_path):
    """Local, free, and no credential -- which is the whole reason this source
    exists rather than an API one."""
    session(tmp_path, [turn()])
    budget = Budget()
    ClaudeCodeThreads(root=tmp_path).fetch([], budget)
    assert budget.made == 0


def test_a_half_written_last_line_does_not_lose_the_session(tmp_path):
    """A session file is appended to while a session runs, so the last line may
    be half-written at the moment this reads it. Refusing the whole file for
    that would make an in-progress session unreadable -- the one it is most
    interesting to read.

    Mutation: raise on a malformed line and this fails.
    """
    path = session(tmp_path, [turn(uuid="u-1")])
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "assistant", "mess')
    threads = ClaudeCodeThreads(root=tmp_path).fetch([], Budget())
    assert len(threads) == 1
    assert len(threads[0].turns) == 1


def test_a_file_that_is_not_a_session_is_skipped_not_called_unreadable(tmp_path):
    """A `.jsonl` under this root is not necessarily a session -- workflow
    journals live here too. Calling one unreadable puts a scary count on the
    page for something entirely fine, and there were seven of them.

    Mutation: report anything without turns as unreadable and this fails.
    """
    session(tmp_path, [{"type": "workflow-step", "step": "one"}],
            name="journal.jsonl")
    source = ClaudeCodeThreads(root=tmp_path)
    assert source.fetch([], Budget()) == []
    assert source.skipped == ["journal.jsonl"]
    assert source.unreadable == []


def test_a_session_file_with_no_turn_is_unreadable(tmp_path):
    """The other branch. Something carrying session records and no turn is a
    session that failed to read, and that is worth a count."""
    session(tmp_path, [{"type": "ai-title", "sessionId": "s-1",
                        "aiTitle": "a session with nothing in it"}])
    source = ClaudeCodeThreads(root=tmp_path)
    assert source.fetch([], Budget()) == []
    assert source.skipped == []
    assert "no turn" in source.unreadable[0].why


def test_machinery_records_are_skipped_not_rendered(tmp_path):
    """A tool invocation rendered as its repr becomes searchable prose that says
    nothing, and would then be scanned for decision markers it cannot contain."""
    session(tmp_path, [
        turn(uuid="u-1", text="real"),
        {"type": "queue-operation", "operation": "add", "content": "not a turn"},
        {"type": "attachment", "attachment": {"x": 1}},
    ])
    turns = ClaudeCodeThreads(root=tmp_path).fetch([], Budget())[0].turns
    assert [t.text for t in turns] == ["real"]


def test_a_non_text_block_is_skipped(tmp_path):
    session(tmp_path, [{
        "type": "assistant", "uuid": "u-1", "sessionId": "s-1",
        "message": {"content": [{"type": "tool_use", "name": "Bash"},
                                {"type": "text", "text": "kept"}]}}])
    assert ClaudeCodeThreads(root=tmp_path).fetch([], Budget())[0].turns[0].text \
        == "kept"


def test_the_title_comes_from_the_session_when_it_has_one(tmp_path):
    session(tmp_path, [turn(), {"type": "ai-title", "sessionId": "s-1",
                                "aiTitle": "Naming the ask kind"}])
    assert ClaudeCodeThreads(root=tmp_path).fetch([], Budget())[0].title \
        == "Naming the ask kind"


# --- the joins, which are the point -------------------------------------------


def test_the_project_comes_from_the_session_not_a_default(tmp_path):
    """THE ONE THAT MATTERS.

    A conversation belongs to no repository, so a web export's deltas go to a
    project somebody chose. A session says which repository it was working in.

    Mutation: return `self.project` unconditionally and this fails, which is
    every session's work filed under one repository.
    """
    session(tmp_path, [turn(),
                       {"type": "pr-link", "sessionId": "s-1",
                        "prRepository": "quaternionmedia/rad", "prNumber": 4}])
    source = ClaudeCodeThreads(root=tmp_path)
    thread = source.fetch([], Budget())[0]
    assert source.project_of(thread) == "quaternionmedia/rad"


def test_a_session_that_names_no_repository_falls_back_and_that_is_a_guess(tmp_path):
    session(tmp_path, [turn()])
    source = ClaudeCodeThreads(root=tmp_path)
    thread = source.fetch([], Budget())[0]
    assert source.project_of(thread) == source.project


def test_a_pull_request_becomes_an_address_on_the_thread_delta(tmp_path):
    """What a web export cannot do: the conversation is linked to the pull
    request it produced, by an address the control panel already uses."""
    session(tmp_path, [turn(),
                       {"type": "pr-link", "sessionId": "s-1",
                        "prRepository": "quaternionmedia/qm", "prNumber": 80}])
    source = ClaudeCodeThreads(root=tmp_path)
    payload = source.deltas(source.fetch([], Budget())[0], Budget())[0]
    links = {(link["link_type"], link["target_name"]) for link in payload["links"]}
    assert (PR_LINK, "quaternionmedia/qm/pr/80") in links


def test_a_branch_becomes_an_address_too(tmp_path):
    session(tmp_path, [turn(branch="evolve/active-repos"),
                       {"type": "pr-link", "sessionId": "s-1",
                        "prRepository": "quaternionmedia/qm", "prNumber": 1}])
    source = ClaudeCodeThreads(root=tmp_path)
    payload = source.deltas(source.fetch([], Budget())[0], Budget())[0]
    links = {(link["link_type"], link["target_name"]) for link in payload["links"]}
    assert (BRANCH_LINK, "quaternionmedia/qm/branch/evolve/active-repos") in links


def test_a_branch_address_keeps_its_slashes(tmp_path):
    """`evolve/active-repos` is the branch's real name. The grammar takes
    everything after the kind as the id, verbatim, which is the whole reason
    branches with slashes are addressable at all."""
    session(tmp_path, [turn(branch="evolve/active-repos")])
    source = ClaudeCodeThreads(root=tmp_path)
    payload = source.deltas(source.fetch([], Budget())[0], Budget())[0]
    branch = next(link["target_name"] for link in payload["links"]
                  if link["link_type"] == BRANCH_LINK)
    assert branch.endswith("/branch/evolve/active-repos")


def test_the_deltas_and_relations_agree_about_the_project(tmp_path):
    """A relation pointing at a delta filed under a different project would
    address nothing."""
    session(tmp_path, [
        turn(text="DECISION: split the gate"),
        {"type": "pr-link", "sessionId": "s-1",
         "prRepository": "quaternionmedia/rad", "prNumber": 4}])
    source = ClaudeCodeThreads(root=tmp_path)
    thread = source.fetch([], Budget())[0]
    payloads = source.deltas(thread, Budget())
    relations = source.relations(thread, Budget())
    assert all(p["project"] == "quaternionmedia/rad" for p in payloads)
    assert relations[0]["target"].startswith("quaternionmedia/rad/delta/")


def test_the_perspective_is_its_own(tmp_path):
    """A session and a web conversation are not the same kind of thing, and
    neither is a duplicate of the other."""
    assert ClaudeCodeThreads().perspective == "claude-code/session"


def test_parse_refuses_because_a_session_is_lines_not_an_object(tmp_path):
    """The base class reads whole JSON files. This overrides `fetch` instead,
    and `parse` stays refusing so a subclass that forgot cannot fall through to
    a shape that does not apply."""
    import pytest

    with pytest.raises(NotImplementedError):
        ClaudeCodeThreads(root=tmp_path).parse({}, tmp_path / "x.jsonl")


def test_this_source_writes_nothing_into_the_store_it_reads(tmp_path):
    """It reads somebody else's store. A test on the source, because the day it
    starts writing is the day this matters."""
    from pathlib import Path

    import qmcp.threads.claudecode as module

    text = Path(module.__file__).read_text(encoding="utf-8")
    for writer in ("write_text(", "open(\"w\"", "'w'", "mkdir("):
        assert writer not in text.replace('errors="replace"', ""), writer


# --- sidechains, and the collision that made seven false divergences ---------


def test_a_subagent_file_is_its_own_thread(tmp_path):
    """A SUBAGENT FILE CARRIES ITS PARENT'S sessionId, AND THAT WAS THE TRAP.

    Keying on `sessionId` alone collapsed many files into one id, each
    overwriting the last, and the index read every overwrite as the thread
    diverging. Seven "divergences" on a first index of a real machine, none of
    them real -- and a false divergence is worse than a missed one, because a
    reader who finds the first seven are noise stops looking at the eighth.

    Mutation: drop the agentId from the identity and this fails.
    """
    session(tmp_path, [turn(uuid="u-1", session_id="s-1")], name="main.jsonl")
    session(tmp_path, [dict(turn(uuid="u-2", session_id="s-1"),
                            agentId="a-9", isSidechain=True)],
            name="agent.jsonl")
    threads = ClaudeCodeThreads(root=tmp_path).fetch([], Budget())
    assert len({t.id for t in threads}) == 2, "two files, two threads"
    assert any(t.id == "s-1/agent-a-9" for t in threads)


def test_two_subagents_of_one_session_do_not_collide(tmp_path):
    """Forty-six files shared one sessionId on the machine this was found on."""
    session(tmp_path, [turn(uuid="u-1", session_id="s-1")], name="main.jsonl")
    for agent in ("a-1", "a-2", "a-3"):
        session(tmp_path, [dict(turn(uuid=f"u-{agent}", session_id="s-1"),
                                agentId=agent, isSidechain=True)],
                name=f"{agent}.jsonl")
    threads = ClaudeCodeThreads(root=tmp_path).fetch([], Budget())
    assert len({t.id for t in threads}) == 4


def test_a_subagent_is_part_of_the_session_that_launched_it(tmp_path):
    """Stated rather than derived from the shared id prefix: a consumer must
    not infer containment from two rows looking alike."""
    session(tmp_path, [dict(turn(uuid="u-2", session_id="s-1"),
                            agentId="a-9", isSidechain=True)],
            name="agent.jsonl")
    source = ClaudeCodeThreads(root=tmp_path)
    thread = source.fetch([], Budget())[0]
    relations = source.relations(thread, Budget())
    assert any(r["relation"] == "part-of"
               and r["target"].endswith("/delta/thread-s-1")
               for r in relations)


def test_a_session_with_no_subagent_is_related_to_nothing(tmp_path):
    """Mutation: relate every thread to a parent and this fails, which is a
    board asserting a hierarchy that is not there."""
    session(tmp_path, [turn(uuid="u-1", session_id="s-1")])
    source = ClaudeCodeThreads(root=tmp_path)
    thread = source.fetch([], Budget())[0]
    assert source.relations(thread, Budget()) == []


def test_a_file_with_no_session_id_falls_back_to_its_name(tmp_path):
    """Two files with neither id would otherwise be one thread."""
    for name in ("one.jsonl", "two.jsonl"):
        session(tmp_path, [{"type": "user", "uuid": f"u-{name}",
                            "message": {"content": [{"type": "text",
                                                     "text": "hi"}]}}],
                name=name)
    threads = ClaudeCodeThreads(root=tmp_path).fetch([], Budget())
    assert len({t.id for t in threads}) == 2
