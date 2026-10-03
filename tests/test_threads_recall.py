"""Where was I, read from the archive and said in sentences.

The tests worth reading are the two about choosing: the latest session is the
one whose *last turn* is latest, not the one that started last, and a session
that surveyed the whole workspace is not the latest work in any one project.
The fixture store carries several sessions across two projects, one session
about both, and a checkout that no longer exists.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import ModuleType

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qmcp.threads.claudecode import ClaudeCodeThreads
from qmcp.threads.recall import Recall, for_speech, names_for, recall
from qmcp.threads.service import register

NAMES = {"qmcp": "quaternionmedia/qmcp", "vox": "quaternionmedia/vox",
         "rad": "quaternionmedia/rad", "alfred": "quaternionmedia/alfred"}

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)


def record(uid, text, session_id, at, branch=None, cwd=None, role="assistant"):
    row = {"type": role, "uuid": uid, "sessionId": session_id, "timestamp": at,
           "message": {"content": [{"type": "text", "text": text}]}}
    if branch:
        row["gitBranch"] = branch
    if cwd:
        row["cwd"] = str(cwd)
    return row


def write(store, name, rows):
    store.mkdir(parents=True, exist_ok=True)
    (store / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                              encoding="utf-8")


@pytest.fixture
def store(tmp_path):
    """Several sessions, two projects, one about both, one checkout gone.

    Session   about   last turn            branch        cwd
    s-old     qmcp    2026-09-30 10:00     feat/old      <exists>
    s-new     qmcp    2026-10-02 09:30     feat/recall   <does not exist>
                      (its newest turn is a tool call with no text)
    s-vox     vox     2026-10-02 18:00     main          <exists>
    s-both    both    2026-10-01 12:00     fix/seam      <exists>
    s-survey  all     2026-10-03 08:00     main          <exists>  (a roster sweep)
    s-undated qmcp    (no timestamps)      -             -
    s-new/agent-a1  qmcp  2026-10-03 10:00  feat/recall  <exists>  (a sidechain)
    """
    here = tmp_path / "checkout"
    here.mkdir()
    gone = tmp_path / "worktree-removed"
    sessions = tmp_path / "sessions" / "proj"

    write(sessions, "old.jsonl", [
        record("o1", "in qmcp, starting the archive", "s-old",
               "2026-09-30T09:00:00Z", "feat/old", here),
        record("o2", "qmcp index written", "s-old",
               "2026-09-30T10:00:00Z", "feat/old", here),
    ])
    write(sessions, "new.jsonl", [
        record("n1", "qmcp recall: read the store", "s-new",
               "2026-10-01T08:00:00Z", "feat/recall", gone),
        record("n2", "the qmcp route is registered before the source route",
               "s-new", "2026-10-02T09:00:00Z", "feat/recall", gone),
        # A tool call, newest of all, with no text block: what a live session's
        # last record almost always is.
        {"type": "assistant", "uuid": "n3", "sessionId": "s-new",
         "timestamp": "2026-10-02T09:30:00Z", "gitBranch": "feat/recall",
         "cwd": str(gone),
         "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
        {"type": "pr-link", "sessionId": "s-new",
         "prRepository": "quaternionmedia/qmcp", "prNumber": 55},
        {"type": "ai-title", "sessionId": "s-new", "aiTitle": "Where was I"},
    ])
    write(sessions, "side.jsonl", [
        dict(record("a1", "qmcp: the subagent reads the store", "s-new",
                    "2026-10-03T09:00:00Z", "feat/recall", here),
             agentId="a1", isSidechain=True),
        dict(record("a2", "qmcp route tested", "s-new",
                    "2026-10-03T10:00:00Z", "feat/recall", here),
             agentId="a1", isSidechain=True),
    ])
    write(sessions, "vox.jsonl", [
        record("v1", "vox pause parameter", "s-vox",
               "2026-10-02T17:00:00Z", "main", here),
        record("v2", "vox contract updated", "s-vox",
               "2026-10-02T18:00:00Z", "main", here),
    ])
    write(sessions, "both.jsonl", [
        record("b1", "qmcp loads vox lazily", "s-both",
               "2026-10-01T11:00:00Z", "fix/seam", here),
        record("b2", "vox is imported by qmcp only on --speak", "s-both",
               "2026-10-01T12:00:00Z", "fix/seam", here),
    ])
    write(sessions, "survey.jsonl", [
        record("w1", "status of qmcp, vox, rad and alfred", "s-survey",
               "2026-10-03T07:00:00Z", "main", here),
        record("w2", "qmcp green, vox green, rad stale, alfred idle", "s-survey",
               "2026-10-03T08:00:00Z", "main", here),
    ])
    write(sessions, "undated.jsonl", [
        {"type": "user", "uuid": "u1", "sessionId": "s-undated",
         "message": {"content": [{"type": "text", "text": "qmcp question"}]}},
        {"type": "assistant", "uuid": "u2", "sessionId": "s-undated",
         "message": {"content": [{"type": "text", "text": "qmcp answer"}]}},
    ])
    return tmp_path / "sessions"


@pytest.fixture
def corpus(tmp_path):
    """A roster naming the same four projects, so the survey rule can fire.

    Without a roster the project is the only name `about` is read against, a
    sweep of the workspace cannot be told from work in it, and `s-survey` --
    the newest session in the store -- is what the route and the command
    answer with. That is the documented weaker reading, and the route and
    command tests pass a roster so they test the same choice the unit tests do.
    """
    where = tmp_path / "corpus"
    (where / "ci").mkdir(parents=True)
    (where / "ci" / "workspace.yaml").write_text(
        "repositories:\n" + "".join(f"  - name: {name}\n" for name in NAMES),
        encoding="utf-8")
    return where


def recalled(store, project, names=NAMES, now=NOW) -> Recall:
    return recall(project, [ClaudeCodeThreads(root=store)], names, now)


# --- choosing -----------------------------------------------------------------


def test_the_latest_session_is_the_one_whose_last_turn_is_latest(store):
    """THE ONE THAT MATTERS.

    `s-old` started a day before `s-new` finished; `s-survey` is newer than
    both and is passed over. The answer is the session a person was most
    recently *in*, which is the last turn and not the first.

    Mutation: sort by `started_at` and this fails on a store where the oldest
    session ran longest; choose `candidates[-1]` and it picks `s-old`.
    """
    found = recalled(store, "qmcp")
    assert found.chosen
    assert found.thread == "s-new"
    assert found.last_activity == "2026-10-02T09:30:00Z"


def test_a_workspace_survey_is_not_the_latest_work_in_a_project(store):
    """`s-survey` names every project in the roster and is the newest file in
    the store. It took stock of the workspace; it did not work in qmcp, and
    `consolidate` already declines to relate it to any one project.

    Mutation: drop the `relation is None` check and this chooses `s-survey`.
    """
    assert recalled(store, "qmcp").thread == "s-new"
    assert recalled(store, "vox").thread == "s-vox"


def test_a_subagent_s_sidechain_is_a_step_and_not_where_the_person_was(store):
    """`s-new/agent-a1` is the newest thread about qmcp in the store and is a
    subagent's conversation. The person was in `s-new`, which launched it;
    "where was I" names the session and not the step it ran. On a real store
    the newest thread about a project was a sidechain of the very session
    asking the question.

    Mutation: drop the `sidechain` check and this chooses `s-new/agent-a1`.
    """
    found = recalled(store, "qmcp")
    assert found.thread == "s-new"
    assert "/agent-" not in found.thread


def test_a_session_about_two_projects_is_a_candidate_for_each(store):
    """`s-both` is about qmcp and vox. With the two dedicated sessions removed
    it is what each project recalls."""
    (store / "proj" / "new.jsonl").unlink()
    (store / "proj" / "old.jsonl").unlink()
    (store / "proj" / "vox.jsonl").unlink()
    assert recalled(store, "qmcp").thread == "s-both"
    assert recalled(store, "vox").thread == "s-both"


def test_how_many_were_considered_is_reported_beside_how_many_were_read(store):
    """Four sessions are about qmcp: `s-old`, `s-new`, `s-both` and the undated
    one. The survey and the sidechain are not counted among them; every thread
    in the store was read to find them."""
    found = recalled(store, "qmcp")
    assert found.read == 7
    assert found.considered == 4


def test_an_undated_session_is_chosen_only_when_nothing_dated_is_about_it(store):
    """A thread with no timestamp cannot be latest. It is still an answer when
    it is the only one.

    Mutation: sort `None` first and `s-undated` wins over `s-new`.
    """
    assert recalled(store, "qmcp").thread == "s-new"
    for name in ("old", "new", "both", "survey"):
        (store / "proj" / f"{name}.jsonl").unlink()
    found = recalled(store, "qmcp")
    assert found.thread == "s-undated"
    assert found.last_activity is None


def test_nothing_about_the_project_is_an_answer_not_an_error(store):
    """Mutation: raise when no candidate is found and this fails -- which at a
    speaker reads as the archive being broken rather than empty of this."""
    found = recalled(store, "rad")
    assert not found.chosen
    assert found.thread is None
    assert found.read == 7
    assert found.considered == 0
    assert "Nothing in the archive is about rad" in found.spoken()
    assert found.rule in found.spoken()


def test_the_rule_is_consolidate_s_rule_and_is_reported(store):
    """Matching is `consolidate.about`'s, not a second matcher's. The rule it
    names is carried through, with the one addition this module makes."""
    from qmcp.threads.base import Thread
    from qmcp.threads.consolidate import about

    found = recalled(store, "qmcp")
    assert found.rule.startswith(about(Thread(id="-"), NAMES).rule)
    assert "surveying the workspace" in found.rule
    assert "sidechain" in found.rule


# --- what the chosen session carries ------------------------------------------


def test_the_chosen_session_carries_its_branch_checkout_and_pull_requests(store):
    found = recalled(store, "qmcp")
    assert found.branches == ("feat/recall",)
    assert found.cwd is not None and found.cwd.endswith("worktree-removed")
    assert found.pulls == (("quaternionmedia/qmcp", 55),)
    assert found.title == "Where was I"
    assert found.started == "2026-10-01T08:00:00Z"
    assert found.source == "claude-code"


def test_whether_the_checkout_exists_is_measured_not_assumed(store):
    """`s-new` ran in a worktree that has since been removed; `s-vox` ran in a
    directory that is still there. The field is a measurement of this disk.

    Mutation: set `cwd_exists=True` whenever `cwd` is set and the first
    assertion fails; set it to `None` and both do.
    """
    assert recalled(store, "qmcp").cwd_exists is False
    assert recalled(store, "vox").cwd_exists is True


def test_no_checkout_means_no_claim_about_one(store):
    for name in ("old", "new", "both", "survey"):
        (store / "proj" / f"{name}.jsonl").unlink()
    found = recalled(store, "qmcp")
    assert found.cwd is None
    assert found.cwd_exists is None


def test_the_last_turns_are_the_last_few_that_said_something(store):
    """`s-new`'s newest record is a tool call with no text. Taking the last
    turns by position quoted an empty string -- "Its last turn said: ." was
    what the first run on a real store spoke.

    Mutation: slice `thread.turns[-turns:]` without dropping empty text and
    the last entry is "".
    """
    found = recalled(store, "qmcp")
    assert [said.text for said in found.last_turns] == [
        "qmcp recall: read the store",
        "the qmcp route is registered before the source route",
    ]
    assert all(said.truncated is False for said in found.last_turns)


def test_a_long_turn_is_cut_on_a_word_and_marked():
    """Mutation: slice at `chars` without the `rsplit` and the cut lands
    mid-word; drop the ellipsis and `truncated` is the only sign."""
    text, truncated = for_speech("alpha beta gamma delta epsilon", chars=16)
    assert text == "alpha beta gamma..."
    assert truncated is True
    assert for_speech("alpha beta gamma delta epsilon", chars=14) == ("alpha beta...", True)
    assert for_speech("short", chars=16) == ("short", False)


def test_markup_is_not_read_aloud():
    text, _ = for_speech("# Title\n\nsome `code` and **bold** [link]")
    assert text == "Title some code and bold link"


# --- rendering ----------------------------------------------------------------


def test_spoken_names_branch_checkout_pull_request_and_last_turn(store):
    said = recalled(store, "qmcp").spoken()
    assert said.startswith("In qmcp, the last session, titled Where was I, was 1 day ago")
    assert "on branch feat/recall in " in said
    assert "which is no longer on disk" in said
    assert "It opened pull request 55 in quaternionmedia/qmcp." in said
    assert "Its last turn said: the qmcp route is registered before the source route." in said
    assert said.endswith("4 sessions about qmcp were read, of 7 in all.")


def test_spoken_says_when_a_checkout_is_still_there_by_saying_nothing(store):
    said = recalled(store, "vox").spoken()
    assert "no longer on disk" not in said
    assert "It opened no pull request." in said


def test_as_dict_carries_every_field_and_the_sentences(store):
    body = recalled(store, "qmcp").as_dict()
    assert body["chosen"] is True
    assert body["pulls"] == [{"repository": "quaternionmedia/qmcp", "number": 55}]
    assert body["cwd_exists"] is False
    assert body["last_turns"][-1]["truncated"] is False
    assert body["spoken"] == recalled(store, "qmcp").spoken()
    assert body["as_of"] == "2026-10-03T12:00:00Z"
    json.dumps(body)


def test_reading_spends_nothing(store):
    """Every source is read with a budget of nothing; a source that needed a
    paid call would refuse rather than bill whoever asked where they were."""
    from qmcp.spend import Budget

    source = ClaudeCodeThreads(root=store)
    spent = []
    original = source.fetch

    def watched(ids, budget: Budget):
        spent.append(budget.authorised)
        return original(ids, budget)

    source.fetch = watched
    recall("qmcp", [source], NAMES, NOW)
    assert spent == [0]


# --- the roster ---------------------------------------------------------------


def test_names_for_reads_the_roster_and_adds_the_project(tmp_path):
    """Without a roster the project is the only name; with one, the project is
    read against every repository so the survey rule can fire."""
    assert names_for("made-up", tmp_path) == {"made-up": "made-up"}

    corpus = tmp_path / "corpus"
    (corpus / "ci").mkdir(parents=True)
    (corpus / "ci" / "workspace.yaml").write_text(
        "repositories:\n  - name: qmcp\n  - name: vox\n", encoding="utf-8")
    found = names_for("qmcp", corpus)
    assert set(found) == {"qmcp", "vox"}
    assert names_for("other", corpus)["other"] == "other"


# --- the route ----------------------------------------------------------------


def client_for(store, corpus):
    app = FastAPI()
    register(app, store, store, corpus=corpus)
    return TestClient(app)


def test_the_route_answers_with_the_same_dict(store, corpus):
    body = client_for(store, corpus).get("/v1/threads/recall/qmcp").json()
    assert body["thread"] == "s-new"
    assert body["branches"] == ["feat/recall"]
    assert body["chosen"] is True


def test_without_a_roster_a_workspace_sweep_cannot_be_told_from_work(store):
    """The weaker reading, stated rather than hidden. With the project as the
    only name the survey rule has nothing to count against, and the newest
    session -- the sweep -- is the answer. A roster is what makes the better
    answer possible, which is why the route reads the embedded corpus.

    Mutation: make `names_for` return the roster when the corpus is absent and
    this fails -- which is a roster invented from nothing.
    """
    body = client_for(store, store / "no-corpus-here").get(
        "/v1/threads/recall/qmcp").json()
    assert body["thread"] == "s-survey"


def test_the_route_is_not_swallowed_by_the_source_route(store, corpus):
    """`/v1/threads/recall/{project}` and `/v1/threads/{source}/{identifier}`
    have the same shape. Registered in the wrong order, `recall` is read as a
    source nobody declared and the 404 looks like an empty archive.

    Mutation: move the registration below `get_thread` and this fails with 404.
    A project nothing is about is used so the only thing distinguishing the
    two answers is the route that produced it.
    """
    response = client_for(store, corpus).get("/v1/threads/recall/rad")
    assert response.status_code == 200
    assert response.json()["chosen"] is False
    assert response.json()["project"] == "rad"


def test_the_route_is_absent_off_loopback(monkeypatch):
    """The archive is somebody's conversations. Off loopback the routes are
    not registered at all, this one included."""
    import qmcp.server

    class OffLoopback:
        host = "0.0.0.0"
        port = 3141
        debug = False
        database_url = "sqlite+aiosqlite:///:memory:"
        log_level = "WARNING"
        voice_engine = "joe"
        voice_engine_url = None

    monkeypatch.setattr(qmcp.server, "get_settings", lambda: OffLoopback())
    app = qmcp.server.create_app()
    assert not [r for r in app.routes if "recall" in getattr(r, "path", "")]


def test_the_app_mounts_the_route_on_loopback():
    from qmcp import server

    app = server.create_app()
    assert "/v1/threads/recall/{project}" in {
        getattr(r, "path", "") for r in app.routes}


# --- the command --------------------------------------------------------------


class _FakeTTS:
    spoken: list[str] = []

    def speak(self, text, out_path=None):
        _FakeTTS.spoken.append(text)
        return out_path or "spoken.wav"


def _install_fake_vox(monkeypatch):
    """`--speak` imports vox lazily through `_load_vox`; stand it in."""
    _FakeTTS.spoken = []
    fake_vox = ModuleType("vox")
    fake_vox.HttpSTT = lambda *a, **k: pytest.fail("an engine client was built")
    fake_pyttsx3 = ModuleType("vox.adapters.pyttsx3")
    fake_pyttsx3.Pyttsx3TTS = _FakeTTS
    monkeypatch.setitem(sys.modules, "vox", fake_vox)
    monkeypatch.setitem(sys.modules, "vox.adapters.pyttsx3", fake_pyttsx3)


def test_the_command_prints_the_sentences(store, corpus):
    from qmcp.cli import cli

    result = CliRunner().invoke(cli, [
        "threads", "recall", "qmcp", "--sessions", str(store),
        "--root", str(store / "no-exports"), "--corpus", str(corpus)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("In qmcp, the last session, titled Where was I,")


def test_the_command_prints_data_with_json(store, corpus):
    from qmcp.cli import cli

    result = CliRunner().invoke(cli, [
        "threads", "recall", "qmcp", "--json", "--sessions", str(store),
        "--root", str(store / "no-exports"), "--corpus", str(corpus)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["thread"] == "s-new"


def test_speak_says_the_printed_sentences_and_nothing_else(store, corpus, monkeypatch):
    """Mutation: speak `found.title` instead and the first assertion fails;
    build an `HttpSTT` and the fake fails the test from inside the import."""
    from qmcp.cli import cli

    _install_fake_vox(monkeypatch)
    result = CliRunner().invoke(cli, [
        "threads", "recall", "qmcp", "--speak", "--sessions", str(store),
        "--root", str(store / "no-exports"), "--corpus", str(corpus)])
    assert result.exit_code == 0, result.output
    assert _FakeTTS.spoken == [result.output.strip()]


def test_without_speak_nothing_is_synthesized(store, corpus, monkeypatch):
    from qmcp.cli import cli

    _install_fake_vox(monkeypatch)
    CliRunner().invoke(cli, [
        "threads", "recall", "qmcp", "--sessions", str(store),
        "--root", str(store / "no-exports"), "--corpus", str(corpus)])
    assert _FakeTTS.spoken == []
