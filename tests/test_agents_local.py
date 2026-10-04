"""The local model as a runtime: the read protocol, the read-only tools, and the bounds.

The model service is stood in for by an `httpx.MockTransport` that answers as
the service's chat route does, so each test decides what the model replies and
sees exactly what was sent. Nothing here needs a model, a GPU or a network.
"""

from __future__ import annotations

import json

import httpx

from qmcp.integrations.agents import Brief, Turn
from qmcp.integrations.agents.adapters import ollama
from qmcp.localmodel import ENDPOINT, MODEL


def _reads(tool, **arguments):
    """A reply that reads, in the protocol `SYSTEM` states."""
    return {"content": json.dumps({"tool": tool, **arguments})}


class _Service:
    """The chat route: answers from a script, keeps every request it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("the runtime asked the model more often than the script allows")
        reply = self.replies.pop(0)
        return httpx.Response(200, json={"message": {"role": "assistant", **reply}, "done": True})

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def _project(tmp_path):
    (tmp_path / "README.md").write_text("# qmcp\n\nqmcp is the local backend.\n", encoding="utf-8")
    (tmp_path / "qmcp").mkdir()
    (tmp_path / "qmcp" / "server.py").write_text("app = 'the server'\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[secret]\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("the local backend, vendored\n", encoding="utf-8")
    return tmp_path


def _brief(cwd, history=()):
    return Brief(instruction="What is qmcp?", cwd=cwd, project="qmcp", history=tuple(history))


def _results(request):
    """What the runtime sent back to the model as tool results."""
    return [m["content"] for m in request["messages"]
            if m["role"] == "user" and m["content"].startswith("Result of ")]


# --- the protocol -----------------------------------------------------------------


def test_the_model_reads_with_a_tool_and_then_answers(tmp_path):
    """Mutation: drop the tool result from the messages -- red, the second
    request carries nothing the model read; return the first reply as the
    answer -- red, the answer is a JSON object."""
    service = _Service(_reads("read_file", path="README.md"),
                       {"content": "qmcp is the local backend."})
    events: list[tuple[str, str]] = []

    outcome = ollama.Runtime(client=service.client()).run(
        _brief(_project(tmp_path)), on_event=lambda s, t: events.append((s, t)))

    assert outcome.text == "qmcp is the local backend." and outcome.exit_code == 0
    assert outcome.spent == 0
    assert outcome.detail == {"model": MODEL, "endpoint": ENDPOINT, "model_calls": 2,
                              "read": ["read_file(README.md)"], "nudged": False, "repeats": 0,
                              "recovered": 0}
    (result,) = _results(service.requests[1])
    assert result.startswith("Result of read_file(README.md):\n")
    assert "qmcp is the local backend." in result
    assert ("output", "read_file(README.md)") in events
    assert events[0][0] == "started" and events[-1] == ("finished", "0")


def test_the_call_is_read_from_the_shapes_a_model_emits(tmp_path):
    """The protocol's shape, the shape the model was trained on, fenced or
    tagged, and the structured field a service may fill. Mutation: read only
    `tool` -- red on `name`; drop the fence stripping -- red on the fenced one;
    drop the structured branch -- red on `tool_calls`."""
    shapes = [
        {"content": '{"tool": "read_file", "path": "README.md"}'},
        {"content": '{"name": "read_file", "arguments": {"path": "README.md"}}'},
        {"content": '```json\n{"tool": "read_file", "path": "README.md"}\n```'},
        {"content": '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'},
        {"content": "", "tool_calls": [{"function": {"name": "read_file",
                                                      "arguments": '{"path": "README.md"}'}}]},
    ]
    for shape in shapes:
        assert ollama.tool_call_in(shape) == ("read_file", {"path": "README.md"}), shape


def test_prose_that_contains_braces_is_an_answer():
    """Mutation: read any JSON object found in the text -- red."""
    for content in ('The config is {"a": 1}.', "{not json}", '{"path": "x"}', "[1, 2]", ""):
        assert ollama.tool_call_in({"content": content}) is None, content


def test_an_answer_wrapped_in_json_is_unwrapped_to_prose(tmp_path):
    """Seen live: told to read in JSON, the model answered in it too, and the
    summary would have been read aloud as braces. Mutation: return the
    content as it came -- red."""
    service = _Service(_reads("read_file", path="README.md"),
                       {"content": '```json\n{"answer": "README.md", "first_sentence": '
                                   '"qmcp is the local backend."}\n```'})

    outcome = ollama.Runtime(client=service.client()).run(_brief(_project(tmp_path)))

    assert outcome.text == "README.md. qmcp is the local backend."
    assert ollama.answer_text("Plain words, {with braces}.") == "Plain words, {with braces}."
    assert ollama.answer_text('{"a": ""}') == '{"a": ""}'


def test_a_read_already_made_is_not_made_again_and_the_model_is_asked_to_answer(tmp_path):
    """Seen live: the model alternated between rereading one file and searching
    for an invented phrase until the step bound ended the run. Mutation: drop
    the repeat check -- red, the file is read twice and no one asks for the
    answer."""
    service = _Service(_reads("read_file", path="README.md"),
                       _reads("read_file", path="README.md"),
                       {"content": "qmcp is the local backend."})

    outcome = ollama.Runtime(client=service.client()).run(_brief(_project(tmp_path)))

    assert outcome.text == "qmcp is the local backend." and outcome.exit_code == 0
    assert outcome.detail["read"] == ["read_file(README.md)"]
    assert outcome.detail["repeats"] == 1
    assert service.requests[2]["messages"][-1] == {"role": "user", "content": ollama.AGAIN}


def test_an_answer_before_any_read_is_sent_back_once(tmp_path):
    """A small model's likeliest failure is answering from nothing. Mutation:
    drop the nudge -- red, the unread answer is returned."""
    service = _Service({"content": "qmcp is a quantum Monte Carlo project."},
                       _reads("read_file", path="README.md"),
                       {"content": "qmcp is the local backend."})

    outcome = ollama.Runtime(client=service.client()).run(_brief(_project(tmp_path)))

    assert outcome.text == "qmcp is the local backend."
    assert outcome.detail["nudged"] is True and outcome.detail["model_calls"] == 3
    assert service.requests[1]["messages"][-1] == {"role": "user", "content": ollama.NUDGE}


def test_a_model_that_still_will_not_read_has_its_answer_kept_and_flagged(tmp_path):
    """One nudge, not a loop: the second unread answer is returned, and the
    record says nothing was read."""
    service = _Service({"content": "a guess"}, {"content": "the same guess"})

    outcome = ollama.Runtime(client=service.client()).run(_brief(tmp_path))

    assert outcome.text == "the same guess"
    assert outcome.detail["read"] == [] and outcome.detail["nudged"] is True


# --- what is sent ----------------------------------------------------------------------


def test_every_request_is_the_pinned_model_capped_with_the_tools_in_the_prompt(tmp_path):
    """Mutation: drop `num_predict` -- red; an uncapped request can hold the
    service long after its caller has gone. The service's tool field is not
    sent: the protocol is in `SYSTEM`."""
    service = _Service({"content": "done"}, {"content": "done"})

    ollama.Runtime(client=service.client()).run(_brief(tmp_path))

    request = service.requests[0]
    assert request["model"] == MODEL and request["stream"] is False
    assert request["options"]["num_predict"] == ollama.MAX_TOKENS
    assert request["options"]["temperature"] == 0  # so a run can be repeated
    assert "tools" not in request
    assert request["messages"][0] == {"role": "system", "content": ollama.SYSTEM}
    for name in ollama.TOOL_NAMES:
        assert f'"tool": "{name}"' in ollama.SYSTEM
    assert request["messages"][1] == {"role": "user", "content": _brief(tmp_path).prompt()}


def test_the_history_from_qmcps_record_reaches_the_model(tmp_path):
    """Continuity comes from qmcp, not the model."""
    service = _Service({"content": "done"}, {"content": "done"})
    earlier = Turn(id="t-1", instruction="Find the README.", status="done",
                   outcome="README.md, at the root.", runtime="local")

    ollama.Runtime(client=service.client()).run(_brief(tmp_path, [earlier]))

    assert "README.md, at the root." in service.requests[0]["messages"][1]["content"]


# --- the tools -----------------------------------------------------------------------


def test_a_path_outside_the_clone_or_inside_git_is_refused_and_the_refusal_goes_back(tmp_path):
    """Mutation: drop the `parents` check in `inside` -- red, `../` reads
    outside the clone; drop the skipped-directory check -- red on `.git`."""
    (tmp_path / "work").mkdir()
    project = _project(tmp_path / "work")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    service = _Service(_reads("read_file", path="../secret.txt"),
                       _reads("read_file", path=".git/config"),
                       _reads("list_files", path=".venv"),
                       {"content": "nothing readable"})

    ollama.Runtime(client=service.client()).run(_brief(project))

    replies = [r.split("\n", 1)[1] for r in _results(service.requests[-1])]
    assert replies == ["refused: '../secret.txt' is outside the project",
                       "refused: '.git/config' is not read",
                       "refused: '.venv' is not read"]


def test_a_leading_slash_means_the_top_of_the_project(tmp_path):
    """Models write `/` for the project's root; on Windows that is the disk's.
    Mutation: drop the `lstrip` -- red, `/README.md` is refused as outside."""
    project = _project(tmp_path)

    assert "qmcp is the local backend." in ollama.read_file(project, "/README.md")
    assert ollama.list_files(project, "/").splitlines() == ["README.md", "qmcp/"]


def test_listing_and_searching_skip_what_is_never_read(tmp_path):
    """Mutation: walk every directory in `search` -- red, the environment's
    copy is found."""
    project = _project(tmp_path)

    assert ollama.list_files(project, ".").splitlines() == ["README.md", "qmcp/"]
    assert ollama.search(project, "LOCAL BACKEND") == "README.md:3: qmcp is the local backend."
    assert ollama.search(project, "nowhere") == "(no matches)"
    assert ollama.call_tool(project, "write_file", {"path": "x"}).startswith("refused:")


def test_a_long_file_is_cut_and_says_so(tmp_path):
    (tmp_path / "big.txt").write_text("x" * (ollama.READ_CHARS + 50), encoding="utf-8")

    text = ollama.read_file(tmp_path, "big.txt")

    assert text.endswith(f"(cut at {ollama.READ_CHARS} of {ollama.READ_CHARS + 50} characters)")


# --- the bounds ----------------------------------------------------------------------


def test_a_model_that_never_stops_reading_is_a_failed_run_at_the_bound(tmp_path):
    """Mutation: drop the step bound -- red, the stand-in runs out of replies."""
    project = _project(tmp_path)
    service = _Service(*[_reads("list_files", path=".")] * 3)

    outcome = ollama.Runtime(client=service.client(), max_steps=3).run(_brief(project))

    assert outcome.exit_code == 1 and "did not finish within 3 calls" in outcome.text
    assert outcome.detail["model_calls"] == 3


def test_an_empty_answer_is_a_failed_run(tmp_path):
    service = _Service(_reads("list_files", path="."), {"content": "   "})

    assert ollama.Runtime(client=service.client()).run(_brief(_project(tmp_path))).exit_code == 1


def test_a_service_that_does_not_answer_is_a_failed_run_naming_the_endpoint(tmp_path):
    """Mutation: let the transport error out of `run` -- red; the act would
    record an exception rather than where to look."""
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    outcome = ollama.Runtime(client=httpx.Client(transport=httpx.MockTransport(refuse))).run(
        _brief(tmp_path))

    assert outcome.exit_code == 1 and outcome.spent == 0
    assert ENDPOINT in outcome.text and "qmcp localmodel check" in outcome.text


# --- readiness -----------------------------------------------------------------------


def _tags(*names):
    def answer(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": n} for n in names]})
    return httpx.Client(transport=httpx.MockTransport(answer))


def test_a_service_serving_the_pinned_model_is_ready():
    assert ollama.Runtime(client=_tags(MODEL, "other:1b")).ready() is None


def test_a_service_without_the_pinned_model_says_how_to_get_it():
    """Mutation: check only that the service answers -- red."""
    reason = ollama.Runtime(client=_tags("other:1b")).ready()

    assert MODEL in reason and "qmcp localmodel plan" in reason


def test_a_service_that_does_not_answer_is_not_ready():
    """Mutation: let the transport error out of `ready` -- red."""
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    reason = ollama.Runtime(client=httpx.Client(transport=httpx.MockTransport(refuse))).ready()

    assert ENDPOINT in reason and "qmcp localmodel plan" in reason


# --- a stalled call ------------------------------------------------------------------------


class _Stalling:
    """The service, stalling on its first chat: times out, then answers after an unload."""

    def __init__(self, stalls=1):
        self.stalls = stalls
        self.paths: list[tuple[str, dict]] = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.paths.append((request.url.path, body))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        if self.stalls:
            self.stalls -= 1
            raise httpx.ReadTimeout("stalled", request=request)
        return httpx.Response(200, json={"message": {"role": "assistant",
                                                     "content": "qmcp is the local backend."}})


def test_a_stalled_call_unloads_the_model_and_is_made_once_more(tmp_path):
    """Seen live: the runner stopped producing tokens, and every later call
    queued behind it until the model was unloaded. Mutation: drop the retry --
    red, the run fails on the first stall."""
    service = _Stalling(stalls=1)
    client = httpx.Client(transport=httpx.MockTransport(service))

    outcome = ollama.Runtime(client=client).run(_brief(_project(tmp_path)))

    assert [path for path, _ in service.paths][:3] == ["/api/chat", "/api/generate", "/api/chat"]
    assert service.paths[1][1] == {"model": MODEL, "keep_alive": 0}
    assert outcome.detail["recovered"] >= 1
    # The run went on after the recovery and returned the answer it was given.
    assert outcome.text == "qmcp is the local backend." and outcome.exit_code == 0


def test_a_second_stall_is_a_failed_run_naming_the_endpoint(tmp_path):
    """One recovery, not a loop. Mutation: retry until it answers -- red, the
    stand-in stalls for ever and the run never ends."""
    service = _Stalling(stalls=2)

    outcome = ollama.Runtime(client=httpx.Client(transport=httpx.MockTransport(service))).run(
        _brief(tmp_path))

    assert outcome.exit_code == 1 and "ReadTimeout" in outcome.text and ENDPOINT in outcome.text
    assert [path for path, _ in service.paths] == ["/api/chat", "/api/generate", "/api/chat"]


# --- the tools' other refusals, and the search through the dispatch ------------------------


def test_a_directory_is_not_read_as_a_file_nor_a_file_listed_as_a_directory(tmp_path):
    """Mutation: drop either kind check -- red, the tool raises instead of
    answering, or reads what it was not asked to."""
    project = _project(tmp_path)

    assert ollama.call_tool(project, "read_file", {"path": "qmcp"}) == "refused: 'qmcp' is not a file"
    assert ollama.call_tool(project, "list_files", {"path": "README.md"}) == (
        "refused: 'README.md' is not a directory")


def test_search_is_reached_through_the_dispatch_and_skips_what_it_cannot_read(tmp_path):
    """The one tool no other test calls through `call_tool`. Mutation: pass the
    path as the text -- red; drop the decode guard -- red, a binary file
    raises."""
    project = _project(tmp_path)
    (project / "qmcp" / "blob.bin").write_bytes(b"\xff\xfe the server \x00")

    found = ollama.call_tool(project, "search", {"text": "the server", "path": "qmcp"})

    assert found == "qmcp/server.py:1: app = 'the server'"


def test_search_stops_at_its_cap_and_says_there_is_more(tmp_path):
    """Mutation: drop the cap -- red; a model's small window would fill with
    one search."""
    (tmp_path / "many.txt").write_text("hit\n" * (ollama.SEARCH_HITS + 5), encoding="utf-8")

    found = ollama.search(tmp_path, "hit")

    assert found.count("many.txt:") == ollama.SEARCH_HITS
    assert found.endswith("(more matches not shown)")


def test_arguments_that_are_not_json_are_read_as_none():
    assert ollama.tool_call_in({"tool_calls": [{"function": {
        "name": "read_file", "arguments": "{broken"}}]}) == ("read_file", {})


def test_an_answer_that_only_looks_like_json_is_kept_as_it_is():
    assert ollama.answer_text("{not json}") == "{not json}"
    assert ollama.answer_text("  plain  ") == "plain"


def test_localmodel_check_says_whether_the_model_is_served(monkeypatch):
    """Where a person is sent to find out, so it asks the service and not only
    the disk. Mutation: drop the `served` line -- red."""
    from click.testing import CliRunner

    from qmcp import cli
    from qmcp.localmodel import Check

    monkeypatch.setattr("qmcp.localmodel.look", lambda: Check())
    monkeypatch.setattr(ollama.Runtime, "ready", lambda self: None)
    served = CliRunner().invoke(cli.cli, ["localmodel", "check"])
    monkeypatch.setattr(ollama.Runtime, "ready", lambda self: "the service does not answer")
    missing = CliRunner().invoke(cli.cli, ["localmodel", "check"])

    assert served.exit_code == 0, served.output
    assert f"served:     {MODEL} at {ENDPOINT}" in served.output
    assert "served:     no; the service does not answer" in missing.output
