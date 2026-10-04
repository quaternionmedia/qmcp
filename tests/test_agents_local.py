"""The local model as a runtime: the chat loop, the read-only tools, and the bounds.

The model service is stood in for by an `httpx.MockTransport` that answers as
the service's chat route does, so each test decides what the model asks for and
sees exactly what was sent. Nothing here needs a model, a GPU or a network.
"""

from __future__ import annotations

import json

import httpx

from qmcp.integrations.agents import Brief, Turn
from qmcp.integrations.agents.adapters import ollama
from qmcp.localmodel import ENDPOINT, MODEL


def _call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


class _Service:
    """The chat route: answers from a script, keeps every request it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0) if self.replies else self.replies_exhausted()
        return httpx.Response(200, json={"message": {"role": "assistant", **reply}, "done": True})

    def replies_exhausted(self):
        raise AssertionError("the runtime asked the model more often than the script allows")

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


def test_the_model_reads_with_a_tool_and_then_answers(tmp_path):
    """Mutation: drop the tool result from the messages -- red, the second
    request carries nothing the model read; skip the loop and return the
    first reply -- red, the answer is empty."""
    service = _Service({"content": "", "tool_calls": [_call("read_file", path="README.md")]},
                       {"content": "qmcp is the local backend."})
    events: list[tuple[str, str]] = []

    outcome = ollama.Runtime(client=service.client()).run(
        _brief(_project(tmp_path)), on_event=lambda s, t: events.append((s, t)))

    assert outcome.text == "qmcp is the local backend." and outcome.exit_code == 0
    assert outcome.spent == 0
    assert outcome.detail == {"model": MODEL, "endpoint": ENDPOINT, "model_calls": 2,
                              "read": ["read_file(README.md)"]}
    tool_message = service.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_name"] == "read_file"
    assert "qmcp is the local backend." in tool_message["content"]
    assert ("output", "read_file(README.md)") in events
    assert events[0][0] == "started" and events[-1] == ("finished", "0")


def test_every_request_is_the_pinned_model_capped_and_offered_only_reading_tools(tmp_path):
    """Mutation: drop `num_predict` -- red; an uncapped request can hold the
    service long after its caller has gone."""
    service = _Service({"content": "done"})

    ollama.Runtime(client=service.client()).run(_brief(tmp_path))

    request = service.requests[0]
    assert request["model"] == MODEL and request["stream"] is False
    assert request["options"]["num_predict"] == ollama.MAX_TOKENS
    assert {t["function"]["name"] for t in request["tools"]} == {"list_files", "read_file", "search"}
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1] == {"role": "user", "content": _brief(tmp_path).prompt()}


def test_the_history_from_qmcps_record_reaches_the_model(tmp_path):
    """Continuity comes from qmcp, not the model."""
    service = _Service({"content": "done"})
    earlier = Turn(id="t-1", instruction="Find the README.", status="done",
                   outcome="README.md, at the root.", runtime="local")

    ollama.Runtime(client=service.client()).run(_brief(tmp_path, [earlier]))

    assert "README.md, at the root." in service.requests[0]["messages"][1]["content"]


def test_a_path_outside_the_clone_or_inside_git_is_refused_and_the_refusal_goes_back(tmp_path):
    """Mutation: drop the `parents` check in `inside` -- red, `../` reads
    outside the clone; drop the skipped-directory check -- red on `.git`."""
    (tmp_path / "work").mkdir()
    project = _project(tmp_path / "work")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    service = _Service({"content": "", "tool_calls": [_call("read_file", path="../secret.txt"),
                                                     _call("read_file", path=".git/config"),
                                                     _call("list_files", path=".venv")]},
                       {"content": "nothing readable"})

    ollama.Runtime(client=service.client()).run(_brief(project))

    replies = [m["content"] for m in service.requests[1]["messages"] if m["role"] == "tool"]
    assert replies[0] == "refused: '../secret.txt' is outside the project"
    assert replies[1] == "refused: '.git/config' is not read"
    assert replies[2] == "refused: '.venv' is not read"
    assert "outside" not in "".join(replies[1:]) and "[secret]" not in "".join(replies)


def test_listing_and_searching_skip_what_is_never_read(tmp_path):
    """Mutation: walk every directory in `search` -- red, the environment's
    copy is found."""
    project = _project(tmp_path)

    listed = ollama.list_files(project, ".")
    found = ollama.search(project, "LOCAL BACKEND")

    assert listed.splitlines() == ["README.md", "qmcp/"]
    assert found == "README.md:3: qmcp is the local backend."
    assert ollama.search(project, "nowhere") == "(no matches)"
    assert ollama.call_tool(project, "write_file", {"path": "x"}).startswith("refused:")


def test_a_long_file_is_cut_and_says_so(tmp_path):
    (tmp_path / "big.txt").write_text("x" * (ollama.READ_CHARS + 50), encoding="utf-8")

    text = ollama.read_file(tmp_path, "big.txt")

    assert text.endswith(f"(cut at {ollama.READ_CHARS} of {ollama.READ_CHARS + 50} characters)")


def test_arguments_given_as_a_json_string_are_read(tmp_path):
    """Some models send `arguments` as a string. Mutation: drop the string
    branch -- red, the path is lost and the tool refuses."""
    project = _project(tmp_path)
    service = _Service({"content": "", "tool_calls": [
                           {"function": {"name": "read_file", "arguments": '{"path": "README.md"}'}}]},
                       {"content": "done"})

    ollama.Runtime(client=service.client()).run(_brief(project))

    assert "qmcp is the local backend." in service.requests[1]["messages"][-1]["content"]


def test_a_model_that_never_stops_calling_tools_is_a_failed_run_at_the_bound(tmp_path):
    """Mutation: drop the step bound -- red, the stand-in runs out of replies."""
    project = _project(tmp_path)
    service = _Service(*[{"content": "", "tool_calls": [_call("list_files")]}] * 3)

    outcome = ollama.Runtime(client=service.client(), max_steps=3).run(_brief(project))

    assert outcome.exit_code == 1 and "did not finish within 3 calls" in outcome.text
    assert outcome.detail["model_calls"] == 3


def test_an_empty_answer_is_a_failed_run(tmp_path):
    outcome = ollama.Runtime(client=_Service({"content": "   "}).client()).run(_brief(tmp_path))

    assert outcome.exit_code == 1


def test_a_service_that_does_not_answer_is_a_failed_run_naming_the_endpoint(tmp_path):
    """Mutation: let the transport error out of `run` -- red; the act would
    record an exception rather than where to look."""
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    outcome = ollama.Runtime(client=httpx.Client(transport=httpx.MockTransport(refuse))).run(
        _brief(tmp_path))

    assert outcome.exit_code == 1 and outcome.spent == 0
    assert ENDPOINT in outcome.text and "qmcp localmodel check" in outcome.text
