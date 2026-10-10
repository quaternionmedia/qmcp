"""The local model, as an agent runtime: `--runtime local`.

    uv run qmcp instructions act <id> --runtime local --budget 1

It talks to the service at `qmcp.localmodel.ENDPOINT` with the model
`qmcp.localmodel.MODEL` pins, spends nothing, and reads the clone through three
tools that cannot write and cannot leave the brief's directory. Each run makes
at most `MAX_STEPS` calls, each capped at `MAX_TOKENS` tokens and `TIMEOUT`
seconds, and a timed-out call is retried once against a fresh load. The
product is named here and in `qmcp.localmodel`, and nowhere else.

`docs/integrations/voice.md`, *The local runtime*, explains the tool protocol, the bounds and the
stall recovery.

What this cannot do: know that the answer is right; the person who said
approve hears what it found and judges it.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from qmcp.integrations.agents import AgentOutcome, Brief, OnEvent
from qmcp.localmodel import ENDPOINT, MODEL

NAME = "local"
PRODUCT = "Ollama"

MAX_STEPS = 8
MAX_TOKENS = 512
TIMEOUT = 90.0

# What a tool hands back, at most. Enough of a file for a model with a
# four-thousand-token window to read it and still answer.
READ_CHARS = 6_000
LIST_ENTRIES = 200
SEARCH_HITS = 40

# Never listed, read or searched: version control, environments, and the
# dependency trees that would drown a search.
SKIPPED = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
                     ".pytest_cache", ".ruff_cache", "dist", "build"})

SYSTEM = """You answer questions about a software project by reading its files. You cannot \
change anything.

To read, reply with exactly one JSON object and nothing else. The three shapes are:
  {"tool": "list_files", "path": "."}
  {"tool": "read_file", "path": "README.md"}
  {"tool": "search", "text": "words to find", "path": "."}
Paths are relative to the project's directory. What the tool returns comes back in the \
next message.

Read before you answer. When you have read enough, answer in plain sentences, not JSON: \
start with the answer, keep it short, and say only what the files you read support."""

AGAIN = ("You have already read that, and it has not changed. Answer now from what you"
         " have read, in plain sentences.")

FINAL = ("That was the last read this run allows. Answer the instruction now from what"
         " you have read, in plain sentences, without asking for a tool.")

NUDGE = ("You have not read anything yet. Reply with one JSON object that reads the project"
         " -- list_files, read_file or search -- before you answer.")

# The same three tools as the service's tool field would define them. Not sent:
# kept so `tool_call_in` and the tests name the tools in one place.
TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List a directory of the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "a directory, relative to the project; '.' for its root"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file of the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "a file, relative to the project"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Find lines containing some text, across the project's files.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the text to find, matched without regard to case"},
            "path": {"type": "string", "description": "a directory to search, relative to the project"}},
            "required": ["text"]}}},
]


class Refused(Exception):
    """A tool call the clone does not allow; its message goes back to the model."""


def inside(root: Path, relative: str | None) -> Path:
    """`relative` resolved under `root`, or `Refused` if it leaves it or names a
    directory that is never read."""
    base = root.resolve()
    # A leading slash is a model meaning "from the top of the project", not the
    # top of the disk, which on Windows would land outside the clone.
    target = (base / (relative or ".").lstrip("/\\")).resolve()
    if target != base and base not in target.parents:
        raise Refused(f"{relative!r} is outside the project")
    if any(part in SKIPPED for part in target.relative_to(base).parts):
        raise Refused(f"{relative!r} is not read")
    return target


def list_files(root: Path, path: str | None = None) -> str:
    target = inside(root, path)
    if not target.is_dir():
        raise Refused(f"{path!r} is not a directory")
    names = sorted(entry.name + ("/" if entry.is_dir() else "")
                   for entry in target.iterdir() if entry.name not in SKIPPED)
    shown = names[:LIST_ENTRIES]
    more = f"\n... and {len(names) - len(shown)} more" if len(names) > len(shown) else ""
    return "\n".join(shown) + more or "(empty)"


def read_file(root: Path, path: str) -> str:
    target = inside(root, path)
    if not target.is_file():
        raise Refused(f"{path!r} is not a file")
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > READ_CHARS:
        return text[:READ_CHARS] + f"\n... (cut at {READ_CHARS} of {len(text)} characters)"
    return text


def search(root: Path, text: str, path: str | None = None) -> str:
    base = inside(root, path)
    needle = re.compile(re.escape(text), re.IGNORECASE)
    hits: list[str] = []
    for directory, subdirectories, files in os.walk(base):
        subdirectories[:] = sorted(d for d in subdirectories if d not in SKIPPED)
        for name in sorted(files):
            file = Path(directory) / name
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(lines, start=1):
                if needle.search(line):
                    hits.append(f"{file.relative_to(root.resolve()).as_posix()}:{number}:"
                                f" {line.strip()[:160]}")
                    if len(hits) >= SEARCH_HITS:
                        return "\n".join(hits) + "\n... (more matches not shown)"
    return "\n".join(hits) or "(no matches)"


TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_TAGS = re.compile(r"</?tool_call>")


def tool_call_in(message: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The tool call a reply makes, as `(name, arguments)`, or None for an answer.

    Read from the structured field where the service filled it, and otherwise
    from the text: one JSON object, alone, optionally fenced, carrying `tool`
    (the protocol `SYSTEM` states) or `name` and `arguments` (the shape the
    model was trained to emit). Prose that merely contains braces is an answer.
    """
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        return str(function.get("name", "")), arguments if isinstance(arguments, dict) else {}
    text = _TAGS.sub("", _FENCE.sub("", (message.get("content") or "").strip())).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict):
        return None
    name = document.get("tool") or document.get("name")
    if not isinstance(name, str):
        return None
    arguments = document.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {k: v for k, v in document.items() if k not in ("tool", "name", "arguments")}
    return name, arguments


def answer_text(content: str) -> str:
    """An answer as prose. A model told to read in JSON sometimes answers in it
    too -- `{"answer": "..."}`, fenced -- and that would be read aloud as
    braces; its string values are the answer, joined as sentences."""
    text = _TAGS.sub("", _FENCE.sub("", content.strip())).strip()
    if not (text.startswith("{") and text.endswith("}")):
        return content.strip()
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return content.strip()
    if not isinstance(document, dict):
        return content.strip()
    parts = [" ".join(str(v).split()) for v in document.values() if isinstance(v, (str, int, float))]
    return " ".join(p if p[-1:] in ".!?" else p + "." for p in parts if p) or content.strip()


def call_tool(root: Path, name: str, arguments: dict[str, Any]) -> str:
    """One tool call, answered as text; a refusal is an answer, not an error."""
    try:
        if name == "list_files":
            return list_files(root, arguments.get("path"))
        if name == "read_file":
            return read_file(root, str(arguments.get("path", "")))
        if name == "search":
            return search(root, str(arguments.get("text", "")), arguments.get("path"))
        raise Refused(f"there is no tool named {name!r}")
    except Refused as refusal:
        return f"refused: {refusal}"


class Runtime:
    """The local model, reading the clone with tools that cannot write.

    `max_steps` bounds tool-loop calls; a run that reaches the bound may make
    one final request to answer from what it read. `recover_timeouts` is off
    only for callers that need a strict request budget.
    """

    name = NAME

    def __init__(self, endpoint: str = ENDPOINT, model: str = MODEL,
                 client: httpx.Client | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 max_steps: int = MAX_STEPS, recover_timeouts: bool = True,
                 max_tokens: int = MAX_TOKENS, timeout: float = TIMEOUT) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.client = client
        self.clock = clock
        self.max_steps = max_steps
        self.recover_timeouts = recover_timeouts
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Stalled calls recovered in the current run; reported in its detail.
        self.recovered = 0

    def ready(self) -> str | None:
        """None when the service answers and serves the pinned model; otherwise
        what is missing, in words a person can act on. Asked before a run that
        would otherwise discover it one timeout later."""
        client = self.client or httpx.Client()
        try:
            response = client.get(f"{self.endpoint}/api/tags", timeout=5.0)
            response.raise_for_status()
            served = {m.get("name") for m in response.json().get("models") or []}
        except (httpx.HTTPError, ValueError) as exc:
            return (f"the local model service does not answer at {self.endpoint}"
                    f" ({type(exc).__name__}); `uv run qmcp localmodel plan` gives the"
                    " commands that install and start it")
        finally:
            if self.client is None:
                client.close()
        if self.model not in served:
            return (f"the service at {self.endpoint} does not serve {self.model};"
                    " `uv run qmcp localmodel plan` gives the commands that pull it")
        return None

    def _chat(self, client: httpx.Client, messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "num_predict": self.max_tokens}}
        try:
            response = client.post(f"{self.endpoint}/api/chat", json=body, timeout=self.timeout)
        except httpx.ReadTimeout:
            if not self.recover_timeouts:
                raise
            # The runner stalled: unload it, as the service's `stop` does, and
            # ask once more of a fresh load. A second timeout propagates.
            self.recovered += 1
            client.post(f"{self.endpoint}/api/generate",
                        json={"model": self.model, "keep_alive": 0}, timeout=30.0)
            response = client.post(f"{self.endpoint}/api/chat", json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("message") or {}

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        if on_event:
            on_event("started", f"{self.model} at {self.endpoint}, with {len(brief.history)}"
                                " earlier instruction(s) from qmcp's record")
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": brief.prompt()}]
        read: list[str] = []
        calls = 0
        nudged = False
        self.recovered = 0
        asked: set[str] = set()
        repeats = 0
        started = self.clock()
        client = self.client or httpx.Client()
        try:
            text, code = "", 1
            while calls < self.max_steps:
                calls += 1
                message = self._chat(client, messages)
                content = message.get("content") or ""
                messages.append({"role": "assistant", "content": content})
                call = tool_call_in(message)
                if call is None:
                    if not read and not nudged:
                        nudged = True
                        messages.append({"role": "user", "content": NUDGE})
                        continue
                    text = answer_text(content)
                    code = 0 if text else 1
                    break
                name, arguments = call
                key = json.dumps([name, arguments], sort_keys=True)
                if key in asked:
                    repeats += 1
                    messages.append({"role": "user", "content": AGAIN})
                    continue
                asked.add(key)
                read.append(f"{name}({', '.join(f'{v}' for v in arguments.values())})")
                if on_event:
                    on_event("output", read[-1])
                messages.append({"role": "user", "content": f"Result of {read[-1]}:\n"
                                                            f"{call_tool(brief.cwd, name, arguments)}"})
            else:
                # The bound is spent, and what was read is still worth an
                # answer: one more call, told that no read follows.
                text = (f"The local model did not finish within {self.max_steps} calls,"
                        " nor when told to answer from what it had read; nothing it found"
                        " is reported as an answer.")
                if read:
                    calls += 1
                    messages.append({"role": "user", "content": FINAL})
                    message = self._chat(client, messages)
                    content = message.get("content") or ""
                    if tool_call_in(message) is None and answer_text(content):
                        text, code = answer_text(content), 0
        except httpx.ReadTimeout as exc:
            # Only the retry, or the unload before it, lets a read timeout out
            # of `_chat`: the service answers and is not generating, which
            # `localmodel check` would report as served.
            if self.recover_timeouts:
                text = (f"The local model at {self.endpoint} stopped replying"
                        f" ({type(exc).__name__}), and again after it was reloaded."
                        " The service answers but is not generating; `uv run qmcp"
                        " localmodel plan` prints the commands that restart it.")
            else:
                text = (f"The local model at {self.endpoint} stopped replying"
                        f" ({type(exc).__name__}); this run did not retry the request.")
            code = 1
        except httpx.HTTPError as exc:
            text, code = (f"The local model did not answer at {self.endpoint}"
                          f" ({type(exc).__name__}). `uv run qmcp localmodel check` says"
                          " whether it is installed and served."), 1
        finally:
            if self.client is None:
                client.close()
        elapsed = self.clock() - started
        if on_event:
            on_event("finished", str(code))
        return AgentOutcome(text=text, exit_code=code, elapsed_seconds=elapsed, spent=0,
                            detail={"model": self.model, "endpoint": self.endpoint,
                                    "model_calls": calls, "read": read, "nudged": nudged,
                                    "repeats": repeats, "recovered": self.recovered})
