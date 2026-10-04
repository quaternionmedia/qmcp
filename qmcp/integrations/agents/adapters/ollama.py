"""The local model, as an agent runtime: `--runtime local`.

    uv run qmcp instructions act <id> --runtime local --budget 1

**QMCP IS THE LOCAL MODEL BACKEND.** `qmcp.localmodel` stands the model up and
pins it; this adapter is how an instruction reaches it. It talks to the
service `qmcp.localmodel.ENDPOINT` names, with the model `qmcp.localmodel.MODEL`
pins, over the service's own chat route -- no command line, no account, and
nothing paid, so `spent` is zero and that zero is a count: no call left this
machine. The product is named here and in `qmcp.localmodel`, and nowhere else.

**IT READS THE CLONE AND CHANGES NOTHING.** The model is offered three tools --
list a directory, read a file, search for text -- each confined to the brief's
directory: a path that resolves outside it is refused, and so is anything under
`.git` or an environment directory. There is no tool that writes, and so no way
for the model to change the clone. What it read is kept in the outcome's
`detail`, so the record says what the answer was based on.

**THE TOOLS ARE A PROTOCOL IN THE PROMPT, NOT THE SERVICE'S TOOL FIELD.** Asked
with the service's own tool definitions, the pinned model on the service this
was built against put its call in the reply's text rather than in the
structured field, and most often skipped the tools and answered from nothing.
So `SYSTEM` states the three tools and the one shape a call takes -- a single
JSON object -- and `tool_call_in` reads that shape from the text, or from the
structured field where a service fills it. It works with any model the service
serves. A model that answers before it has read anything is sent back once to
read: an answer grounded in no file is the failure a small model is likeliest
to commit, and the cheapest one to refuse.

**THE BRIEF IS THE MEMORY.** Each run starts with nothing but `Brief.prompt()`:
the place, the project's earlier instructions and outcomes from qmcp's record,
and the instruction. That is the same prompt any other runtime is given, which
is the point -- continuity comes from qmcp, not the model.

**EVERY CALL IS BOUNDED.** At most `MAX_STEPS` model calls a run, each capped
at `MAX_TOKENS` tokens and `TIMEOUT` seconds. A service that does not answer is
a failed run with the endpoint named, never a hang: the service finishes a
request it has started even after its caller gives up, so an uncapped request
can occupy it long after anyone is waiting.

WHAT THIS CANNOT DO. Know that the answer is right. A seven-billion-parameter
model reading a few files is a quick reader, not a reviewer; the person who
said approve hears what it found and judges it.
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
    """The local model, reading the clone with tools that cannot write."""

    name = NAME

    def __init__(self, endpoint: str = ENDPOINT, model: str = MODEL,
                 client: httpx.Client | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 max_steps: int = MAX_STEPS) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.client = client
        self.clock = clock
        self.max_steps = max_steps

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
                    f" ({type(exc).__name__}); `uv run qmcp localmodel check` says whether it"
                    " is installed and served")
        finally:
            if self.client is None:
                client.close()
        if self.model not in served:
            return (f"the service at {self.endpoint} does not serve {self.model};"
                    " `uv run qmcp localmodel plan` gives the commands that pull it")
        return None

    def _chat(self, client: httpx.Client, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = client.post(f"{self.endpoint}/api/chat", json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": 0.2, "num_predict": MAX_TOKENS}}, timeout=TIMEOUT)
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
                read.append(f"{name}({', '.join(f'{v}' for v in arguments.values())})")
                if on_event:
                    on_event("output", read[-1])
                messages.append({"role": "user", "content": f"Result of {read[-1]}:\n"
                                                            f"{call_tool(brief.cwd, name, arguments)}"})
            else:
                text = (f"The local model did not finish within {self.max_steps} calls;"
                        " nothing it found is reported as an answer.")
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
                                    "model_calls": calls, "read": read, "nudged": nudged})
