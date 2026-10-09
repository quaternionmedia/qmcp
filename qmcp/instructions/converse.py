"""One standing conversation, so the loop is spoken from end to end.

    uv run qmcp serve --converse --runtime local    # the server, and the conversation beside it
    uv run qmcp converse --runtime local            # the conversation, against a running server

**ONCE THE SERVERS ARE UP, NOTHING IS TYPED.** With the speech engine running
and `qmcp serve --converse`, this says it is ready and asks what should be
done. An instruction is spoken, read back and recorded -- tacitly when the
engine heard it confidently, otherwise on `agree`
(`qmcp.instructions.dialog`); consent is asked aloud and given with `approve`
(`qmcp.instructions.act`); the runtime carries it out in the project's clone;
the outcome is said back (`qmcp.instructions.spoken`); and it asks whether
there is anything else. No id is read off a screen, no flag is typed, no
button is pressed.

**WHAT IS WAITING IS ASKED FIRST.** Before each instruction is taken, questions
an agent has put on the human queue are asked aloud, oldest first, through the
same loop `qmcp human voice` runs -- closed choices by their options, open
questions read back. A question nobody answers stays pending for anyone, and
is not asked again in this conversation.

**NOTHING RUNS WITHOUT A SPOKEN APPROVE.** The instruction is agreed to once it
is read back -- with `agree`, or, heard confidently, by a moment's uninterrupted
silence -- and consent is given with `approve`, after it says what will run
where. Every
runtime is asked, the local model included. Continuity comes from qmcp, not
the model: each run is handed the project's earlier instructions and what
they found, from this server's record.

**THE CLONE IS FOUND WITHOUT A PATH.** A project acted on before runs in the
clone its last act ran in, from the record. A project acted on for the first
time runs in a clone named for it beside this checkout -- the workspace keeps
its repositories as siblings -- or in `--clones`. With neither, the act refuses
before it asks, and the refusal is said.

**SILENCE IS WAITED THROUGH, AND IT ENDS WHEN TOLD.** Nobody speaking is not an
error: the microphone stays open, take after take, until somebody does. "No"
or "that's all" after "Anything else?" goes back to waiting; "yes" asks for the
instruction; "stop listening" or "goodbye" ends the conversation. A turn that
fails is said to have failed and the conversation goes on.

WHAT THIS CANNOT DO. Tell speech meant for it from speech in the room. Every
utterance is read back before it is recorded, and nothing runs without
`approve`; `wake` requires a word to begin an instruction, for a room where
people talk. Nor can it interrupt itself: the microphone is closed while it
speaks, so a person waits for it to finish.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from qmcp.instructions.dialog import LISTEN_DURATION, PAUSE_MS, PROMPT, VOCABULARY, InstructionDialog
from qmcp.integrations.voice import vocabulary
from qmcp.integrations.voice.adapter import REPEAT, ask_over, speakably

READY = "Ready. What should be done?"
ANYTHING_ELSE = "Anything else?"
WAITING = "Listening. Say an instruction whenever you are ready."
OKAY = "Listening."
STOPPING = "Stopping."
QUESTION_WAITING = "A question is waiting."
FAILED = "That failed. Nothing ran."
NOT_RECORDED = "Nothing recorded."
NOTHING_TO_TRY = "Nothing to try again."
NO_SUCH_PROJECT = "No such project."
NOTHING_HEARD = "Nothing heard yet."
NOTHING_RAN = "Nothing has run yet."
HEARING_YOU = "I can hear you."
HELP_SAID = ("Say an instruction. Or say: try again, same in a project, what did you hear, "
             "how did that go, what is waiting, or stop. To add a phrase of your own, say "
             "add vocabulary phrase, the words, to, and the command. For topologies, say "
             "list topologies in a project, show topology and its name in a project, create "
             "a topology or a component, compose topologies, or run a topology in a project "
             "about a task; every change and every run asks for approval.")

# The most entries a hint carries with the projects' terms in it: the engine
# keeps no more than this many (joe's `HINT_WORDS`).
HINT_ENTRIES = 12

# What ends the conversation, what goes back to waiting, and what asks for an
# instruction, matched on the whole utterance with punctuation and case gone
# (`vocabulary.toml`, `conversation.*`).
STOP = vocabulary.phrases("conversation.stop")
DONE = vocabulary.phrases("conversation.done")
MORE = vocabulary.phrases("conversation.more")
# What the conversation does itself, without a model, between instructions:
# the last instruction again, for its project or another, and what it can say
# of how it stands (`vocabulary.toml`, `iteration.*` and `diagnostic.*`).
TRY_AGAIN = vocabulary.phrases("iteration.try_again")
SAME_IN = vocabulary.phrases("iteration.same_in")
NEVER_MIND = vocabulary.phrases("iteration.never_mind")
WHAT_HEARD = vocabulary.phrases("diagnostic.what_heard")
HOW_DID_IT_GO = vocabulary.phrases("diagnostic.how_did_it_go")
WHATS_WAITING = vocabulary.phrases("diagnostic.whats_waiting")
TEST_VOICE = vocabulary.phrases("diagnostic.test_voice")
WHICH_PROJECTS = vocabulary.phrases("diagnostic.which_projects")
HELP = vocabulary.phrases("diagnostic.help")


def _said(declared: tuple[str, ...], key: str) -> tuple[str, ...]:
    """A command's declared phrases and the ones a person has added, read as
    they stand so an approved edit is heard on the next utterance."""
    return (*declared, *vocabulary.overrides(key))


# "add vocabulary phrase <words> to <command>", and its reverse. The command is
# the last "to" or "from" and one of `vocabulary.TARGETS`, so a phrase may
# itself contain "to".
_COMMANDS = "|".join(sorted(map(re.escape, vocabulary.TARGETS), key=len, reverse=True))
ADD_PHRASE = re.compile(rf"add vocabulary phrase (.+) to ({_COMMANDS})")
REMOVE_PHRASE = re.compile(rf"remove vocabulary phrase (.+) from ({_COMMANDS})")
UNDO_PHRASE = vocabulary.phrases("vocabulary.undo")
EDIT_OPTIONS = ("approve", "hold")
# The answers "Anything else?" offers as keys and buttons, and hints to the engine.
MORE_OPTIONS = ("yes", "no")

# Request ids this conversation puts on the queue itself, asked inside the act
# that created them rather than as a waiting question.
OWN = ("instruction-",)


def plain(text: str) -> str:
    """An utterance as words alone: lower case, no punctuation, single spaces,
    and a word said over and over read once ("No. No. No." is "no")."""
    words = re.sub(r"[^\w\s]", "", (text or "").lower().replace("'", "")).split()
    if len(words) > 1 and len(set(words)) == 1:
        words = words[:1]
    return " ".join(words)


def sibling_clones() -> Path:
    """Where the workspace keeps repositories: beside this checkout."""
    return Path(__file__).resolve().parents[3]


@dataclass
class Turn:
    """One instruction taken in the conversation, and how it ended."""

    instruction_id: str | None
    text: str
    status: str
    summary: str
    carried: tuple[str, ...] = ()


@dataclass
class Ended:
    """What a conversation did, for a caller that ran it to an end."""

    turns: list[Turn] = field(default_factory=list)
    answered: list[str] = field(default_factory=list)
    """Waiting questions answered by voice, by request id."""
    reason: str = ""


class Conversation:
    """Takes instructions by voice, acts on each with consent, and says what happened."""

    def __init__(self, stt: Any, tts: Any, client: Any, runtime: Any, names: Iterable[str], *,
                 rows: Any = None, clones: Path | None = None, wake: str | None = None,
                 listen_duration: float = LISTEN_DURATION, pause_ms: int = PAUSE_MS,
                 answer_duration: float = 5.0, max_retries: int = 2,
                 idle_limit: int | None = None, poll_interval: float = 1.0,
                 echo: Callable[[str], None] | None = None,
                 tacit_above: float | None = None, hint_terms: bool = False) -> None:
        self.stt = stt
        self.tts = speakably(tts)
        self.client = client
        self.hint_terms = hint_terms
        self.runtime = runtime
        self.names = tuple(names)
        self.rows = rows
        self.clones = Path(clones) if clones is not None else sibling_clones()
        self.wake = plain(wake) if wake else None
        self.listen_duration = listen_duration
        self.pause_ms = pause_ms
        self.answer_duration = answer_duration
        self.max_retries = max_retries
        # Consecutive silent takes before the conversation ends on its own. None
        # -- the default -- waits for ever, which is what a standing conversation
        # does; a check sets it so a script that runs out of words ends.
        self.idle_limit = idle_limit
        self.poll_interval = poll_interval
        self.echo = echo or (lambda line: None)
        # The engine confidence at or above which an instruction's read-back
        # asks nothing and silence agrees (`InstructionDialog`); None always asks.
        self.tacit_above = tacit_above
        self._confidence: float | None = None
        self._asked: set[str] = set()
        # A phrase edit waiting for "approve" or "hold", and how often the
        # answer was neither.
        self._pending_edit: vocabulary.PhraseEdit | None = None
        self._edit_retries = 0
        # The question the next take answers, and the options it offered, so
        # "repeat" can say it again and the engine can be hinted.
        self._question: tuple[str, tuple[str, ...]] = (READY, ())
        # The last utterance taken as an instruction or an iteration, for
        # "what did you hear".
        self._last_heard = ""
        self._vocabulary: list[str] = []

    # --- speaking ---------------------------------------------------------------

    def say(self, text: str) -> None:
        from qmcp.instructions.spoken import say

        self.echo(f"said: {text}")
        say(text, self.tts, self.stt)

    def vocabulary(self) -> list[str]:
        """The project names an instruction is likely to carry: those of the
        most recent instructions first, from the record, then the roster's,
        without repeats and at most `VOCABULARY`. A record that cannot be read
        leaves the roster's. With `hint_terms`, the core projects' terms follow
        the names, in the same order, within the engine's bound on a hint."""
        recent: list[str] = []
        try:
            for row in self.client.list_instructions(limit=20):
                if row.get("project"):
                    recent.append(row["project"])
        except Exception:  # noqa: BLE001 -- a hint is no reason to stop listening
            pass
        names = list(dict.fromkeys([*recent, *self.names]))[:VOCABULARY]
        if not self.hint_terms:
            return names
        return list(dict.fromkeys([*names, *vocabulary.terms(names)]))[:HINT_ENTRIES]

    def ask(self, text: str, options: tuple[str, ...] = ()) -> None:
        """Say a question and leave the turn to the person: announced as
        `speaking`, with its options, and not closed with `idle`, so the
        engine knows the next take answers it -- joe cues the person to
        speak -- and the page can offer the options."""
        from qmcp.instructions.spoken import announce

        self._question = (text, tuple(options))
        self.echo(f"said: {text}")
        announce(self.stt, "speaking", text, options=list(options) or None)
        # Answerable before it ends, watched for the take the loop makes next.
        ask_over(self.stt, self.tts, text, duration=self.listen_duration, pause_ms=self.pause_ms,
                 hint=self._hint())

    def _hint(self) -> list[str]:
        """The question's own options, then the names an instruction is
        likely to carry: a take after it may be either."""
        return list(dict.fromkeys([*self._question[1], *self._vocabulary]))[:VOCABULARY + 2]

    def _announce(self, state: str, text: str) -> None:
        from qmcp.instructions.spoken import announce

        announce(self.stt, state, text)

    # --- the conversation -------------------------------------------------------

    def run(self) -> Ended:
        """Talk until told to stop, or until `idle_limit` silent takes in a row."""
        from qmcp.integrations.voice.adapter import listen_for

        ended = Ended()
        self._vocabulary = self.vocabulary()
        self.ask(READY)
        idle = 0
        while True:
            self._ask_waiting(ended)
            heard, recording = listen_for(self.stt, self.listen_duration, pause_ms=self.pause_ms,
                                          hint=self._hint())
            # An answer with no recording behind it came from a key or a button
            # on the engine's page: it was meant for this conversation.
            keyed = not recording
            self._confidence = getattr(self.stt, "last_confidence", None)
            words = plain(heard)
            if not words:
                idle += 1
                if idle == 1:
                    self._announce("idle", WAITING)
                if self.idle_limit is not None and idle >= self.idle_limit:
                    ended.reason = f"{idle} silent takes in a row"
                    return ended
                continue
            idle = 0
            self.echo(f"heard: {heard.strip()}")
            if self.wake and not keyed:
                if not words.startswith(self.wake):
                    continue
                heard = _after(heard, self.wake)
                words = plain(heard)
                if not words:
                    self.say(PROMPT)
                    continue
            if self._pending_edit is not None:
                self._confirm_edit(heard)
                continue
            if words in _said(STOP, "conversation.stop"):
                self.say(STOPPING)
                ended.reason = "told to stop"
                return ended
            if words in _said(REPEAT, "conversation.repeat"):
                self.ask(*self._question)
                continue
            if words in _said(DONE, "conversation.done"):
                self.ask(OKAY)
                continue
            if words in _said(MORE, "conversation.more"):
                self.ask(PROMPT)
                continue
            if words in _said(NEVER_MIND, "iteration.never_mind"):
                self.ask(OKAY)
                continue
            if self._diagnose(words, ended):
                continue
            if self._browse(heard):
                continue
            if self._edit_vocabulary(words):
                continue
            self._last_heard = heard.strip()
            if self._iterate(words, ended):
                continue
            try:
                ended.turns.append(self._take(heard))
                self._vocabulary = self.vocabulary()
            except Exception as exc:  # noqa: BLE001 -- a standing conversation outlives a turn
                self.echo(f"turn failed: {type(exc).__name__}: {exc}")
                self.ask(FAILED)
                continue
            self.ask(ANYTHING_ELSE, MORE_OPTIONS)

    def _take(self, heard: str) -> Turn:
        """One instruction: read back and recorded, then acted on with consent, then said."""
        from qmcp.instructions.act import RULE_SIBLING, act
        from qmcp.instructions.spoken import starting, summarise
        from qmcp.integrations.voice.adapter import UnclearResponse
        from qmcp.spend import Budget

        dialog = InstructionDialog(stt=self.stt, tts=self.tts, client=self.client,
                                   names=self.names, max_retries=self.max_retries,
                                   listen_duration=self.listen_duration,
                                   pause_ms=self.pause_ms, answer_duration=self.answer_duration,
                                   tacit_above=self.tacit_above, vocabulary=self._vocabulary)
        try:
            row = dialog.run_once(heard=heard, confidence=self._confidence)
        except UnclearResponse:
            self.say(NOT_RECORDED)
            return Turn(None, heard.strip(), "not recorded", NOT_RECORDED)
        return self._act_on(row)

    def _act_on(self, row: dict[str, Any]) -> Turn:
        """A recorded instruction, acted on with consent, and its outcome said."""
        from qmcp.instructions.act import RULE_SIBLING, act
        from qmcp.instructions.spoken import starting, summarise
        from qmcp.spend import Budget

        from qmcp.integrations.agents.check import CheckRuntime

        project = row.get("project")
        self.echo(f"recorded: {row['id']} for {project or 'no project'}")
        # An instruction naming one of the project's declared checks runs that
        # command, behind the same consent, instead of the model. One that is a
        # topology command acts on saved designs, and one that starts with an
        # executable topology's phrase runs that topology.
        text = row.get("text", "")
        check = vocabulary.match_check(plain(text), project)
        design_command = vocabulary.match_topology_command(text) if check is None else None
        topology_attempt = plain(text).startswith((
            "create topology", "create component", "edit component",
            "add component", "remove component", "compose topology", "run topology",
        ))
        topology = (vocabulary.match_topology(text)
                    if check is None and design_command is None else None)
        if check:
            runtime = CheckRuntime(check)
        elif design_command:
            if project is None or project.casefold() != design_command.project.casefold():
                message = (f"The topology command names {design_command.project}, but the "
                           f"instruction was recorded for {project or 'no project'}. Nothing ran.")
                self.say(message)
                recorded = self.client.get_instruction(row["id"])
                return Turn(row["id"], text, recorded["status"], message)
            from qmcp.integrations.agents.topology_design import TopologyDesignRuntime

            try:
                runtime = TopologyDesignRuntime(self.client, design_command)
            except (httpx.HTTPError, ValueError, OSError, RuntimeError) as exc:
                message = (f"I cannot prepare topology {design_command.name}: "
                           f"{type(exc).__name__}: {exc}. Nothing ran.")
                self.echo(f"topology preparation failed: {message}")
                self.say(message)
                recorded = self.client.get_instruction(row["id"])
                return Turn(row["id"], text, recorded["status"], message)
            self.echo(f"topology command: {design_command.action} "
                      f"{design_command.name} in {design_command.project}")
        elif topology_attempt:
            message = ("I could not parse that topology command. Say help for the exact "
                       "create, edit, compose and run forms. Nothing ran.")
            self.say(message)
            recorded = self.client.get_instruction(row["id"])
            return Turn(row["id"], text, recorded["status"], message)
        elif topology and topology.name == "crosscheck":
            from qmcp.integrations.agents.crosscheck import CrossCheckRuntime

            runtime = CrossCheckRuntime(topology.prompt)
            self.echo(f"topology: {topology.name}: {topology.prompt}")
        else:
            runtime = self.runtime
        if check:
            self.echo(f"check: {check.project}.{check.name}: {check.command}")

        def event(state: str, text: str) -> None:
            if state == "acting":
                self.say(starting(project))
            elif state == "output":
                self.echo(f"read: {text}")

        done = act(row["id"], runtime, Budget(authorised=1), client=self.client,
                   rows=self.rows, cwd=self._clone_for(project, row["id"]),
                   cwd_rule=RULE_SIBLING, stt=self.stt, tts=self.tts, on_event=event,
                   poll_interval=self.poll_interval)
        recorded = self.client.get_instruction(row["id"])
        summary = summarise(recorded, why=done.why)
        if done.why and not done.ran:
            self.echo(f"why: {done.why}")
        self.say(summary)
        return Turn(row["id"], row["text"], recorded["status"], summary, done.carried)

    def _iterate(self, words: str, ended: Ended) -> bool:
        """"try again", and "same in <project>": the last instruction taken,
        recorded anew -- for the project named, or its own -- and acted on
        behind a new consent. False when the words are neither."""
        from qmcp.instructions import resolve

        named = None
        if words not in _said(TRY_AGAIN, "iteration.try_again"):
            named = same_in(words)
            if named is None:
                return False
        last = next((t for t in reversed(ended.turns) if t.instruction_id), None)
        if last is None:
            self.ask(NOTHING_TO_TRY)
            return True
        if named is None:
            project = (self.client.get_instruction(last.instruction_id) or {}).get("project")
        else:
            project = resolve(named, self.names).project
            if project is None:
                self.ask(NO_SUCH_PROJECT)
                return True
        try:
            row = self.client.create_instruction(last.text, source="voice", project=project,
                                                 heard=[self._last_heard])
            self.echo(f"recorded again: {row['id']} for {row.get('project') or 'no project'}")
            ended.turns.append(self._act_on(row))
        except Exception as exc:  # noqa: BLE001 -- a standing conversation outlives a turn
            self.echo(f"turn failed: {type(exc).__name__}: {exc}")
            self.ask(FAILED)
            return True
        self.ask(ANYTHING_ELSE, MORE_OPTIONS)
        return True

    def _diagnose(self, words: str, ended: Ended) -> bool:
        """What the conversation can say of how it stands, said and the turn
        left open; False when the words ask none of it."""
        from qmcp.integrations.voice.adapter import counted

        if words in _said(WHAT_HEARD, "diagnostic.what_heard"):
            said = f"I heard: {self._last_heard.rstrip('.!?')}." if self._last_heard else NOTHING_HEARD
        elif words in _said(HOW_DID_IT_GO, "diagnostic.how_did_it_go"):
            said = ended.turns[-1].summary if ended.turns else NOTHING_RAN
        elif words in _said(WHATS_WAITING, "diagnostic.whats_waiting"):
            waiting = self._waiting()
            said = ("Nothing is waiting." if waiting == 0 else
                    f"{counted(waiting, 'question').capitalize()} {'is' if waiting == 1 else 'are'} waiting.")
        elif words in _said(TEST_VOICE, "diagnostic.test_voice"):
            said = HEARING_YOU
        elif words in _said(WHICH_PROJECTS, "diagnostic.which_projects"):
            said = f"I know {_listed(list(self.names))}." if self.names else "I know no projects."
        elif words in _said(HELP, "diagnostic.help"):
            said = HELP_SAID
        else:
            return False
        self.ask(said)
        return True

    def _browse(self, heard: str) -> bool:
        """Say what saved designs and components a project holds. Read-only:
        no consent, no model, no record. False when the words ask none of it."""
        query = vocabulary.match_topology_query(heard)
        if query is None:
            return False
        try:
            said = self._describe(query)
        except (httpx.HTTPError, ValueError, OSError) as exc:
            self.echo(f"browse failed: {type(exc).__name__}: {exc}")
            said = f"I could not read {query.project}'s designs. Nothing changed."
        self.ask(said)
        return True

    def _describe(self, query: vocabulary.TopologyCommand) -> str:
        project = query.project
        if query.action == "list_topologies":
            items = self.client.list_topologies(project=project)["topologies"]
            names = [f"{i['name']}, a {i['topology_type']}" for i in items]
            return _named_list(names, "topology", "topologies", project)
        if query.action == "list_components":
            items = self.client.list_topology_components(project=project)["components"]
            return _named_list([i["name"] for i in items], "component", "components", project)
        if query.action == "show_topology":
            try:
                row = self.client.get_topology(query.name, project=project)
            except Exception as exc:  # noqa: BLE001 -- only a 404 is an answer
                if _status(exc) != 404:
                    raise
                return f"No topology {query.name} in {project}."
            used = [c.get("name") if isinstance(c, dict) else str(c)
                    for c in row.get("config", {}).get("components", [])]
            said = f"Topology {row['name']} in {project} is a {row['topology_type']}"
            said += f" with {_listed(used)}." if used else "."
            return said
        try:
            row = self.client.get_topology_component(query.name, project=project)
        except Exception as exc:  # noqa: BLE001 -- only a 404 is an answer
            if _status(exc) != 404:
                raise
            return f"No component {query.name} in {project}."
        return f"Component {row['name']} in {project}: {row['instruction']}"

    def _edit_vocabulary(self, words: str) -> bool:
        """Prepare one phrase edit and ask for approval; nothing is saved
        until it comes. False when the words ask for no edit."""
        add, remove = ADD_PHRASE.fullmatch(words), REMOVE_PHRASE.fullmatch(words)
        if not (add or remove or words in UNDO_PHRASE):
            return False
        try:
            if add:
                edit = vocabulary.prepare_add(add.group(2), add.group(1))
            elif remove:
                edit = vocabulary.prepare_remove(remove.group(2), remove.group(1))
            else:
                edit = vocabulary.prepare_undo()
        except (ValueError, OSError) as exc:
            self.echo(f"vocabulary edit refused: {exc}")
            self.ask(f"That cannot be changed: {exc}")
            return True
        if edit is None:
            self.ask("There is no vocabulary change to undo.")
            return True
        self._pending_edit, self._edit_retries = edit, 0
        self.ask(f"{_edit_said(edit)} Approve or hold?", EDIT_OPTIONS)
        return True

    def _confirm_edit(self, heard: str) -> None:
        """Save the pending edit on "approve", drop it on "hold" or a no, and
        ask again on anything else until `max_retries`."""
        from qmcp.integrations.voice.adapter import match_option, parse_yes_no

        edit = self._pending_edit
        choice = match_option(heard, list(EDIT_OPTIONS))
        both = set(EDIT_OPTIONS) <= set(plain(heard).split())
        decision = None if both else parse_yes_no(heard)
        if not both and (choice == "hold" or decision is False):
            self._pending_edit = None
            self.ask("Held. The vocabulary is unchanged.")
            return
        if not both and (choice == "approve" or decision is True):
            self._pending_edit = None
            try:
                vocabulary.apply_edit(edit)
            except (OSError, RuntimeError, ValueError) as exc:
                self.echo(f"vocabulary edit failed: {type(exc).__name__}: {exc}")
                self.ask("The vocabulary is unchanged: the edit could not be saved.")
                return
            self.echo(f"vocabulary: {edit.action} {edit.phrase!r} for {edit.key}")
            self.ask(f"Saved. {_edit_said(edit, past=True)}")
            return
        self._edit_retries += 1
        if self._edit_retries > self.max_retries:
            self._pending_edit = None
            self.ask("No clear answer. The vocabulary is unchanged.")
            return
        self.ask("Say approve to save the change, or hold to leave it.", EDIT_OPTIONS)

    def _waiting(self) -> int:
        """How many questions on the human queue are waiting for a person."""
        try:
            pending = self.client.list_human_requests(status_filter="pending", limit=20,
                                                      oldest_first=True)
        except Exception:  # noqa: BLE001 -- an unreachable queue holds nothing to say
            return 0
        return sum(1 for request in pending if not request.id.startswith(OWN))

    def _clone_for(self, project: str | None, instruction_id: str) -> Path | None:
        """None when the record knows the project's clone -- the act reads it
        there -- else the sibling clone named for the project, if one exists."""
        from qmcp.instructions import act as act_module
        from qmcp.instructions.continuity import last_clone

        if not project:
            return None
        rows = self.rows or act_module.configured_rows()
        self.rows = rows
        if last_clone(rows, project, before=instruction_id) is not None:
            return None
        candidate = self.clones / project.rsplit("/", 1)[-1]
        return candidate if candidate.is_dir() else None

    def _ask_waiting(self, ended: Ended) -> None:
        """Ask aloud what agents have put on the human queue, oldest first, once each."""
        from qmcp.integrations.voice.adapter import UnclearResponse, VoiceApprovalLoop

        try:
            pending = self.client.list_human_requests(status_filter="pending", limit=20,
                                                      oldest_first=True)
        except Exception:  # noqa: BLE001 -- an unreachable queue costs this turn nothing
            return
        for request in pending:
            if request.id in self._asked or request.id.startswith(OWN):
                continue
            self._asked.add(request.id)
            self.say(QUESTION_WAITING)
            try:
                answer = VoiceApprovalLoop(stt=self.stt, tts=self.tts,
                                           client=self.client).run_once(request.id)
            except UnclearResponse:
                continue
            ended.answered.append(request.id)
            self.echo(f"answered: {request.id} -> {answer.response}")


def wait_for(client: Any, stt: Any, echo: Callable[[str], None] = print,
             sleep: Callable[[float], None] | None = None, every: float = 2.0,
             limit: int | None = None) -> bool:
    """Wait until this server and the speech engine both answer, in either order.

    The conversation is started with the server, and the engine may come up
    before it or long after; neither is a reason to fail. Says once what it is
    waiting for. `limit` bounds the waits for a test; None waits for ever.
    Returns True once both answer, False when `limit` ran out first.
    """
    import time

    sleep = sleep or time.sleep
    said = False
    waits = 0
    while True:
        try:
            client.health()
            server = True
        except Exception:  # noqa: BLE001 -- not up yet is the expected answer
            server = False
        engine = bool(stt.reachable())
        if server and engine:
            return True
        if not said:
            missing = [name for name, up in (("qmcp", server), ("the speech engine", engine))
                       if not up]
            echo(f"waiting for {' and '.join(missing)}")
            said = True
        if limit is not None and waits >= limit:
            return False
        waits += 1
        sleep(every)


def same_in(words: str) -> str | None:
    """The project words of "same in <project>" and its kin, or None."""
    for phrase in _said(SAME_IN, "iteration.same_in"):
        lead = phrase.replace("{project}", "").strip()
        if words.startswith(lead + " ") and words[len(lead):].strip():
            return words[len(lead):].strip()
    return None


def _status(exc: Exception) -> int | None:
    """The HTTP status an error carries, whichever client library raised it."""
    return getattr(getattr(exc, "response", None), "status_code", None)


def _named_list(names: list[str], one: str, many: str, project: str) -> str:
    if not names:
        return f"No {many} in {project}."
    noun = one if len(names) == 1 else many
    return f"{len(names)} {noun} in {project}: {_listed(names)}."


def _edit_said(edit: vocabulary.PhraseEdit, *, past: bool = False) -> str:
    """A phrase edit as a sentence: "Add 'halt' to stop." or, saved, "Added"."""
    command, phrase = vocabulary.target(edit.key), repr(vocabulary.shown(edit.phrase))
    if edit.action == "undo":
        return f"{'Undid' if past else 'Undo'} the last change, to {phrase} in {command}."
    if edit.action == "add":
        return f"{'Added' if past else 'Add'} {phrase} to {command}."
    return f"{'Removed' if past else 'Remove'} {phrase} from {command}."


def _listed(names: list[str]) -> str:
    """"qmcp, joe and vox", the first six named and how many more."""
    shown, more = names[:6], len(names) - 6
    said = shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"
    return f"{said}, and {more} more" if more > 0 else said


def _after(heard: str, wake: str) -> str:
    """The utterance with its wake word, and the punctuation after it, taken off."""
    words = heard.strip().split()
    taken = len(wake.split())
    return " ".join(words[taken:]).lstrip(",.!? ")


__all__ = [
    "ANYTHING_ELSE",
    "DONE",
    "MORE",
    "READY",
    "STOP",
    "STOPPING",
    "Conversation",
    "Ended",
    "Turn",
    "plain",
    "sibling_clones",
    "wait_for",
]
