# Voice loop: cleanup before new capabilities

This is a work sequence for future contributors, not a decision record or a
claim that any phase is underway. Its baseline is the integrated demo snapshot
at qmcp `5c65815` and joe `25890e7` on their
`demo/voice-loop-2026-10-04` branches. Those snapshots are integration
evidence, not merge bases or proof that the work has landed on the default
branches. Before acting, re-establish the current branch, open pull-request
stack, demo state, and gates.

The order is deliberate: make the existing behavior understandable and
bounded before expanding what a spoken command can change.

## Baseline already established

The integrated voice work provides:

- A declared qmcp vocabulary for conversation controls, answers, diagnostics,
  iteration, project terms, and named project checks.
- Joe-specific phrases that edit or label a transcript take, plus level
  reporting.
- A transcript view beside the visualiser, with live updates, history, labels,
  struck words, and take playback with an approximate word lane.
- Voice-triggered execution of declared project checks, after consent.

The implementation and integration rehearsal are evidence for the baseline;
they do not make every behavior production-ready or establish a release claim.
Re-read the source and tests before changing it. The two known limitations to
carry forward are that transcript word positions are estimated from segment
boundaries, and the append-only datapoints file is reread in full at page load
and take completion. The project terms remain opt-in as transcription hints;
do not change that default without new evidence and an explicit decision.

## Phase 0 — Re-establish the baseline

**Goal:** distinguish current behavior from plans, historical handoffs, and
claims made on integration branches.

**Work:**

- Read the current `docs/ROADMAP.md`, `docs/integrations/voice.md`, the
  implementation, and the relevant tests. Use the repository's current
  checkouts; do not infer landed status from a handoff or a demo branch.
- Confirm each dependency's branch and PR relationship before starting a
  change. Keep independent changes separate and dependent work stacked.
- Run the relevant offline walkthroughs first. Then use a live speech-engine
  rehearsal only where the phase needs real microphone, speaker, or
  transcription evidence.
- Record exactly what was exercised and what the test could not cover.

**Complete when:** the agent can name the code paths and tests that implement
the feature being changed, and can distinguish deterministic offline evidence
from live-device evidence.

## Cleanup, before feature work

### Phase 1 — Reconcile onboarding and architecture documentation

**Goal:** give a new reader one current path through setup and accurately
describe where voice orchestration runs.

**Work:**

- First let the existing roadmap corrections land in their prescribed order;
  do not recreate their changes on another branch.
- Reconcile the architecture description with the standing conversation,
  the voice engine, and the process that runs declared checks. State which
  component owns control flow and which component executes a requested action.
- Reconcile deployment examples, configured defaults, health checks, and
  documented ports against the current configuration.
- Explain the three separate meanings of “vocabulary”: qmcp conversation
  commands, project terms supplied as optional transcription hints, and joe
  commands that edit a take. Do not imply that users can teach persistent
  commands by speaking.
- Keep machine-specific measurements dated and in their historical handoff.
  Put reusable setup instructions in the maintained onboarding/reference
  pages; avoid copying volatile PR state into them.
- Where an existing roadmap or decision document is canonical, update or link
  that source rather than maintaining a competing status checklist here.

**Complete when:** every onboarding command is checked against the current CLI;
the architecture and deployment docs agree with code; and a reader can tell
which voice actions record, which edit a take, and which execute a command.

### Phase 2 — Bound transcript storage and rendering cost

**Goal:** preserve the transcript's useful append-only evidence without making
every page load and completed take cost grow with the entire history.

**Work:**

- Trace every reader and writer of `Data/Voice/segments.jsonl`, including
  reload, live reconnect, end-of-take history refresh, transcript corrections,
  and take playback.
- Measure load and update cost against generated files of increasing size
  before selecting an optimization. Report the file size, operation, and
  measured result so a performance threshold has a reproducible basis.
- Choose an incremental read, indexed view, or other bounded-read design only
  after the measurements. Keep the durable source of truth and its ordering
  explicit.
- Do not silently prune or rewrite recorded takes. Any retention or deletion
  policy needs a separate decision, an explicit user action, and a recoverable
  path.
- Preserve the existing protections against late history overwriting live
  events and reconnects duplicating them. Keep word timing described as
  approximate unless the capture/transcription pipeline begins providing
  measured word timestamps.

**Complete when:** reload and reconnect recover the same ordered conversation,
live events are neither lost nor duplicated, corrections and labels remain
visible, and a large-history test or benchmark demonstrates bounded work for
the chosen operation.

### Phase 3 — Make declared-check boundaries explicit

**Goal:** describe and test what consent-gated checks guarantee, without
mistaking a fixed command for a sandbox.

**Work:**

- Audit each declared check's target, working directory, effects, timeout, and
  output. The check process runs as the current user; fixed `argv` and no shell
  do not isolate it from the user's files or permissions.
- Reconcile any promise that a check leaves a clone unchanged with what is
  actually measured. Distinguish tracked changes, untracked files, caches,
  database changes, and external effects.
- If immutability is a requirement, add a bounded before/after check and a
  useful failure report. Never automatically reset, clean, or restore a
  user's working tree to make a check appear safe.
- Retain and extend tests for exact consent text and target, held requests,
  phrase ambiguity, command-not-found, timeout, nonzero exit, and output
  reporting. Mutation-test the guard with a command that changes a disposable
  fixture.
- Keep commands and arguments trusted declarations. Do not append recognized
  speech to `argv`, invoke a shell, or let the model invent executable
  arguments.

**Complete when:** each operation's effects and limits are documented, and
tests show that refusal, ambiguity, failure, and timeout do not produce a
success-shaped result or silently discard working-tree changes.

## New capabilities, after cleanup

### Phase 4 — Design persistent vocabulary authoring

**Goal:** let a person propose a new phrase without conflating transcript
correction, transcription hints, and executable operations.

**Design and implementation sequence:**

1. Decide which vocabulary is editable: qmcp conversation phrases, project
   terms, joe take commands, or a deliberately smaller subset. Keep these
   scopes distinct in the interface and storage.
2. Choose persistence and precedence. Prefer a validated user/workspace
   overlay over editing package-owned defaults at runtime; define how changes
   survive upgrades and how the running process reloads them.
3. Make speech produce a proposed edit, not an immediate write. Read back the
   exact normalized phrase, its meaning, and its scope; require explicit
   confirmation, and provide a clear cancel and undo path.
4. Reject malformed phrases and conflicts, especially phrases that map to
   different whole-utterance actions or overlap yes/no decisions. Keep
   executable check declarations and their `argv` outside voice editing.
5. Add deterministic validation and a walkthrough before connecting the
   authoring path to real speech. Test restart, rollback, conflicting edits,
   mishearing, and failure to persist.

**Complete when:** a confirmed phrase change is durable and auditable, an
unconfirmed or ambiguous proposal changes nothing, conflicts are explained,
and the prior vocabulary can be restored without editing source files.

**Open decisions:** editable scopes; whether changes are per user or workspace;
how changes are audited and backed up; whether restart or safe reload applies
them; and which phrase conflicts are errors versus warnings. Resolve these
before implementation rather than embedding assumptions in a voice command.

### Phase 5 — Define a capability model for voice-operated systems

**Goal:** expand from declared checks only if the desired operation is clear,
bounded, and distinguishable from arbitrary command execution.

**Work:**

- Inventory actual user goals before adding operations. Separate observation
  (status, health, list) from state changes (start, stop, restart, deploy).
- Define named capabilities with a fixed target and operation, declared
  arguments, permissions, timeout, expected effects, and a postcondition.
  Start with a read-only operation or one low-risk local lifecycle action.
- For state-changing actions, speak back the exact system, action, and
  consequence, then require an explicit approval tied to that request. A wake
  word is not identity or authorization; ambiguous targets fail closed.
- Keep execution in a trusted registry and use argument arrays rather than a
  shell. Never translate arbitrary speech or model output directly into a
  terminal command. Do not elevate privileges implicitly.
- Define duplicate-request and retry behavior, cancellation, partial failure,
  process ownership, and recovery before adding long-running service controls.
  Persist the request, approval, operation, and result so a future agent can
  explain what happened.

**Complete when:** a deterministic test and walkthrough demonstrate the
operation's target, approval boundary, success postcondition, timeout/failure
path, and audit record. A refused or uncertain request must not execute.

**Open decisions:** which systems are in scope, whether operations are local
only, who may declare capabilities, which actions need stronger approval, and
whether any long-running process should be supervised by qmcp or by an
existing service manager. Do not infer answers from the current check registry.

### Phase 6 — Measure recognition and usability before changing defaults

**Goal:** improve phrase recognition and interaction without weakening consent
or relying on a few rehearsals.

**Work:**

- Gather opt-in, appropriately handled examples across speakers, microphones,
  rooms, and supported speech engines. Keep synthesized test clips separate
  from real speech evidence.
- Measure command recognition, false activations, correction rate, confidence,
  and latency. Report denominators and conditions; a higher recognition score
  alone does not justify a less safe confirmation policy.
- Evaluate project-term hints independently from command grammar and from
  consent parsing. Keep hints off by default until evidence supports a change.
- Preserve accessible alternatives: keys, page controls, and typed commands
  for setup and recovery.

**Complete when:** a proposed default change has reproducible evidence across
the conditions it claims to support, known failure modes, and explicit
human approval. Consent must never be inferred from a lower confidence
threshold.

## Working rule for future agents

Take one phase at a time. A phase that needs a human decision stops at the
decision; do not smuggle a default into code or a PR body. Keep each
implementation change atomic, based on the correct parent branch, and
validated by the repository's actual preflight plus focused tests. A green
offline suite does not replace the live rehearsal when a phase changes
microphone, speaker, or transcription behavior.
