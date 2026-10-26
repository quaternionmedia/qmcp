# qmcp documentation

Every page here, listed by the role it serves. [`roles.md`](roles.md) says
what each role is, and [`setup/README.md`](setup/README.md) sets qmcp up for
one of them.

## Every role

- [`roles.md`](roles.md): the roles, what each owns, and what qmcp leaves to the programs around it
- [`setup/README.md`](setup/README.md): the shared install, and which server and which queue a command reaches
- [`http-api.md`](http-api.md): every route, by role, and which are served off loopback
- [`architecture.md`](architecture.md): the client, server and storage planes
- [`deployment.md`](deployment.md): running qmcp as a service, and `qmcp db`
- [`ROADMAP.md`](ROADMAP.md): the phases, built and planned

## Tool server and human gate

- [`setup/tool-server.md`](setup/tool-server.md): serve, connect a caller, answer what is waiting
- [`tools.md`](tools.md): tool discovery, invocation and history
- [`human_in_loop.md`](human_in_loop.md): the human queue's API and lifecycle
- [`client.md`](client.md): the Python client

## Harness

- [`setup/harness.md`](setup/harness.md): run the gates recorded, and hand them to dossier
- [`self-checks.md`](self-checks.md): the dashboard, audit, self-check, the feedback loop and intake
- [`threads.md`](threads.md): conversations with an assistant, read as units of work

## Instruction loop

- [`voice-loop-demo.md`](voice-loop-demo.md): the onboarding, proved in three tiers, and what to say
- [`integrations/voice.md`](integrations/voice.md): the reference for every spoken step
- [`voice-loop-next-phases.md`](voice-loop-next-phases.md): what is planned next for the loop

## Topology designer

- [`agentframework/topologies.md`](agentframework/topologies.md): the shapes, saved designs and components, and running a design by voice

## Flow host

- [`setup/flows.md`](setup/flows.md): run the cookbook's flows in Docker
- [`integrations/alfred.md`](integrations/alfred.md): alfred's renders and captions, behind a person's approval
- [`agentframework/runners.md`](agentframework/runners.md): how a flow's steps are built

## Agent library

- [`integrations/index.md`](integrations/index.md): the integrations and when to use each
- [`integrations/pydantic-ai.md`](integrations/pydantic-ai.md): PydanticAI agents with their tool calls recorded here
- [`agentframework/overview.md`](agentframework/overview.md), [`models.md`](agentframework/models.md), [`mixins.md`](agentframework/mixins.md): the agent schemas
- [`adrs/pydantic-ai-comparison.md`](adrs/pydantic-ai-comparison.md): the comparison behind the PydanticAI integration

## Changing qmcp

- [`contributing.md`](contributing.md): the setup and the loop for a change
- [`testing.md`](testing.md): what the suite isolates, and the harness an offline check stands a qmcp up with
- [`overview.md`](overview.md): what qmcp does and does not do
