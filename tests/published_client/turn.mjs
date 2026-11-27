// One turn on an Agent Host Protocol host, driven by the protocol's published
// TypeScript client, `@microsoft/agent-host-protocol`.
//
//   node turn.mjs <ws url> <working directory uri> <provider> <answer>
//
// The client negotiates its own version, creates a session and a chat, starts
// a turn, and answers the turn's input request with <answer>. Every action
// the host sends is reduced with the package's reducers. During the turn,
// when the input request arrives, and again after the turn, the host is asked
// for its snapshots, and each is compared with the reduced state.
//
// The last line printed is one JSON object, the verdict. The exit status is
// 0 when the turn completed and every comparison agreed, and 1 otherwise.
//
// What this cannot do: hold the host's messages to the protocol's JSON
// schemas, which the package does not carry.

if (globalThis.WebSocket === undefined) {
  globalThis.WebSocket = (await import('ws')).WebSocket;
}

const types = await import('@microsoft/agent-host-protocol');
const { AhpClient, AhpStateMirror } = await import('@microsoft/agent-host-protocol/client');
const { WebSocketTransport } = await import('@microsoft/agent-host-protocol/ws');

const [url, cwd, provider, answer] = process.argv.slice(2);
const TURN_MS = 120000;

// Key order and absent-versus-undefined are not differences of state.
const canon = value => Array.isArray(value) ? value.map(canon)
  : value && typeof value === 'object'
    ? Object.fromEntries(Object.keys(value).sort()
      .filter(key => value[key] !== undefined).map(key => [key, canon(value[key])]))
    : value;
const same = (mine, theirs) => JSON.stringify(canon(mine)) === JSON.stringify(canon(theirs));

const verdict = {
  package: types.PROTOCOL_VERSION, negotiated: null, agents: [], actions: [],
  problems: [], agree: { during: null, after: null, session: null },
  turn: null, request: null,
};

const transport = await WebSocketTransport.connect(url);
const client = new AhpClient(transport);
const mirror = new AhpStateMirror();
client.connect();

const init = await client.initialize({
  clientId: 'published-client', protocolVersions: [types.PROTOCOL_VERSION],
  initialSubscriptions: ['ahp-root://'],
});
verdict.negotiated = init.protocolVersion;
for (const snapshot of init.snapshots ?? []) mirror.applySnapshot(snapshot);
verdict.agents = mirror.root.agents.map(agent => agent.provider);

let chatState;
let finish;
const finished = new Promise(resolve => { finish = resolve; });

function follow(uri) {
  const events = client.attachSubscription(uri);
  (async () => {
    for await (const event of events) {
      if (event.type !== 'action') continue;
      const envelope = event.params;
      const action = envelope.action;
      verdict.actions.push(action.type);
      try {
        mirror.apply(envelope);
        if (uri.startsWith('ahp-chat:') && !envelope.rejectionReason) {
          chatState = types.chatReducer(chatState, action,
            message => verdict.problems.push(`${action.type}: ${message}`));
        }
      } catch (error) {
        verdict.problems.push(`${action.type}: ${error.message}`);
      }
      if (envelope.rejectionReason) {
        verdict.problems.push(`${action.type} rejected: ${envelope.rejectionReason}`);
      }
      if (action.type === 'chat/inputRequested') {
        verdict.request = action.request;
        const during = await client.request('subscribe', { channel: uri });
        verdict.agree.during = same(chatState, during.snapshot.state);
        client.dispatch(uri, {
          type: 'chat/inputCompleted', requestId: action.request.id, response: 'accept',
          answers: { [action.request.questions[0].id]: {
            state: 'submitted', value: { kind: 'selected', value: answer } } },
        });
      }
      if (action.type === 'chat/turnComplete' || action.type === 'chat/error') finish();
    }
  })();
}

const session = `ahp-session:/${crypto.randomUUID()}`;
const chat = `ahp-chat:/${crypto.randomUUID()}`;
follow(session);
follow(chat);

await client.request('createSession', { channel: session, provider, workingDirectories: [cwd] });
const subscribed = (await client.subscribe(session)).result;
if (subscribed.snapshot) mirror.applySnapshot(subscribed.snapshot);
await client.request('createChat', { channel: session, chat });
chatState = (await client.subscribe(chat)).result.snapshot?.state;

client.dispatch(chat, {
  type: 'chat/turnStarted', turnId: crypto.randomUUID(),
  startedAt: new Date().toISOString(),
  message: { text: 'Say what qmcp is.', origin: { kind: 'user' } },
});

const timer = new Promise(resolve => setTimeout(resolve, TURN_MS, 'timeout'));
if (await Promise.race([finished, timer]) === 'timeout') {
  verdict.problems.push(`the turn did not end in ${TURN_MS} ms`);
}
// The session is told of the chat's last change after the chat is.
await new Promise(resolve => setTimeout(resolve, 300));

const after = await client.request('subscribe', { channel: chat });
const sessionAfter = await client.request('subscribe', { channel: session });
verdict.agree.after = same(chatState, after.snapshot.state);
verdict.agree.session = same(mirror.getSession(session), sessionAfter.snapshot.state);

const turn = chatState?.turns?.at(-1);
if (turn) {
  const asked = turn.responseParts.find(part => part.kind === 'inputRequest');
  verdict.turn = {
    state: turn.state,
    text: turn.responseParts.filter(part => part.kind === 'markdown')
      .map(part => part.content).join('\n'),
    response: asked?.response ?? null,
    answers: asked?.request.answers ?? null,
  };
}

const held = verdict.problems.length === 0 && verdict.turn !== null
  && Object.values(verdict.agree).every(value => value === true);
console.log(JSON.stringify(verdict));
await client.shutdown();
process.exit(held ? 0 : 1);
