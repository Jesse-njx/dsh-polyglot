/**
 * Config resolution tests: chain validation, preset resolution, and an
 * end-to-end cordis mount (LlmRuntime + plugin + sessions) proving fallback
 * through the real runtime with durable served events.
 */

import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { Context } from '@deepseek-ai/cordis';
import LlmRuntime from '@deepseek-ai/dsh-llm';
import CommandRuntime from '@deepseek-ai/dsh-commands';
import SessionStore from '@deepseek-ai/dsh-session';
import { Config, resolveState } from '../src/index.ts';
import type { ConfigType } from '../src/index.ts';
import * as Polyglot from '../src/index.ts';
import { MockProvider, okStream } from './mock-server.ts';
import { generateOptions } from './helpers.ts';

const servers: MockProvider[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.close()));
});

async function startMock(scenarios: Parameters<MockProvider['script']>[0][] = []): Promise<MockProvider> {
  const server = new MockProvider(scenarios);
  servers.push(server);
  await server.start();
  return server;
}

function baseConfig(): ConfigType {
  return {
    chains: { default: [{ preset: 'custom', provider: 'mock-a', model: 'model-a' }] },
    defaultChain: 'default',
    virtualProvider: 'polyglot',
    cooldown: { baseMs: 30, maxMs: 900, factor: 2, jitterRatio: 0 },
    maxTokens: 8192,
    defaultContextWindow: 131072,
    streamIdleTimeoutMs: 5000,
  };
}

test('Config normalizes defaults through the schema', () => {
  const raw = { chains: { default: [{ preset: 'custom' }] } };
  const normalized = Config(raw);
  assert.equal(normalized.defaultChain, 'default');
  assert.equal(normalized.virtualProvider, 'polyglot');
  assert.equal(normalized.maxTokens, 256000);
  assert.equal(normalized.cooldown?.baseMs, 30000);
  // The schema leaves per-entry model unset; resolveState materializes the
  // preset's first model when a chain entry omits one.
  assert.equal(normalized.chains.default[0]!.model, undefined);
  const state = resolveState(normalized);
  assert.equal(state.chains.get('default')![0]!.model, 'model', 'custom preset first model');
});

test('resolveState resolves presets into providers and chains', () => {
  const config = baseConfig();
  config.chains.default = [
    { preset: 'nous-portal' },
    { preset: 'deepseek-official', model: 'deepseek-v4-flash' },
  ];
  const state = resolveState(config);
  assert.ok(state.providers.has('nous-portal'));
  assert.ok(state.providers.has('deepseek-official'));
  const entries = state.chains.get('default')!;
  assert.equal(entries.length, 2);
  assert.equal(entries[0]!.provider, 'nous-portal');
  assert.equal(entries[0]!.model, 'deepseek/deepseek-v4-flash:free');
  assert.equal(entries[1]!.model, 'deepseek-v4-flash');
  assert.equal(entries[1]!.connection.baseURL, 'https://api.deepseek.com');
});

test('resolveState rejects invalid configurations', () => {
  assert.throws(() => resolveState({ ...baseConfig(), chains: {} }), /at least one chain/);
  assert.throws(() => resolveState({ ...baseConfig(), defaultChain: 'missing' }), /defaultChain/);
  assert.throws(() => resolveState({ ...baseConfig(), chains: { default: [] } }), /is empty/);
  assert.throws(() => resolveState({ ...baseConfig(), chains: { default: [{ preset: 'nope' }] } }), /not found/);
  assert.throws(
    () => resolveState({
      ...baseConfig(),
      chains: {
        default: [{ preset: 'custom', provider: 'shared', model: 'a', baseUrl: 'http://x/v1' }],
        other: [{ preset: 'custom', provider: 'shared', model: 'b', baseUrl: 'http://y/v1' }],
      },
    }),
    /configured differently in multiple chains/,
  );
});

test('end-to-end: cordis mount serves a chain with fallback through ctx.llm.stream', async () => {
  const limited = await startMock([{ status: 429, body: { error: { message: 'free tier ceiling' } } }]);
  const healthy = await startMock([okStream('hello from the second provider', { prompt_tokens: 12, completion_tokens: 4 })]);

  const ctx = new Context();
  await ctx.plugin(LlmRuntime);
  await ctx.plugin(CommandRuntime);
  await ctx.plugin(SessionStore);
  const fiber = await ctx.plugin(Polyglot, {
    ...baseConfig(),
    chains: {
      default: [
        { preset: 'custom', provider: 'mock-a', model: 'model-a', baseUrl: limited.url },
        { preset: 'custom', provider: 'mock-b', model: 'model-b', baseUrl: healthy.url },
      ],
    },
  } as never);
  try {
    const session = ctx.sessions.create();
    const chunks: unknown[] = [];
    for await (const chunk of ctx.llm.stream({ ...generateOptions({ provider: 'polyglot' }), sessionId: session.id })) {
      chunks.push(chunk);
    }

    // Fallback happened: both providers were hit, second one served.
    assert.equal(limited.count, 1);
    assert.equal(healthy.count, 1);
    const texts = chunks.filter((chunk) => (chunk as { type: string }).type === 'text-delta').map((chunk) => (chunk as { text: string }).text);
    assert.equal(texts.join(''), 'hello from the second provider');
    const finish = chunks.at(-1) as { type: string; reason: { kind: string } };
    assert.equal(finish.type, 'finish');
    assert.equal(finish.reason.kind, 'stop');

    // Durable served events landed in the session log.
    const served = session.events.filter((event) => event.type === 'polyglot/served');
    assert.equal(served.length, 2);
    assert.equal(served[0]!.data.status, 'failed');
    assert.equal(served[0]!.data.provider, 'mock-a');
    assert.equal(served[1]!.data.status, 'ok');
    assert.equal(served[1]!.data.provider, 'mock-b');
    assert.equal(served[1]!.data.usage?.inputTokens, 12);
  } finally {
    await fiber.dispose();
  }
});

test('end-to-end: a chain of two healthy providers serves the first', async () => {
  const first = await startMock([okStream('from first')]);
  const second = await startMock([okStream('from second')]);
  const ctx = new Context();
  await ctx.plugin(LlmRuntime);
  await ctx.plugin(CommandRuntime);
  const fiber = await ctx.plugin(Polyglot, {
    ...baseConfig(),
    chains: {
      default: [
        { preset: 'custom', provider: 'mock-a', model: 'model-a', baseUrl: first.url },
        { preset: 'custom', provider: 'mock-b', model: 'model-b', baseUrl: second.url },
      ],
    },
  } as never);
  try {
    const chunks: unknown[] = [];
    for await (const chunk of ctx.llm.stream(generateOptions({ provider: 'polyglot' }))) {
      chunks.push(chunk);
    }
    assert.equal(first.count, 1);
    assert.equal(second.count, 0);
  } finally {
    await fiber.dispose();
  }
});

test('end-to-end: unconfigured provider route falls through without a key', async () => {
  const healthy = await startMock([okStream('ok')]);
  const ctx = new Context();
  await ctx.plugin(LlmRuntime);
  await ctx.plugin(CommandRuntime);
  const fiber = await ctx.plugin(Polyglot, {
    ...baseConfig(),
    chains: {
      default: [
        // deepseek-official with no key configured anywhere: MISSING_CREDENTIAL → skip
        { preset: 'deepseek-official', model: 'deepseek-v4-flash' },
        { preset: 'custom', provider: 'mock-b', model: 'model-b', baseUrl: healthy.url },
      ],
    },
  } as never);
  try {
    const chunks: unknown[] = [];
    for await (const chunk of ctx.llm.stream(generateOptions({ provider: 'polyglot' }))) {
      chunks.push(chunk);
    }
    assert.equal(healthy.count, 1, 'fell through the key-less provider to the mock');
    const finish = chunks.at(-1) as { type: string; reason: { kind: string } };
    assert.equal(finish.reason.kind, 'stop');
  } finally {
    await fiber.dispose();
  }
});

test('Config export shape is stable', () => {
  assert.ok(Config !== undefined);
  assert.ok(Polyglot.name === 'polyglot');
  assert.deepEqual(Polyglot.inject, ['llm', 'commands']);
});
