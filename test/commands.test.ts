/**
 * Command tests: /model chain switching (with durable polyglot/chain events)
 * and /polyglot status + per-provider usage tally from session-log events.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import type { Context } from '@deepseek-ai/cordis';
import type { CommandDefinition, CommandInvocation } from '@deepseek-ai/dsh-commands';
import type { Session } from '@deepseek-ai/dsh-session';
import { registerPolyglotCommands } from '../src/commands.ts';
import { estimateCost, formatCost } from '../src/commands.ts';
import type { PolyglotCommandDeps } from '../src/commands.ts';
import type { ServedRecord } from '../src/types.ts';

function makeHarness(events: unknown[] = []) {
  const definitions = new Map<string, CommandDefinition>();
  const ctx = {
    commands: {
      register: (definition: CommandDefinition) => {
        definitions.set(definition.name, definition);
        return () => definitions.delete(definition.name);
      },
    },
    logger: { warn: () => undefined },
  } as unknown as Context;
  const appended: Array<[string, unknown]> = [];
  const session = {
    events,
    append: (type: string, data: unknown) => {
      appended.push([type, data]);
    },
  } as unknown as Session;
  return { ctx, definitions, session, appended };
}

function invoke(definitions: Map<string, CommandDefinition>, name: string, rawInput: string, session: Session) {
  const definition = definitions.get(name);
  assert.ok(definition, `command /${name} registered`);
  const invocation = {
    commandId: 'cmd-1' as never,
    agent: { session },
    rawInput,
    signal: new AbortController().signal,
  } as CommandInvocation;
  return definition.handler(invocation);
}

function deps(overrides: Partial<PolyglotCommandDeps> = {}): PolyglotCommandDeps {
  return {
    chainNames: () => ['default', 'paid'],
    activeChain: () => 'default',
    setActiveChain: (name) => name === 'default' || name === 'paid',
    defaultChain: () => 'default',
    virtualProvider: () => 'polyglot',
    entriesSummary: () => [
      { provider: 'nous-portal', model: 'deepseek/deepseek-v4-flash:free', preset: 'nous-portal', free: { tier: true } },
      { provider: 'deepseek-official', model: 'deepseek-v4-flash', preset: 'deepseek-official' },
    ],
    cooldownStatus: () => ({ 'nous-portal': 25 }),
    pricingFor: (provider) => provider === 'deepseek-official'
      ? { inputPerMTokens: 0.14, outputPerMTokens: 0.28, currency: 'USD' }
      : undefined,
    ...overrides,
  };
}

function served(provider: string, status: 'ok' | 'failed', usage?: ServedRecord['usage']): unknown {
  return {
    type: 'polyglot/served',
    seq: 0,
    time: 0,
    data: { chain: 'default', provider, model: 'm', preset: 'custom', attempt: 1, status, latencyMs: 10, ...usage !== undefined ? { usage } : {} },
  };
}

test('/model with no input lists chains and marks the active one', async () => {
  const { ctx, definitions, session } = makeHarness();
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'model', '', session);
  assert.equal(result.kind, 'success');
  const text = (result as { text: string }).text ?? '';
  assert.match(text, /\* default \(active\)/);
  assert.match(text, /polyglot/);
});

test('/model switches to an existing chain and records a durable event', async () => {
  const { ctx, definitions, session, appended } = makeHarness();
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'model', 'paid', session);
  assert.equal(result.kind, 'success');
  assert.match((result as { text: string }).text ?? '', /Switched dsh-polyglot to chain "paid"/);
  assert.deepEqual(appended, [['polyglot/chain', { chain: 'paid' }]]);
});

test('/model rejects an unknown chain', async () => {
  const { ctx, definitions, session, appended } = makeHarness();
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'model', 'bogus', session);
  assert.equal(result.kind, 'error');
  assert.match((result as { text: string }).text ?? '', /Unknown chain "bogus"/);
  assert.equal(appended.length, 0);
});

test('/polyglot usage tallies calls, tokens, and estimated cost', async () => {
  const { ctx, definitions, session } = makeHarness([
    served('nous-portal', 'failed', { inputTokens: 100, outputTokens: 10 }),
    served('deepseek-official', 'ok', { inputTokens: 1000, outputTokens: 500, cacheReadTokens: 200 }),
    served('deepseek-official', 'ok', { inputTokens: 400, outputTokens: 300 }),
  ]);
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'polyglot', 'usage', session);
  assert.equal(result.kind, 'success');
  const text = (result as { text: string }).text ?? '';
  assert.match(text, /nous-portal: 1 calls \(0 ok \/ 1 failed\)/);
  assert.match(text, /deepseek-official: 2 calls \(2 ok \/ 0 failed\)/);
  assert.match(text, /1,400 in \/ 800 out/);
  assert.match(text, /200 cache/);
  // 1400 * 0.14/M + 800 * 0.28/M = 0.000196 + 0.000224 = 0.00042
  assert.match(text, /est 0\.000420 USD/);
});

test('/polyglot with no served calls reports none', async () => {
  const { ctx, definitions, session } = makeHarness([]);
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'polyglot', 'usage', session);
  const text = (result as { text: string }).text ?? '';
  assert.match(text, /No dsh-polyglot provider calls recorded/);
});

test('/polyglot status shows the active chain and cooldowns', async () => {
  const { ctx, definitions, session } = makeHarness([served('nous-portal', 'ok')]);
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'polyglot', '', session);
  const text = (result as { text: string }).text ?? '';
  assert.match(text, /active chain: default/);
  assert.match(text, /nous-portal: 25s/);
  assert.match(text, /deepseek-official \(deepseek-v4-flash\)/);
});

test('/polyglot presets lists free-tier posture', async () => {
  const { ctx, definitions, session } = makeHarness();
  registerPolyglotCommands(ctx, deps());
  const result = await invoke(definitions, 'polyglot', 'presets', session);
  const text = (result as { text: string }).text ?? '';
  assert.match(text, /nous-portal \(nous-portal\/deepseek\/deepseek-v4-flash:free\) \[free\]/);
  assert.match(text, /deepseek-official \(deepseek-official\/deepseek-v4-flash\) \[paid\]/);
});

test('estimateCost and formatCost are stable', () => {
  assert.equal(estimateCost({ inputTokens: 1_000_000, outputTokens: 0, cacheReadTokens: 0 }, { inputPerMTokens: 0.14, currency: 'USD' }), 0.14);
  assert.equal(formatCost(0, 'USD'), '0 USD');
  assert.equal(formatCost(0.00042, 'USD'), '0.000420 USD');
  assert.equal(formatCost(1.23456, 'USD'), '1.23 USD');
});
