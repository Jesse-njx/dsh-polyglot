/**
 * PolyglotRouterAdapter tests: fallback on pre-content failures, cooldown with
 * Retry-After honoring, terminal failures after content flowed, abort
 * handling, chunk ordering, and durable served records.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PolyglotRouterAdapter, isFallbackEligible } from '../src/router.ts';
import type { RouterAdapterOptions } from '../src/router.ts';
import type { ActiveChain, ServedRecord } from '../src/types.ts';
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm';
import { chainEntry, collect, generateOptions } from './helpers.ts';

/** A dispatch that returns scripted per-provider chunk streams. */
function scriptedDispatch(spec: Record<string, Array<StreamChunk | (() => StreamChunk[])>>) {
  const calls: string[] = [];
  const dispatch = (options: GenerateOptions) => {
    calls.push(options.provider);
    const chunks = spec[options.provider] ?? [];
    return (async function* () {
      for (const item of chunks) {
        yield typeof item === 'function' ? item() : item;
      }
    })();
  };
  return { dispatch, calls };
}

const errorFinish = (code: string, message = 'provider failed') => ({
  type: 'finish' as const,
  reason: { kind: 'error' as const, failure: { message, code } },
});

const okFinish = () => ({ type: 'finish' as const, reason: { kind: 'stop' as const } });

const textChunk = (text: string) => ({ type: 'text-delta' as const, index: 0, text });

function makeRouter(options: Partial<RouterAdapterOptions> & { dispatch: RouterAdapterOptions['dispatch'] }, entries: ReturnType<typeof chainEntry>[], now = () => 1000000) {
  const served: ServedRecord[] = [];
  const chain = (): ActiveChain => ({ name: 'default', entries });
  const router = new PolyglotRouterAdapter({
    chain,
    dispatch: options.dispatch,
    cooldown: () => options.cooldown ?? { baseMs: 30000, maxMs: 900000, factor: 2, jitterRatio: 0 },
    now,
    onServed: (record) => served.push(record),
  });
  return { router, served };
}

const entryA = chainEntry('http://a.test/v1', 'provider-a', 'model-a');
const entryB = chainEntry('http://b.test/v1', 'provider-b', 'model-b');

test('falls back to the next provider on a pre-content rate limit', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [errorFinish('RATE_LIMIT')],
    'provider-b': [textChunk('served by b'), okFinish()],
  });
  const { router, served } = makeRouter({ dispatch }, [entryA, entryB]);
  const chunks = await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a', 'provider-b']);
  assert.deepEqual(chunks, [textChunk('served by b'), okFinish()]);
  assert.equal(served.length, 2);
  assert.equal(served[0]!.status, 'failed');
  assert.equal(served[0]!.provider, 'provider-a');
  assert.equal(served[0]!.attempt, 1);
  assert.equal(served[0]!.failureCode, 'RATE_LIMIT');
  assert.equal(served[1]!.status, 'ok');
  assert.equal(served[1]!.provider, 'provider-b');
  assert.equal(served[1]!.attempt, 2);
});

test('a cooling provider is skipped without dispatch', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [errorFinish('RATE_LIMIT')],
    'provider-b': [textChunk('ok'), okFinish()],
  });
  const now = { value: 1000000 };
  const { router } = makeRouter({ dispatch }, [entryA, entryB], () => now.value);
  await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a', 'provider-b']);
  assert.equal(router.cooldown.cooling('provider-a'), true, 'provider-a marked cooling');

  calls.length = 0;
  await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-b'], 'cooling provider-a skipped');
});

test('cooldown honors providerRetryAfterMs above the local backoff', async () => {
  const { dispatch } = scriptedDispatch({
    'provider-a': [{ type: 'finish', reason: { kind: 'error', failure: { message: 'slow down', code: 'RATE_LIMIT', providerRetryAfterMs: 600000 } } }],
    'provider-b': [textChunk('ok'), okFinish()],
  });
  const now = { value: 0 };
  const { router } = makeRouter({ dispatch }, [entryA, entryB], () => now.value);
  await collect(router.stream(generateOptions()));
  assert.equal(router.cooldown.cooling('provider-a'), true);
  now.value = 599000;
  assert.equal(router.cooldown.cooling('provider-a'), true, 'still cooling before retry-after');
  now.value = 600001;
  assert.equal(router.cooldown.cooling('provider-a'), false, 'cooled after retry-after');
});

test('a failure after content flowed is terminal — no fallback', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [textChunk('partial answer'), errorFinish('RATE_LIMIT')],
    'provider-b': [textChunk('should not run'), okFinish()],
  });
  const { router, served } = makeRouter({ dispatch }, [entryA, entryB]);
  const chunks = await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a']);
  assert.deepEqual(chunks, [textChunk('partial answer'), errorFinish('RATE_LIMIT')]);
  assert.equal(served.length, 1);
  assert.equal(served[0]!.status, 'failed');
});

test('a non-eligible failure (invalid request) is terminal', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [errorFinish('INVALID_REQUEST')],
    'provider-b': [textChunk('nope'), okFinish()],
  });
  const { router } = makeRouter({ dispatch }, [entryA, entryB]);
  const chunks = await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a']);
  assert.deepEqual(chunks, [errorFinish('INVALID_REQUEST')]);
});

test('every provider failing yields the last failure', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [errorFinish('RATE_LIMIT', 'a rate limited')],
    'provider-b': [errorFinish('SERVER', 'b exploded')],
  });
  const { router, served } = makeRouter({ dispatch }, [entryA, entryB]);
  const chunks = await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a', 'provider-b']);
  const finish = chunks[chunks.length - 1] as Extract<StreamChunk, { type: 'finish' }>;
  assert.equal(finish.reason.kind, 'error');
  assert.equal(finish.reason.failure.code, 'SERVER');
  assert.equal(served.length, 2);
  assert.ok(served.every((record) => record.status === 'failed'));
});

test('missing credential skips to the next provider', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [errorFinish('MISSING_CREDENTIAL')],
    'provider-b': [textChunk('ok'), okFinish()],
  });
  const { router } = makeRouter({ dispatch }, [entryA, entryB]);
  const chunks = await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a', 'provider-b']);
  assert.deepEqual(chunks, [textChunk('ok'), okFinish()]);
});

test('caller abort stops the chain with an aborted finish', async () => {
  const { dispatch, calls } = scriptedDispatch({
    'provider-a': [{ type: 'finish', reason: { kind: 'aborted', failure: { message: 'canceled', code: 'ABORTED' } } }],
    'provider-b': [textChunk('nope'), okFinish()],
  });
  const { router } = makeRouter({ dispatch }, [entryA, entryB]);
  const chunks = await collect(router.stream(generateOptions()));
  assert.deepEqual(calls, ['provider-a']);
  const finish = chunks[chunks.length - 1] as Extract<StreamChunk, { type: 'finish' }>;
  assert.equal(finish.reason.kind, 'aborted');
});

test('usage is forwarded before finish and nothing follows finish', async () => {
  const usage = { type: 'usage' as const, usage: { inputTokens: 3, outputTokens: 2 } };
  const { dispatch } = scriptedDispatch({
    'provider-a': [textChunk('hi'), usage, okFinish()],
  });
  const { router } = makeRouter({ dispatch }, [entryA]);
  const chunks = await collect(router.stream(generateOptions()));
  const usageIndex = chunks.findIndex((chunk) => chunk.type === 'usage');
  const finishIndex = chunks.findIndex((chunk) => chunk.type === 'finish');
  assert.ok(usageIndex >= 0 && finishIndex > usageIndex, 'usage before finish');
  assert.equal(finishIndex, chunks.length - 1, 'finish is last');
});

test('empty chain yields NO_PROVIDER', async () => {
  const { dispatch } = scriptedDispatch({});
  const { router } = makeRouter({ dispatch }, []);
  const chunks = await collect(router.stream(generateOptions()));
  const finish = chunks[0] as Extract<StreamChunk, { type: 'finish' }>;
  assert.equal(finish.reason.kind, 'error');
  assert.equal(finish.reason.failure.code, 'NO_PROVIDER');
});

test('success clears the cooldown for that provider', async () => {
  // provider-a fails once (rate limit), succeeds on later calls.
  let calls = 0;
  const dispatch = (options: GenerateOptions) => {
    calls += 1;
    const chunks = options.provider === 'provider-a' && calls === 1
      ? [errorFinish('RATE_LIMIT')]
      : [textChunk('ok'), okFinish()];
    return (async function* () {
      for (const chunk of chunks) yield chunk;
    })();
  };
  const now = { value: 0 };
  const { router } = makeRouter({ dispatch }, [entryA, entryB], () => now.value);
  await collect(router.stream(generateOptions()));
  assert.equal(router.cooldown.cooling('provider-a'), true);
  now.value = 100000;
  await collect(router.stream(generateOptions()));
  assert.equal(router.cooldown.cooling('provider-a'), false, 'success clears cooldown');
});

test('isFallbackEligible classifies failure codes', () => {
  for (const code of ['RATE_LIMIT', 'QUOTA', 'SERVER', 'TRANSPORT', 'TIMEOUT', 'STREAM_CLOSED', 'MISSING_CREDENTIAL', 'NO_ADAPTER', 'AUTH', 'EMPTY_RESPONSE']) {
    assert.equal(isFallbackEligible({ message: '', code }), true, code);
  }
  for (const code of ['INVALID_REQUEST', 'CONTEXT_WINDOW_EXCEEDED', 'ABORTED', 'UNSUPPORTED', 'UNSUPPORTED_CONTENT', 'INVALID_PREPARED_CALL']) {
    assert.equal(isFallbackEligible({ message: '', code }), false, code);
  }
  assert.equal(isFallbackEligible({ message: '', code: 'HTTP_503' }), true);
  assert.equal(isFallbackEligible({ message: '', code: 'HTTP_418' }), false);
});
