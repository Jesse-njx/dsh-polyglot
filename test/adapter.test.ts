/**
 * OpenAiCompatAdapter tests against the scripted mock server: streaming,
 * reasoning, tool calls, usage modes, error mapping, and stream integrity.
 */

import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { OpenAiCompatAdapter } from '../src/adapter.ts';
import { resolveQuirks } from '../src/quirk.ts';
import { collect, generateOptions, providerConfig } from './helpers.ts';
import { MockProvider, finishDelta, okStream, reasoningDelta, textDelta, toolCallDelta, usageChunk } from './mock-server.ts';
import { LlmError } from '@deepseek-ai/dsh-llm';
import type { StreamChunk } from '@deepseek-ai/dsh-llm';

const servers: MockProvider[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.close()));
});

function startMock(scenarios: Parameters<MockProvider['script']>[0][] = []): Promise<MockProvider> {
  const server = new MockProvider(scenarios);
  servers.push(server);
  return server.start().then(() => server);
}

function adapterFor(server: MockProvider, quirks?: Parameters<typeof resolveQuirks>[0], baseURL = '') {
  const connection = providerConfig({
    baseURL: baseURL || server.url,
    quirks: resolveQuirks(quirks, 'test'),
  });
  const adapter = new OpenAiCompatAdapter({
    options: () => connection,
    resolveApiKey: async () => 'sk-test',
    resolveUserId: () => 42 as never,
  });
  return { adapter, connection };
}

test('streams text, reasoning, usage, and finish in order', async () => {
  const server = await startMock([{
    chunks: [
      reasoningDelta('think...'),
      textDelta('hello '),
      textDelta('world'),
      finishDelta('stop'),
      usageChunk({ prompt_tokens: 10, completion_tokens: 5, prompt_tokens_details: { cached_tokens: 4 } }),
    ],
  }]);
  const { adapter } = adapterFor(server);
  const chunks = await collect(adapter.stream(generateOptions()));
  const types = chunks.map((chunk) => chunk.type);
  assert.deepEqual(types, ['block-start', 'reasoning-delta', 'block-start', 'text-delta', 'text-delta', 'block-end', 'block-end', 'usage', 'finish']);
  const usage = chunks.find((chunk): chunk is Extract<StreamChunk, { type: 'usage' }> => chunk.type === 'usage')!;
  assert.deepEqual(usage.usage, { inputTokens: 6, outputTokens: 5, cacheReadTokens: 4 });
  const finish = chunks.find((chunk): chunk is Extract<StreamChunk, { type: 'finish' }> => chunk.type === 'finish')!;
  assert.deepEqual(finish.reason, { kind: 'stop' });
  const text = chunks.filter((chunk) => chunk.type === 'text-delta').map((chunk) => (chunk as Extract<StreamChunk, { type: 'text-delta' }>).text).join('');
  assert.equal(text, 'hello world');
});

test('sends the authorization header and attribution headers', async () => {
  const server = await startMock([okStream('hi')]);
  const { adapter, connection } = adapterFor(server);
  const connectionWithKey = { ...connection, authKind: 'key' as const };
  const adapterKeyed = new OpenAiCompatAdapter({
    options: () => connectionWithKey,
    resolveApiKey: async () => 'sk-secret',
    resolveUserId: () => 1 as never,
  });
  await collect(adapterKeyed.stream(generateOptions()));
  assert.equal(server.count, 1);
  const request = server.requests[0]!;
  assert.equal(request.headers.authorization, 'Bearer sk-secret');
  assert.ok(request.headers['user-agent']?.includes('deepseek-harness'), 'attribution user-agent present');
  assert.equal(request.body.stream, true);
  assert.equal(request.body.model, 'model-a');
});

test('deepseek usage mode subtracts cache hits folded into prompt_tokens', async () => {
  const server = await startMock([{
    chunks: [finishDelta('stop'), usageChunk({ prompt_tokens: 10, completion_tokens: 5, prompt_cache_hit_tokens: 4 })],
  }]);
  const { adapter } = adapterFor(server, { usage: 'deepseek' });
  const chunks = await collect(adapter.stream(generateOptions()));
  const usage = chunks.find((chunk): chunk is Extract<StreamChunk, { type: 'usage' }> => chunk.type === 'usage')!;
  assert.deepEqual(usage.usage, { inputTokens: 6, outputTokens: 5, cacheReadTokens: 4 });
});

test('usage none emits no usage chunk', async () => {
  const server = await startMock([{ chunks: [textDelta('hi'), finishDelta('stop'), usageChunk({ prompt_tokens: 3, completion_tokens: 1 })] }]);
  const { adapter } = adapterFor(server, { usage: 'none' });
  const chunks = await collect(adapter.stream(generateOptions()));
  assert.equal(chunks.some((chunk) => chunk.type === 'usage'), false);
  assert.equal(server.requests[0]!.body.stream_options, undefined);
});

test('tool call fragments assemble into one tool-call block', async () => {
  const server = await startMock([{
    chunks: [
      toolCallDelta({ id: 'call_1', name: 'read_file' }),
      toolCallDelta({ arguments: '{"path":' }),
      toolCallDelta({ arguments: '"a.txt"}' }),
      finishDelta('tool_calls'),
    ],
  }]);
  const { adapter } = adapterFor(server);
  const chunks = await collect(adapter.stream(generateOptions()));
  const blockEnd = chunks.find((chunk): chunk is Extract<StreamChunk, { type: 'block-end' }> => chunk.type === 'block-end')!;
  assert.deepEqual(blockEnd.block, { type: 'tool-call', id: 'call_1', name: 'read_file', arguments: '{"path":"a.txt"}' });
  const finish = chunks.find((chunk): chunk is Extract<StreamChunk, { type: 'finish' }> => chunk.type === 'finish')!;
  assert.deepEqual(finish.reason, { kind: 'tool-calls' });
});

test('HTTP 429 maps to RATE_LIMIT with Retry-After honored', async () => {
  const server = await startMock([{
    status: 429,
    headers: { 'retry-after': '30', 'x-request-id': 'req-123' },
    body: { error: { message: 'rate limited', type: 'rate_limit_error' } },
  }]);
  const { adapter } = adapterFor(server);
  await assert.rejects(
    async () => { for await (const _ of adapter.stream(generateOptions())) { /* drain */ } },
    (error: unknown) => {
      assert.ok(error instanceof LlmError);
      assert.equal(error.code, 'RATE_LIMIT');
      assert.equal(error.failure.status, 429);
      assert.equal(error.failure.providerRetryAfterMs, 30000);
      assert.equal(error.failure.requestId, 'req-123');
      return true;
    },
  );
});

test('HTTP 500 maps to SERVER', async () => {
  const server = await startMock([{ status: 500, body: { error: { message: 'boom' } } }]);
  const { adapter } = adapterFor(server);
  await assert.rejects(
    async () => { for await (const _ of adapter.stream(generateOptions())) { /* drain */ } },
    (error: unknown) => error instanceof LlmError && error.code === 'SERVER',
  );
});

test('quota wording maps to QUOTA', async () => {
  const server = await startMock([{ status: 402, body: { error: { message: 'insufficient balance' } } }]);
  const { adapter } = adapterFor(server);
  await assert.rejects(
    async () => { for await (const _ of adapter.stream(generateOptions())) { /* drain */ } },
    (error: unknown) => error instanceof LlmError && error.code === 'QUOTA',
  );
});

test('truncated stream without [DONE] throws STREAM_CLOSED', async () => {
  const server = await startMock([{ status: 200, chunks: [textDelta('partial')], noDone: true }]);
  const { adapter } = adapterFor(server);
  await assert.rejects(
    async () => { for await (const _ of adapter.stream(generateOptions())) { /* drain */ } },
    (error: unknown) => error instanceof LlmError && error.code === 'STREAM_CLOSED',
  );
});

test('reasoningField null ignores reasoning deltas', async () => {
  const server = await startMock([{ chunks: [reasoningDelta('think'), textDelta('answer'), finishDelta('stop')] }]);
  const { adapter } = adapterFor(server, { reasoningField: null });
  const chunks = await collect(adapter.stream(generateOptions()));
  assert.equal(chunks.some((chunk) => chunk.type === 'reasoning-delta'), false);
  const types = chunks.map((chunk) => chunk.type);
  assert.ok(types.includes('text-delta'));
});

test('caller abort maps to ABORTED', async () => {
  const server = await startMock([{ chunks: [textDelta('hi'), finishDelta('stop')], delayMs: 5000 }]);
  const { adapter } = adapterFor(server);
  const controller = new AbortController();
  const options = generateOptions({ signal: controller.signal });
  const promise = (async () => { for await (const _ of adapter.stream(options)) { /* drain */ } })();
  setTimeout(() => controller.abort(), 20);
  await assert.rejects(
    promise,
    (error: unknown) => error instanceof LlmError && error.code === 'ABORTED',
  );
});

test('empty completion maps to EMPTY_RESPONSE finish', async () => {
  const server = await startMock([{ chunks: [finishDelta('stop'), usageChunk({ prompt_tokens: 1, completion_tokens: 0 })] }]);
  const { adapter } = adapterFor(server);
  const chunks = await collect(adapter.stream(generateOptions()));
  const finish = chunks.find((chunk): chunk is Extract<StreamChunk, { type: 'finish' }> => chunk.type === 'finish')!;
  assert.equal(finish.reason.kind, 'error');
  assert.equal((finish.reason as Extract<StreamChunk, { type: 'finish' }>['reason'] & { failure: { code: string } }).failure.code, 'EMPTY_RESPONSE');
});
