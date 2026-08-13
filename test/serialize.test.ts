/**
 * Golden wire-request assertions per quirk flag. The adapter's provider
 * deviations are declarative; these tests pin exactly what each flag puts on
 * the wire.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { serializeMessages, serializeRequest } from '../src/serialize.ts';
import { resolveQuirks } from '../src/quirk.ts';
import { messages, CallId, ReasoningEffortId } from './helpers.ts';
import { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm';

test('baseline request: stream + usage + optional fields', () => {
  const quirks = resolveQuirks({}, 'test');
  const body = serializeRequest({
    provider: 'mock',
    model: 'model-a',
    messages: messages('hi'),
    system: 'be brief',
    temperature: 0.2,
    maxTokens: 123,
    stop: ['END'],
    tools: [{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } } }],
  }, quirks);
  assert.equal(body.model, 'model-a');
  assert.equal(body.stream, true);
  assert.deepEqual(body.stream_options, { include_usage: true });
  assert.equal(body.temperature, 0.2);
  assert.equal(body.max_tokens, 123);
  assert.deepEqual(body.stop, ['END']);
  assert.deepEqual(body.messages, [
    { role: 'system', content: 'be brief' },
    { role: 'user', content: 'hi' },
  ]);
  const tool = (body.tools as Array<{ function: Record<string, unknown> }>)[0];
  assert.equal(tool.function.name, 'read');
  assert.equal(tool.function.strict, undefined, 'strict is off by default');
});

test('quirk maxTokensField: max_completion_tokens', () => {
  const quirks = resolveQuirks({ maxTokensField: 'max_completion_tokens' }, 'test');
  const body = serializeRequest({ provider: 'mock', model: 'm', messages: messages(), maxTokens: 99 }, quirks);
  assert.equal(body.max_completion_tokens, 99);
  assert.equal(body.max_tokens, undefined);
});

test('quirk strictToolSchemas: strict: true on every tool', () => {
  const quirks = resolveQuirks({ strictToolSchemas: true }, 'test');
  const body = serializeRequest({
    provider: 'mock', model: 'm', messages: messages(),
    tools: [
      { name: 'a', description: 'A', parameters: {} },
      { name: 'b', description: 'B', parameters: {} },
    ],
  }, quirks);
  const tools = body.tools as Array<{ function: { strict?: boolean } }>;
  assert.equal(tools.length, 2);
  for (const tool of tools) assert.equal(tool.function.strict, true);
});

test('quirk usage none: no stream_options on the wire', () => {
  const quirks = resolveQuirks({ usage: 'none' }, 'test');
  const body = serializeRequest({ provider: 'mock', model: 'm', messages: messages() }, quirks);
  assert.equal(body.stream_options, undefined);
});

test('assistant reasoning replays under the quirk reasoning field', () => {
  const quirks = resolveQuirks({ reasoningField: 'reasoning_content' }, 'test');
  const assistant = createAssistantMessage({
    content: [
      { type: 'reasoning', text: 'think hard' },
      { type: 'text', text: 'answer' },
    ],
    source: { kind: 'model', provider: 'mock', model: 'm' },
  });
  const wire = serializeMessages([assistant], quirks);
  assert.deepEqual(wire, [{ role: 'assistant', content: 'answer', reasoning_content: 'think hard' }]);
});

test('quirk reasoningField null drops reasoning blocks entirely', () => {
  const quirks = resolveQuirks({ reasoningField: null }, 'test');
  const assistant = createAssistantMessage({
    content: [
      { type: 'reasoning', text: 'think hard' },
      { type: 'text', text: 'answer' },
    ],
    source: { kind: 'model', provider: 'mock', model: 'm' },
  });
  const wire = serializeMessages([assistant], quirks);
  assert.deepEqual(wire, [{ role: 'assistant', content: 'answer' }]);
});

test('tool results expand into separate tool messages', () => {
  const quirks = resolveQuirks({}, 'test');
  const callId = CallId('call_1');
  const toolResult = createUserMessage({
    content: [{
      type: 'tool-result',
      toolCallId: callId,
      content: [{ type: 'text', text: 'file contents' }],
    }],
    source: { kind: 'tool', callId },
  });
  const wire = serializeMessages([toolResult], quirks);
  assert.deepEqual(wire, [{ role: 'tool', tool_call_id: 'call_1', content: 'file contents' }]);
});

test('quirk thinkingField: off disables thinking, high/max send effort', () => {
  const quirks = resolveQuirks({ thinkingField: true, reasoningEffortField: true }, 'test');
  const off = serializeRequest({ provider: 'mock', model: 'm', messages: messages(), reasoningEffort: ReasoningEffortId('off') }, quirks);
  assert.deepEqual(off.thinking, { type: 'disabled' });
  const high = serializeRequest({ provider: 'mock', model: 'm', messages: messages(), reasoningEffort: ReasoningEffortId('high') }, quirks);
  assert.deepEqual(high.thinking, { type: 'enabled' });
  assert.equal(high.reasoning_effort, 'high');
});

test('quirk reasoningEffortField only: effort without thinking envelope', () => {
  const quirks = resolveQuirks({ thinkingField: false, reasoningEffortField: true }, 'test');
  const body = serializeRequest({ provider: 'mock', model: 'm', messages: messages(), reasoningEffort: ReasoningEffortId('max') }, quirks);
  assert.equal(body.thinking, undefined);
  assert.equal(body.reasoning_effort, 'max');
});

test('unsupported reasoning effort throws', () => {
  const quirks = resolveQuirks({ thinkingField: false, reasoningEffortField: false }, 'test');
  assert.throws(
    () => serializeRequest({ provider: 'mock', model: 'm', messages: messages(), reasoningEffort: ReasoningEffortId('high') }, quirks),
    (error: unknown) => error instanceof Error && error.message.includes('reasoning effort'),
  );
});
