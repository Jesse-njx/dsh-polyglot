/**
 * Serialize harness messages into generic OpenAI-compatible chat completions.
 * User text is joined; assistant text becomes `content`, tool calls become
 * `tool_calls`, and tool results become separate tool messages. Reasoning
 * blocks replay under the quirk-configured reasoning field (DeepSeek's
 * `reasoning_content`, or another host's spelling), and are dropped entirely
 * when the provider exposes none (`reasoningField: null`).
 *
 * @module dsh-polyglot/serialize
 */
import { LlmError, contentHasImage } from '@deepseek-ai/dsh-llm';
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm';
import type { ResolvedQuirks } from './types.ts';

/** Wire message shapes accepted by OpenAI-compatible endpoints. */
export type WireMessage =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string; reasoning_content?: string; tool_calls?: WireToolCall[] }
  | { role: 'tool'; tool_call_id: string; content: string };

/** Wire tool-call shape inside an assistant message. */
export interface WireToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

/** Join the text blocks of a message (used for user/tool-result content). */
export function flattenText(blocks: readonly ContentBlock[]): string {
  return blocks.filter((block) => block.type === 'text').map((block) => block.text).join('');
}

/** Reject core image content before any text-flattening path can silently erase it. */
export function assertTextOnly(blocks: readonly ContentBlock[]): void {
  if (contentHasImage(blocks)) {
    throw new LlmError('The dsh-polyglot OpenAI-compatible adapter does not support image content.', 'UNSUPPORTED_CONTENT');
  }
}

/** Serialize one assistant message (text + reasoning + tool calls). */
function serializeAssistant(message: Message, quirks: ResolvedQuirks): WireMessage {
  const text = flattenText(message.content);
  const reasoning = message.content.filter((block) => block.type === 'reasoning').map((block) => block.text).join('');
  const toolCalls: WireToolCall[] = message.content.filter((block) => block.type === 'tool-call').map((block) => ({
    id: block.id,
    type: 'function',
    function: {
      name: block.name,
      arguments: block.arguments,
    },
  }));
  return {
    role: 'assistant',
    content: text,
    ...quirks.reasoningField !== null && reasoning.length > 0 ? { [quirks.reasoningField]: reasoning } : {},
    ...toolCalls.length > 0 ? { tool_calls: toolCalls } : {},
  };
}

/**
 * Serialize the conversation. `tool-result` blocks become standalone
 * `{role: 'tool'}` messages; a mixed user message contributes its text first
 * and its tool results as separate wire messages after.
 * @param messages - the harness conversation, in order.
 * @param quirks - resolved provider deviations.
 * @returns the wire messages; order preserved, each tool result expanded into its own entry.
 */
export function serializeMessages(messages: readonly Message[], quirks: ResolvedQuirks): WireMessage[] {
  const wire: WireMessage[] = [];
  for (const message of messages) {
    assertTextOnly(message.content);
    if (message.role === 'system') {
      wire.push({ role: 'system', content: flattenText(message.content) });
      continue;
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message, quirks));
      continue;
    }
    const toolResults = message.content.filter((block) => block.type === 'tool-result');
    const text = flattenText(message.content);
    if (text.length > 0 || toolResults.length === 0) wire.push({ role: 'user', content: text });
    for (const result of toolResults) {
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        content: flattenText(result.content) || '(no output)',
      });
    }
  }
  return wire;
}

/** Resolve one legal thinking/effort pair without exposing `off` as a wire effort. */
function resolveThinking(options: GenerateOptions, quirks: ResolvedQuirks): Record<string, unknown> {
  if (options.purpose === 'session-title') return {};
  const effort = options.reasoningEffort;
  if (effort === undefined || effort === 'off') {
    if (quirks.thinkingField) return { thinking: { type: 'disabled' } };
    return {};
  }
  if (effort === 'high' || effort === 'max') {
    if (quirks.thinkingField) {
      return { thinking: { type: 'enabled' }, reasoning_effort: effort };
    }
    if (quirks.reasoningEffortField) return { reasoning_effort: effort };
    throw new LlmError(`Provider does not support reasoning effort "${effort}"`, 'UNSUPPORTED_REASONING_EFFORT');
  }
  throw new LlmError(`Provider does not support reasoning effort "${effort}"`, 'UNSUPPORTED_REASONING_EFFORT');
}

/**
 * Build the full wire request. Always streaming; optional fields are omitted
 * rather than sent as null, so provider defaults apply.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param quirks - resolved provider deviations.
 * @returns the chat-completions request body.
 */
export function serializeRequest(options: GenerateOptions, quirks: ResolvedQuirks): Record<string, unknown> {
  const messages: WireMessage[] = [];
  if (options.system !== undefined) messages.push({ role: 'system', content: options.system });
  messages.push(...serializeMessages(options.messages, quirks));
  const tools = options.tools?.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      ...quirks.strictToolSchemas ? { strict: true } : {},
    },
  }));
  const thinking = resolveThinking(options, quirks);
  return {
    model: options.model,
    messages,
    stream: true,
    ...quirks.usage !== 'none' && quirks.streamOptions ? { stream_options: { include_usage: true } } : {},
    ...thinking,
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...options.temperature !== undefined ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { [quirks.maxTokensField]: options.maxTokens },
    ...options.stop !== undefined ? { stop: options.stop } : {},
  };
}
