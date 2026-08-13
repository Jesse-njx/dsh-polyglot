/**
 * Translate OpenAI-compatible SSE payloads into the harness `StreamChunk`
 * protocol, with one stateful harness block per content, reasoning, or tool
 * call index. Finish reason and the latest usage are deferred until `[DONE]`,
 * covering both finish-attached and trailing usage-only shapes while ensuring
 * no chunk follows `finish`.
 *
 * @module dsh-polyglot/translate
 */
import { CallId, EMPTY_RESPONSE_CODE, LlmError } from '@deepseek-ai/dsh-llm';
import type { FinishReason, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm';
import type { ResolvedQuirks } from './types.ts';

/** Wire usage object from a chat-completions chunk. */
interface WireUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
  prompt_cache_hit_tokens?: number;
  completion_tokens_details?: { reasoning_tokens?: number };
}

/** Map the wire finish_reason vocabulary to the harness FinishReason. */
function mapFinishReason(reason: string): FinishReason {
  switch (reason) {
    case 'stop': return { kind: 'stop' };
    case 'tool_calls': return { kind: 'tool-calls' };
    case 'length': return { kind: 'max-tokens' };
    default: return {
      kind: 'error',
      failure: {
        message: `model stopped: ${reason}`,
        code: reason.toUpperCase(),
      },
    };
  }
}

/**
 * Map wire usage fields to disjoint harness counts. In `'deepseek'` mode the
 * wire `prompt_tokens` INCLUDES cache hits, so cache reads are subtracted out;
 * `'standard'` mode treats the wire counts as disjoint already. Cache/reasoning
 * fields are present only when the wire reported them.
 */
export function mapUsage(usage: WireUsage, mode: ResolvedQuirks['usage']): TokenUsage | undefined {
  if (mode === 'none') return undefined;
  const cacheRead = mode === 'deepseek'
    ? usage.prompt_cache_hit_tokens
    : usage.prompt_tokens_details?.cached_tokens;
  const reasoning = usage.completion_tokens_details?.reasoning_tokens;
  const inputTokens = usage.prompt_tokens ?? 0;
  const outputTokens = usage.completion_tokens ?? 0;
  if (inputTokens === 0 && outputTokens === 0 && cacheRead === undefined && reasoning === undefined) return undefined;
  return {
    inputTokens: inputTokens - (cacheRead ?? 0),
    outputTokens,
    ...cacheRead !== undefined ? { cacheReadTokens: cacheRead } : {},
    ...reasoning !== undefined ? { reasoningTokens: reasoning } : {},
  };
}

/** Assemble the final ContentBlock for one open block. */
function closeBlock(block: OpenBlock): Extract<StreamChunk, { type: 'block-end' }>['block'] {
  switch (block.kind) {
    case 'text': return { type: 'text', text: block.text };
    case 'reasoning': return { type: 'reasoning', text: block.text };
    case 'tool-call': return {
      type: 'tool-call',
      id: CallId(block.callId ?? ''),
      name: block.name ?? '',
      arguments: block.text,
    };
  }
}

interface OpenBlock {
  index: number;
  kind: 'text' | 'reasoning' | 'tool-call';
  text: string;
  callId?: string;
  name?: string;
}

/**
 * Consume SSE data payloads (ending with `[DONE]`) and yield StreamChunks.
 * Malformed JSON payloads abort the stream with `MALFORMED_RESPONSE`.
 * @param payloads - SSE data payloads from {@link parseSse}, `[DONE]`-terminated.
 * @param quirks - resolved provider deviations (reasoning field name, usage mode).
 * @returns deltas as they arrive; `block-end`s, `usage`, and `finish` are all deferred to the `[DONE]` sentinel.
 *   A `stop` (or absent) finish with no opened blocks is a degenerate provider completion and maps to an
 *   `EMPTY_RESPONSE` error finish instead of a successful empty message.
 */
export async function* translate(payloads: AsyncIterable<string>, quirks: ResolvedQuirks): AsyncGenerator<StreamChunk> {
  let nextIndex = 0;
  let textBlock: OpenBlock | undefined;
  let reasoningBlock: OpenBlock | undefined;
  const toolBlocks = new Map<number, OpenBlock>();
  const order: OpenBlock[] = [];
  let pendingFinish: FinishReason | undefined;
  let pendingUsage: TokenUsage | undefined;

  function open(kind: OpenBlock['kind']): OpenBlock {
    const block: OpenBlock = { index: nextIndex++, kind, text: '' };
    order.push(block);
    return block;
  }

  for await (const payload of payloads) {
    if (payload === '[DONE]') {
      for (const block of order) {
        yield { type: 'block-end', index: block.index, block: closeBlock(block) };
      }
      if (pendingUsage) yield { type: 'usage', usage: pendingUsage };
      const reason = pendingFinish ?? { kind: 'stop' };
      yield {
        type: 'finish',
        reason: reason.kind === 'stop' && order.length === 0 ? {
          kind: 'error',
          failure: {
            message: 'model returned a completed response with no content',
            code: EMPTY_RESPONSE_CODE,
          },
        } : reason,
      };
      return;
    }
    let chunk: Record<string, unknown>;
    try {
      chunk = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      throw new LlmError(`malformed SSE payload: ${payload.slice(0, 120)}`, 'MALFORMED_RESPONSE');
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices as Record<string, unknown>[] : [];
    for (const choice of choices) {
      const delta = (choice.delta ?? {}) as Record<string, unknown>;
      const reasoningField = quirks.reasoningField;
      const reasoning = reasoningField !== null && typeof delta[reasoningField] === 'string' ? delta[reasoningField] as string : undefined;
      if (typeof reasoning === 'string' && reasoning.length > 0) {
        if (!reasoningBlock) {
          reasoningBlock = open('reasoning');
          yield { type: 'block-start', index: reasoningBlock.index, blockType: 'reasoning' };
        }
        reasoningBlock.text += reasoning;
        yield { type: 'reasoning-delta', index: reasoningBlock.index, text: reasoning };
      }
      const content = typeof delta.content === 'string' ? delta.content : undefined;
      if (typeof content === 'string' && content.length > 0) {
        if (!textBlock) {
          textBlock = open('text');
          yield { type: 'block-start', index: textBlock.index, blockType: 'text' };
        }
        textBlock.text += content;
        yield { type: 'text-delta', index: textBlock.index, text: content };
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls as Record<string, unknown>[] : [];
      for (const call of calls) {
        const callIndex = typeof call.index === 'number' ? call.index : 0;
        let block = toolBlocks.get(callIndex);
        if (!block) {
          block = open('tool-call');
          toolBlocks.set(callIndex, block);
          yield { type: 'block-start', index: block.index, blockType: 'tool-call' };
        }
        if (typeof call.id === 'string') block.callId = call.id;
        const fn = (call.function ?? {}) as Record<string, unknown>;
        if (typeof fn.name === 'string') block.name = fn.name;
        const fragment = typeof fn.arguments === 'string' ? fn.arguments : '';
        block.text += fragment;
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: CallId(block.callId ?? ''),
          ...block.name !== undefined ? { name: block.name } : {},
          argumentsDelta: fragment,
        };
      }
      if (typeof choice.finish_reason === 'string') pendingFinish = mapFinishReason(choice.finish_reason);
    }
    if (chunk.usage !== undefined && typeof chunk.usage === 'object') {
      const mapped = mapUsage(chunk.usage as WireUsage, quirks.usage);
      if (mapped) pendingUsage = mapped;
    }
  }
  throw new LlmError('SSE payload stream ended without [DONE]', 'STREAM_CLOSED');
}
