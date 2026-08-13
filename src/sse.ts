/**
 * SSE transport parsing for OpenAI-compatible streaming. Same framing contract
 * the reference adapter (llm-deepseek) was verified against: each event's data
 * payload is yielded verbatim, `[DONE]` terminates, and a stream ending
 * without the sentinel is a protocol failure.
 *
 * @module dsh-polyglot/sse
 */
import { LlmError } from '@deepseek-ai/dsh-llm';
import { EventSourceParserStream } from 'eventsource-parser/stream';

/**
 * Parse an SSE byte stream into data payloads. Yields `[DONE]` as the final
 * value and returns; throws `LlmError('STREAM_CLOSED')` when the stream ends
 * without it (truncated response — the model call cannot be trusted).
 * @param stream - raw SSE bytes; reads may split anywhere, including mid-UTF-8 sequence.
 * @param onComment - optional transport-activity callback; comments never enter the yielded payload stream.
 * @returns each event's data payload in arrival order, the `[DONE]` sentinel last.
 */
export async function* parseSse(stream: ReadableStream<Uint8Array>, onComment?: () => void): AsyncGenerator<string> {
  const events = stream
    .pipeThrough(new TextDecoderStream())
    .pipeThrough(new EventSourceParserStream({ onComment }));
  for await (const { data } of events) {
    yield data;
    if (data === '[DONE]') return;
  }
  throw new LlmError('SSE stream ended without [DONE]', 'STREAM_CLOSED');
}
