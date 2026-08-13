/**
 * `OpenAiCompatAdapter`: fetch + SSE against any OpenAI-compatible
 * chat-completions endpoint, emitting harness StreamChunks. The adapter is
 * transport-only: connection facts arrive through a thunk resolved once per
 * operation and the bearer token through a per-request resolver, so the
 * registering plugin owns validation, layering, and credential policy.
 * Provider deviations ride the declarative `quirks` flags, never code paths.
 *
 * @module dsh-polyglot/adapter
 */
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  EMPTY_RESPONSE_CODE,
  LlmAdapter,
  LlmError,
  ProviderRequestId,
  QUOTA_EXCEEDED_CODE,
  ReasoningEffortId,
  attributionHeaders,
  isContextWindowExceededError,
  isQuotaExceededError,
} from '@deepseek-ai/dsh-llm';
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
} from '@deepseek-ai/dsh-llm';
import type { AnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id';
import { idleWatchdog, timeoutOf } from '@deepseek-ai/dsh-timeout';
import { serializeRequest } from './serialize.ts';
import { parseSse } from './sse.ts';
import { translate } from './translate.ts';
import type { ProviderConfig } from './types.ts';

/** Constructor options for {@link OpenAiCompatAdapter}: the operation-local resolution hooks the plugin owns. */
export interface OpenAiCompatAdapterOptions {
  /** Current validated connection facts; called once per operation. */
  options: () => ProviderConfig;
  /**
   * Resolve the bearer token for the connection facts of one request. The
   * snapshot is passed in — never re-read — so the key can only ever come
   * from the same resolution as the endpoint it is sent to. Throws `LlmError`
   * `MISSING_CREDENTIAL` when no key is available anywhere.
   */
  resolveApiKey: (connection: ProviderConfig) => Promise<string>;
  /** Resolve the harness-home anonymous id shared with telemetry and feedback. */
  resolveUserId: () => AnonymousUserId;
}

const STREAM_IDLE_TIMEOUT_CODE = 'LLM_STREAM_IDLE_TIMEOUT';

const OFF_REASONING_EFFORT = ReasoningEffortId('off');
const HIGH_REASONING_EFFORT = ReasoningEffortId('high');
const MAX_REASONING_EFFORT = ReasoningEffortId('max');
const REASONING_EFFORTS = [
  { id: OFF_REASONING_EFFORT, name: 'Off' },
  { id: HIGH_REASONING_EFFORT, name: 'High' },
  { id: MAX_REASONING_EFFORT, name: 'Max' },
] as const;

function modelInfo(provider: string, model: ProviderConfig['catalog'][number]): LlmModelInfo {
  return {
    provider,
    id: model.id,
    name: model.name ?? model.id,
    ...model.description === undefined ? {} : { description: model.description },
    inputModalities: ['text'],
  };
}

function providerRetryAfterMs(value: string | null): number | undefined {
  if (value === null) return undefined;
  if (/^\d+$/.test(value)) {
    const delay = Number(value) * 1000;
    return Number.isFinite(delay) && delay > 0 ? delay : undefined;
  }
  const delay = Date.parse(value) - Date.now();
  return Number.isFinite(delay) && delay > 0 ? delay : undefined;
}

function requestId(headers: Headers): ProviderRequestId | undefined {
  const value = headers.get('x-request-id') ?? headers.get('x-deepseek-request-id');
  return value === null || value.length === 0 ? undefined : ProviderRequestId(value);
}

/** Map an HTTP status to a stable LlmError code. */
export function httpErrorCode(status: number, error?: { code?: string; type?: string; message?: string }): string {
  if (status === 401 || status === 403) return 'AUTH';
  const detail = [error?.code, error?.type, error?.message].filter(Boolean).join(' ');
  if (isQuotaExceededError(detail)) return QUOTA_EXCEEDED_CODE;
  if (status === 429) return 'RATE_LIMIT';
  if (status === 400) {
    if (isContextWindowExceededError(detail)) return CONTEXT_WINDOW_EXCEEDED_CODE;
    return 'INVALID_REQUEST';
  }
  if (status >= 500) return 'SERVER';
  return `HTTP_${status}`;
}

/**
 * The generic OpenAI-compatible `LlmAdapter`. One instance serves every model
 * name it was registered under (the harness model name IS the wire model name).
 *
 * One stable signal reaches both initial fetch and body reads. Caller aborts
 * map to `ABORTED`; the configured per-read idle watchdog maps to `TIMEOUT`.
 */
export class OpenAiCompatAdapter extends LlmAdapter {
  readonly #config: OpenAiCompatAdapterOptions;

  constructor(config: OpenAiCompatAdapterOptions) {
    super();
    this.#config = config;
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: this.#config.options().displayName };
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy {
    return this.#config.options().retryPolicy;
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return Promise.resolve(this.#config.options().catalog.map((model) => modelInfo(provider, model)));
  }

  override resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const connection = this.#config.options();
    const configured = connection.catalog.find((entry) => entry.id === model);
    const contextWindow = configured?.contextWindow ?? connection.defaultContextWindow;
    const supportsReasoning = (configured?.reasoning ?? false)
      && (connection.quirks.thinkingField || connection.quirks.reasoningEffortField);
    return Promise.resolve({
      ...configured === undefined ? { provider, id: model, name: model, inputModalities: ['text'] as const } : modelInfo(provider, configured),
      context: { contextWindow },
      defaultMaxTokens: configured?.maxTokens ?? connection.maxTokens,
      ...supportsReasoning ? { reasoning: { efforts: REASONING_EFFORTS, defaultEffort: OFF_REASONING_EFFORT } } : {},
    });
  }

  override async *stream(options: GenerateOptions): AsyncGenerator<StreamChunk> {
    const connection = this.#config.options();
    const apiKey = connection.authKind === 'none' ? '' : await this.#config.resolveApiKey(connection);
    const userId = this.#config.resolveUserId();
    const consumer = new AbortController();
    const watchdog = idleWatchdog(
      options.signal === undefined ? consumer.signal : AbortSignal.any([options.signal, consumer.signal]),
      connection.streamIdleTimeoutMs,
      STREAM_IDLE_TIMEOUT_CODE,
    );
    const iterator = this.request(options, watchdog.signal, connection, apiKey, userId, () => {
      watchdog.pulse();
    })[Symbol.asyncIterator]();
    let exhausted = false;
    try {
      while (true) {
        const result = await watchdog.next(iterator);
        if (result.done) {
          exhausted = true;
          return;
        }
        yield result.value;
      }
    } catch (error) {
      if (timeoutOf(watchdog.signal, STREAM_IDLE_TIMEOUT_CODE) !== undefined) {
        throw new LlmError(`Provider stream idle timeout after ${connection.streamIdleTimeoutMs}ms`, 'TIMEOUT', { cause: error });
      }
      if (options.signal?.aborted) throw new LlmError('Request aborted by caller', 'ABORTED', { cause: error });
      if (error instanceof LlmError) throw error;
      throw new LlmError(`Provider API stream from ${connection.baseURL} failed`, 'TRANSPORT', { cause: error });
    } finally {
      consumer.abort('OpenAiCompat stream consumer stopped');
      if (!exhausted && iterator.return !== undefined) {
        try {
          await iterator.return(undefined);
        } catch {
          // transport teardown after the consumer stopped — nothing to report
        }
      }
      watchdog[Symbol.dispose]();
    }
  }

  async *request(
    options: GenerateOptions,
    signal: AbortSignal,
    connection: ProviderConfig,
    apiKey: string,
    userId: AnonymousUserId,
    onComment: () => void,
  ): AsyncGenerator<StreamChunk> {
    const body = serializeRequest(options, connection.quirks);
    const payload = JSON.stringify(body);
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      'accept': 'text/event-stream',
      ...attributionHeaders(),
      'x-deepseek-harness-user-id': String(userId),
      ...connection.headers,
      ...options.sessionId !== undefined ? { 'x-deepseek-harness-session-id': String(options.sessionId) } : {},
      ...options.purpose === 'compaction' ? { 'x-deepseek-harness-compact': '1' } : {},
    };
    if (connection.authKind !== 'none') headers['authorization'] = `Bearer ${apiKey}`;
    let response: Response;
    try {
      response = await fetch(`${connection.baseURL}/chat/completions`, {
        method: 'POST',
        headers,
        body: payload,
        signal,
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new LlmError(`Provider API request to ${connection.baseURL} failed`, 'TRANSPORT', { cause: error });
    }
    if (!response.ok) {
      let message = `Provider API error (HTTP ${response.status})`;
      let providerError: { code?: string; type?: string; message?: string } | undefined;
      try {
        const parsed = (await response.json()) as { error?: { code?: string; type?: string; message?: string } };
        providerError = parsed.error;
        if (providerError?.message) message = providerError.message;
      } catch {
        // non-JSON error body — keep the status line
      }
      const delay = providerRetryAfterMs(response.headers.get('retry-after'));
      const id = requestId(response.headers);
      throw new LlmError(message, httpErrorCode(response.status, providerError), {
        status: response.status,
        ...delay === undefined ? {} : { providerRetryAfterMs: delay },
        ...id === undefined ? {} : { requestId: id },
      });
    }
    if (!response.body) throw new LlmError('Provider API returned no response body', EMPTY_RESPONSE_CODE);
    yield* translate(parseSse(response.body, onComment), connection.quirks);
  }
}
