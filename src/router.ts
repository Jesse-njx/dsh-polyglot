/**
 * `PolyglotRouterAdapter`: the meta-adapter that owns fallback. It registers
 * on `ctx.llm` under a virtual provider route and serves an ordered chain of
 * concrete OpenAI-compatible providers: on a fallback-eligible failure before
 * any content flowed, the failing provider is marked cooling-down
 * (exponential, honoring `Retry-After`) and the request is retried on the
 * next provider in the chain. Each attempt is recorded as a durable
 * `polyglot/served` session event.
 *
 * @module dsh-polyglot/router
 */
import { LlmAdapter, LlmError } from '@deepseek-ai/dsh-llm';
import type {
  FinishReason,
  GenerateOptions,
  LlmFailure,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  ResolvedRetryPolicy,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm';
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm';
import type { SessionId } from '@deepseek-ai/dsh-session';
import type { ActiveChain, CooldownConfig, ResolvedChainEntry, ServedRecord } from './types.ts';

/** Minimal logger surface (cordis logger satisfies it). */
export interface RouterLogger {
  warn(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
}

/** Constructor options for {@link PolyglotRouterAdapter}. */
export interface RouterAdapterOptions {
  /** The active chain snapshot; read per operation so a `/model` switch lands on the next request. */
  chain: () => ActiveChain;
  /** Dispatch one call to a concrete provider route (binds `ctx.llm.stream`). */
  dispatch: (options: GenerateOptions) => AsyncIterable<StreamChunk>;
  /** Per-provider cooldown behavior; read per failure so settings edits land live. */
  cooldown: () => CooldownConfig;
  /** Wall clock; injectable for tests. */
  now?: () => number;
  /** Durable per-attempt record sink; the session id of the call travels alongside. */
  onServed: (record: ServedRecord, sessionId: SessionId | undefined) => void;
  /** Optional logger. */
  logger?: RouterLogger;
}

/** Failure codes that justify trying the next provider in the chain. */
const FALLBACK_CODES = new Set([
  'RATE_LIMIT',
  'QUOTA',
  'SERVER',
  'TRANSPORT',
  'TIMEOUT',
  'STREAM_CLOSED',
  'MALFORMED_RESPONSE',
  'EMPTY_RESPONSE',
  'MISSING_CREDENTIAL',
  'NO_ADAPTER',
  'INVALID_CREDENTIAL',
  'AUTH',
]);

/** Whether one failure is eligible for provider fallback. */
export function isFallbackEligible(failure: LlmFailure): boolean {
  if (FALLBACK_CODES.has(failure.code)) return true;
  if (failure.code.startsWith('HTTP_')) {
    const status = Number(failure.code.slice('HTTP_'.length));
    return Number.isInteger(status) && status >= 500;
  }
  return false;
}

/** Extract a provider-requested delay from a failure, in milliseconds. */
function retryAfterOf(failure: LlmFailure): number | undefined {
  return failure.providerRetryAfterMs;
}

/** Bounded exponential per-provider cooldown with symmetric jitter. */
export class ProviderCooldown {
  readonly #config: () => CooldownConfig;
  readonly #now: () => number;
  readonly #state = new Map<string, { until: number; consecutive: number }>();

  constructor(config: () => CooldownConfig, now: () => number) {
    this.#config = config;
    this.#now = now;
  }

  /** Whether `provider` is currently cooling down and must be skipped. */
  cooling(provider: string): boolean {
    const entry = this.#state.get(provider);
    return entry !== undefined && entry.until > this.#now();
  }

  /** Record one fallback-eligible failure, growing the cooldown exponentially. */
  recordFailure(provider: string, failure?: LlmFailure): void {
    const previous = this.#state.get(provider);
    const consecutive = (previous?.consecutive ?? 0) + 1;
    const { baseMs, maxMs, factor, jitterRatio } = this.#config();
    const backoff = Math.min(maxMs, baseMs * Math.pow(factor, consecutive - 1));
    const honored = retryAfterOf(failure ?? {} as LlmFailure) ?? 0;
    const delay = Math.min(maxMs, Math.max(backoff, honored));
    const jitter = delay * jitterRatio * (Math.random() * 2 - 1);
    const until = this.#now() + Math.max(0, delay + jitter);
    this.#state.set(provider, { until, consecutive });
  }

  /** Record a successful call, clearing the provider's cooldown. */
  recordSuccess(provider: string): void {
    this.#state.delete(provider);
  }

  /** Snapshot for diagnostics: provider → seconds until cool. */
  status(now: number): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [provider, entry] of this.#state) {
      const remaining = entry.until - now;
      if (remaining > 0) out[provider] = Math.ceil(remaining / 1000);
    }
    return out;
  }
}

const OFF_REASONING_EFFORT = ReasoningEffortId('off');
const HIGH_REASONING_EFFORT = ReasoningEffortId('high');
const MAX_REASONING_EFFORT = ReasoningEffortId('max');
const REASONING_EFFORTS = [
  { id: OFF_REASONING_EFFORT, name: 'Off' },
  { id: HIGH_REASONING_EFFORT, name: 'High' },
  { id: MAX_REASONING_EFFORT, name: 'Max' },
] as const;

/** The outcome of one chain entry attempt. */
type EntryOutcome =
  | { kind: 'served'; finish: FinishReason; usage?: TokenUsage }
  | { kind: 'failed'; finish: FinishReason; usage?: TokenUsage; terminal: boolean };

/**
 * The router `LlmAdapter`. `stream()` walks the active chain, forwarding the
 * first attempt that completes. A fallback-eligible failure that arrives
 * before any content chunk was forwarded swaps to the next provider; a failure
 * after content flowed cannot be unwritten and is forwarded as-is.
 */
export class PolyglotRouterAdapter extends LlmAdapter {
  readonly #options: RouterAdapterOptions;
  readonly #cooldown: ProviderCooldown;

  constructor(options: RouterAdapterOptions) {
    super();
    this.#options = options;
    this.#cooldown = new ProviderCooldown(options.cooldown, options.now ?? Date.now);
  }
  /** The cooldown tracker, exposed for diagnostics. */
  get cooldown(): ProviderCooldown {
    return this.#cooldown;
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Polyglot' };
  }

  override providerRetryPolicy(_provider: string): ResolvedRetryPolicy | undefined {
    return undefined;
  }

  override listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    const chain = this.#options.chain();
    const seen = new Set<string>();
    const models: LlmModelInfo[] = [];
    for (const entry of chain.entries) {
      for (const model of entry.connection.catalog) {
        if (seen.has(model.id)) continue;
        seen.add(model.id);
        models.push({
          provider,
          id: model.id,
          name: model.name ?? model.id,
          ...model.description === undefined ? {} : { description: model.description },
          inputModalities: ['text'],
        });
      }
    }
    return Promise.resolve(models);
  }

  override resolveModel(provider: string, model: string, _signal?: AbortSignal): Promise<LlmResolvedModelInfo> {
    const chain = this.#options.chain();
    for (const entry of chain.entries) {
      const found = entry.connection.catalog.find((candidate) => candidate.id === model);
      if (found) return Promise.resolve(this.#resolveFrom(provider, model, entry, found));
    }
    const fallback = chain.entries[0];
    if (fallback === undefined) {
      throw new LlmError(`dsh-polyglot: chain "${chain.name}" has no configured providers`, 'NO_PROVIDER');
    }
    return Promise.resolve({
      provider,
      id: model,
      name: model,
      inputModalities: ['text'],
      context: { contextWindow: fallback.connection.defaultContextWindow },
      defaultMaxTokens: fallback.connection.maxTokens,
    });
  }

  #resolveFrom(
    provider: string,
    model: string,
    entry: ResolvedChainEntry,
    catalog: ResolvedChainEntry['connection']['catalog'][number],
  ): LlmResolvedModelInfo {
    const connection = entry.connection;
    const supportsReasoning = (catalog.reasoning ?? false)
      && (connection.quirks.thinkingField || connection.quirks.reasoningEffortField);
    return {
      provider,
      id: model,
      name: catalog.name ?? model,
      ...catalog.description === undefined ? {} : { description: catalog.description },
      inputModalities: ['text'],
      context: { contextWindow: catalog.contextWindow ?? connection.defaultContextWindow },
      defaultMaxTokens: catalog.maxTokens ?? connection.maxTokens,
      ...supportsReasoning ? { reasoning: { efforts: REASONING_EFFORTS, defaultEffort: OFF_REASONING_EFFORT } } : {},
    };
  }

  override async *stream(options: GenerateOptions): AsyncGenerator<StreamChunk> {
    const chain = this.#options.chain();
    const entries = chain.entries;
    if (entries.length === 0) {
      yield {
        type: 'finish',
        reason: {
          kind: 'error',
          failure: { message: `dsh-polyglot: chain "${chain.name}" has no configured providers`, code: 'NO_PROVIDER' },
        },
      };
      return;
    }
    let attempt = 0;
    let lastFailure: FinishReason | undefined;
    for (const entry of entries) {
      attempt += 1;
      if (this.#cooldown.cooling(entry.provider)) {
        this.#options.logger?.info(`dsh-polyglot: ${entry.provider} cooling down — skipping`);
        continue;
      }
      const outcome = yield* this.#tryEntry(options, entry, attempt);
      if (outcome.kind === 'served') {
        if (outcome.usage) yield { type: 'usage', usage: outcome.usage };
        yield { type: 'finish', reason: outcome.finish };
        return;
      }
      if (outcome.terminal) {
        // Content already flowed (or caller aborted): cannot unwrite, and the
        // chain must not continue after a finish was delivered.
        if (outcome.usage) yield { type: 'usage', usage: outcome.usage };
        yield { type: 'finish', reason: outcome.finish };
        return;
      }
      lastFailure = outcome.finish;
    }
    const reason = lastFailure ?? {
      kind: 'error' as const,
      failure: {
        message: `dsh-polyglot: every provider in chain "${chain.name}" failed or is cooling down`,
        code: 'NO_PROVIDER',
      },
    };
    yield { type: 'finish', reason };
  }

  /**
   * Attempt one chain entry. Content chunks are forwarded live; the usage and
   * finish chunks are held until the outcome is known, so a pre-content
   * failure can be swapped to the next provider without leaking a finish-less
   * usage record. Returns `served` on a normal completion, `failed` with
   * `terminal: false` on a fallback-eligible pre-content failure, and
   * `failed` with `terminal: true` once content flowed or the caller aborted.
   */
  async *#tryEntry(
    options: GenerateOptions,
    entry: ResolvedChainEntry,
    attempt: number,
  ): AsyncGenerator<StreamChunk, EntryOutcome> {
    const started = this.#options.now?.() ?? Date.now();
    let forwardedContent = false;
    let usage: TokenUsage | undefined;
    try {
      const stream = this.#options.dispatch({
        ...options,
        provider: entry.provider,
        model: entry.model,
      });
      for await (const chunk of stream) {
        if (chunk.type === 'usage') {
          usage = chunk.usage;
          continue;
        }
        if (chunk.type === 'finish') {
          const finish = chunk.reason;
          const ok = finish.kind !== 'error' && finish.kind !== 'aborted';
          if (ok) {
            this.#cooldown.recordSuccess(entry.provider);
            this.#record(options, entry, attempt, started, {
              status: 'ok',
              finishKind: finish.kind,
              usage: usage === undefined ? undefined : {
                inputTokens: usage.inputTokens,
                outputTokens: usage.outputTokens,
                ...usage.cacheReadTokens === undefined ? {} : { cacheReadTokens: usage.cacheReadTokens },
                ...usage.cacheWriteTokens === undefined ? {} : { cacheWriteTokens: usage.cacheWriteTokens },
                ...usage.reasoningTokens === undefined ? {} : { reasoningTokens: usage.reasoningTokens },
              },
            });
            return { kind: 'served', finish, usage };
          }
          const failure = finish.kind === 'aborted'
            ? { message: 'request aborted', code: 'ABORTED' }
            : finish.failure;
          this.#record(options, entry, attempt, started, {
            status: 'failed',
            failureCode: failure.code,
            failureMessage: failure.message,
          });
          const eligible = isFallbackEligible(failure) && !forwardedContent;
          if (eligible) {
            this.#cooldown.recordFailure(entry.provider, failure);
            this.#options.logger?.warn(
              `dsh-polyglot: ${entry.provider} failed (${failure.code}) before content — falling back`,
            );
            return { kind: 'failed', finish, terminal: false };
          }
          return { kind: 'failed', finish, usage, terminal: true };
        }
        if (isContentChunk(chunk)) forwardedContent = true;
        yield chunk;
      }
      // Stream ended without a finish chunk — protocol violation from the
      // underlying dispatch (itself an error finish we already normalized).
      const failure: LlmFailure = {
        message: `dsh-polyglot: ${entry.provider} stream ended without a finish chunk`,
        code: 'STREAM_CLOSED',
      };
      this.#record(options, entry, attempt, started, { status: 'failed', failureCode: failure.code, failureMessage: failure.message });
      this.#cooldown.recordFailure(entry.provider, failure);
      return { kind: 'failed', finish: { kind: 'error', failure }, terminal: false };
    } catch (error) {
      const failure = normalizeThrown(error);
      this.#record(options, entry, attempt, started, {
        status: 'failed',
        failureCode: failure.code,
        failureMessage: failure.message,
      });
      const eligible = isFallbackEligible(failure) && !forwardedContent && failure.code !== 'ABORTED';
      if (eligible) {
        this.#cooldown.recordFailure(entry.provider, failure);
        this.#options.logger?.warn(
          `dsh-polyglot: ${entry.provider} threw (${failure.code}) before content — falling back`,
        );
        return { kind: 'failed', finish: { kind: 'error', failure }, terminal: false };
      }
      const aborted = options.signal?.aborted || failure.code === 'ABORTED';
      return {
        kind: 'failed',
        finish: aborted ? { kind: 'aborted', failure } : { kind: 'error', failure },
        terminal: true,
      };
    }
  }

  #record(
    options: GenerateOptions,
    entry: ResolvedChainEntry,
    attempt: number,
    started: number,
    partial: Pick<ServedRecord, 'status'> & Partial<ServedRecord>,
  ): void {
    const latencyMs = (this.#options.now?.() ?? Date.now()) - started;
    this.#options.onServed({
      chain: this.#options.chain().name,
      provider: entry.provider,
      model: entry.model,
      preset: entry.preset,
      attempt,
      latencyMs,
      ...partial,
    } as ServedRecord, options.sessionId);
  }
}

/** Whether a chunk carries visible model output (content that cannot be unwritten). */
function isContentChunk(chunk: StreamChunk): boolean {
  switch (chunk.type) {
    case 'block-start':
    case 'text-delta':
    case 'reasoning-delta':
    case 'tool-call-delta':
    case 'block-end':
      return true;
    default:
      return false;
  }
}

/** Normalize a thrown value into a stable failure record. */
export function normalizeThrown(error: unknown): LlmFailure {
  if (error instanceof LlmError) {
    return {
      message: error.message,
      code: error.code,
      ...error.failure.status === undefined ? {} : { status: error.failure.status },
      ...error.failure.providerRetryAfterMs === undefined ? {} : { providerRetryAfterMs: error.failure.providerRetryAfterMs },
      ...error.failure.requestId === undefined ? {} : { requestId: error.failure.requestId },
    };
  }
  if (error instanceof Error) {
    return { message: error.message, code: 'TRANSPORT' };
  }
  return { message: String(error), code: 'TRANSPORT' };
}
