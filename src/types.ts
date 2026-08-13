/**
 * Shared dsh-polyglot types: provider connection facts, declarative quirks,
 * chain configuration, and the durable per-call record.
 *
 * @module dsh-polyglot/types
 */
import type { CredentialRef } from '@deepseek-ai/dsh-credentials';
import type { ResolvedRetryPolicy } from '@deepseek-ai/dsh-llm';

/**
 * Declarative per-provider deviations. Providers stay data — a preset or a
 * chain entry tweaks these flags instead of forking the adapter.
 */
export interface QuirksConfig {
  /**
   * Wire field inside each `delta` that carries reasoning/thinking text.
   * `'reasoning_content'` is the DeepSeek spelling; some OpenAI-compatible
   * hosts use `'reasoning'`. `null` disables reasoning handling entirely —
   * the provider exposes no reasoning (or rejects the field), so reasoning
   * blocks are dropped and no reasoning metadata is advertised.
   */
  reasoningField?: string | null;
  /** Wire field for the output-token cap (OpenAI `max_tokens` vs newer `max_completion_tokens`). */
  maxTokensField?: 'max_tokens' | 'max_completion_tokens';
  /**
   * Usage accounting shape: `'standard'` — `prompt_tokens`/`completion_tokens`
   * are disjoint; `'deepseek'` — `prompt_tokens` INCLUDES cache hits, so they
   * are subtracted out to keep harness counts disjoint; `'none'` — the
   * provider reports no usage (no usage chunk is emitted).
   */
  usage?: 'standard' | 'deepseek' | 'none';
  /** Whether the endpoint accepts `stream_options: { include_usage: true }`. */
  streamOptions?: boolean;
  /** Add `strict: true` to tool function schemas (some hosts require it). */
  strictToolSchemas?: boolean;
  /** Send `thinking: { type: 'enabled' | 'disabled' }` for reasoning efforts (DeepSeek spelling). */
  thinkingField?: boolean;
  /** Send `reasoning_effort` for high/max efforts (OpenAI spelling). */
  reasoningEffortField?: boolean;
}

/** Fully resolved quirks; every field materialized. */
export interface ResolvedQuirks {
  reasoningField: string | null;
  maxTokensField: 'max_tokens' | 'max_completion_tokens';
  usage: 'standard' | 'deepseek' | 'none';
  streamOptions: boolean;
  strictToolSchemas: boolean;
  thinkingField: boolean;
  reasoningEffortField: boolean;
}

/** One advisory catalog model; requests remain unrestricted by it. */
export interface CatalogModel {
  /** Wire model id accepted by the configured endpoint. */
  id: string;
  /** Selector label; defaults to {@link id}. */
  name?: string;
  /** Optional selector detail for similar model variants. */
  description?: string;
  /** Known combined request/response context capacity. */
  contextWindow?: number;
  /** Per-request output cap; omission falls back to the connection default. */
  maxTokens?: number;
  /** Whether the provider exposes reasoning efforts for this model. */
  reasoning?: boolean;
}

/**
 * Validated connection facts for one provider route. The plugin's
 * `resolveProviderConfig` is the one explicit resolve step producing this
 * shape; the adapter trusts it and re-reads it per operation, so a
 * configuration change reaches the next request without re-registration.
 */
export interface ProviderConfig {
  /** Preset id this provider was built from. */
  preset: string;
  /** Registered provider route on `ctx.llm`. */
  provider: string;
  /** Human-readable provider name for selectors and diagnostics. */
  displayName: string;
  /** Endpoint base; `/chat/completions` is appended. */
  baseURL: string;
  /** Credential reference resolved per request. */
  apiKeyEnv: CredentialRef;
  /** How the endpoint authenticates. */
  authKind: 'key' | 'oauth' | 'none';
  /** Extra per-provider headers (OpenRouter identity headers etc.). */
  headers: Record<string, string>;
  /** Declarative provider deviations. */
  quirks: ResolvedQuirks;
  /** Advisory models exposed to discovery consumers. */
  catalog: readonly CatalogModel[];
  /** Default per-request output cap; explicit request values win. */
  maxTokens: number;
  /** Positive context capacity used when the selected model has no exact value. */
  defaultContextWindow: number;
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs: number;
  /** Provider-owned model-request retry policy, already resolved. */
  retryPolicy: ResolvedRetryPolicy;
}

/** A chain entry as configured: a preset plus optional per-entry overrides. */
export interface ChainEntryConfig {
  /** Preset id from the registry (presets/*.json). */
  preset: string;
  /** Override the registered provider route (defaults to the preset id). */
  provider?: string;
  /** Override the wire model (defaults to the preset's first model). */
  model?: string;
  /** Override the endpoint base (defaults to the preset's baseUrl). */
  baseUrl?: string;
  /** Override the credential reference (defaults to the preset's env name). */
  apiKeyEnv?: string;
  /** Extra per-provider headers merged over the preset's. */
  headers?: Record<string, string>;
  /** Per-entry quirks merged over the preset's. */
  quirks?: Partial<QuirksConfig>;
}

/** Bounded exponential cooldown with symmetric jitter, per failed provider. */
export interface CooldownConfig {
  /** Initial per-provider cooldown in milliseconds. */
  baseMs: number;
  /** Maximum per-provider cooldown in milliseconds. */
  maxMs: number;
  /** Exponential growth factor per consecutive failure. */
  factor: number;
  /** Symmetric random multiplier range around one. */
  jitterRatio: number;
}

/** The plugin configuration resolved from the entry + user settings section. */
export interface PolyglotConfig {
  /** Ordered fallback chains by name. */
  chains: Record<string, ChainEntryConfig[]>;
  /** Chain used when none is selected explicitly. */
  defaultChain: string;
  /** Registered route of the router meta-adapter (a virtual provider). */
  virtualProvider: string;
  /** Per-provider cooldown behavior after a fallback-eligible failure. */
  cooldown?: CooldownConfig;
  /** Default per-request output cap materialized for catalog models. */
  maxTokens: number;
  /** Positive context capacity used when a model has no exact value. */
  defaultContextWindow: number;
  /** Maximum provider idle time while one stream read is outstanding. */
  streamIdleTimeoutMs: number;
}

/** JSON-safe usage projection for the durable per-call record. */
export interface ServedUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
  reasoningTokens?: number;
}

/**
 * One provider attempt in a chain, recorded as a durable `polyglot/served`
 * session event so the usage command and any replay can tally per provider.
 */
export interface ServedRecord {
  /** Chain name that served (or attempted). */
  chain: string;
  /** Registered provider route that served. */
  provider: string;
  /** Wire model id that served. */
  model: string;
  /** Preset id the provider was built from. */
  preset: string;
  /** 1-based position in the chain that served. */
  attempt: number;
  /** Whether the attempt completed or failed. */
  status: 'ok' | 'failed';
  /** Finish kind for successful attempts (`stop`, `tool-calls`, `max-tokens`). */
  finishKind?: string;
  /** Stable failure code for failed attempts. */
  failureCode?: string;
  /** Human-readable failure summary. */
  failureMessage?: string;
  /** Token accounting reported by the provider, when any. */
  usage?: ServedUsage;
  /** Wall-clock duration of the attempt in milliseconds. */
  latencyMs: number;
}

/** The active chain snapshot the router serves from. */
export interface ActiveChain {
  name: string;
  /** Fully resolved entries in fallback order. */
  entries: readonly ResolvedChainEntry[];
}

/** A chain entry resolved against the preset registry and connection options. */
export interface ResolvedChainEntry {
  /** Preset id. */
  preset: string;
  /** Registered provider route. */
  provider: string;
  /** Wire model id used for this entry. */
  model: string;
  /** Connection facts for the adapter instance serving this entry. */
  connection: ProviderConfig;
  /** Optional preset pricing for cost estimation in the usage command. */
  pricing?: PresetPricing;
}

/** Pricing metadata carried by a preset, used for cost tallies. */
export interface PresetPricing {
  inputPerMTokens?: number;
  outputPerMTokens?: number;
  cacheReadPerMTokens?: number;
  cacheWritePerMTokens?: number;
  currency: string;
}
