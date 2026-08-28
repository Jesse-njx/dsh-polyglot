/**
 * dsh-polyglot plugin entry: registers one generic OpenAI-compatible adapter
 * per configured provider route and the router meta-adapter that serves
 * fallback chains. Connection facts resolve per request instead of freezing at
 * load: the plugin layers its `cordis.yml` entry config under the optional
 * `polyglot` user-settings section and resolves API keys through the optional
 * credential seam, so a changed base URL, catalog, key, or chain reaches the
 * very next request without restarting anything.
 *
 * @module @dsh-polyglot/bundle
 */
import { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
import { assertUsableApiKey, LlmError } from '@deepseek-ai/dsh-llm';
import type { LlmConfigurableProvider } from '@deepseek-ai/dsh-llm';
import { credentialRef } from '@deepseek-ai/dsh-credentials';
import { launchEnvironmentOf } from '@deepseek-ai/dsh-launch-environment';
import { deepEqualJson, installSettingsSection, settingsNamespace } from '@deepseek-ai/dsh-settings';
import { getOrCreateAnonymousUserId } from '@deepseek-ai/dsh-anonymous-user-id';
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout';
import { OpenAiCompatAdapter } from './adapter.ts';
import { loadPreset } from './preset.ts';
import { resolveQuirks } from './quirk.ts';
import { PolyglotRouterAdapter } from './router.ts';
import type { ResolvedChainEntry } from './types.ts';
import type { ChainEntryConfig, PolyglotConfig, ProviderConfig, ServedRecord } from './types.ts';
import { appendServed } from './session-events.ts';
import { registerPolyglotCommands } from './commands.ts';
import type { PolyglotCommandDeps } from './commands.ts';

export const name = 'polyglot';
// `commands` is required: `registerPolyglotCommands` touches the `ctx.commands`
// proxy accessor, which cordis rejects unless the service is declared here.
export const inject = ['llm', 'commands'] as const;

const NS = settingsNamespace('polyglot');

/** Default combined request/response context capacity. */
export const DEFAULT_CONTEXT_WINDOW = 1000000;
/** Default per-request output-token cap. */
export const DEFAULT_MAX_TOKENS = 256000;
/** Default maximum idle interval while an adapter stream read is outstanding. */
export const DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300000;
/** Default per-provider cooldown after a fallback-eligible failure. */
export const DEFAULT_COOLDOWN_BASE_MS = 30000;
/** Default maximum per-provider cooldown. */
export const DEFAULT_COOLDOWN_MAX_MS = 900000;
/** Default cooldown growth factor per consecutive failure. */
export const DEFAULT_COOLDOWN_FACTOR = 2;
/** Default cooldown jitter ratio. */
export const DEFAULT_COOLDOWN_JITTER_RATIO = 0.1;

const quirkSchema = z.object({
  reasoningField: z.union([z.string(), z.const(null)]),
  maxTokensField: z.union(['max_tokens', 'max_completion_tokens']),
  usage: z.union(['standard', 'deepseek', 'none']),
  streamOptions: z.boolean(),
  strictToolSchemas: z.boolean(),
  thinkingField: z.boolean(),
  reasoningEffortField: z.boolean(),
});

const entrySchema = z.object({
  preset: z.string().required(),
  provider: z.string(),
  model: z.string(),
  baseUrl: z.string(),
  apiKeyEnv: z.string().role('credential-ref'),
  headers: z.dict(z.string()),
  quirks: quirkSchema,
});

/** Raw input shape of the plugin configuration. */
export interface ConfigInput {
  chains: Record<string, ChainEntryConfig[]>;
  defaultChain?: string;
  virtualProvider?: string;
  cooldown?: {
    baseMs?: number;
    maxMs?: number;
    factor?: number;
    jitterRatio?: number;
  };
  maxTokens?: number;
  defaultContextWindow?: number;
  streamIdleTimeoutMs?: number;
}

/** Normalized output shape of the plugin configuration (defaults applied). */
export interface ConfigOutput {
  chains: Record<string, ChainEntryConfig[]>;
  defaultChain: string;
  virtualProvider: string;
  cooldown?: {
    baseMs: number;
    maxMs: number;
    factor: number;
    jitterRatio: number;
  };
  maxTokens: number;
  defaultContextWindow: number;
  streamIdleTimeoutMs: number;
}

export const Config: z<ConfigInput, ConfigOutput> = z.object({
  chains: z.dict(z.array(entrySchema)).required(),
  defaultChain: z.string().default('default'),
  virtualProvider: z.string().default('polyglot'),
  cooldown: z.object({
    baseMs: z.number().min(1).max(MAX_TIMER_DELAY_MS).default(DEFAULT_COOLDOWN_BASE_MS),
    maxMs: z.number().min(1).max(MAX_TIMER_DELAY_MS).default(DEFAULT_COOLDOWN_MAX_MS),
    factor: z.number().min(1).max(10).default(DEFAULT_COOLDOWN_FACTOR),
    jitterRatio: z.number().min(0).max(1).default(DEFAULT_COOLDOWN_JITTER_RATIO),
  }),
  maxTokens: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_MAX_TOKENS),
  defaultContextWindow: z.number().step(1).min(1).max(Number.MAX_SAFE_INTEGER).default(DEFAULT_CONTEXT_WINDOW),
  streamIdleTimeoutMs: z.number().min(Number.MIN_VALUE).max(MAX_TIMER_DELAY_MS).default(DEFAULT_STREAM_IDLE_TIMEOUT_MS),
});

/** The normalized output type of the {@link Config} schema. */
export type ConfigType = ConfigOutput;

/** The resolved plugin state: every provider route plus every resolved chain. */
interface ResolvedState {
  /** provider route → validated connection facts. */
  providers: Map<string, ProviderConfig>;
  /** chain name → resolved entries in fallback order. */
  chains: Map<string, ResolvedChainEntry[]>;
}

/** The default credential reference for a preset when it declares none. */
function defaultEnvFor(presetId: string): string {
  return `${presetId.replace(/[^A-Za-z0-9]/g, '_').toUpperCase()}_API_KEY`;
}

/** Resolve, validate, and detach one chain entry against the preset registry. */
function resolveEntry(
  raw: ChainEntryConfig,
  chainName: string,
  index: number,
  config: PolyglotConfig,
): ResolvedChainEntry {
  const preset = loadPreset(raw.preset);
  const provider = raw.provider ?? preset.id;
  if (provider.length === 0) throw new Error(`dsh-polyglot: chain "${chainName}" entry ${index} has an empty provider route`);
  const model = raw.model ?? preset.models[0]?.id;
  if (model === undefined || model.length === 0) {
    throw new Error(`dsh-polyglot: chain "${chainName}" entry ${index} (${preset.id}) has no model; add one to the entry or to the preset`);
  }
  const quirks = resolveQuirks({ ...preset.quirks, ...raw.quirks }, preset.id);
  const apiKeyEnv = credentialRef(raw.apiKeyEnv ?? preset.auth.apiKeyEnv ?? defaultEnvFor(preset.id));
  const connection: ProviderConfig = {
    preset: preset.id,
    provider,
    displayName: preset.displayName,
    baseURL: raw.baseUrl ?? preset.baseUrl,
    apiKeyEnv,
    authKind: preset.auth.kind,
    headers: { ...preset.headers, ...raw.headers },
    quirks,
    catalog: preset.models,
    maxTokens: config.maxTokens,
    defaultContextWindow: config.defaultContextWindow,
    streamIdleTimeoutMs: config.streamIdleTimeoutMs,
    retryPolicy: {
      mode: 'normal',
      maxRetries: 2,
      retryableCodes: ['RATE_LIMIT', 'QUOTA', 'SERVER', 'TIMEOUT', 'STREAM_CLOSED', 'TRANSPORT'],
      initialDelayMs: 500,
      maxDelayMs: 10000,
      jitterRatio: 0.1,
    },
  };
  return {
    preset: preset.id,
    provider,
    model,
    connection,
    ...preset.pricing !== undefined ? { pricing: { currency: 'USD', ...preset.pricing } } : {},
  };
}

/** Resolve the whole configuration into provider facts and chains, failing loud. */
export function resolveState(raw: PolyglotConfig): ResolvedState {
  const chainNames = Object.keys(raw.chains);
  if (chainNames.length === 0) throw new Error('dsh-polyglot: at least one chain is required');
  if (!(raw.defaultChain in raw.chains)) {
    throw new Error(`dsh-polyglot: defaultChain "${raw.defaultChain}" is not one of the configured chains: ${chainNames.join(', ')}`);
  }
  const providers = new Map<string, ProviderConfig>();
  const chains = new Map<string, ResolvedChainEntry[]>();
  for (const [chainName, entries] of Object.entries(raw.chains)) {
    if (entries.length === 0) throw new Error(`dsh-polyglot: chain "${chainName}" is empty`);
    const resolved = entries.map((entry, index) => resolveEntry(entry, chainName, index, raw));
    for (const entry of resolved) {
      const existing = providers.get(entry.provider);
      if (existing !== undefined) {
        if (!deepEqualJson(existing, entry.connection)) {
          throw new Error(
            `dsh-polyglot: provider route "${entry.provider}" is configured differently in multiple chains; use distinct provider routes or identical entries`,
          );
        }
      } else {
        providers.set(entry.provider, entry.connection);
      }
    }
    chains.set(chainName, resolved);
  }
  return { providers, chains };
}

export function apply(ctx: Context, raw: ConfigInput | ConfigType): void {
  // Normalize once through the schema: a direct ctx.plugin mount skips the
  // loader's schema pass, so defaults must be materialized here to keep
  // config.defaultChain etc. defined no matter how the bundle is loaded.
  const config = Config(raw as ConfigInput);
  let current = (): ConfigType => config;
  let lastRaw: ConfigType | undefined;
  let lastGood: ResolvedState | undefined;

  const state = (): ResolvedState => {
    const raw = current();
    if (raw === lastRaw && lastGood !== undefined) return lastGood;
    try {
      const next = resolveState(raw as unknown as PolyglotConfig);
      lastRaw = raw;
      lastGood = next;
      return next;
    } catch (error) {
      if (lastGood === undefined) throw error;
      lastRaw = raw;
      ctx.logger.error('dsh-polyglot: keeping the last good configuration after an invalid settings section');
      ctx.logger.error(error);
      return lastGood;
    }
  };
  state();

  const resolveUserId = () => getOrCreateAnonymousUserId();
  const adapterByProvider = new Map<string, OpenAiCompatAdapter>();
  const registrationByProvider = new Map<string, () => void>();
  let configurable: LlmConfigurableProvider[] = [];
  let directoryHandle: { replace(entries: readonly LlmConfigurableProvider[]): void } | undefined;

  const makeAdapter = (provider: string): OpenAiCompatAdapter => new OpenAiCompatAdapter({
    options: () => state().providers.get(provider) as ProviderConfig,
    resolveApiKey: async (connection) => {
      if (connection.authKind === 'none') return '';
      const ref = connection.apiKeyEnv;
      const credentials = ctx.get('credentials');
      if (credentials !== undefined) {
        const hit = await credentials.resolve(ref);
        if (hit !== undefined) return assertUsableApiKey(hit.value, 'polyglot', ref);
      } else {
        const ambient = launchEnvironmentOf(ctx).get(ref);
        if (ambient !== undefined && ambient.value.length > 0) return assertUsableApiKey(ambient.value, 'polyglot', ref);
      }
      throw new LlmError(
        `dsh-polyglot: no API key for provider route "${provider}"; store ${ref} through the credentials service (the web Models page writes it), or export ${ref} in the launching environment`,
        'MISSING_CREDENTIAL',
      );
    },
    resolveUserId,
  });

  const ensureProvider = (provider: string): void => {
    if (adapterByProvider.has(provider)) return;
    const connection = state().providers.get(provider);
    if (connection === undefined) return;
    const adapter = makeAdapter(provider);
    const registration = ctx.llm.registerAdapter([provider], adapter);
    adapterByProvider.set(provider, adapter);
    registrationByProvider.set(provider, registration);
    configurable.push({
      provider,
      displayName: connection.displayName,
      settingsNs: NS,
      settingsPath: [],
      declared: true,
    });
  };

  /** Bring provider registrations in line with the current configuration. */
  const syncProviders = (): void => {
    const routes = new Set(state().providers.keys());
    for (const [provider, registration] of registrationByProvider) {
      if (routes.has(provider)) continue;
      registration();
      registrationByProvider.delete(provider);
      adapterByProvider.delete(provider);
      const index = configurable.findIndex((entry) => entry.provider === provider);
      if (index >= 0) configurable.splice(index, 1);
    }
    for (const provider of routes) ensureProvider(provider);
    if (directoryHandle !== undefined) directoryHandle.replace(configurable);
  };

  for (const provider of state().providers.keys()) ensureProvider(provider);
  directoryHandle = ctx.llm.registerConfigurableProviders(configurable);

  // Router: serves the virtual provider route over the active chain.
  let activeChainName = config.defaultChain;
  const router = new PolyglotRouterAdapter({
    chain: () => {
      const resolved = state();
      const name = activeChainName in resolved.chains ? activeChainName : config.defaultChain;
      const entries = resolved.chains.get(name) ?? [];
      return { name, entries };
    },
    dispatch: (options) => ctx.llm.stream(options),
    cooldown: () => {
      const raw = current() as unknown as PolyglotConfig;
      return {
        baseMs: raw.cooldown?.baseMs ?? DEFAULT_COOLDOWN_BASE_MS,
        maxMs: raw.cooldown?.maxMs ?? DEFAULT_COOLDOWN_MAX_MS,
        factor: raw.cooldown?.factor ?? DEFAULT_COOLDOWN_FACTOR,
        jitterRatio: raw.cooldown?.jitterRatio ?? DEFAULT_COOLDOWN_JITTER_RATIO,
      };
    },
    onServed: (record: ServedRecord, sessionId) => appendServed(ctx, sessionId, record),
    logger: ctx.logger,
  });
  ctx.llm.registerAdapter([config.virtualProvider], router);

  installSettingsSection(ctx, NS, Config, config, {
    setSource: (source) => {
      current = source as () => ConfigType;
    },
    onChange: () => {
      const resolved = state();
      if (!(activeChainName in resolved.chains)) activeChainName = config.defaultChain;
      syncProviders();
    },
  });

  const deps: PolyglotCommandDeps = {
    chainNames: () => Object.keys(current().chains),
    activeChain: () => {
      const resolved = state();
      return activeChainName in resolved.chains ? activeChainName : config.defaultChain;
    },
    setActiveChain: (name) => {
      if (!(name in current().chains)) return false;
      activeChainName = name;
      return true;
    },
    defaultChain: () => config.defaultChain,
    virtualProvider: () => config.virtualProvider,
    entriesSummary: () => {
      const resolved = state();
      const name = activeChainName in resolved.chains ? activeChainName : config.defaultChain;
      return (resolved.chains.get(name) ?? []).map((entry) => ({
        provider: entry.provider,
        model: entry.model,
        preset: entry.preset,
        free: loadPreset(entry.preset).free,
      }));
    },
    cooldownStatus: () => router.cooldown.status(Date.now()),
    pricingFor: (provider) => {
      const resolved = state();
      for (const entries of resolved.chains.values()) {
        const entry = entries.find((candidate) => candidate.provider === provider);
        if (entry?.pricing !== undefined) return entry.pricing;
      }
      return undefined;
    },
  };
  registerPolyglotCommandsSafe(ctx, deps);
}

/** Mount the human commands when the commands service is present. */
function registerPolyglotCommandsSafe(ctx: Context, deps: PolyglotCommandDeps): void {
  const commands = ctx.get('commands');
  if (commands === undefined) return;
  registerPolyglotCommands(ctx, deps);
}
