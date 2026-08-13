/**
 * Preset registry: presets/*.json loaded as data, never code, so community
 * PRs can add providers without touching the adapter. Each preset is
 * validated against {@link PresetFile} at load.
 *
 * @module dsh-polyglot/preset
 */
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { CatalogModel, PresetPricing, QuirksConfig } from './types.ts';

/** The registry shape of one presets/*.json file. */
export interface PresetFile {
  /** Stable preset id; also the default provider route and file name. */
  id: string;
  /** Human-readable provider name for selectors and diagnostics. */
  displayName: string;
  /** Endpoint base; `/chat/completions` is appended. */
  baseUrl: string;
  /** Authentication facts: kind plus the credential reference to resolve. */
  auth: {
    kind: 'key' | 'oauth' | 'none';
    /** Credential reference (env name) resolved through the credentials seam. */
    apiKeyEnv?: string;
    /** Free-form auth guidance (OAuth device flow, manual token…). */
    note?: string;
  };
  /** Free-tier facts: whether a free tier exists and its known limits. */
  free?: {
    tier: boolean | 'mixed';
    grant?: string;
    limits?: string;
  };
  /** Advisory model catalog. */
  models: CatalogModel[];
  /** Declarative provider deviations applied to every entry built from this preset. */
  quirks?: Partial<QuirksConfig>;
  /** Extra per-provider request headers (OpenRouter identity headers etc.). */
  headers?: Record<string, string>;
  /** Optional pricing for cost tallies in the usage command. */
  pricing?: Omit<PresetPricing, 'currency'> & { currency?: string };
  /** Free-form notes; ToS caveats belong here (surfaced at configure time). */
  notes?: string;
  /** ISO date the preset facts were last verified against the endpoint. */
  verifiedAt?: string;
}

/** Absolute URL of the presets directory, same from src/ (tests) and lib/ (build). */
const PRESETS_URL = new URL('../presets/', import.meta.url);
const PRESETS_DIR = fileURLToPath(PRESETS_URL);

/** Load one preset by id, validating its shape. */
export function loadPreset(id: string): PresetFile {
  const file = fileURLToPath(new URL(`${id}.json`, PRESETS_URL));
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(file, 'utf8'));
  } catch (error) {
    throw new Error(`dsh-polyglot: preset "${id}" not found or not valid JSON (${file}): ${String(error)}`);
  }
  const preset = validatePreset(raw, id);
  return preset;
}

/** Every preset id shipped with the bundle, in file order. */
export function listPresetIds(): string[] {
  return readdirSync(PRESETS_DIR)
    .filter((name) => name.endsWith('.json'))
    .map((name) => name.slice(0, -'.json'.length))
    .sort();
}

/** Validate one raw preset document; every field is re-judged because presets are external data. */
export function validatePreset(raw: unknown, id: string): PresetFile {
  if (typeof raw !== 'object' || raw === null) throw new Error(`dsh-polyglot: preset "${id}" must be a JSON object`);
  const doc = raw as Record<string, unknown>;
  const presetId = typeof doc.id === 'string' && doc.id.length > 0 ? doc.id : id;
  if (typeof doc.displayName !== 'string' || doc.displayName.length === 0) throw new Error(`dsh-polyglot: preset "${id}" lacks a displayName`);
  if (typeof doc.baseUrl !== 'string' || doc.baseUrl.length === 0) throw new Error(`dsh-polyglot: preset "${id}" lacks a baseUrl`);
  const auth = doc.auth;
  if (typeof auth !== 'object' || auth === null) throw new Error(`dsh-polyglot: preset "${id}" lacks auth`);
  const authKind = (auth as Record<string, unknown>).kind;
  if (authKind !== 'key' && authKind !== 'oauth' && authKind !== 'none') throw new Error(`dsh-polyglot: preset "${id}" auth.kind must be key|oauth|none`);
  const apiKeyEnv = (auth as Record<string, unknown>).apiKeyEnv;
  if (apiKeyEnv !== undefined && (typeof apiKeyEnv !== 'string' || apiKeyEnv.length === 0)) throw new Error(`dsh-polyglot: preset "${id}" auth.apiKeyEnv must be a non-empty string`);
  if (!Array.isArray(doc.models) || doc.models.some((m) => typeof m !== 'object' || m === null)) throw new Error(`dsh-polyglot: preset "${id}" models must be an array of objects`);
  const models: CatalogModel[] = (doc.models as Record<string, unknown>[]).map((m) => {
    const entry = m as Record<string, unknown>;
    if (typeof entry.id !== 'string' || entry.id.length === 0) throw new Error(`dsh-polyglot: preset "${id}" has a model without an id`);
    return {
      id: entry.id,
      ...typeof entry.name === 'string' && entry.name.length > 0 ? { name: entry.name } : {},
      ...typeof entry.description === 'string' ? { description: entry.description } : {},
      ...typeof entry.contextWindow === 'number' && Number.isInteger(entry.contextWindow) && entry.contextWindow > 0 ? { contextWindow: entry.contextWindow } : {},
      ...typeof entry.maxTokens === 'number' && Number.isInteger(entry.maxTokens) && entry.maxTokens > 0 ? { maxTokens: entry.maxTokens } : {},
      ...entry.reasoning === true ? { reasoning: true } : {},
    };
  });
  const pricingRaw = doc.pricing;
  let pricing: PresetPricing | undefined;
  if (pricingRaw !== undefined) {
    if (typeof pricingRaw !== 'object' || pricingRaw === null) throw new Error(`dsh-polyglot: preset "${id}" pricing must be an object`);
    const p = pricingRaw as Record<string, unknown>;
    const num = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
    pricing = {
      ...num(p.inputPerMTokens) !== undefined ? { inputPerMTokens: num(p.inputPerMTokens) } : {},
      ...num(p.outputPerMTokens) !== undefined ? { outputPerMTokens: num(p.outputPerMTokens) } : {},
      ...num(p.cacheReadPerMTokens) !== undefined ? { cacheReadPerMTokens: num(p.cacheReadPerMTokens) } : {},
      ...num(p.cacheWritePerMTokens) !== undefined ? { cacheWritePerMTokens: num(p.cacheWritePerMTokens) } : {},
      currency: typeof p.currency === 'string' && p.currency.length > 0 ? p.currency : 'USD',
    };
  }
  const free = doc.free;
  let freeOut: PresetFile['free'] | undefined;
  if (free !== undefined && typeof free === 'object' && free !== null) {
    const freeDoc = free as Record<string, unknown>;
    freeOut = {
      tier: validateFreeTier(freeDoc.tier, id),
      ...typeof freeDoc.grant === 'string' ? { grant: freeDoc.grant } : {},
      ...typeof freeDoc.limits === 'string' ? { limits: freeDoc.limits } : {},
    };
  }
  return {
    id: presetId,
    displayName: doc.displayName,
    baseUrl: doc.baseUrl,
    auth: {
      kind: authKind,
      ...apiKeyEnv !== undefined ? { apiKeyEnv } : {},
      ...typeof (auth as Record<string, unknown>).note === 'string' ? { note: (auth as Record<string, unknown>).note as string } : {},
    },
    ...freeOut !== undefined ? { free: freeOut } : {},
    models,
    ...parseQuirks(doc.quirks, presetId),
    ...parseHeaders(doc.headers, presetId),
    ...pricing !== undefined ? { pricing } : {},
    ...typeof doc.notes === 'string' ? { notes: doc.notes } : {},
    ...typeof doc.verifiedAt === 'string' ? { verifiedAt: doc.verifiedAt } : {},
  };
}

/** Validate the free-tier marker: a boolean or the literal `mixed`. */
function validateFreeTier(raw: unknown, id: string): boolean | 'mixed' {
  if (raw === true || raw === false) return raw;
  if (raw === 'mixed') return 'mixed';
  throw new Error(`dsh-polyglot: preset "${id}" free.tier must be a boolean or "mixed"`);
}

/** Parse the optional preset quirk flags, validating the well-known ones. */
function parseQuirks(raw: unknown, id: string): { quirks: Partial<QuirksConfig> } | Record<string, never> {
  if (raw === undefined) return {};
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`dsh-polyglot: preset "${id}" quirks must be an object`);
  }
  const source = raw as Record<string, unknown>;
  const quirks: Partial<QuirksConfig> = {};
  if (source.reasoningField !== undefined) {
    if (source.reasoningField !== null && typeof source.reasoningField !== 'string') {
      throw new Error(`dsh-polyglot: preset "${id}" quirks.reasoningField must be a string or null`);
    }
    quirks.reasoningField = source.reasoningField as string | null;
  }
  if (source.maxTokensField !== undefined) {
    if (source.maxTokensField !== 'max_tokens' && source.maxTokensField !== 'max_completion_tokens') {
      throw new Error(`dsh-polyglot: preset "${id}" quirks.maxTokensField must be max_tokens|max_completion_tokens`);
    }
    quirks.maxTokensField = source.maxTokensField as 'max_tokens' | 'max_completion_tokens';
  }
  if (source.usage !== undefined) {
    if (source.usage !== 'standard' && source.usage !== 'deepseek' && source.usage !== 'none') {
      throw new Error(`dsh-polyglot: preset "${id}" quirks.usage must be standard|deepseek|none`);
    }
    quirks.usage = source.usage as 'standard' | 'deepseek' | 'none';
  }
  for (const flag of ['streamOptions', 'strictToolSchemas', 'thinkingField', 'reasoningEffortField'] as const) {
    if (source[flag] !== undefined) {
      if (typeof source[flag] !== 'boolean') throw new Error(`dsh-polyglot: preset "${id}" quirks.${flag} must be a boolean`);
      quirks[flag] = source[flag] as boolean;
    }
  }
  return { quirks };
}

/** Parse the optional extra headers, validating string keys/values. */
function parseHeaders(raw: unknown, id: string): { headers: Record<string, string> } | Record<string, never> {
  if (raw === undefined) return {};
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error(`dsh-polyglot: preset "${id}" headers must be an object`);
  }
  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof value !== 'string') throw new Error(`dsh-polyglot: preset "${id}" headers.${key} must be a string`);
    headers[key] = value;
  }
  return { headers };
}
