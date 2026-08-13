/**
 * Quirk resolution: per-provider deviations stay declarative flags, resolved
 * once per provider config so the adapter never branches per provider name.
 *
 * @module dsh-polyglot/quirk
 */
import type { QuirksConfig, ResolvedQuirks } from './types.ts';

/** Defaults every quirk falls back to (OpenAI-compatible baseline). */
export const DEFAULT_QUIRKS: ResolvedQuirks = {
  reasoningField: 'reasoning_content',
  maxTokensField: 'max_tokens',
  usage: 'standard',
  streamOptions: true,
  strictToolSchemas: false,
  thinkingField: false,
  reasoningEffortField: true,
};

/** Resolve a partial quirk set against the defaults, validating values. */
export function resolveQuirks(raw: Partial<QuirksConfig> | undefined, presetId: string): ResolvedQuirks {
  const source = raw ?? {};
  const reasoningField = source.reasoningField === undefined ? DEFAULT_QUIRKS.reasoningField : source.reasoningField;
  if (reasoningField !== null && (typeof reasoningField !== 'string' || reasoningField.length === 0)) {
    throw new Error(`dsh-polyglot: preset "${presetId}" quirks.reasoningField must be a non-empty string or null`);
  }
  const maxTokensField = source.maxTokensField ?? DEFAULT_QUIRKS.maxTokensField;
  if (maxTokensField !== 'max_tokens' && maxTokensField !== 'max_completion_tokens') {
    throw new Error(`dsh-polyglot: preset "${presetId}" quirks.maxTokensField must be max_tokens|max_completion_tokens`);
  }
  const usage = source.usage ?? DEFAULT_QUIRKS.usage;
  if (usage !== 'standard' && usage !== 'deepseek' && usage !== 'none') {
    throw new Error(`dsh-polyglot: preset "${presetId}" quirks.usage must be standard|deepseek|none`);
  }
  return {
    reasoningField,
    maxTokensField,
    usage,
    streamOptions: source.streamOptions ?? DEFAULT_QUIRKS.streamOptions,
    strictToolSchemas: source.strictToolSchemas ?? DEFAULT_QUIRKS.strictToolSchemas,
    thinkingField: source.thinkingField ?? DEFAULT_QUIRKS.thinkingField,
    reasoningEffortField: source.reasoningEffortField ?? DEFAULT_QUIRKS.reasoningEffortField,
  };
}
