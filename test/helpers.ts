/**
 * Test helpers: minimal ProviderConfig/chain fixtures and stream collection.
 */

import { CallId, MessageId, ReasoningEffortId, createUserMessage } from '@deepseek-ai/dsh-llm';
import type { GenerateOptions, Message, StreamChunk } from '@deepseek-ai/dsh-llm';
import { credentialRef } from '@deepseek-ai/dsh-credentials';
import type { ProviderConfig, ResolvedChainEntry, ResolvedQuirks } from '../src/types.ts';
import { resolveQuirks } from '../src/quirk.ts';

/** A connection config pointing at one mock (or real) endpoint. */
export function providerConfig(overrides: Partial<ProviderConfig> & { baseURL: string; provider?: string }): ProviderConfig {
  return {
    preset: 'custom',
    provider: overrides.provider ?? 'mock',
    displayName: 'Mock',
    apiKeyEnv: credentialRef('MOCK_API_KEY'),
    authKind: 'none',
    headers: {},
    quirks: DEFAULT_TEST_QUIRKS,
    catalog: [
      { id: 'model-a', name: 'Model A', contextWindow: 131072 },
      { id: 'model-b', name: 'Model B', contextWindow: 262144, reasoning: true },
    ],
    maxTokens: 8192,
    defaultContextWindow: 131072,
    streamIdleTimeoutMs: 5000,
    retryPolicy: {
      mode: 'normal',
      maxRetries: 2,
      retryableCodes: ['RATE_LIMIT', 'QUOTA', 'SERVER', 'TIMEOUT', 'STREAM_CLOSED', 'TRANSPORT'],
      initialDelayMs: 500,
      maxDelayMs: 10000,
      jitterRatio: 0.1,
    },
    ...overrides,
  };
}

/** Fully resolved quirks used by tests that do not care about specific flags. */
export const DEFAULT_TEST_QUIRKS: ResolvedQuirks = resolveQuirks({}, 'test');

/** One chain entry pointing at a mock endpoint. */
export function chainEntry(baseURL: string, provider: string, model: string, quirks?: Parameters<typeof resolveQuirks>[0]): ResolvedChainEntry {
  const connection = providerConfig({ baseURL, provider, quirks: resolveQuirks(quirks, provider) });
  return { preset: 'custom', provider, model, connection };
}

/** A minimal conversation for GenerateOptions. */
export function messages(text = 'hello'): Message[] {
  return [createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })];
}

/** Collect a chunk stream into an array. */
export async function collect(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

/** Build GenerateOptions for a direct adapter/router call. */
export function generateOptions(overrides: Partial<GenerateOptions> = {}): GenerateOptions {
  return {
    provider: 'mock',
    model: 'model-a',
    messages: messages(),
    ...overrides,
  };
}

export { CallId, MessageId, ReasoningEffortId };
