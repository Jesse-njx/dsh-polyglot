/**
 * Scripted OpenAI-compatible mock server for adapter/router tests. Each
 * scenario is consumed in request order; the last scenario repeats.
 */

import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import type { Server } from 'node:http';

/** One scripted response. */
export interface MockScenario {
  /** HTTP status; 2xx streams the `chunks`, anything else returns the error body. */
  status?: number;
  /** Extra response headers (Retry-After etc.). */
  headers?: Record<string, string>;
  /** JSON error body; default `{error: {message: 'mock error'}}`. */
  body?: unknown;
  /** SSE data payloads (objects or raw strings), streamed when status is 2xx. */
  chunks?: Array<Record<string, unknown> | string>;
  /** Omit the terminal `data: [DONE]` (truncated-stream scenarios). */
  noDone?: boolean;
  /** Delay before responding, milliseconds. */
  delayMs?: number;
}

/** One recorded request. */
export interface RecordedRequest {
  body: Record<string, unknown>;
  headers: Record<string, string>;
  url: string;
}

/** SSE helpers for building chunk payloads. */
export function textDelta(text: string, index = 0, finish: string | null = null): Record<string, unknown> {
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion.chunk',
    choices: [{ index, delta: { content: text }, finish_reason: finish }],
  };
}

export function reasoningDelta(text: string, index = 0): Record<string, unknown> {
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion.chunk',
    choices: [{ index, delta: { reasoning_content: text }, finish_reason: null }],
  };
}

export function toolCallDelta(fragment: { id?: string; name?: string; arguments?: string }, callIndex = 0): Record<string, unknown> {
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion.chunk',
    choices: [{
      index: 0,
      delta: {
        tool_calls: [{
          index: callIndex,
          ...fragment.id !== undefined ? { id: fragment.id } : {},
          ...fragment.name !== undefined || fragment.arguments !== undefined ? {
            function: {
              ...fragment.name !== undefined ? { name: fragment.name } : {},
              ...fragment.arguments !== undefined ? { arguments: fragment.arguments } : {},
            },
          } : {},
        }],
      },
      finish_reason: null,
    }],
  };
}

export function finishDelta(reason: string, index = 0): Record<string, unknown> {
  return {
    id: 'chatcmpl-mock',
    object: 'chat.completion.chunk',
    choices: [{ index, delta: {}, finish_reason: reason }],
  };
}

export function usageChunk(usage: Record<string, unknown>): Record<string, unknown> {
  return { id: 'chatcmpl-mock', object: 'chat.completion.chunk', choices: [], usage };
}

/** A running mock endpoint. */
export class MockProvider {
  readonly requests: RecordedRequest[] = [];
  #scenarios: MockScenario[] = [];
  #server: Server;
  #baseUrl = '';

  constructor(scenarios: MockScenario[] = []) {
    this.#scenarios = scenarios;
    this.#server = createServer((req, res) => this.#handle(req, res));
  }

  async start(): Promise<string> {
    await new Promise<void>((resolve) => this.#server.listen(0, '127.0.0.1', resolve));
    const { port } = this.#server.address() as AddressInfo;
    this.#baseUrl = `http://127.0.0.1:${port}/v1`;
    return this.#baseUrl;
  }

  async close(): Promise<void> {
    await new Promise<void>((resolve, reject) => this.#server.close((error) => (error ? reject(error) : resolve())));
  }

  /** Append scripted scenarios. */
  script(...scenarios: MockScenario[]): void {
    this.#scenarios.push(...scenarios);
  }

  /** The endpoint base URL (`http://host:port/v1`); valid after {@link start}. */
  get url(): string {
    return this.#baseUrl;
  }

  /** The number of requests received. */
  get count(): number {
    return this.requests.length;
  }

  #nextScenario(): MockScenario {
    const scenario = this.#scenarios.shift();
    if (scenario !== undefined) return scenario;
    // Repeat the last scenario forever; default to a plain error.
    return this.#scenarios.at(-1) ?? { status: 500, body: { error: { message: 'mock: no scenario left' } } };
  }

  async #handle(req: import('node:http').IncomingMessage, res: import('node:http').ServerResponse): Promise<void> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const raw = Buffer.concat(chunks).toString('utf8');
    let body: Record<string, unknown> = {};
    try {
      body = raw.length > 0 ? JSON.parse(raw) as Record<string, unknown> : {};
    } catch {
      body = { raw };
    }
    this.requests.push({
      body,
      headers: Object.fromEntries(Object.entries(req.headers).map(([k, v]) => [k, String(v)])),
      url: req.url ?? '',
    });
    const scenario = this.#nextScenario();
    if (scenario.delayMs !== undefined) await new Promise((resolve) => setTimeout(resolve, scenario.delayMs));
    if (scenario.status !== undefined && (scenario.status < 200 || scenario.status >= 300)) {
      res.writeHead(scenario.status, { 'content-type': 'application/json', ...scenario.headers });
      res.end(JSON.stringify(scenario.body ?? { error: { message: `mock error ${scenario.status}` } }));
      return;
    }
    const chunksToSend = scenario.chunks ?? [textDelta('hello from mock')];
    res.writeHead(200, { 'content-type': 'text/event-stream', ...scenario.headers });
    for (const chunk of chunksToSend) {
      const payload = typeof chunk === 'string' ? chunk : JSON.stringify(chunk);
      res.write(`data: ${payload}\n\n`);
    }
    if (scenario.noDone !== true) res.end('data: [DONE]\n\n');
    else res.end();
  }
}

/** A completed streaming call with an empty message and usage — common tail. */
export function okStream(text: string, usage?: Record<string, unknown>): MockScenario {
  const chunks: Array<Record<string, unknown> | string> = [textDelta(text), finishDelta('stop')];
  if (usage !== undefined) chunks.push(usageChunk(usage));
  return { status: 200, chunks };
}
