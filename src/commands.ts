/**
 * dsh-polyglot human commands: `/model` switches the active chain mid-session,
 * `/polyglot` reports status, per-provider usage (read from the session log),
 * and the preset catalog. Registered through the plugin-owned command
 * registry; every switch is recorded as a durable `polyglot/chain` event.
 *
 * @module dsh-polyglot/commands
 */
import type { Context } from '@deepseek-ai/cordis';
import type { CommandInvocation, CommandResult } from '@deepseek-ai/dsh-commands';
import type { Session } from '@deepseek-ai/dsh-session';
import { appendChainSwitch } from './session-events.ts';
import type { PresetPricing, ServedRecord } from './types.ts';

/** Facts the commands need from the plugin; injected so the commands stay testable. */
export interface PolyglotCommandDeps {
  /** Every configured chain name. */
  chainNames(): string[];
  /** The currently active chain name. */
  activeChain(): string;
  /** Activate a chain by name; false when unknown. */
  setActiveChain(name: string): boolean;
  /** The configured default chain name. */
  defaultChain(): string;
  /** The virtual provider route the router serves. */
  virtualProvider(): string;
  /** Active chain entries, for status display. */
  entriesSummary(): { provider: string; model: string; preset: string; free?: { tier: boolean | 'mixed'; grant?: string; limits?: string } }[];
  /** provider → seconds remaining in cooldown. */
  cooldownStatus(): Record<string, number>;
  /** Pricing metadata for one provider route, when the preset declared it. */
  pricingFor(provider: string): PresetPricing | undefined;
}

/** Register the `/model` and `/polyglot` commands on `ctx.commands`. */
export function registerPolyglotCommands(ctx: Context, deps: PolyglotCommandDeps): void {
  ctx.commands.register({
    name: 'model',
    description: 'Switch the active dsh-polyglot provider chain (or show the current one)',
    input: { hint: '<chain> | list' },
    handler: (invocation) => modelCommand(ctx, invocation, deps),
  });
  ctx.commands.register({
    name: 'polyglot',
    description: 'dsh-polyglot status, per-provider usage, and preset list',
    input: { hint: 'status | usage | presets' },
    handler: (invocation) => polyglotCommand(invocation, deps),
  });
}

async function modelCommand(ctx: Context, invocation: CommandInvocation, deps: PolyglotCommandDeps): Promise<CommandResult> {
  const raw = invocation.rawInput.trim();
  const chains = deps.chainNames();
  if (raw === '' || raw === 'list') {
    const current = deps.activeChain();
    const lines = chains.map((name) => (name === current ? `* ${name} (active)` : `  ${name}`));
    return {
      kind: 'success',
      text: `dsh-polyglot chains (virtual provider ${deps.virtualProvider()}):\n${lines.join('\n')}\nUse /model <chain> to switch.`,
    };
  }
  if (deps.setActiveChain(raw)) {
    appendChainSwitch(ctx, invocation.agent.session, raw);
    return {
      kind: 'success',
      text: `Switched dsh-polyglot to chain "${raw}". The next model request is served from that chain.`,
    };
  }
  return {
    kind: 'error',
    text: `Unknown chain "${raw}". Available chains: ${chains.join(', ')}.`,
  };
}

async function polyglotCommand(invocation: CommandInvocation, deps: PolyglotCommandDeps): Promise<CommandResult> {
  const raw = invocation.rawInput.trim();
  if (raw === 'usage' || raw === 'stats') {
    return { kind: 'success', text: usageSummary(invocation.agent.session, deps) };
  }
  if (raw === 'presets') {
    return { kind: 'success', text: presetSummary(deps) };
  }
  return { kind: 'success', text: statusSummary(invocation.agent.session, deps) };
}

function statusSummary(session: Session, deps: PolyglotCommandDeps): string {
  const current = deps.activeChain();
  const entries = deps.entriesSummary();
  const entryLines = entries.map((entry) => {
    const free = entry.free?.tier === true ? ' [free tier]' : entry.free?.tier === 'mixed' ? ' [free+paid]' : '';
    return `  - ${entry.provider} (${entry.model})${free}`;
  });
  const cooldown = deps.cooldownStatus();
  const cooldownLines = Object.keys(cooldown).map((provider) => `  - ${provider}: ${cooldown[provider]}s`);
  return [
    `dsh-polyglot active chain: ${current} (default: ${deps.defaultChain()})`,
    `chain entries:`,
    ...entryLines,
    cooldownLines.length > 0 ? `cooldowns:` : '',
    ...cooldownLines,
    '',
    `Last ${servedCount(session)} provider calls this session. /polyglot usage for the tally, /model to switch chains.`,
  ].filter((line) => line !== '').join('\n');
}

function servedCount(session: Session): number {
  let count = 0;
  for (const event of session.events) {
    if (event.type === 'polyglot/served') count += 1;
  }
  return count;
}

/** Tally `polyglot/served` events per provider with token and cost estimates. */
function usageSummary(session: Session, deps: PolyglotCommandDeps): string {
  interface Tally {
    calls: number;
    ok: number;
    failed: number;
    inputTokens: number;
    outputTokens: number;
    cacheReadTokens: number;
    latencyMs: number;
  }
  const per = new Map<string, Tally>();
  for (const event of session.events) {
    if (event.type !== 'polyglot/served') continue;
    const record = event.data as ServedRecord;
    let tally = per.get(record.provider);
    if (tally === undefined) {
      tally = { calls: 0, ok: 0, failed: 0, inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, latencyMs: 0 };
      per.set(record.provider, tally);
    }
    tally.calls += 1;
    if (record.status === 'ok') tally.ok += 1; else tally.failed += 1;
    tally.inputTokens += record.usage?.inputTokens ?? 0;
    tally.outputTokens += record.usage?.outputTokens ?? 0;
    tally.cacheReadTokens += record.usage?.cacheReadTokens ?? 0;
    tally.latencyMs += record.latencyMs;
  }
  if (per.size === 0) {
    return 'No dsh-polyglot provider calls recorded in this session yet.';
  }
  const lines: string[] = ['Per-provider usage this session (from the session log):'];
  for (const [provider, tally] of per) {
    const pricing = deps.pricingFor(provider);
    const cost = pricing === undefined ? undefined : estimateCost(tally, pricing);
    const costText = cost === undefined ? '' : ` | est ${formatCost(cost, pricing?.currency ?? 'USD')}`;
    lines.push(
      `  ${provider}: ${tally.calls} calls (${tally.ok} ok / ${tally.failed} failed)` +
      ` | ${tally.inputTokens.toLocaleString()} in / ${tally.outputTokens.toLocaleString()} out` +
      (tally.cacheReadTokens > 0 ? ` / ${tally.cacheReadTokens.toLocaleString()} cache` : '') +
      (costText),
    );
  }
  lines.push('', 'Costs are estimates from preset pricing; a preset without pricing shows tokens only.');
  return lines.join('\n');
}

/** Estimate USD cost from a tally and preset pricing. */
export function estimateCost(tally: { inputTokens: number; outputTokens: number; cacheReadTokens: number }, pricing: PresetPricing): number {
  const input = (tally.inputTokens / 1_000_000) * (pricing.inputPerMTokens ?? 0);
  const output = (tally.outputTokens / 1_000_000) * (pricing.outputPerMTokens ?? 0);
  const cacheRead = (tally.cacheReadTokens / 1_000_000) * (pricing.cacheReadPerMTokens ?? 0);
  return input + output + cacheRead;
}

/** Format a small dollar amount without floating noise. */
export function formatCost(cost: number, currency: string): string {
  if (cost === 0) return `0 ${currency}`;
  const digits = cost < 0.01 ? 6 : cost < 1 ? 4 : 2;
  return `${cost.toFixed(digits)} ${currency}`;
}

function presetSummary(deps: PolyglotCommandDeps): string {
  const activeChain = deps.activeChain();
  const entries = deps.entriesSummary();
  const lines = entries.map((entry) => {
    const free = entry.free?.tier === true ? 'free' : entry.free?.tier === 'mixed' ? 'free+paid' : 'paid';
    const limits = entry.free?.limits !== undefined ? ` — ${entry.free.limits}` : '';
    return `  - ${entry.preset} (${entry.provider}/${entry.model}) [${free}]${limits}`;
  });
  return [
    `Presets in active chain "${activeChain}":`,
    ...lines,
    '',
    'ToS: free tiers may be gated for evaluation use — check each preset notes before relying on one.',
  ].join('\n');
}
