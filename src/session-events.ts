/**
 * Durable session events for dsh-polyglot. Every provider attempt lands in
 * the append-only session log as `polyglot/served` (log-only, never model
 * surface), which is what makes "which provider served each turn" visible in
 * the session log and what the usage command tallies. Chain switches land as
 * `polyglot/chain` so replay can reconstruct which chain was active.
 *
 * @module dsh-polyglot/session-events
 */
import type { Context } from '@deepseek-ai/cordis';
import type { Session, SessionId } from '@deepseek-ai/dsh-session';
import type { ServedRecord } from './types.ts';

declare module '@deepseek-ai/dsh-session/types' {
  interface SessionEventMap {
    /**
     * One provider attempt in a dsh-polyglot chain (success or failure).
     * Log-only: never model surface, never derived history.
     * @mode emit
     * @param record - the attempt facts.
     */
    'polyglot/served': ServedRecord;
    /**
     * The active dsh-polyglot chain changed (via `/model`). Log-only.
     * @mode emit
     * @param data - the newly active chain name.
     */
    'polyglot/chain': { chain: string };
  }
}

/**
 * Append a `polyglot/served` record to the session that owns one call, when
 * that session is live. Fire-and-forget: a missing session or a rejected
 * append is logged, never thrown into the streaming path.
 */
export function appendServed(ctx: Context, sessionId: SessionId | undefined, record: ServedRecord): void {
  if (sessionId === undefined) return;
  const sessions = ctx.get('sessions');
  if (sessions === undefined) return;
  const session = sessions.get(sessionId);
  if (session === undefined) return;
  try {
    session.append('polyglot/served', record);
  } catch (error) {
    ctx.logger.warn('dsh-polyglot: could not append polyglot/served event', error);
  }
}

/** Append a `polyglot/chain` event recording an activated chain. */
export function appendChainSwitch(ctx: Context, session: Session | undefined, chain: string): void {
  if (session === undefined) return;
  try {
    session.append('polyglot/chain', { chain });
  } catch (error) {
    ctx.logger.warn('dsh-polyglot: could not append polyglot/chain event', error);
  }
}
