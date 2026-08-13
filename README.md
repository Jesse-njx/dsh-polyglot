# dsh-polyglot

**The model switch for DSH.** Point DeepSeek Harness at any OpenAI-compatible
endpoint — with curated presets for free and cheap DeepSeek providers and
automatic fallback when a free tier rate-limits you.

What claude-code-router is to Claude Code, dsh-polyglot is to DSH — except
DSH's `ctx.llm` is a sanctioned extension seam, so there is no request
interception: the generic adapter and the router are both real `LlmAdapter`
registrations.

- **One generic adapter.** A single OpenAI-compatible `ctx.llm` adapter
  parameterized by `{baseUrl, apiKey, model, headers?, quirks?}`. Streaming,
  tool calls, and usage extraction are all handled; per-provider deviations
  (reasoning field names, strict tool schemas, cache-folded usage) are small
  declarative `quirks` flags, never per-provider code.
- **A router with fallback.** On 429 / quota-exceeded / 5xx (or a missing
  key), the failing provider is marked cooling-down (exponential backoff,
  honoring `Retry-After`) and the request is retried on the next provider in
  the chain. Free tiers rate-limit constantly — automatic failover is the
  whole product.
- **Provider presets as data.** `presets/*.json` — community PRs add providers
  without touching the adapter. Each preset carries `verifiedAt` and free-tier
  notes so rot is visible.
- **Usage you can see.** Every attempt lands in the append-only session log as
  `polyglot/served`; `/polyglot usage` tallies per provider with token counts
  and estimated cost from preset pricing.

## Quick start

Install the bundle into a profile (a DSH profile is an ordered stack of
plugin-bundle patch layers):

```bash
dsh plugin --profile web add @dsh-polyglot/bundle
```

The bundle's patch registers the `polyglot` plugin with the recommended
default chain — *"code all day for free until something rate-limits, then
degrade gracefully to cheapest-paid"*:

```
nous-portal → opencode-zen → deepseek-official (5M grant) → kilo
```

Configure keys through the credentials seam (the web Models page writes them),
or export the env names each preset declares:

```bash
export NOUS_PORTAL_TOKEN=...      # nous-portal (bearer, manual token for v0.1)
export OPENCODE_API_KEY=...       # opencode-zen
export DEEPSEEK_API_KEY=...       # deepseek-official (new accounts: 5M free tokens, 30 days, no card)
export KILO_API_KEY=...           # kilo (paid fallback rung)
```

Pick the virtual provider `polyglot` in the model selector. A provider without
a configured key is skipped automatically — the chain degrades, it never
fails hard.

### Day-to-day commands

| Command | What it does |
|---|---|
| `/model` | show chains and the active one |
| `/model <chain>` | switch the active chain mid-session (logged as `polyglot/chain`) |
| `/polyglot` | status: active chain, entries, provider cooldowns |
| `/polyglot usage` | per-provider tally from the session log: calls, ok/failed, tokens, est. cost |
| `/polyglot presets` | free-tier posture of the active chain's presets |

## Configuration

Override `chains` and `cooldown` from your profile patch:

```yaml
# your profile's cordis.patch.yml (or --patch overlay)
- patch:
    - id: polyglot
      config:
        chains:
          default:
            - preset: nous-portal
            - preset: opencode-zen
            - preset: deepseek-official
              model: deepseek-v4-flash
            - preset: kilo
          paid:
            - preset: deepseek-official
              model: deepseek-v4-pro
        cooldown:
          baseMs: 30000        # initial per-provider cooldown after a failure
          maxMs: 900000        # ceiling (also honors provider Retry-After)
          factor: 2            # exponential growth per consecutive failure
          jitterRatio: 0.1     # symmetric jitter around each delay
```

Per-entry overrides: `provider` (route name), `model`, `baseUrl`, `apiKeyEnv`,
`headers`, `quirks` — the `custom` preset is the escape hatch for
vLLM/Ollama/SGLang localhost and any other OpenAI-compatible endpoint
(Qwen/GLM/Kimi official APIs included).

### Quirks reference

| Flag | Default | Meaning |
|---|---|---|
| `reasoningField` | `'reasoning_content'` | wire delta field carrying reasoning text; `null` disables reasoning entirely |
| `maxTokensField` | `'max_tokens'` | output-cap wire field (`max_completion_tokens` for newer hosts) |
| `usage` | `'standard'` | `'deepseek'` subtracts cache hits folded into `prompt_tokens`; `'none'` when the host reports none |
| `streamOptions` | `true` | send `stream_options: {include_usage: true}` |
| `strictToolSchemas` | `false` | add `strict: true` to tool schemas |
| `thinkingField` | `false` | send `thinking: {type}` (DeepSeek spelling) |
| `reasoningEffortField` | `true` | send `reasoning_effort` for high/max efforts |

## Preset registry

All figures were re-verified 2026-08-14 against provider docs; **these move
weekly** — every preset carries `verifiedAt`, and a CI job pinging each
`baseUrl` with a 1-token request is the planned trust loop.

| Preset | What you get | Cost / limits | Notes |
|---|---|---|---|
| `deepseek-official` | V4-Flash, V4-Pro | $0.14/$0.28 per M (Flash); **5M free tokens new accounts, 30 days, no card** | Baseline; prices trending up |
| `opencode-zen` | `deepseek-v4-flash-free` (+ Qwen 3.6 Plus, MiniMax M3, MiMo…) | Free, no card, 200k context; rate limits undocumented | Commercial terms unclear — flagged in the preset `notes` |
| `nous-portal` | `deepseek/deepseek-v4-flash:free` | Free, OAuth-gated, hard rate ceiling that returns errors | The poster child for fallback; put it first in a chain |
| `kilo` | V4-Pro, V4-Flash, V3.1 Terminus | Pay-as-you-go at **no markup** over provider rates | Good paid-fallback rung |
| `openrouter` | `:free` DeepSeek variants + everything else | Free variants throttled; paid at listed rates | Widest catalog, one key |
| `custom` | anything OpenAI-compatible | — | vLLM/Ollama/SGLang localhost; Qwen/GLM/Kimi official endpoints |
| `groq` / `together` / `fireworks` | DeepSeek hosting | fast but pricier | Latency upgrades, not savings |

## How it works

```
profile ──> provider route "polyglot" (the router meta-adapter)
              │  chain: nous-portal → opencode-zen → deepseek-official → kilo
              ▼
        ctx.llm.stream({provider: "nous-portal", ...})
              │  adapter per real route (OpenAiCompatAdapter, one per preset)
              ▼
        POST {baseUrl}/chat/completions   (SSE, usage, tools)
```

The router forwards the first attempt that completes. A fallback-eligible
failure that arrives **before any content flowed** — the free-tier ceiling
case — swaps to the next provider seamlessly; a failure after content flowed
cannot be unwritten and surfaces as a normal error finish. Which provider
actually served each turn is durable in the session log (`polyglot/served`),
so `/polyglot usage` is a pure fold over the log, not plugin-side accounting.

## ToS note

Free tiers are often gated for evaluation use (OpenCode Zen's commercial
terms are undocumented). Preset `notes` surface this at configure time —
dsh-polyglot does not silently launder usage.

## Development

```bash
pnpm install
pnpm typecheck    # strict TS
pnpm build        # tsc → lib/
pnpm test         # 56 tests: mock OpenAI-compat server with scripted 429/500/
                  # stream scenarios, golden wire assertions per quirk, and
                  # end-to-end cordis mounts proving fallback + session events
```

## Roadmap

- **v0.2** — per-role chains (planner → paid V4-Pro, executor/summarizer →
  free Flash); OAuth device flow for Nous Portal; preset auto-update check;
  provider benchmark/arena integration.
- **Non-goals** — proxying non-chat modalities; silent key laundering.
