# GodLocal

> **Your AI. Your machine.**

[![Deploy](https://img.shields.io/badge/Vercel-LIVE-brightgreen?logo=vercel)](https://godlocal.vercel.app)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Groq-llama--3.3--70b-orange)](https://groq.com)

GodLocal is an **open-source AI inference platform** that lets you run powerful AI entirely on your own machine — or in hybrid mode. No subscriptions, no data leaving your device, no dependency on centralized cloud.

---

## What it is

| Feature | Details |
|---|---|
| **4-tier inference stack** | WASM → Groq → Cerebras → AirLLM |
| **Peak speed** | ~17,000 tokens/sec (Taalas HC1 spec) |
| **iPhone support** | CoreML + ANE — real on-device speed |
| **Autonomous agents** | ReAct loop, tool calling, cron ticks — no constant supervision |
| **API surface** | 18+ endpoints: OSINT, SparkNet, Solana CLI, kill switch, market data |
| **Codebase** | ~2,900 lines — clean, auditable, zero vendor lock-in |

---

## Architecture

```
+---------------------------------------------------+
|               INFERENCE TIERS                     |
|                                                   |
|  WASM      -> micro tasks, <50 tokens, offline    |
|  Groq      -> 270-875 tok/s, cloud burst          |
|  Cerebras  -> ~3,000 tok/s, low latency           |
|  AirLLM    -> any model, any RAM, layer-by-layer  |
+---------------------+-----------------------------+
                      |
+---------------------v-----------------------------+
|                 AGENT CORE                        |
|                                                   |
|  ReAct loop   -> think -> tool -> observe -> loop |
|  SparkNet     -> Q-learning signal scoring         |
|  /agent/tick  -> autonomous market cycle           |
|  kill switch  -> circuit breaker for xzero        |
+---------------------+-----------------------------+
                      |
+---------------------v-----------------------------+
|            TOOLS (18+ endpoints)                  |
|                                                   |
|  /think         -> ReAct agent (llama-3.3-70b)    |
|  /agent/tick    -> autonomous analysis cycle       |
|  /market        -> live BTC/ETH/SOL/BNB/SUI        |
|  /status        -> kill switch + circuit breaker   |
|  /mobile/*      -> iPhone PWA control surface     |
+---------------------+-----------------------------+
                      |
+---------------------v-----------------------------+
|             iPhone (CoreML + ANE)                 |
|                                                   |
|  PWA at godlocal.vercel.app                       |
|  NexaSDK  MobileOBridge  AudioBridgeMLX           |
|  SwiftUI XZeroControlView                         |
+---------------------------------------------------+
```

---

## Quick Start

```bash
git clone https://github.com/GODLOCAL/godlocal.git
cd godlocal
export GROQ_API_KEY=your_key_here
python api/index.py
```

Or hit the live deployment:

```bash
# Ask the agent
curl -X POST https://godlocal.vercel.app/think \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyse current BTC market and assess risk"}'

# Autonomous tick — 5 tools, sparks, kill switch decision
curl -X POST https://godlocal.vercel.app/agent/tick

# Live prices
curl https://godlocal.vercel.app/market
```

---

## Agent Tools

| Tool | What it does |
|---|---|
| `get_market_data` | Live BTC/ETH/SOL/BNB/SUI prices (CoinGecko, 5-min cache) |
| `get_system_status` | Kill switch state, circuit breaker, consecutive losses |
| `get_recent_thoughts` | Agent's own recent analysis history |
| `set_kill_switch` | Enable/disable xzero trading — agent decides autonomously |
| `add_spark` | Log trading signal to SparkNet with confidence score |

---

## Model Fallback Chain

Rate limits handled automatically — no 429 errors reach the user:

```
llama-3.3-70b-versatile   <- primary (best reasoning)
        | 429
llama-3.1-8b-instant      <- fallback (~483 tok/s)
        | 429
llama3-8b-8192            <- emergency
```

---

## Inference Speed Benchmarks

| Model | Speed | Use case |
|---|---|---|
| Taalas HC1 | ~17,000 tok/s | WASM tier (spec) |
| Cerebras llama3.1-8b | ~3,000 tok/s | Micro tasks |
| Groq gpt-oss-20b | ~875 tok/s | Summarize / fast |
| Groq llama-3.1-8b | ~483 tok/s | Classify |
| Groq llama-3.3-70b | ~270 tok/s | Full reasoning |
| AirLLM | CPU/RAM dependent | On-device, offline |

---

## iPhone / PWA

Add to Home Screen via Safari — works as a native app:

```
https://godlocal.vercel.app/static/pwa/index.html
```

Tabs: **Status** · **Think** · **SparkNet** · **Controls**
PWA polls every 5 seconds. Kill switch toggle is one tap.

---

## Environment Variables

```env
GROQ_API_KEY=           # Required — get at console.groq.com
CEREBRAS_API_KEY=       # Optional — Cerebras Inference
XZERO_KILL_SWITCH=      # true/false — default false
XZERO_MAX_DRAWDOWN_PCT=
XZERO_MAX_CONSECUTIVE_L=
XZERO_DAILY_LOSS_SOL=
```

---

## Roadmap

- [x] ReAct agent loop with tool calling
- [x] Live market data (CoinGecko)
- [x] Model fallback chain (429 protection)
- [x] iPhone PWA
- [x] SparkNet Q-learning signal scoring
- [ ] Vercel Cron -> /agent/tick every 5 min
- [ ] VPS deploy (Hetzner CAX11)
- [ ] godlocal.io domain
- [ ] Taalas HC1 API key -> 17k tok/s benchmark
- [ ] On-device CoreML (NexaSDK)

---

## Community

Telegram: [@godlocalai](https://t.me/godlocalai)
Twitter: [@kitbtc](https://twitter.com/kitbtc)

---

## License

MIT — use it, fork it, ship it.
