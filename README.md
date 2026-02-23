# GodLocal

**Your AI. Your machine. Your soul. No cloud.**

GodLocal is a sovereign local AI agent that runs entirely on your own hardware.  
Not an API wrapper. Not a chatbot. A living system with a soul, memory, and tools — that gets smarter while you sleep.

```bash
python godlocal_v5.py
```

---

## Why

| Cloud AI (GPT, Claude) | GodLocal |
|---|---|
| Your data leaves your machine | Zero egress, zero cloud |
| $20–200/mo subscription | One-time model download |
| Personality set by the company | You define the soul |
| Tools limited to what they allow | Any tool you can code |
| Gets dumber if you stop paying | Memory consolidates nightly |

---

## Architecture

```
┌─────────────────────────────────────┐
│             SOUL                    │
│   soul/*.md — who your AI is        │
│   "I am a cold precise quant..."    │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│             BRAIN                   │
│   AirLLM (layer-by-layer, any RAM)  │
│   or Ollama (faster, daemon mode)   │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│             BODY (tools)            │
│  Files · Calendar · Shell · Web     │
│  Speech · MRI · Custom plugins      │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│             SLEEP                   │
│  Nightly memory consolidation       │
│  Hippocampal replay → long-term     │
└─────────────────────────────────────┘
```

---

## What's New in v5

- **ImageGen** — Stable Diffusion / SDXL-Turbo / Flux (local, no API)
- **VideoGen** — CogVideoX-2b text-to-video (4-6s clips)
- **AppGen** — Build full apps from descriptions (DeepSeek-Coder / Qwen-Coder)
- **AudioGen** — Bark TTS + MusicGen (multilingual, music clips)
- **KnowledgeBase** — Import URLs, PDFs, YouTube → long-term memory
- **SecretsVault** — Encrypted local secrets (Fernet AES-128)
- **MultiAgentRunner** — Parallel sub-agents with different souls
- **OCREngine** — Image/PDF text extraction (Tesseract)
- **SolanaDEX** — Jupiter API: prices + swap quotes (no API key)

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/GODLOCAL/godlocal
cd godlocal

# 2. Install
pip install chromadb sentence-transformers fastapi uvicorn

# 3. Choose your brain:
# Option A — Ollama (recommended, faster)
brew install ollama && ollama pull qwen2.5:7b

# Option B — AirLLM (any VRAM, huge models on small hardware)
pip install airllm

# 4. Run
python godlocal_v5.py
# → http://localhost:8000/docs
```

Or with Docker:
```bash
cp .env.example .env
docker-compose up -d
docker exec godlocal-ollama ollama pull qwen2.5:7b
```

---

## Soul Files

A soul is a markdown file that defines who your AI is.

```
godlocal_data/souls/
├── default.md     # calm, precise, private assistant
├── warrior.md     # X-ZERO — cold Solana quant agent
└── sovereign.md   # full autonomy, max agency
```

**Switch souls at runtime:**
```bash
curl -X POST http://localhost:8000/souls/load -d '{"soul_name": "warrior"}'
```

Create your own using `god_soul.example.md` as template.

---

## REST API

```
GET  /status              — capabilities, current soul, device
POST /chat                — send a message
POST /create/image        — generate image (Stable Diffusion)
POST /create/video        — generate video (CogVideoX)
POST /create/app          — build an app from description
POST /create/audio        — TTS or music generation
POST /execute             — run whitelisted shell command
POST /sleep               — trigger memory consolidation
GET  /souls               — list souls
POST /souls/load          — switch soul
POST /knowledge/import    — import URL/PDF/YouTube to memory
POST /solana/price        — token prices (Jupiter)
POST /solana/quote        — swap quote (Jupiter)
GET  /docs                — Swagger UI
```

---

## Sleep Cycle

GodLocal consolidates memories every night at 01:00:

```python
god.run_sleep_cycle()
# or: POST /sleep
```

Samples recent memories → LLM extracts insights → promotes to long-term ChromaDB.  
Mimics hippocampal replay during slow-wave sleep. The model gets *wiser* — not just bigger.

---

## Support

| Method | Link |
|--------|------|
| ☕ Ko-fi | [ko-fi.com/godlocal](https://ko-fi.com/godlocal) |
| 🪙 SOL | `EWcSFdC3eERL6mAbwbdX3W9eFfYZJbFvaix1J3JcGM1r` |
| 💼 Commercial License | [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) |
| 🌐 Website | [godlocal.ai](https://godlocal.ai) *(coming soon)* |

---

## Roadmap

- [x] v5 — ImageGen, VideoGen, AppGen, AudioGen, KnowledgeBase, SecretsVault, MultiAgentRunner, OCR, SolanaDEX
- [ ] v5.1 — ConnectorsModule (Composio SDK, 500+ service integrations)
- [ ] v5.2 — Computer Use + TradingView webhooks
- [ ] v5.3 — Notifications, email, translate
- [ ] X100 OASIS integration (archetype soul ↔ $X100 token gate)

---

## License

**AGPL-3.0** for open-source use.  
**Commercial License** for closed-source / SaaS / enterprise.  
See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md).

---

*Built by Rostyslav Oliinyk*  
*Part of the [X100 OASIS](https://x100-app.vercel.app) ecosystem*
