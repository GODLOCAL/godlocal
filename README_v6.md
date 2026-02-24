# БОГ || OASIS v6
### Sovereign Local AI Studio — Your AI. Your Machine. Getting Smarter While You Sleep.

```
╬══════════════════════════════════════════════════════════╖
║        БОГ || OASIS v6 — Sovereign AI Studio             ║
║  Your AI. On your machine. Getting smarter while sleep.  ║
╚══════════════════════════════════════════════════════════╝
```

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)

---

## What is БОГ || OASIS?

A sovereign AI engine that runs entirely on your hardware — Mac (Apple Silicon/MPS), Steam Deck (ROCm), or any Linux box. No cloud. No subscription. It learns from your interactions, evolves its own code nightly while you sleep, and connects to 500+ external services via Composio.

**Core loop:**
```
Wake → Think → Act → Sleep (memory consolidation + self-evolution) → Repeat
```

---

## Architecture v6

```
godlocal/
├── core/
│   ├── settings.py        # pydantic-settings config (GODLOCAL_ env prefix)
│   ├── brain.py           # LLMBridge + MemoryEngine + Brain singleton
│   └── __init__.py
├── agents/
│   ├── autogenesis_v2.py  # FEP + DockerSafeApply + Plan-and-Execute
│   ├── agent_pool.py      # 6-slot hot-swap AgentPool (MLX RAM-efficient)
│   └── __init__.py
├── extensions/
│   ├── sandbox/
│   │   ├── Dockerfile.sandbox   # isolated Python 3.12 test runner
│   │   └── safe_apply.py        # Docker → pytest → apply or rollback
│   └── xzero/
│       ├── hyperliquid_connector.py
│       ├── jupiter_enhanced.py   # Jupiter v6 swap + DCA + sniper
│       ├── kalshi_connector.py
│       └── manifold_connector.py
├── mobile/                # SwiftUI OasisApp (iOS)
│   ├── OasisApp.swift
│   ├── StatusView.swift
│   ├── EvolveView.swift
│   ├── LogView.swift
│   └── AgentView.swift
├── models/
│   └── schemas.py         # Pydantic v2 request/response models
├── utils/
│   └── logger.py          # color console + JSON mode
├── tests/                 # pytest coverage for all modules
├── godlocal_v6.py         # FastAPI entrypoint (lifespan + all routes)
├── sleep_scheduler_v6.py  # async nightly scheduler (4 phases)
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/GODLOCAL/godlocal
cd godlocal

# 2. Install
cp .env.example .env
pip install -r requirements.txt

# 3. Start Ollama + pull model
ollama serve
ollama pull qwen3:8b

# 4. Start sandbox (optional — needed for live code patching)
docker compose up -d --build

# 5. Run
python godlocal_v6.py

# Dashboard → http://localhost:8000
# API docs  → http://localhost:8000/docs
```

**Apple Silicon (MLX):**
```bash
pip install mlx-lm
GODLOCAL_MODEL=mlx-community/Qwen2.5-32B-Instruct-4bit python godlocal_v6.py
```

**Steam Deck (ROCm):**
```bash
GODLOCAL_MODEL=qwen3:8b GODLOCAL_DEVICE=cpu python godlocal_v6.py
```

---

## API Reference

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/` | HTML dashboard (FEP metrics, uptime) |
| `GET`  | `/status` | JSON status snapshot |
| `POST` | `/think` | `{"task": "..."}` → LLM response with memory |
| `POST` | `/evolve` | `{"task": "...", "apply": false}` → AutoGenesis |
| `GET`  | `/agent/status` | AgentPool state |
| `POST` | `/agent/swap/{type}` | Hot-swap to specialist agent |
| `GET`  | `/mobile/status` | Compact snapshot for SwiftUI |
| `POST` | `/mobile/evolve` | Trigger evolution from iPhone |
| `POST` | `/feedback` | `?was_corrected=true` → FEP signal |
| `POST` | `/memory/add` | Add item to memory |
| `POST` | `/rollback/{file}` | Rollback file to backup |

---

## Key Features

### 🧠 Brain — Unified LLM
Single `Brain.get()` singleton across the entire app. `LLMBridge` auto-detects:
- **Ollama** (default) — `qwen3:8b`, `llama3`, any Ollama model
- **MLX** — any `mlx-community/` model (Apple Silicon only), runs in thread executor (async-safe)

Memory: ChromaDB `short_term` (50-item rolling) + `long_term` (consolidated nightly).

### 🌙 Sleep Cycle (Nightly at 01:00 UTC)
```
Phase 1  Memory consolidation  short_term → long_term
Phase 2  Self-evolve           self_evolve.py (code quality loop)
Phase 3  Performance analysis  performance_logger.py
Phase 4  AutoGenesis           Plan-and-Execute + DockerSafeApply
```
Override: `python sleep_scheduler_v6.py --now`

### ⚡ AutoGenesis v2
**Free Energy Principle (FEP)** — tracks `correction_rate` across interactions.  
**DockerSafeApply** — patches run in isolated Alpine container → pytest → apply or rollback.  
**Plan-and-Execute** — LLM generates `[PLAN]` JSON first, then `SEARCH/REPLACE` surgical patches.

```bash
# Dry run (default)
curl -X POST http://localhost:8000/evolve   -H "Content-Type: application/json"   -d '{"task": "Add type hints to core/brain.py", "apply": false}'

# Live patch (requires GODLOCAL_AUTOGENESIS_APPLY=true)
curl -X POST http://localhost:8000/evolve   -d '{"task": "...", "apply": true}'
```

### 🤖 AgentPool
6 specialist agents, one in RAM at a time:
| Agent | Model |
|-------|-------|
| `coding` | DeepSeek-Coder-V2-Lite-4bit |
| `trading` | Qwen2.5-72B-4bit |
| `writing` | Mistral-7B-Instruct-4bit |
| `research` | Qwen2.5-32B-4bit |
| `ocr` | LLaVA-1.5-7B-4bit |
| `medical` | Qwen2.5-32B-4bit |

```bash
curl -X POST http://localhost:8000/agent/swap/coding
```

### 📱 SwiftUI Mobile (iOS)
`OasisApp.swift` connects to your local server via Tailscale or ngrok:
- `StatusView` — live FEP metrics
- `EvolveView` — trigger AutoGenesis from iPhone
- `AgentView` — swap agents remotely

---

## Configuration (.env)

```env
GODLOCAL_MODEL=qwen3:8b
GODLOCAL_API_KEY=your-secret-key    # leave empty to disable auth
GODLOCAL_AUTOGENESIS_APPLY=false    # true = live code patching
GODLOCAL_SLEEP_HOUR=1
GODLOCAL_LOG_JSON=false
```

Full reference: [`.env.example`](.env.example)

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v --tb=short
```

Coverage report:
```bash
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

---

## License

**AGPL-3.0** for open source use.  
**Commercial license** available — [contact](mailto:provodnikro@gmail.com) for Developer Pro / Enterprise / Medical B2B tiers.

---

*БОГ || OASIS — Ти тут. Він тут. Разом — Бог.*
