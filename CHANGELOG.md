# Changelog

All notable changes to GodLocal are documented here.  
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) | Versioning: [SemVer](https://semver.org/)

---

## [Unreleased]

## [5.1.0] — 2026-02-24

### Added
- `utils.py` — Centralised `detect_device()`, `Capabilities` dataclass, `format_status()`, `atomic_write()`
- `install.sh` — One-command installer: venv + pip + Ollama + model pull + directory setup
- `CONTRIBUTING.md` — Contributor guide with architecture overview and PR checklist
- `CHANGELOG.md` — This file
- `tests/test_utils.py` — Initial test suite for shared utilities
- `.github/workflows/ci.yml` — GitHub Actions CI: ruff lint + pytest
- `POST /clear` — Reset conversation history endpoint (fixes Telegram `/clear` command)
- `GET /` health check now returns tagline + docs URL
- Graceful shutdown handler (`@app.on_event("shutdown")`) — closes DB, releases GPU memory
- Optional API-key authentication on mutating routes (`/chat`, `/execute`, `/sleep`, `/evolve`) via `GODLOCAL_API_KEY` env var
- `SelfEvolveEngine._init_db()` — auto-creates `conversations.db` with `messages` table on first run
- `SelfEvolveEngine.log_message()` — helper for `godlocal_v5.py` to persist chat turns
- `[LEARNED_PATTERNS]` placeholder in `god_soul.example.md` with backup/rollback documentation
- `extensions/` directory — X-ZERO and Polymarket modules moved here (optional, not imported by core)
- `paroquant_backend.py` — ParoQuant 4-bit LLM backend with `OllamaCompatAdapter`; ROCm/CUDA/MPS/CPU auto-detect
- `polymarket_connector.py` — Polymarket READ-ONLY prediction market signals (`assess_for_trade`, `market_digest`)
- `xzero_delegation.py` — DeepMind-style Eliza→Picobot delegation with `LOCKED_LIMITS` and audit log

### Changed
- `godlocal_v5.py` — Device detection delegates to `utils.detect_device()`
- `godlocal_v5.py` — Capability flags use `utils.Capabilities`
- `godlocal_v5.py` — `Depends` + `APIKeyHeader` imports added from `fastapi`
- `godlocal_v5.py` — `except` blocks now use `logger.exception()` for full stack traces
- `self_evolve.py` — `conversations.db` auto-created on `SelfEvolveEngine.__init__`
- `self_evolve.py` — DB errors use `logger.exception()` instead of `print()`
- `CLAUDE.md` — Added `utils.py` contract, `performance_logger` contract, and `extensions/` guidance

### Fixed
- Telegram `/clear` command no longer returns 404 (missing endpoint added)
- Self-improve loop now receives real conversation data (not demo-only gaps)
- `sleep_cycle_state.json` writes are now crash-safe (atomic write)
- Two independent device-detection implementations consolidated into `utils.detect_device()`

## [5.0.0] — 2026-02-23

### Added
- Initial v5 release
- `godlocal_v5.py` (657 lines) — FastAPI server + `GodLocalAgent` monolith
- `self_evolve.py` (555 lines) — Autonomous knowledge-gap resolution
- `godlocal_telegram.py` (475 lines) — Telegram bridge
- `performance_logger.py` (308 lines) — Session telemetry with LOCKED/snapshot/rollback
- `tasks/lessons.md` — Weekly Monday pattern snapshot
- `.claude/` system (8 files) — Agents, Rules, Skills, post-commit hook
- `god_soul.example.md` — Onboarding soul template
- AGPL-3.0 + Commercial dual license (MongoDB/Grafana model)
- `https://godlocal.vercel.app` landing page
- Show HN launch + 6-tweet thread @kitbtc
- ParoQuant model stack: `z-lab/Qwen3-4B-PARO` (1.8GB daily) + `z-lab/Qwen3-8B-PARO` (4GB sleep_cycle)
