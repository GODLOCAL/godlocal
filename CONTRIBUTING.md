# Contributing to GodLocal

Thanks for your interest in GodLocal — a sovereign, self-improving AI that runs on your machine.  
GodLocal is AGPL-3.0 + Commercial dual-licensed. By contributing, you agree your code may be used under both licenses.

---

## Quick Start

```bash
git clone https://github.com/GODLOCAL/godlocal.git
cd godlocal
bash install.sh          # sets up venv + Ollama + model
source ~/godlocal-env/bin/activate
cp god_soul.example.md god_soul.md   # personalise your soul file
```

## Development Rules (see also CLAUDE.md)

1. **Two-terminal stack** — `godlocal_v5.py` must start before `godlocal_telegram.py`
2. **Never commit `god_soul.md`** — it's user-private and `.gitignore`-d
3. **Never hardcode secrets** — use `.env` / environment variables only
4. **Device detection** — import `detect_device()` from `utils.py`, never re-implement
5. **Capability flags** — use `Capabilities.*` from `utils.py`, never re-check inline
6. **Log exceptions fully** — use `logger.exception(...)`, not `print(f"Error: {e}")`
7. **Atomic writes** — use `atomic_write()` from `utils.py` for any state files

## Code Style

```bash
pip install ruff
ruff check .     # lint
ruff format .    # format
```

- Line length: 100
- Type hints on all public functions
- Docstrings on all public classes and non-trivial methods

## Pull Request Checklist

- [ ] `ruff check .` passes with no errors
- [ ] New feature has at least one test in `tests/`
- [ ] `CLAUDE.md` updated if you change AI workflow logic
- [ ] No secrets committed (run `git diff --cached` to check)
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`

## Architecture Overview

```
godlocal_v5.py          ← FastAPI server + GodLocalAgent (core)
godlocal_telegram.py    ← Telegram bridge (calls localhost:8000)
self_evolve.py          ← Autonomous knowledge-gap resolution
performance_logger.py   ← Session telemetry → soul pattern updates
paroquant_backend.py    ← ParoQuant 4-bit LLM backend (Qwen3 family)
sleep_scheduler.py      ← Nightly sleep_cycle() scheduler (run standalone)
utils.py                ← Shared helpers: DeviceDetector, Capabilities, format_status
extensions/             ← Optional modules (X-ZERO trading, Polymarket)
tests/                  ← pytest test suite
```

## What We Actively Need

| Area | Priority |
|------|----------|
| ConnectorsModule (Composio SDK, 500+ integrations) | 🔴 High |
| Streaming Telegram responses (token-by-token) | 🟠 Medium |
| Unit tests for MemoryEngine | 🟠 Medium |
| Unit tests for SelfEvolveEngine | 🟠 Medium |
| API documentation (mkdocs) | ⬜ Low |

## Reporting Bugs

Open a GitHub Issue with:
- GodLocal version (`GET http://localhost:8000/status`)
- OS + Python version
- Full error traceback from logs
- Steps to reproduce
