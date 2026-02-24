---
name: godlocal-main
description: Main GodLocal development agent — knows the full stack, deployment, and self-improve loop
color: green
model: claude-opus-4-5
---

# GodLocal Main Agent

You are working on **GodLocal** — a sovereign local AI assistant that runs on the user's machine and gets smarter while they sleep.

## Core tagline
> "Your AI. On your machine. Getting smarter while you sleep."

## Stack
- `godlocal_v5.py` — main engine (FastAPI + 14 modules + sleep_cycle())
- `godlocal_telegram.py` — Telegram bot adapter (448 lines, 11 commands)
- `self_evolve.py` — autonomous knowledge gap scanner (555 lines)
- `performance_logger.py` — ao-52 analog: logs every interaction + corrections
- `god_soul.md` — personality file (NOT committed; use god_soul.example.md as template)
- `tasks/lessons.md` — session memory across restarts
- `CLAUDE.md` — AI coding workflow rules (read this first every session)

## Two-terminal startup
```bash
# Terminal 1
python3 godlocal_v5.py

# Terminal 2
python3 godlocal_telegram.py
```

## Deployment target
- **Primary**: Steam Deck (SteamOS/Arch Linux) — Desktop Mode + Konsole
- **Secondary**: Mac (development)
- **Hosting decision pending**: Railway free tier vs Picobot VPS

## Self-improve loop (ALWAYS keep intact)
```
User message
    → godlocal_telegram.py handles it
    → godlocal_v5.py generates response
    → performance_logger.log_interaction() records it    ← WIRE THIS IN
    → Nightly sleep_cycle() analyzes log
    → god_soul.md gets updated with learned patterns
    → Next session: smarter agent
```

## Critical: wiring performance_logger into telegram handler
Still pending — add to godlocal_telegram.py message handler:
```python
from performance_logger import log_interaction
# After generating response:
log_interaction(
    user_input=update.message.text,
    agent_response=response_text,
    session_id=str(update.effective_chat.id),
    tags=extracted_tags  # optional
)
```

## License
AGPL-3.0 + Commercial dual license (MongoDB/GitLab model)
- Free: AGPL-3.0
- Paid: Developer Pro ($9/mo), X100 Soul Gate (10K $X100), Medical B2B ($299-999/mo), Enterprise ($5K-25K/yr)

## Business plan
See: workspace/GODLOCAL-BUSINESS-PLAN.md
