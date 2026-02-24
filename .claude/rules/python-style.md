---
name: python-style
description: Python coding standards for GodLocal codebase
---

# Python Style Rules — GodLocal

## Core principles
- Python 3.11+
- Type hints on all public functions
- Docstrings on all classes and public methods
- Max function length: 50 lines (split if longer)
- Max file length: 600 lines (split into modules)

## Error handling
- Always wrap external calls (Ollama, ChromaDB, Telegram API) in try/except
- Log warnings with logger.warning(), errors with logger.error()
- Never let exceptions bubble up to the user silently — always return a fallback

## Imports
- Standard library first, then third-party, then local
- Use `from pathlib import Path` (not `os.path`)
- Lazy imports inside functions for optional heavy deps (e.g., `import chromadb`)

## Async
- FastAPI endpoints: async def
- Background tasks (sleep_cycle): run via asyncio.new_event_loop() in threads
- Telegram handlers: async def

## Data storage
- ChromaDB for vector memory (godlocal_data/chroma/)
- SQLite for structured memory (godlocal_data/memory.db)
- JSONL for logs (godlocal_data/performance_log.jsonl, godlocal_data/patterns_history.jsonl)
- Markdown for human-readable state (god_soul.md, tasks/lessons.md)

## Configuration
- All config via environment variables with sensible defaults
- Use CFG dataclass pattern (see godlocal_v5.py Config class)
- Never hardcode paths — use Path(os.getenv("X", "default"))

## Testing
- Every new module gets a `if __name__ == "__main__":` smoke test block
- Integration tests go in tasks/lessons.md as manual verification steps
