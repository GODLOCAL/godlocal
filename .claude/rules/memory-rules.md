---
name: memory-rules
description: Rules for working with GodLocal memory system (ChromaDB + SQLite + performance_log)
---

# Memory System Rules — GodLocal

## Three memory layers

| Layer | Storage | Purpose | Lifespan |
|-------|---------|---------|---------|
| Short-term | SQLite memories table | Recent interactions, raw facts | 24–72h |
| Long-term | ChromaDB vectors | Promoted high-importance memories | Permanent |
| Soul | god_soul.md | Personality + learned patterns | Permanent |

## sleep_cycle() phases
- **Phase 1**: Hippocampal replay — promote memories (importance ≥ threshold), prune low-signal (< 0.2)
- **Phase 2**: Self-evolve — scan gaps, fill with web research → ChromaDB
- **Phase 3**: Performance analysis — analyze performance_log.jsonl → update god_soul.md

## performance_log.jsonl schema
```json
{
  "ts": "ISO8601",
  "date": "YYYY-MM-DD",
  "session_id": "telegram_chat_id",
  "user_input": "string (max 500 chars)",
  "response_preview": "string (max 300 chars)",
  "was_corrected": false,
  "correction_note": null,
  "tags": ["solana", "trading"],
  "response_len": 1234
}
```

## Correction detection (to implement)
A response was "corrected" if the user's NEXT message:
- Starts with "нет", "не", "no", "wrong", "неправильно"
- Contains "я имел в виду", "I meant", "на самом деле"
- Is a re-ask of essentially the same question

## Do NOT
- Store PII in ChromaDB (user names, wallet addresses, phone numbers)
- Let performance_log.jsonl grow unbounded — archive monthly
- Edit memory.db directly — always use the Memory class methods
- Hardcode soul file path — use SOUL_PATH from performance_logger.py
