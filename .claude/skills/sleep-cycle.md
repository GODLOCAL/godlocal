---
name: sleep-cycle
description: Run GodLocal sleep_cycle() — hippocampal replay + self-evolution + performance analysis
invocation: /sleep-cycle
---

# /sleep-cycle

Triggers GodLocal nightly consolidation cycle manually.

## What it does
- **Phase 1** — Hippocampal replay: promote high-importance memories, prune low-signal ones, extract insights
- **Phase 2** — Self-evolution: scan knowledge gaps (self_evolve.py), resolve with web research
- **Phase 3** — Performance analysis: analyze performance_log.jsonl, update god_soul.md with learned patterns

## Steps

1. Check that godlocal_v5.py process is running:
   ```bash
   curl -s http://localhost:8000/health | python3 -m json.tool
   ```

2. Trigger sleep_cycle via FastAPI endpoint:
   ```bash
   curl -s -X POST http://localhost:8000/sleep | python3 -m json.tool
   ```

3. Read the result — check `self_evolve` and `performance` keys in response.

4. If `performance.soul_updated == true` — open god_soul.md and verify the new [LEARNED_PATTERNS] section at the bottom.

5. Open `tasks/lessons.md` — verify new entry was appended.

## Expected output
```json
{
  "status": "ok",
  "promoted_count": N,
  "pruned_count": N,
  "self_evolve": {"gaps_found": N, "gaps_resolved": N},
  "performance": {"correction_rate": 0.XX, "soul_updated": true}
}
```

## If endpoint is down
Run directly:
```bash
python3 -c "
from godlocal_v5 import GodLocalAgent
import os
agent = GodLocalAgent()
result = agent.run_sleep_cycle()
print(result)
"
```
