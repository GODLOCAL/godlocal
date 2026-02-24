---
name: evolve
description: Run self_evolve.py knowledge gap scanner — finds what GodLocal doesn't know and fills gaps
invocation: /evolve
---

# /evolve

Runs the autonomous knowledge gap scanner (self_evolve.py, 555 lines).

## What it does
1. Scans recent chat interactions for unanswered or poorly-answered questions
2. Identifies knowledge gaps (topics the agent struggled with)
3. Fetches web research to fill those gaps
4. Stores new knowledge in ChromaDB memory

## How to run

```bash
python3 self_evolve.py --hours 24 --max-gaps 10
```

Or via Python:
```python
import asyncio
from self_evolve import SelfEvolveEngine

engine = SelfEvolveEngine()
result = asyncio.run(engine.run_evolution_cycle(
    llm_generate=None,  # uses default
    max_gaps=10,
    hours_back=24
))
print(f"Resolved {result.gaps_resolved}/{result.gaps_found} gaps")
print(f"Topics: {result.topics_resolved}")
```

## After running
Check performance_log stats:
```bash
python3 -c "from performance_logger import get_stats; import json; print(json.dumps(get_stats(7), indent=2))"
```
