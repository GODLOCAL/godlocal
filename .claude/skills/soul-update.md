---
name: soul-update
description: Update god_soul.md with new learned patterns, preferences, or behavioral rules
invocation: /soul-update
---

# /soul-update

Manually triggers god_soul.md update based on latest performance_log analysis.

## Steps

1. Check current performance stats:
```bash
python3 -c "
from performance_logger import get_stats, get_correction_rate
import json
print('Stats (7d):', json.dumps(get_stats(7), indent=2))
print('Correction rate (24h):', f'{get_correction_rate(24):.1%}')
"
```

2. Run pattern analysis and update soul:
```bash
python3 -c "
from performance_logger import analyze_patterns, update_soul_with_patterns
from godlocal_v5 import LLMEngine

llm = LLMEngine()
patterns = analyze_patterns(llm.complete, hours_back=48)
print('Patterns found:', patterns.get('patterns'))
print('Soul updates:', patterns.get('soul_updates'))

updated = update_soul_with_patterns(patterns)
print('Soul updated:', updated)
"
```

3. Open god_soul.md and review the [LEARNED_PATTERNS] section at the bottom.
   Edit manually if needed — the LLM output is a starting point, not gospel.

4. If correction_rate > 20%: investigate which topics cause corrections:
```bash
python3 -c "
import json
from performance_logger import get_recent_interactions
corrections = [i for i in get_recent_interactions(48) if i['was_corrected']]
for c in corrections:
    print(f'Q: {c["user_input"][:80]}')
    print(f'  Fix: {c["correction_note"]}')
    print()
"
```

## god_soul.md structure reminder
- Top section: personality, name, communication style (edit manually)
- [LEARNED_PATTERNS]: auto-appended by sleep_cycle() Phase 3
- Keep under 500 lines total — summarize/prune [LEARNED_PATTERNS] monthly
