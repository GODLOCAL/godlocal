---
name: soul-rules
description: Rules for god_soul.md — the personality file that makes GodLocal feel human
---

# Soul File Rules — GodLocal

## What god_soul.md is
The personality definition file. Every response GodLocal generates starts by reading this file.
It's the equivalent of a system prompt — but it evolves over time via sleep_cycle() Phase 3.

## Structure
```markdown
# [Name] Soul File
[Personality description]

## Communication Style
[How to speak]

## Knowledge & Expertise
[What the agent knows about]

## Values
[What matters to this agent]

## [LEARNED_PATTERNS] — Auto-updated YYYY-MM-DD
- [Pattern auto-appended by sleep_cycle()]
```

## Rules
1. **Top section** (above [LEARNED_PATTERNS]): edit manually only
2. **[LEARNED_PATTERNS]**: auto-appended by performance_logger.update_soul_with_patterns()
3. Keep total file under 500 lines — summarize [LEARNED_PATTERNS] monthly
4. Use god_soul.example.md as the public onboarding template (no private data)
5. Never commit actual god_soul.md to repo — it's in .gitignore
6. Each GodLocal instance has its own unique soul — that's the product differentiator

## Editing guidelines
- Personality should sound like a real person the user would enjoy talking to
- Avoid corporate language, jargon, or AI-sounding phrases
- Include specific interests, quirks, preferred topics
- Communication style: match user's language preference and complexity level

## [LEARNED_PATTERNS] entry format
```
- User prefers concise answers under 3 sentences for factual questions
- User often asks about Solana/DeFi — load solana context proactively
- Correction rate peaked on memory consolidation topics — improve Phase 1 explanation
```
