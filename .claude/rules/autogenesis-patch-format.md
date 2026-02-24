# AutoGenesis Patch Format

When generating code patches for GodLocal AutoGenesis, always use SEARCH/REPLACE format.

## Format

```
### filename.py
<<<<<<< SEARCH
exact verbatim code to replace (no omissions, no "..." shortcuts)
=======
new replacement code
>>>>>>> REPLACE
```

## Rules

1. **SEARCH must be verbatim** — copy-paste exact code from the file, including whitespace and indentation
2. **Multiple blocks** — use multiple SEARCH/REPLACE blocks per file if needed
3. **Surgical** — only include the changed section, not the whole file
4. **Fallback** — use full ```python``` block only if SEARCH/REPLACE is not applicable (new files, full rewrites)
5. **[PLAN] first** — always output a [PLAN] block before any patches

## [PLAN] Format

```
[PLAN]
- Mode: CODING|TRADING|WRITING|MEDICAL|ANALYSIS
- Prediction error: what does the current code get wrong?
- Minimal change: one sentence on what changes
- Risk: LOW|MEDIUM|HIGH — reason
- Files: filename.py, other.py
[/PLAN]
```

## Why SEARCH/REPLACE

- Cursor, Windsurf, Devin all use this format — proven in production
- 90% fewer tokens than full-file rewrites
- Clean rollback: backup diff is surgical, not full-file
- Clearer review: exactly what changed is visible immediately
