# CLAUDE.md — GodLocal

> Auto-updated by Claude after every correction.
> Target: ≤2.5K tokens. Every line earns its place.

## Project Identity

GodLocal v6 — sovereign local AI layer. Stack:
- Core: `godlocal_v6.py`, `core/brain.py` (Brain / LLMBridge / MemoryEngine)
- Agents: `agents/` — AutoGenesisV2, AgentPool 6-slot, GoalExecutor, ClawFeedAgent
- X-ZERO: `agents/xzero/` — XZeroAgentSoul, XZeroHeartbeat
- Extensions: `extensions/xzero/` (CIMD connectors: SolscanFree, VinextReplicate, MoonPayAgents)
- Mobile: `mobile/` — NexaSDK, LLMBridgeNexa.swift, AudioBridgeMLX.swift
- Web: `index.html` (Three.js), `server.js`, `contributors.html`
- Skills: `.claude/skills/` — 57 curated + 930+ via antigravity

## Code Style (LOCKED)

- Python 3.11+, type hints on all public methods
- `from __future__ import annotations` at top of every Python file
- Connectors: always extend `CIMDConnectorBase` from `extensions/xzero/cimd_connector_base.py`
- Commit format: `БОГ || vX.Y.Z: <feature> [<domain>]`
- Patch format: SEARCH/REPLACE surgical blocks (never full-file rewrite unless new file)
- New file commits: use `GITHUB_COMMIT_MULTIPLE_FILES` with `upserts[]`
- Always `encoding="utf-8"` on file writes

## Architecture Rules (LOCKED)

- Every new connector: (1) `openapi_schema()`, (2) `registration_manifest()`, (3) `run_tool(tool, params)`
- Env vars in connectors — raise `EnvironmentError` with setup instructions if missing
- Sleep cycle phases order: Phase1 Memory → Phase2 Self-evolve → Phase3 Perf → Phase3b Soul prune → Phase4 AutoGenesis → Phase5 XZeroHeartbeat
- `god_soul.md` LOCKED sections: never overwrite. LLM synthesis triggers at >50 learned patterns
- Auto-rollback if `correction_rate` degrades >10 percentage points
- SYSTEM_PROMPT requires `[PLAN]` block before patches (Devin-style, 256-token JSON)

## Parallelism Patterns

- Use `git worktrees` + separate Claude sessions for parallel tracks (Roblox / GodLocal / X100 content)
- Batch independent GitHub commits in one `GITHUB_COMMIT_MULTIPLE_FILES` call
- Run heartbeat + sleep_scheduler as non-blocking background tasks

## Naming Conventions

- Connectors: `<Service>Connector` in `extensions/xzero/<service>_connector.py`
- Agent souls: `<Name>AgentSoul` in `agents/<name>/agent_soul.py`
- Heartbeats: `<Name>Heartbeat` in `agents/<name>/heartbeat.py`
- Skills: kebab-case dirs in `.claude/skills/<skill-name>/`
- Tweet→implement commits: include source handle in commit body, e.g. `[@crypsaf tweet 2026-02-25]`

## X100 OASIS Context

- Token: $X100, Solana, 100M supply; tiers Explorer(100)→Sovereign(1M)
- Agent wallet: non-custodial, funded via MoonPayAgentsConnector
- Swaps: Jupiter API (slippage ≤3% enforced in XZeroAgentSoul locked rules)
- Monitoring: SolscanFreeConnector — `token_holders_total`, `top_address_transfers`, `token_data`
- Alerts: Telegram only, gated by XZeroHeartbeat (quiet hours UTC 2–6)
- Company structure: 3+ agents → Squads multisig → ROI leaderboard

## Pending Wiring (Next AI Tasks)

1. GoalExecutor + ClawFeedAgent → `godlocal_v6.py` (`/goals/*`, `/feed*` endpoints)
2. XZeroHeartbeat → `sleep_scheduler_v6.py` Phase 5 (non-blocking asyncio)
3. NexaSDK xcframework → activate from stub (`./setup_nexa.sh`)

## Self-Update Rule

After every correction or new pattern:
```
Tell Claude: "Update CLAUDE.md so you don't make that mistake again"
```
Keep total size ≤ 2.5K tokens. Synthesize, don't append.
