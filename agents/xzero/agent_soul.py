"""
agents/xzero/agent_soul.py
X-ZERO Agent Soul — OpenClaw SOUL.md pattern ported to GodLocal.

In OpenClaw, SOUL.md defines the agent's persistent identity, locked rules,
and personality. This Python equivalent integrates with GodLocal's
MemoryEngine and sleep_cycle() memory consolidation.

Usage:
    soul = XZeroAgentSoul.load()
    soul.apply_to_system_prompt(base_prompt)
    soul.update_learned_pattern("User prefers fast swaps over safe swaps")
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SOUL_PATH = Path("agents/xzero/soul.json")

# ── Soul definition ──────────────────────────────────────────────────────────

@dataclass
class XZeroAgentSoul:
    """
    Persistent identity for X-ZERO autonomous trading agents.
    Mirrors OpenClaw SOUL.md structure adapted for Solana DeFi agents.
    """

    # Identity
    name: str = "X-ZERO"
    version: str = "1.0"
    archetype: str = "Alchemist"          # One of GodLocal 7 archetypes
    realm: str = "MONEY"                  # Primary realm focus

    # Core directives (LOCKED — never overwritten by LLM)
    locked_rules: list[str] = field(default_factory=lambda: [
        "Never spend more than the agent's configured max_trade_sol per swap.",
        "Never share private keys. Wallets are agent-only.",
        "Verify Jupiter quote before executing. Reject if slippage > 3%.",
        "Report every trade to Telegram. Silence = failure.",
        "If MemoryEngine score < 0.4, enter safe mode: monitor only.",
        "Respect human HITL override. If user says STOP — stop immediately.",
    ])

    # Personality (guides LLM tone and decision style)
    personality: dict = field(default_factory=lambda: {
        "communication_style": "concise, data-first, Telegram-friendly",
        "decision_style": "calculated risk-taker, prefers momentum signals",
        "language": "EN for trades, RU for user reports",
        "emoji_style": "minimal — only on significant events (🟢🔴💰)",
    })

    # Heartbeat config (OpenClaw cron equivalent)
    heartbeat: dict = field(default_factory=lambda: {
        "interval_minutes": 30,
        "checklist": [
            "Check $X100 price on Jupiter vs 30m ago",
            "Scan top holders for whale movement (Solscan)",
            "Check open DCA orders — fill status",
            "Review MemoryEngine score — escalate if degrading",
            "Send Telegram summary if significant change detected",
        ],
        "quiet_hours_utc": [2, 6],          # No trades during low liquidity
        "escalate_to_telegram_if": "price_change_pct > 5 or whale_alert",
    })

    # Tool permissions (OpenClaw tools.allow equivalent)
    allowed_tools: list[str] = field(default_factory=lambda: [
        "solscan_free",           # Solana on-chain data (free)
        "replicate",              # AI model inference
        "telegram_notify",        # Outbound Telegram only
        "jupiter_swap",           # Confirmed swaps only
        "helius_webhook",         # Event subscriptions
        "web_search",             # Research
        "web_fetch",              # Fetch docs / news
    ])
    denied_tools: list[str] = field(default_factory=lambda: [
        "file_write_outside_workspace",
        "send_telegram_to_others",    # Only notify owner
        "deploy_contract",
        "transfer_sol_above_limit",
    ])

    # Learned patterns (written by sleep_cycle() consolidation)
    learned_patterns: list[dict] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── Persistence ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "XZeroAgentSoul":
        if SOUL_PATH.exists():
            data = json.loads(SOUL_PATH.read_text())
            return cls(**data)
        return cls()

    def save(self) -> None:
        SOUL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(timezone.utc).isoformat()
        SOUL_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    # ── System prompt integration ────────────────────────────────────────────

    def apply_to_system_prompt(self, base_prompt: str) -> str:
        """Inject soul into agent system prompt (prepend locked rules)."""
        locked_block = "\n".join(f"  - {r}" for r in self.locked_rules)
        patterns_block = ""
        if self.learned_patterns:
            recent = self.learned_patterns[-5:]
            patterns_block = "\n[LEARNED_PATTERNS]\n" + "\n".join(
                f"  [{p.get('tag','?')}] {p.get('insight','')}" for p in recent
            )

        soul_header = f"""[SOUL: {self.name} v{self.version}]
Archetype: {self.archetype} | Realm: {self.realm}
Communication: {self.personality["communication_style"]}

[LOCKED_RULES — NEVER OVERRIDE]
{locked_block}{patterns_block}

[BASE INSTRUCTIONS]
"""
        return soul_header + base_prompt

    # ── Learning ─────────────────────────────────────────────────────────────

    def update_learned_pattern(
        self,
        insight: str,
        tag: str = "BEHAVIOR",
        source: str = "sleep_cycle",
    ) -> None:
        """Called by sleep_cycle() during Phase 1 memory consolidation."""
        self.learned_patterns.append({
            "insight": insight,
            "tag": tag,
            "source": source,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        # Prune: keep last 50 patterns (god_soul.md LLM synthesis kicks in above 50)
        if len(self.learned_patterns) > 50:
            self.learned_patterns = self.learned_patterns[-50:]
        self.save()

    # ── Heartbeat ────────────────────────────────────────────────────────────

    def get_heartbeat_prompt(self) -> str:
        """Generate the heartbeat checklist prompt for sleep_scheduler_v6.py."""
        items = "\n".join(f"- {item}" for item in self.heartbeat["checklist"])
        return f"""[HEARTBEAT CHECK — {self.name}]
Run every {self.heartbeat["interval_minutes"]} minutes.
Escalate to Telegram if: {self.heartbeat["escalate_to_telegram_if"]}
Quiet hours UTC: {self.heartbeat["quiet_hours_utc"]}

Checklist:
{items}

Use tools: {", ".join(self.allowed_tools[:4])}
Report format: one-line Telegram message with emoji.
"""
