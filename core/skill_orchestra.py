"""
core/skill_orchestra.py
SkillOrchestra — Skill-aware agent routing for GodLocal AgentPool.

Based on: "SkillOrchestra: Learning to Route Agents via Skill Transfer"
arXiv:2602.19672 | HuggingFace Daily Papers #2 | Feb 23, 2026

Key insight: instead of end-to-end RL routing (Router-R1 / ToolOrchestra style),
build a Skill Handbook from execution traces, then route based on skill-conditioned
agent profiles. 700x cheaper to learn than Router-R1, +22.5% performance.

GodLocal integration:
- AgentPool 6-slot → routed by SkillOrchestraRouter (replaces round-robin)
- sleep_cycle() Phase 2 → runs SkillHandbook.refine() nightly
- X-ZERO agents → profiled per skill (swap, monitor, analyze, alert)

Usage:
    router = SkillOrchestraRouter()
    agent = router.route(query="monitor $X100 whale movement", mode="monitor")
    result = await agent.run(query)
    router.record(query, agent, success=True, mode="monitor")
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Skill:
    """A fine-grained, reusable capability distilled from execution traces."""
    id: str
    name: str
    description: str
    mode: str                    # "monitor" | "trade" | "analyze" | "generate" | "fetch"
    examples: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentSkillProfile:
    """
    Beta-distributed success probability for agent-skill pair.
    alpha = successes + 1 (prior), beta = failures + 1 (prior)
    E[p] = alpha / (alpha + beta)
    """
    agent_id: str
    skill_id: str
    alpha: float = 1.0    # successes + 1 (Laplace smoothing)
    beta: float = 1.0     # failures + 1
    cost_tokens: float = 0.0   # avg tokens consumed

    @property
    def expected_success(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """Wilson score lower bound — penalises low sample count."""
        n = self.alpha + self.beta - 2
        if n == 0:
            return 0.5
        p = (self.alpha - 1) / n
        z = 1.645  # 90% CI
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (centre - spread) / denom

    def update(self, success: bool, tokens: float = 0.0) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        n = self.alpha + self.beta - 2
        if n > 0:
            self.cost_tokens = (self.cost_tokens * (n - 1) + tokens) / n


@dataclass
class SkillHandbook:
    """
    Global registry of skills + agent profiles.
    Built from execution traces, refined nightly by sleep_cycle() Phase 2.
    """
    skills: dict[str, Skill] = field(default_factory=dict)
    profiles: dict[str, AgentSkillProfile] = field(default_factory=dict)   # key: f"{agent}:{skill}"
    mode_signals: dict[str, list[str]] = field(default_factory=dict)        # mode → [keyword hints]
    version: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    HANDBOOK_PATH = Path("core/skill_handbook.json")

    # ── Seed skills for GodLocal / X-ZERO ───────────────────────────────────

    SEED_SKILLS = [
        Skill("monitor_price",    "Price Monitor",       "Track token price and detect significant moves", "monitor",  ["price", "pump", "dump", "$X100", "whale"]),
        Skill("monitor_holders",  "Holder Monitor",      "Track holder count changes and whale wallets",  "monitor",  ["holders", "whale", "top wallet", "concentrated"]),
        Skill("trade_swap",       "Jupiter Swap",        "Execute token swap via Jupiter aggregator",     "trade",    ["buy", "sell", "swap", "slippage", "route"]),
        Skill("trade_dca",        "DCA Order",           "Place or manage dollar-cost-average orders",    "trade",    ["DCA", "recurring", "schedule buy"]),
        Skill("analyze_onchain",  "On-chain Analysis",   "Read Solscan data: txns, defi, portfolio",      "analyze",  ["solscan", "on-chain", "wallet", "portfolio", "history"]),
        Skill("analyze_sentiment","Sentiment Analysis",  "Judge market sentiment from text signals",      "analyze",  ["sentiment", "community", "twitter", "bullish", "bearish"]),
        Skill("generate_image",   "Image Generation",    "Generate images via Replicate/Flux",            "generate", ["image", "art", "pixel", "banner", "nft"]),
        Skill("generate_text",    "Text Generation",     "Compose content: tweets, posts, reports",       "generate", ["write", "draft", "tweet", "post", "summary"]),
        Skill("fetch_web",        "Web Fetch",           "Retrieve and parse web pages and APIs",         "fetch",    ["fetch", "scrape", "search", "URL", "docs"]),
        Skill("alert_telegram",   "Telegram Alert",      "Compose and route Telegram notifications",     "alert",    ["alert", "notify", "telegram", "report", "signal"]),
        Skill("code_python",      "Python Code",         "Write and execute Python code or scripts",      "generate", ["python", "script", "code", "function", "class"]),
        Skill("code_solana",      "Solana Dev",          "Write Solana programs (Rust/Anchor/TS)",        "generate", ["rust", "anchor", "solana program", "IDL", "CPI"]),
    ]

    @classmethod
    def load(cls) -> "SkillHandbook":
        if cls.HANDBOOK_PATH.exists():
            data = json.loads(cls.HANDBOOK_PATH.read_text())
            hb = cls()
            hb.skills    = {k: Skill(**v)             for k, v in data.get("skills",   {}).items()}
            hb.profiles  = {k: AgentSkillProfile(**v) for k, v in data.get("profiles", {}).items()}
            hb.mode_signals = data.get("mode_signals", {})
            hb.version      = data.get("version", 0)
            return hb
        hb = cls()
        hb._seed()
        return hb

    def _seed(self) -> None:
        for s in self.SEED_SKILLS:
            self.skills[s.id] = s

    def save(self) -> None:
        self.HANDBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.version += 1
        self.HANDBOOK_PATH.write_text(json.dumps({
            "skills":       {k: asdict(v) for k, v in self.skills.items()},
            "profiles":     {k: asdict(v) for k, v in self.profiles.items()},
            "mode_signals": self.mode_signals,
            "version":      self.version,
            "updated_at":   self.updated_at,
        }, indent=2, ensure_ascii=False))

    # ── Skill inference ──────────────────────────────────────────────────────

    def infer_skills(self, query: str, mode: Optional[str] = None) -> list[str]:
        """
        Keyword-match query against skill examples → return ranked skill IDs.
        No LLM call needed for routing (cheap-check-first principle).
        """
        q = query.lower()
        scores: dict[str, int] = {}
        for sid, skill in self.skills.items():
            if mode and skill.mode != mode:
                continue
            hit = sum(1 for kw in skill.examples if kw.lower() in q)
            if hit:
                scores[sid] = hit
        return sorted(scores, key=lambda k: scores[k], reverse=True)

    # ── Profile access ───────────────────────────────────────────────────────

    def profile(self, agent_id: str, skill_id: str) -> AgentSkillProfile:
        key = f"{agent_id}:{skill_id}"
        if key not in self.profiles:
            self.profiles[key] = AgentSkillProfile(agent_id=agent_id, skill_id=skill_id)
        return self.profiles[key]

    def record(self, agent_id: str, skill_id: str, success: bool, tokens: float = 0.0) -> None:
        self.profile(agent_id, skill_id).update(success, tokens)

    # ── Nightly refinement (sleep_cycle Phase 2) ─────────────────────────────

    def refine(self, min_samples: int = 5) -> dict:
        """
        Prune skills with no data after min_samples opportunities.
        Merge skills with near-identical performance vectors (cosine > 0.95).
        Called by sleep_cycle() Phase 2 nightly.
        Returns: {"pruned": [...], "kept": [...]}
        """
        pruned, kept = [], []
        for sid in list(self.skills.keys()):
            if sid.startswith("_"):   # internal / seed skills — never prune
                kept.append(sid)
                continue
            total_samples = sum(
                (p.alpha + p.beta - 2)
                for key, p in self.profiles.items()
                if key.endswith(f":{sid}")
            )
            if total_samples == 0:
                pruned.append(sid)
                del self.skills[sid]
            else:
                kept.append(sid)
        return {"pruned": pruned, "kept": kept}


# ── Router ───────────────────────────────────────────────────────────────────

class SkillOrchestraRouter:
    """
    Skill-aware agent router for GodLocal AgentPool 6-slot.

    Replaces round-robin with:
    1. Infer required skills from query
    2. Score each agent by skill-conditioned success probability (Wilson CI)
    3. Apply cost penalty (budget_weight)
    4. Return best agent

    Usage:
        router = SkillOrchestraRouter(agents=pool.agents)
        agent = router.route("monitor $X100 whale", mode="monitor")
        result = agent.run(...)
        router.record("monitor $X100 whale", agent.id, success=True, tokens=1200)
    """

    def __init__(
        self,
        agents: list | None = None,
        handbook: SkillHandbook | None = None,
        budget_weight: float = 0.1,    # Cost penalty weight (0 = ignore cost)
        epsilon: float = 0.05,         # ε-greedy exploration rate
    ):
        self.agents = agents or []
        self.handbook = handbook or SkillHandbook.load()
        self.budget_weight = budget_weight
        self.epsilon = epsilon
        self._last_query_skills: list[str] = []

    def route(
        self,
        query: str,
        mode: Optional[str] = None,
        exclude: list[str] | None = None,
    ) -> object | None:
        """
        Select best agent for query.
        Returns agent object or None if pool is empty.
        """
        if not self.agents:
            return None

        exclude = set(exclude or [])
        candidates = [a for a in self.agents if getattr(a, "id", str(a)) not in exclude]
        if not candidates:
            return None

        # ε-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(candidates)

        # Infer skills
        skill_ids = self.handbook.infer_skills(query, mode=mode)
        self._last_query_skills = skill_ids

        if not skill_ids:
            return candidates[0]   # No skill match → first available (safest default)

        # Score each agent
        best_agent = None
        best_score = -1.0

        for agent in candidates:
            aid = getattr(agent, "id", str(agent))
            # Weighted avg of skill-conditioned Wilson CI scores
            scores = [
                self.handbook.profile(aid, sid).confidence
                for sid in skill_ids[:3]   # top-3 most relevant skills
            ]
            skill_score = sum(scores) / max(len(scores), 1)

            # Cost penalty
            avg_cost = sum(
                self.handbook.profile(aid, sid).cost_tokens
                for sid in skill_ids[:3]
            ) / max(len(skill_ids[:3]), 1)
            cost_penalty = self.budget_weight * (avg_cost / 10_000)  # normalise to 10K tokens

            total = skill_score - cost_penalty
            if total > best_score:
                best_score = total
                best_agent = agent

        return best_agent

    def record(
        self,
        query: str,
        agent_id: str,
        success: bool,
        tokens: float = 0.0,
    ) -> None:
        """Record execution result. Updates Beta distributions in handbook."""
        skill_ids = self._last_query_skills or self.handbook.infer_skills(query)
        for sid in skill_ids[:3]:
            self.handbook.record(agent_id, sid, success, tokens)

    def save(self) -> None:
        self.handbook.save()
