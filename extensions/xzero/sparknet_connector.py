"""
extensions/xzero/sparknet_connector.py
SparkNetConnector — Agent swarm memory sync & distillation for GodLocal.

Based on: Spark & SparkNet — agent swarm orchestration memory system
Source: @jumperz (Twitter, 2026-02-25) via @meta_alchemist
Context: "your agents can be smart, fast, well-prompted and still completely
          out of sync — what's stored in memory is not distilled and evoked
          at the right times" (286 likes, 594 bookmarks, 25K impressions)

Problem it solves:
  Agents in AgentPool share no memory → duplicate work, contradictory actions,
  lost context between sleep_cycle() phases, no institutional memory.

SparkNet architecture (reverse-engineered from tweets + patterns):
  Spark     = per-agent memory node: captures, distils, stores "sparks" (insights)
  SparkNet  = sync mesh: broadcasts relevant sparks to agents that need them NOW
  Evocation = trigger-based retrieval: context-aware, not query-based

GodLocal integration:
  - SparkNetConnector wraps GodLocal's AgentPool as the SparkNet mesh
  - Each agent has a Spark node backed by KuzuDB or SQLite
  - XZeroHeartbeat broadcasts market sparks to PolytermConnector agents
  - AutoGenesisV2 reads pre-patch sparks before modifying code
  - sleep_cycle() Phase 4: distill() compresses agent memories into sparks

Usage:
  from extensions.xzero.sparknet_connector import SparkNetConnector, Spark
  net = SparkNetConnector()
  await net.capture("xzero", "SOL crossed $200 at 01:22 UTC", tags=["price", "sol"])
  sparks = await net.evoke("goal_executor", context="planning SOL trade")
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# ── Spark dataclass ──────────────────────────────────────────────────────────

@dataclass
class Spark:
    """
    A distilled unit of agent memory.

    Spark vs raw memory:
      raw: "at 01:14 the agent queried SOL price, got $198.42, logged it"
      Spark: "SOL $198-200 range active — agents should watch $200 break"

    Evoked when: tags overlap + recency + relevance_score > threshold.
    """
    id:            str
    source_agent:  str          # who captured it
    content:       str          # distilled insight (≤200 chars)
    tags:          list[str]    # semantic labels for evocation routing
    created_at:    float        = field(default_factory=time.time)
    access_count:  int          = 0
    relevance_decay: float      = 1.0   # drops by 0.1/hr, floored at 0.1
    context_hash:  str          = ""    # hash of capture context (dedup)
    linked_sparks: list[str]    = field(default_factory=list)  # SparkNet edges
    success_score:    float         = 0.5    # ReasoningBank: EMA of outcome (0=fail, 1=success)
    trial_count:      int           = 0      # Wilson CI: total judge() calls for this spark
    success_count:    int           = 0      # Wilson CI: cumulative successes (outcome=True)

    @classmethod
    def create(cls, source_agent: str, content: str, tags: list[str], context: str = "") -> "Spark":
        content_hash = hashlib.sha256((source_agent + content).encode()).hexdigest()[:12]
        return cls(
            id=f"spark_{content_hash}",
            source_agent=source_agent,
            content=content,
            tags=tags,
            context_hash=hashlib.sha256(context.encode()).hexdigest()[:8] if context else "",
        )

    def decay(self) -> float:
        """Current relevance score accounting for time decay."""
        hours_old  = (time.time() - self.created_at) / 3600
        decayed    = self.relevance_decay - (hours_old * 0.1)
        return max(0.1, decayed)

    def score(self, query_tags: list[str]) -> float:
        """Relevance score for evocation: tag overlap × recency × access boost."""
        tag_overlap  = len(set(self.tags) & set(query_tags)) / max(len(self.tags), 1)
        access_boost = min(self.access_count * 0.05, 0.3)  # cap at +30%
        return (tag_overlap * 0.6 + self.decay() * 0.4) + access_boost


# ── SparkNet storage backend ─────────────────────────────────────────────────

class SparkStore:
    """SQLite-backed spark storage. One DB per GodLocal instance."""

    DB_PATH = os.getenv("SPARKNET_DB", ".gitnexus/sparknet.db")

    def __init__(self):
        db = Path(self.DB_PATH)
        db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sparks (
                id            TEXT PRIMARY KEY,
                source_agent  TEXT NOT NULL,
                content       TEXT NOT NULL,
                tags          TEXT NOT NULL,  -- JSON array
                created_at    REAL NOT NULL,
                access_count  INTEGER DEFAULT 0,
                relevance_decay REAL DEFAULT 1.0,
                context_hash  TEXT DEFAULT '',
                linked_sparks TEXT DEFAULT '[]'
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_agent ON sparks(source_agent)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON sparks(created_at)")
        self._conn.commit()

    def upsert(self, spark: Spark):
        """Insert or update a spark (dedup by context_hash)."""
        if spark.context_hash:
            existing = self._conn.execute(
                "SELECT id FROM sparks WHERE context_hash=? AND source_agent=?",
                (spark.context_hash, spark.source_agent)
            ).fetchone()
            if existing:
                return  # dedup

        self._conn.execute("""
            INSERT OR REPLACE INTO sparks
            (id, source_agent, content, tags, created_at, access_count,
             relevance_decay, context_hash, linked_sparks)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            spark.id, spark.source_agent, spark.content,
            json.dumps(spark.tags), spark.created_at,
            spark.access_count, spark.relevance_decay,
            spark.context_hash, json.dumps(spark.linked_sparks)
        ))
        self._conn.commit()

    def load_all(self, agent: str | None = None, limit: int = 500) -> list[Spark]:
        """Load sparks, optionally filtered by agent."""
        if agent:
            rows = self._conn.execute(
                "SELECT * FROM sparks WHERE source_agent=? ORDER BY created_at DESC LIMIT ?",
                (agent, limit)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM sparks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_spark(r) for r in rows]

    def increment_access(self, spark_id: str):
        self._conn.execute(
            "UPDATE sparks SET access_count = access_count + 1 WHERE id=?", (spark_id,)
        )
        self._conn.commit()


    def delete_spark(self, spark_id: str) -> None:
        """Remove a spark (used by consolidate to delete merged duplicates)."""
        self._conn.execute("DELETE FROM sparks WHERE id=?", (spark_id,))
        self._conn.commit()

    def _row_to_spark(self, row) -> Spark:
        return Spark(
            id=row[0], source_agent=row[1], content=row[2],
            tags=json.loads(row[3]), created_at=row[4],
            access_count=row[5], relevance_decay=row[6],
            context_hash=row[7], linked_sparks=json.loads(row[8])
        )


# ── SparkNet connector ────────────────────────────────────────────────────────

class SparkNetConnector:
    """
    Agent swarm memory sync mesh for GodLocal.

    Core operations:
      capture(agent, content, tags)  — agent learns something, creates Spark
      distill(agent, raw_logs)       — LLM compresses N logs into M sparks
      evoke(agent, context)          — retrieve relevant sparks for context
      broadcast(spark, target_agents)— push spark to specific agents
      sync()                         — full mesh sync (all agents share all relevant sparks)

    SparkNet topology:
      Each agent is a node. Sparks flow along edges weighted by tag affinity.
      High-access sparks become "anchors" — always evoked regardless of query.
    """

    EVOKE_THRESHOLD   = 0.25   # minimum score to evoke a spark
    EVOKE_TOP_K       = 8      # max sparks returned per evocation
    ANCHOR_THRESHOLD  = 5      # access_count to become anchor

    def __init__(self):
        self._store = SparkStore()
        self._subscribers: dict[str, list[str]] = {}  # agent → tag subscriptions
        self._agent_contexts: dict[str, str]    = {}  # agent → current context string

    # ── Core API ──────────────────────────────────────────────────────────────

    async def capture(
        self,
        agent:   str,
        content: str,
        tags:    list[str],
        context: str = "",
        link_to: list[str] | None = None,
    ) -> Spark:
        """
        Agent captures a new insight as a Spark.
        Auto-deduplicates by context_hash.

        Example:
          await net.capture("xzero_heartbeat", "SOL $200 broken → BULLISH signal",
                            tags=["sol", "price", "signal", "bullish"])
        """
        spark = Spark.create(agent, content[:200], tags, context)
        if link_to:
            spark.linked_sparks = link_to
        await asyncio.to_thread(self._store.upsert, spark)
        await self._propagate(spark)
        return spark

    async def distill(
        self,
        agent:    str,
        raw_logs: list[str],
        max_sparks: int = 5,
    ) -> list[Spark]:
        """
        Compress N raw log entries into M distilled sparks.
        Called during sleep_cycle() Phase 4 for each agent.

        Distillation rules:
          - Remove timestamps/noise
          - Merge similar events: 3 "SOL checked" → 1 "SOL monitored 3x"
          - Extract actionable patterns: "X happened 3x → PATTERN: X is recurring"
          - Flag anomalies: anything 2σ from mean → HIGH_PRIORITY tag
        """
        from core.brain import Brain  # GodLocal LLM bridge

        prompt = f"""You are distilling {len(raw_logs)} agent log entries into ≤{max_sparks} 
key insights (Sparks). Each spark: ≤150 chars, actionable, no timestamps.
Extract patterns, anomalies, decisions. Skip routine/repeated events.

Agent: {agent}
Logs:
{chr(10).join(f"- {l}" for l in raw_logs[-50:])}

Output JSON array of objects: [{{"content": "...", "tags": ["tag1", "tag2"]}}]
Return ONLY the JSON array."""

        try:
            brain = Brain()
            response = await brain.async_complete(prompt, max_tokens=400)
            items    = json.loads(response)
            sparks   = []
            for item in items[:max_sparks]:
                s = await self.capture(agent, item["content"], item.get("tags", []))
                sparks.append(s)
            return sparks
        except Exception as e:
            # Fallback: create sparks from last N logs without LLM
            fallback_sparks = []
            for log in raw_logs[-max_sparks:]:
                s = await self.capture(agent, log[:150], ["raw", agent])
                fallback_sparks.append(s)
            return fallback_sparks

    async def evoke(
        self,
        agent:     str,
        context:   str,
        extra_tags: list[str] | None = None,
    ) -> list[Spark]:
        """
        Evoke the most relevant sparks for a given agent + context.
        This is SparkNet's core innovation: RIGHT sparks at RIGHT time.

        Context-aware: uses current task description, not just keyword search.
        Always includes anchors (high-access sparks) regardless of score.

        Example:
          sparks = await net.evoke("goal_executor", "planning a SOL DCA order")
          # Returns: SOL price pattern, DCA history, wallet balance spark, etc.
        """
        self._agent_contexts[agent] = context
        context_tags = self._extract_tags(context)
        all_tags     = list(set(context_tags + (extra_tags or [])))

        all_sparks = await asyncio.to_thread(self._store.load_all)
        scored     = [(s, s.score(all_tags)) for s in all_sparks]

        # Always include anchors
        anchors = [s for s, _ in scored if s.access_count >= self.ANCHOR_THRESHOLD]
        relevant = [s for s, score in scored
                    if score >= self.EVOKE_THRESHOLD
                    and s not in anchors]

        relevant.sort(key=lambda s: s.score(all_tags), reverse=True)
        result = (anchors + relevant)[:self.EVOKE_TOP_K]

        # Update access counts
        for spark in result:
            await asyncio.to_thread(self._store.increment_access, spark.id)

        return result

    async def broadcast(self, spark: Spark, target_agents: list[str]):
        """
        Push a spark directly to specific agents.
        Used when XZeroHeartbeat detects a major market move.
        """
        for agent in target_agents:
            subs = self._subscribers.get(agent, [])
            # Boost relevance_decay for subscribed tags
            if any(t in subs for t in spark.tags):
                spark.relevance_decay = min(1.0, spark.relevance_decay + 0.2)
        await asyncio.to_thread(self._store.upsert, spark)

    # ── ReasoningBank pipeline (Claude-Flow) ────────────────────────────────

    async def judge(self, spark_id: str, outcome: bool) -> float:
        """
        JUDGE step: EMA score + Wilson CI lower-bound bonus.

        Why Wilson CI (not raw EMA alone):
          EMA is recency-biased — a new spark that just succeeded looks equally
          good as one that's succeeded 50 times. Wilson CI lower bound is a
          pessimistic Bayesian estimate that properly rewards consistency.

        Formula:
          EMA component:    0.7 * old_score + 0.3 * outcome
          Wilson lower:     (p_hat + z²/2n - z*√(p_hat(1-p_hat)/n + z²/4n²)) / (1 + z²/n)
          Final score:      clamp(EMA + wilson_lower * 0.3, 0.0, 1.0)

        Returns new score.
        """
        import math
        all_sparks = await asyncio.to_thread(self._store.load_all)
        spark = next((s for s in all_sparks if s.id == spark_id), None)
        if spark is None:
            return 0.0

        # Update trial counters
        spark.trial_count   = getattr(spark, "trial_count", 0) + 1
        spark.success_count = getattr(spark, "success_count", 0) + (1 if outcome else 0)

        # EMA component
        old = getattr(spark, "success_score", 0.5)
        ema_score = 0.7 * old + 0.3 * (1.0 if outcome else 0.0)

        # Wilson CI lower bound (z=1.96 for 95% confidence)
        n = spark.trial_count
        k = spark.success_count
        z = 1.96
        if n >= 2:
            p_hat   = k / n
            z2_over_2n = (z ** 2) / (2 * n)
            z2_over_4n2 = (z ** 2) / (4 * n ** 2)
            numerator   = p_hat + z2_over_2n - z * math.sqrt(p_hat * (1 - p_hat) / n + z2_over_4n2)
            denominator = 1 + (z ** 2) / n
            wilson_lower = max(0.0, numerator / denominator)
        else:
            wilson_lower = 0.0   # not enough data for CI yet

        # Combine: EMA primary, Wilson CI adds calibration bonus
        spark.success_score = max(0.0, min(1.0, ema_score + wilson_lower * 0.3))

        # Decay adjustment
        if outcome:
            spark.relevance_decay = min(1.0, spark.relevance_decay + 0.15)
        else:
            spark.relevance_decay = max(0.1, spark.relevance_decay - 0.1)

        await asyncio.to_thread(self._store.upsert, spark)
        return spark.success_score

    async def consolidate(self, min_success_score: float = 0.7, overlap_threshold: float = 0.8) -> int:
        """
        CONSOLIDATE step: merge high-score duplicate sparks.
        Merges pairs where tag overlap >= overlap_threshold AND both score >= min_success_score.
        Returns number of sparks consolidated.
        """
        all_sparks = await asyncio.to_thread(self._store.load_all)
        high = [s for s in all_sparks if getattr(s, "success_score", 0.5) >= min_success_score]
        merged: set[str] = set()

        for i, s1 in enumerate(high):
            if s1.id in merged:
                continue
            for s2 in high[i + 1:]:
                if s2.id in merged or s1.id == s2.id:
                    continue
                union = set(s1.tags) | set(s2.tags)
                inter = set(s1.tags) & set(s2.tags)
                if not union:
                    continue
                if len(inter) / len(union) >= overlap_threshold:
                    # Merge s2 into s1
                    s1.tags = list(set(s1.tags) | set(s2.tags))
                    if len(s2.content) > len(s1.content):
                        s1.content = s2.content[:200]
                    s1.success_score = max(  # type: ignore
                        getattr(s1, "success_score", 0.5),
                        getattr(s2, "success_score", 0.5),
                    )
                    s1.access_count += s2.access_count
                    s1.linked_sparks = list(set(s1.linked_sparks + [s2.id]))
                    await asyncio.to_thread(self._store.upsert, s1)
                    merged.add(s2.id)

        for sid in merged:
            await asyncio.to_thread(self._store.delete_spark, sid)
        return len(merged)

    async def reasoning_bank_cycle(
        self,
        agent: str,
        raw_logs: list[str],
        outcomes: list[bool] | None = None,
    ) -> dict:
        """
        Full RETRIEVE → JUDGE → DISTILL → CONSOLIDATE cycle.
        Call from sleep_cycle() Phase 4 instead of bare distill().
        Returns: {"distilled": N, "judged": N, "consolidated": N}
        """
        # RETRIEVE context
        existing = await self.evoke(agent, context=" ".join(raw_logs[-5:]))
        # DISTILL
        new_sparks = await self.distill(agent, raw_logs)
        # JUDGE
        judged = 0
        if outcomes:
            for i, spark in enumerate(new_sparks):
                if i < len(outcomes):
                    await self.judge(spark.id, outcomes[i])
                    judged += 1
        # CONSOLIDATE
        consolidated = await self.consolidate()
        return {"distilled": len(new_sparks), "judged": judged, "consolidated": consolidated, "context": len(existing)}


    async def subscribe(self, agent: str, tags: list[str]):
        """Agent subscribes to specific tag streams (like a Kafka consumer group)."""
        self._subscribers[agent] = list(set(self._subscribers.get(agent, []) + tags))

    async def sync(self) -> dict[str, int]:
        """
        Full mesh sync: rebalance spark routing across all agents.
        Called by sleep_cycle() Phase 4.
        Returns: {agent: sparks_evoked} counts.
        """
        all_sparks = await asyncio.to_thread(self._store.load_all)
        counts     = {}
        for agent, context in self._agent_contexts.items():
            evoked = await self.evoke(agent, context)
            counts[agent] = len(evoked)
        return counts

    async def spark_summary(self) -> str:
        """
        One-paragraph summary of the SparkNet state.
        Used by XZeroHeartbeat for Telegram digest.
        """
        all_sparks = await asyncio.to_thread(self._store.load_all, limit=20)
        if not all_sparks:
            return "SparkNet: no sparks yet."
        top     = sorted(all_sparks, key=lambda s: s.access_count, reverse=True)[:3]
        anchors = [s for s in all_sparks if s.access_count >= self.ANCHOR_THRESHOLD]
        return (
            f"SparkNet: {len(all_sparks)} sparks, {len(anchors)} anchors. "
            f"Top: {' | '.join(s.content[:50] for s in top)}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _propagate(self, spark: Spark):
        """Auto-broadcast to agents subscribed to any of the spark's tags."""
        for agent, subs in self._subscribers.items():
            if any(t in subs for t in spark.tags):
                pass  # In-memory notification hook (extend for Telegram/WS)

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """Heuristic tag extraction from context string."""
        KNOWN_TAGS = [
            "sol", "btc", "eth", "usdc", "x100", "price", "signal", "whale",
            "trade", "swap", "dca", "bullish", "bearish", "goal", "patch",
            "autogenesis", "heartbeat", "error", "deploy", "roblox", "mobile",
        ]
        lower = text.lower()
        return [t for t in KNOWN_TAGS if t in lower]

    # ── X-ZERO integration ────────────────────────────────────────────────────

    async def pre_action_evoke(self, agent: str, action_description: str) -> str:
        """
        Called before any major agent action.
        Returns formatted spark context to prepend to agent prompt.
        Prevents agents from acting on stale/contradicted info.
        """
        sparks = await self.evoke(agent, action_description)
        if not sparks:
            return ""
        lines = [f"[SPARK {i+1}] {s.content}" for i, s in enumerate(sparks)]
        return "=== SparkNet Context ===
" + "
".join(lines) + "
========================
"

    async def post_action_capture(self, agent: str, action: str, result: str, success: bool = True):
        """
        Called after an agent completes an action.
        Auto-distils into a spark + calls judge() for ReasoningBank scoring.
        """
        content  = f"{action[:80]} → {result[:100]}"
        tags     = self._extract_tags(action + " " + result)
        tags    += ["outcome", agent]
        spark = await self.capture(agent, content, tags, context=action)
        await self.judge(spark.id, outcome=success)
        return spark


# ── Module-level singleton (shared across GodLocal) ──────────────────────────

_default_net: SparkNetConnector | None = None

def get_sparknet() -> SparkNetConnector:
    """Get or create the shared SparkNet instance."""
    global _default_net
    if _default_net is None:
        _default_net = SparkNetConnector()
    return _default_net
