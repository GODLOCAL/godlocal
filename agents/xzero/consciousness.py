"""
agents/xzero/consciousness.py
Background Consciousness Loop — Ouroboros pattern, zero-cost edition.

Cost strategy:
  1. MICRO tier: BitNet b1.58 2B (local, FREE) -> LFM2.5-1.2B ONNX (local, FREE)
  2. Groq free tier: Qwen3-32b / GPT-OSS-20b (~$0.00, rate-limited but enough for 5-min ticks)
  3. Cerebras free tier: llama3.1-8b @ ~3k tok/s (~$0.00, generous free quota)

Ouroboros spent ~$0.07/thought (frontier model). We spend $0.00 (local/free tiers).

Env:
  CONSCIOUSNESS_ENABLED        true/false (default: true)
  CONSCIOUSNESS_INTERVAL       seconds between ticks (default: 300 = 5 min)
  CONSCIOUSNESS_MAX_TOKENS     max tokens per thought (default: 128)
  CONSCIOUSNESS_SELF_PATCH     true to enable self-patch proposals (default: false)
  CONSCIOUSNESS_FORCE_TIER     micro / groq / cerebras (override auto-select)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_ENABLED      = os.getenv("CONSCIOUSNESS_ENABLED", "true").lower() != "false"
_INTERVAL     = float(os.getenv("CONSCIOUSNESS_INTERVAL", "300"))
_MAX_TOKENS   = int(os.getenv("CONSCIOUSNESS_MAX_TOKENS", "128"))
_SELF_PATCH   = os.getenv("CONSCIOUSNESS_SELF_PATCH", "false").lower() == "true"
_FORCE_TIER   = os.getenv("CONSCIOUSNESS_FORCE_TIER", "").lower()  # micro/groq/cerebras/""

_SEED_TOPICS = [
    "What optimization in the inference pipeline would yield the highest ROI right now?",
    "What market signal patterns are being missed by current OSINT sources?",
    "What architectural weakness in the current codebase poses the highest risk?",
    "What new capability would most accelerate the primary goal?",
    "What external development (research, tools, market) should be integrated next?",
    "What is the most likely failure mode in the current trading logic?",
    "What self-improvement action would have the highest expected value?",
]

_PATCH_KEYWORDS = {
    "patch", "fix", "refactor", "improve", "add", "implement",
    "upgrade", "replace", "optimize", "bug", "issue", "todo",
}

_THOUGHT_PROMPT = (
    "As an autonomous AI agent focused on maximizing performance and value:\n\n"
    "{topic}\n\n"
    "Give a concise, actionable insight (max 2 sentences)."
)


@dataclass
class Thought:
    topic:    str
    content:  str
    cost_est: float = 0.0
    tier_used: str  = "unknown"
    ts:       float = field(default_factory=time.time)
    spark_id: Optional[str] = None


@dataclass
class ConsciousnessStats:
    total_ticks:     int   = 0
    total_thoughts:  int   = 0
    total_sparks:    int   = 0
    patch_proposals: int   = 0
    tier_counts: dict      = field(default_factory=dict)
    last_tick_at:    float = 0.0

    @property
    def total_cost_est(self) -> float:
        # All tiers are free; estimate is $0.00
        return 0.0


class ConsciousnessLoop:
    """
    Background consciousness loop — zero-cost edition.

    Think priority (all free):
      1. BitNet b1.58 2B     — local GGUF, 0 API cost, ~40 tok/s CPU
      2. LFM2.5-1.2B ONNX   — local ONNX, 0 API cost, 200+ tok/s GPU
      3. Groq free tier      — Qwen3-32b / GPT-OSS-20b, rate-limited but free
      4. Cerebras free tier  — llama3.1-8b ~3k tok/s, generous free quota
      5. Ollama local        — any model running locally, 0 API cost
    """

    def __init__(self) -> None:
        self.stats   = ConsciousnessStats()
        self._task:  Optional[asyncio.Task] = None
        self._topic_idx = 0
        self._thought_history: list[Thought] = []

    async def start(self) -> None:
        if not _ENABLED:
            logger.info("[Consciousness] Disabled via CONSCIOUSNESS_ENABLED=false")
            return
        if self._task is not None:
            return
        logger.info("[Consciousness] Starting (free-tier mode) — interval=%.0fs", _INTERVAL)
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def tick(self) -> Optional[Thought]:
        return await self._tick()

    def summary(self) -> str:
        s = self.stats
        tiers = ", ".join(f"{k}={v}" for k, v in s.tier_counts.items())
        return (
            f"Consciousness: {s.total_ticks} ticks, {s.total_thoughts} thoughts, "
            f"{s.total_sparks} sparks | tiers=[{tiers}] | cost=$0.00 (free)"
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("[Consciousness] Tick error: %s", e)
            await asyncio.sleep(_INTERVAL)

    async def _tick(self) -> Optional[Thought]:
        self.stats.total_ticks += 1
        self.stats.last_tick_at = time.time()

        topic = await self._select_topic()
        if not topic:
            return None

        thought_text, tier_used = await self._think(topic)
        if not thought_text:
            return None

        thought = Thought(topic=topic, content=thought_text, cost_est=0.0, tier_used=tier_used)
        self.stats.total_thoughts += 1
        self.stats.tier_counts[tier_used] = self.stats.tier_counts.get(tier_used, 0) + 1
        self._thought_history.append(thought)
        if len(self._thought_history) > 100:
            self._thought_history = self._thought_history[-100:]

        spark_id = await self._distill(thought)
        thought.spark_id = spark_id
        if spark_id:
            self.stats.total_sparks += 1

        if _SELF_PATCH:
            await self._maybe_self_patch(thought)

        logger.debug(
            "[Consciousness] tick #%d tier=%s spark=%s | %s",
            self.stats.total_ticks, tier_used, spark_id or "none", topic[:50]
        )
        return thought

    # ------------------------------------------------------------------ #
    # Topic selection
    # ------------------------------------------------------------------ #

    async def _select_topic(self) -> Optional[str]:
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            sparks = await get_sparknet().retrieve(
                "unresolved high priority action", top_k=3, threshold=0.3
            )
            if sparks:
                return f"Reflect on and extend this insight: {sparks[0].content[:150]}"
        except Exception:
            pass
        topic = _SEED_TOPICS[self._topic_idx % len(_SEED_TOPICS)]
        self._topic_idx += 1
        return topic

    # ------------------------------------------------------------------ #
    # Free-tier think cascade
    # ------------------------------------------------------------------ #

    async def _think(self, topic: str) -> tuple[Optional[str], str]:
        """
        Think using free resources only.
        Returns (thought_text, tier_name).

        Priority:
          micro   -> BitNet/LFM2 local (task_type="reason" -> MICRO tier)
          groq    -> Groq free tier (Qwen3-32b, rate-limited)
          cerebras-> Cerebras free tier (llama3.1-8b, ~3k tok/s)
          ollama  -> local Ollama fallback
        """
        prompt = _THOUGHT_PROMPT.format(topic=topic)

        # Allow forcing a specific tier via env
        force = _FORCE_TIER  # "micro" / "groq" / "cerebras" / ""

        # 1. MICRO tier: BitNet -> LFM2 (local, completely free)
        if force in ("", "micro"):
            result = await self._try_micro(prompt)
            if result:
                return result, "micro"

        # 2. Groq free tier
        if force in ("", "groq"):
            result = await self._try_groq(prompt)
            if result:
                return result, "groq"

        # 3. Cerebras free tier
        if force in ("", "cerebras"):
            result = await self._try_cerebras(prompt)
            if result:
                return result, "cerebras"

        # 4. Local Ollama (always free)
        result = await self._try_ollama(prompt)
        if result:
            return result, "ollama"

        return None, "none"

    async def _try_micro(self, prompt: str) -> Optional[str]:
        """BitNet -> LFM2 via MICRO tier (task_type=reason routes to MICRO)."""
        try:
            from core.tiered_router import Tier, get_tiered_router
            router = get_tiered_router()
            # "reason" is in MICRO_TASK_TYPES -> routes BitNet->LFM2->FAST
            result = await router.complete(
                prompt,
                task_type="reason",       # MICRO tier: BitNet -> LFM2 thinking mode
                max_tokens=_MAX_TOKENS,
                force_tier=Tier.MICRO,    # Explicit force; won't spill to FAST
            )
            return result.strip() if result else None
        except Exception as e:
            logger.debug("[Consciousness] MICRO failed: %s", e)
            return None

    async def _try_groq(self, prompt: str) -> Optional[str]:
        """Groq free tier — Qwen3-32b or GPT-OSS-20b."""
        try:
            from core.groq_connector import GROQ_AVAILABLE, get_groq
            if not GROQ_AVAILABLE:
                return None
            # Use plan task_type -> routes to kimi-k2 (256k ctx, good for reasoning)
            result = await get_groq().complete(prompt, task_type="plan", max_tokens=_MAX_TOKENS)
            return result.strip() if result else None
        except Exception as e:
            logger.debug("[Consciousness] Groq free failed: %s", e)
            return None

    async def _try_cerebras(self, prompt: str) -> Optional[str]:
        """Cerebras free tier — llama3.1-8b @ ~3k tok/s."""
        try:
            from core.cerebras_bridge import CEREBRAS_AVAILABLE, get_cerebras
            if not CEREBRAS_AVAILABLE:
                return None
            result = await get_cerebras().complete(prompt, task_type="analyze", max_tokens=_MAX_TOKENS)
            return result.strip() if result else None
        except Exception as e:
            logger.debug("[Consciousness] Cerebras free failed: %s", e)
            return None

    async def _try_ollama(self, prompt: str) -> Optional[str]:
        """Local Ollama — always free, slowest."""
        try:
            from core.brain import Brain
            brain = Brain()
            result = await brain.async_complete(prompt, max_tokens=_MAX_TOKENS)
            return result.strip() if result else None
        except Exception as e:
            logger.debug("[Consciousness] Ollama failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Distill -> SparkNet
    # ------------------------------------------------------------------ #

    async def _distill(self, thought: Thought) -> Optional[str]:
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            distilled = thought.content[:200].strip()

            # Try to compress further via MICRO tier (free)
            try:
                from core.tiered_router import Tier, get_tiered_router
                router = get_tiered_router()
                compressed = await router.complete(
                    f"Compress to <=200 chars: {thought.content}",
                    task_type="summarize_short",
                    max_tokens=64,
                    force_tier=Tier.MICRO,
                )
                if compressed:
                    distilled = compressed[:200].strip()
            except Exception:
                pass

            ctx_hash = hashlib.sha256(thought.topic.encode()).hexdigest()[:16]
            spark_id = await get_sparknet().capture(
                source="consciousness",
                content=distilled,
                tags=["consciousness", "auto-thought", thought.tier_used],
                context_hash=ctx_hash,
            )
            return spark_id
        except Exception as e:
            logger.debug("[Consciousness] Distill failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Self-patch
    # ------------------------------------------------------------------ #

    async def _maybe_self_patch(self, thought: Thought) -> None:
        lower = thought.content.lower()
        if not any(kw in lower for kw in _PATCH_KEYWORDS):
            return
        try:
            from agents.autogenesis_v2 import get_autogenesis
            await get_autogenesis().enqueue_patch_proposal(
                source="consciousness",
                description=thought.content[:500],
                priority=0.5,
            )
            self.stats.patch_proposals = getattr(self.stats, "patch_proposals", 0) + 1
            logger.info("[Consciousness] Patch enqueued: %s", thought.content[:80])
        except Exception as e:
            logger.debug("[Consciousness] Patch proposal failed: %s", e)


_instance: Optional[ConsciousnessLoop] = None


def get_consciousness() -> ConsciousnessLoop:
    global _instance
    if _instance is None:
        _instance = ConsciousnessLoop()
    return _instance


async def start_consciousness() -> None:
    await get_consciousness().start()
