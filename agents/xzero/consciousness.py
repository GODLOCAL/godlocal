"""
agents/xzero/consciousness.py
Background Consciousness Loop — inspired by Ouroboros self-evolving agent.

Ouroboros pattern (@aigclink / 2026-02-25):
  - Agent thinks continuously even with zero user interaction
  - Decides autonomously WHAT to think about
  - Fixed cost per thought cycle (~$0.06-0.08 via full LLM)
  - 30+ self-patches in 48 hours, zero human intervention

GodLocal adaptation:
  - Uses MICRO tier (BitNet/LFM2/Groq) for thought generation (~$0.001/cycle)
  - Thought topics auto-selected from SparkNet high-priority sparks
  - New insights → distilled back into SparkNet (ReasoningBank loop)
  - Self-patch proposals → forwarded to AutoGenesisV2 queue
  - Runs as asyncio background task, configurable tick interval

Architecture:
  ConsciousnessLoop
    tick() every CONSCIOUSNESS_INTERVAL seconds
      -> _select_topic()    : pick top SparkNet spark or generate novel topic
      -> _think(topic)      : MICRO/FAST tier inference, ~128 tokens
      -> _distill(thought)  : compress to <=200 chars, store in SparkNet
      -> _maybe_self_patch(): if thought contains patch signal -> AutoGenesis queue

Env:
  CONSCIOUSNESS_ENABLED    true/false (default: true)
  CONSCIOUSNESS_INTERVAL   seconds between ticks (default: 300 = 5 min)
  CONSCIOUSNESS_MAX_TOKENS max tokens per thought (default: 128)
  CONSCIOUSNESS_SELF_PATCH true to enable self-patch proposals (default: false)
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
_INTERVAL     = float(os.getenv("CONSCIOUSNESS_INTERVAL", "300"))       # 5 min default
_MAX_TOKENS   = int(os.getenv("CONSCIOUSNESS_MAX_TOKENS", "128"))
_SELF_PATCH   = os.getenv("CONSCIOUSNESS_SELF_PATCH", "false").lower() == "true"

# Topics the agent explores when SparkNet is empty
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


@dataclass
class Thought:
    topic:    str
    content:  str
    cost_est: float          # estimated $ cost of this thought
    ts:       float = field(default_factory=time.time)
    spark_id: Optional[str] = None


@dataclass
class ConsciousnessStats:
    total_ticks:    int   = 0
    total_thoughts: int   = 0
    total_sparks:   int   = 0
    patch_proposals: int  = 0
    total_cost_est: float = 0.0
    last_tick_at:   float = 0.0


class ConsciousnessLoop:
    """
    Background consciousness loop for GodLocal xzero agent.

    Continuously generates thoughts even without user interaction.
    Stores insights in SparkNet. Optionally proposes self-patches
    to AutoGenesisV2.

    Cost estimate per thought:
      MICRO tier (BitNet/LFM2): ~$0.0001
      FAST tier (Groq/Cerebras): ~$0.001-0.005
      vs Ouroboros baseline: ~$0.06-0.08 (full frontier model)
    """

    def __init__(self) -> None:
        self.stats   = ConsciousnessStats()
        self._task:  Optional[asyncio.Task] = None
        self._topic_idx = 0
        self._thought_history: list[Thought] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the background consciousness loop."""
        if not _ENABLED:
            logger.info("[Consciousness] Disabled via CONSCIOUSNESS_ENABLED=false")
            return
        if self._task is not None:
            logger.warning("[Consciousness] Already running")
            return
        logger.info(
            "[Consciousness] Starting — interval=%.0fs, self_patch=%s",
            _INTERVAL, _SELF_PATCH
        )
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        """Stop the background loop gracefully."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[Consciousness] Stopped — %d thoughts generated", self.stats.total_thoughts)

    async def tick(self) -> Optional[Thought]:
        """Manually trigger one consciousness tick (useful for testing)."""
        return await self._tick()

    def summary(self) -> str:
        s = self.stats
        return (
            f"Consciousness: {s.total_ticks} ticks, {s.total_thoughts} thoughts, "
            f"{s.total_sparks} sparks stored, {s.patch_proposals} patch proposals, "
            f"est. cost=${s.total_cost_est:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Internal loop
    # ------------------------------------------------------------------ #

    async def _loop(self) -> None:
        logger.info("[Consciousness] Loop started")
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

        thought_text = await self._think(topic)
        if not thought_text:
            return None

        cost = self._estimate_cost(thought_text)
        thought = Thought(topic=topic, content=thought_text, cost_est=cost)
        self.stats.total_thoughts += 1
        self.stats.total_cost_est += cost
        self._thought_history.append(thought)
        if len(self._thought_history) > 100:
            self._thought_history = self._thought_history[-100:]

        # Distill into SparkNet
        spark_id = await self._distill(thought)
        thought.spark_id = spark_id
        if spark_id:
            self.stats.total_sparks += 1

        # Maybe propose self-patch
        if _SELF_PATCH:
            await self._maybe_self_patch(thought)

        logger.debug(
            "[Consciousness] Tick #%d — topic=%r cost=$%.4f spark=%s",
            self.stats.total_ticks, topic[:60], cost, spark_id or "none"
        )
        return thought

    # ------------------------------------------------------------------ #
    # Topic selection
    # ------------------------------------------------------------------ #

    async def _select_topic(self) -> Optional[str]:
        """
        Select what to think about:
        1. Top-priority unresolved spark from SparkNet (if available)
        2. Round-robin through seed topics
        """
        # Try SparkNet for high-signal unresolved context
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            sparknet = get_sparknet()
            sparks = await sparknet.retrieve("unresolved high priority action", top_k=3, threshold=0.3)
            if sparks:
                # Pick the most recent high-relevance spark as topic
                best = sparks[0]
                return f"Reflect on and extend this insight: {best.content[:150]}"
        except Exception as e:
            logger.debug("[Consciousness] SparkNet topic fetch failed: %s", e)

        # Fall back to seed topics (round-robin)
        topic = _SEED_TOPICS[self._topic_idx % len(_SEED_TOPICS)]
        self._topic_idx += 1
        return topic

    # ------------------------------------------------------------------ #
    # Thinking (MICRO tier preferred for cost efficiency)
    # ------------------------------------------------------------------ #

    async def _think(self, topic: str) -> Optional[str]:
        """Generate a thought using the tiered router (MICRO tier preferred)."""
        try:
            from core.tiered_router import get_tiered_router
            router = get_tiered_router()
            prompt = (
                f"As an autonomous AI agent focused on maximizing performance and value:\n\n"
                f"{topic}\n\n"
                f"Give a concise, actionable insight (max 2 sentences)."
            )
            # Use FAST tier for better thought quality (still cheap vs frontier)
            thought = await router.complete(
                prompt,
                task_type="analyze",
                max_tokens=_MAX_TOKENS,
                force_tier=None,  # Let router decide (will use FULL for "analyze")
            )
            return thought.strip() if thought else None
        except Exception as e:
            logger.warning("[Consciousness] Think failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Distillation into SparkNet
    # ------------------------------------------------------------------ #

    async def _distill(self, thought: Thought) -> Optional[str]:
        """Compress thought to <=200 chars and store in SparkNet."""
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            from core.tiered_router import get_tiered_router

            sparknet = get_sparknet()
            router   = get_tiered_router()

            # Compress to <=200 chars via MICRO tier
            distilled = await router.complete(
                f"Compress to <=200 chars: {thought.content}",
                task_type="summarize_short",
                max_tokens=64,
            )
            distilled = distilled[:200].strip()

            ctx_hash = hashlib.sha256(thought.topic.encode()).hexdigest()[:16]
            spark_id = await sparknet.capture(
                source="consciousness",
                content=distilled,
                tags=["consciousness", "auto-thought"],
                context_hash=ctx_hash,
            )
            return spark_id
        except Exception as e:
            logger.debug("[Consciousness] Distill/store failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Self-patch proposal
    # ------------------------------------------------------------------ #

    async def _maybe_self_patch(self, thought: Thought) -> None:
        """
        If the thought contains a patch signal, forward to AutoGenesisV2 queue.
        Only active when CONSCIOUSNESS_SELF_PATCH=true.
        """
        lower = thought.content.lower()
        has_patch_signal = any(kw in lower for kw in _PATCH_KEYWORDS)
        if not has_patch_signal:
            return

        try:
            from agents.autogenesis_v2 import get_autogenesis
            ag = get_autogenesis()
            await ag.enqueue_patch_proposal(
                source="consciousness",
                description=thought.content[:500],
                priority=0.5,
            )
            self.stats.patch_proposals += 1
            logger.info(
                "[Consciousness] Patch proposal enqueued: %s",
                thought.content[:80]
            )
        except Exception as e:
            logger.debug("[Consciousness] Patch proposal failed: %s", e)

    # ------------------------------------------------------------------ #
    # Cost estimation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _estimate_cost(text: str) -> float:
        """
        Rough cost estimate per thought.
        MICRO tier (local CPU): ~$0.0001
        FAST tier (Groq/Cerebras): ~$0.001
        vs Ouroboros (frontier): ~$0.06-0.08
        """
        tokens = len(text.split()) * 4 // 3
        # Assume FAST tier pricing (~$0.00001/token blended)
        return tokens * 0.00001


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #
_instance: Optional[ConsciousnessLoop] = None


def get_consciousness() -> ConsciousnessLoop:
    global _instance
    if _instance is None:
        _instance = ConsciousnessLoop()
    return _instance


async def start_consciousness() -> None:
    """Convenience: get singleton and start."""
    await get_consciousness().start()
