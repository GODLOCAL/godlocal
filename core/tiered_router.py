"""
core/tiered_router.py
TieredRouter — 4-tier cost routing for GodLocal LLM calls.

Claude-Flow pattern: WASM (Python) → fast model → full model → GIANT (AirLLM 70B)
When GROQ_API_KEY is set, FAST/FULL tiers use Groq LPU (~500 tok/s) and fall
back to local Ollama only on Groq failure.  ~75-80% token savings on micro-tasks.

Usage:
    from core.tiered_router import TieredRouter, get_tiered_router
    router = get_tiered_router()
    result = await router.complete(prompt, task_type="format")    # WASM tier
    result = await router.complete(prompt, task_type="classify")  # FAST (Groq/Ollama)
    result = await router.complete(prompt, task_type="codegen")   # full tier
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Tier(Enum):
    WASM  = 0   # Pure Python, 0 tokens
    FAST  = 1   # Lightweight Ollama model (~60% token saving)
    FULL  = 2   # Full Brain model (baseline)
    GIANT = 3   # AirLLM layer-by-layer 70B+ on 4GB VRAM


WASM_TASK_TYPES = {
    "format", "parse", "extract_tags", "extract_numbers",
    "bool_check", "json_extract", "url_check", "trim", "count"
}

FAST_TASK_TYPES = {
    "classify", "summarize_short", "sentiment", "tag_infer",
    "translate_short", "yes_no", "single_label"
}

FULL_TASK_TYPES = {
    "codegen", "distill", "analyze", "plan", "reason",
    "generate_long", "multi_step", "creative"
}

GIANT_TASK_TYPES = {
    "giant", "deep_reason", "long_analysis", "research", "expert"
}  # → AirLLM 70B layer-by-layer

FAST_MODEL = os.getenv("GODLOCAL_FAST_MODEL", "qwen3:1.7b")

KNOWN_TAGS = [
    "sol", "btc", "eth", "usdc", "x100", "price", "signal", "whale",
    "trade", "swap", "dca", "bullish", "bearish", "goal", "patch",
    "autogenesis", "heartbeat", "error", "deploy", "roblox", "mobile",
]


class WASMHandlers:
    """Pure Python task handlers — zero LLM calls, zero cost."""

    @staticmethod
    def extract_tags(text: str, known_tags: list[str] = KNOWN_TAGS) -> list[str]:
        lower = text.lower()
        return [t for t in known_tags if t in lower]

    @staticmethod
    def extract_numbers(text: str) -> list[float]:
        return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]

    @staticmethod
    def bool_check(text: str) -> bool:
        pos = {"yes", "true", "success", "ok", "confirmed", "1"}
        return any(p in text.lower() for p in pos)

    @staticmethod
    def json_extract(text: str) -> Any:
        """Extract JSON from text (handles markdown code fences)."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    @staticmethod
    def count_tokens_approx(text: str) -> int:
        return len(text.split()) * 4 // 3

    @staticmethod
    def format_spark_context(sparks: list) -> str:
        """Format spark list into prompt prefix. No LLM needed."""
        if not sparks:
            return ""
        lines = [f"[SPARK {i+1}] {s.content}" for i, s in enumerate(sparks)]
        return "=== SparkNet Context ===\n" + "\n".join(lines) + "\n========================\n"


@dataclass
class TierStats:
    wasm_calls:    int = 0
    fast_calls:    int = 0
    full_calls:    int = 0
    giant_calls:   int = 0           # AirLLM 70B tier (added today)
    wasm_saved_tokens: int = 0
    fast_saved_tokens: int = 0
    sparknet_reports: int = 0        # savings milestone reports emitted

    @property
    def total_calls(self) -> int:
        return self.wasm_calls + self.fast_calls + self.full_calls + self.giant_calls

    @property
    def savings_pct(self) -> float:
        if self.total_calls == 0:
            return 0.0
        wasm_pct  = self.wasm_calls  / self.total_calls
        fast_pct  = self.fast_calls  / self.total_calls
        # WASM=100% savings, FAST=60%, FULL/GIANT=0%
        return wasm_pct * 1.0 + fast_pct * 0.6


class TieredRouter:
    """
    3-tier cost router wrapping GodLocal Brain.

    Tier 0 (WASM):  pure Python regex/JSON handlers, 0 tokens, <0.1ms
    Tier 1 (FAST):  lightweight model (qwen3:1.7b), ~60% token saving
    Tier 2 (FULL):  full Brain model, baseline

    Auto-classifies by task_type or prompt length.
    Falls back gracefully — if fast model unavailable, uses full.
    """

    def __init__(self) -> None:
        self._brain: Any = None
        self._fast_brain: Any = None
        self.wasm = WASMHandlers()
        self.stats = TierStats()

    def _get_brain(self) -> Any:
        if self._brain is None:
            from core.brain import Brain
            self._brain = Brain()
        return self._brain

    def _get_fast_brain(self) -> Any:
        if self._fast_brain is None:
            try:
                from core.brain import Brain
                self._fast_brain = Brain(model=FAST_MODEL)
            except Exception:
                self._fast_brain = self._get_brain()
        return self._fast_brain

    def classify_tier(self, prompt: str, task_type: str | None = None) -> Tier:
        if task_type:
            if task_type in WASM_TASK_TYPES:
                return Tier.WASM
            if task_type in FAST_TASK_TYPES:
                return Tier.FAST
            if task_type in FULL_TASK_TYPES:
                return Tier.FULL
            if task_type in GIANT_TASK_TYPES:
                return Tier.GIANT
        approx = self.wasm.count_tokens_approx(prompt)
        if approx < 50:
            return Tier.WASM
        if approx < 300:
            return Tier.FAST
        if approx > 2000:
            return Tier.GIANT   # Very long prompt → AirLLM 70B
        return Tier.FULL

    async def complete(
        self,
        prompt: str,
        task_type: str | None = None,
        max_tokens: int = 512,
        force_tier: Tier | None = None,
    ) -> str:
        """Route prompt to appropriate tier and return completion."""
        tier = force_tier or self.classify_tier(prompt, task_type)

        if tier == Tier.WASM:
            self.stats.wasm_calls += 1
            self.stats.wasm_saved_tokens += self.wasm.count_tokens_approx(prompt) + max_tokens
            extracted = self.wasm.json_extract(prompt)
            if extracted is not None:
                return json.dumps(extracted)
            tags = self.wasm.extract_tags(prompt)
            return json.dumps(tags) if tags else prompt[:200]

        elif tier == Tier.FAST:
            self.stats.fast_calls += 1
            self.stats.fast_saved_tokens += int(self.wasm.count_tokens_approx(prompt) * 0.6)
            # Try Groq LPU first (~500 tok/s) — fall back to local Ollama on failure/absence
            from core.groq_connector import GROQ_AVAILABLE, get_groq
            if GROQ_AVAILABLE:
                try:
                    return await get_groq().complete(prompt, tier="fast", max_tokens=max_tokens)
                except Exception:
                    logger.warning("Groq FAST failed — falling back to Ollama")
            brain = self._get_fast_brain()
            return await brain.async_complete(prompt, max_tokens=max_tokens)

        elif tier == Tier.GIANT:
            # AirLLM: 70B layer-by-layer on 4-8GB VRAM. pip install airllm on VPS.
            from core.airllm_bridge import get_airllm
            return await get_airllm().complete(prompt, max_new_tokens=max_tokens)

        else:
            self.stats.full_calls += 1
            # Try Groq LPU first (Kimi K2-0905, GPT-4 class) — fall back to local Ollama
            from core.groq_connector import GROQ_AVAILABLE, get_groq
            if GROQ_AVAILABLE:
                try:
                    return await get_groq().complete(prompt, tier="full", max_tokens=max_tokens)
                except Exception:
                    logger.warning("Groq FULL failed — falling back to Ollama")
            brain = self._get_brain()
            return await brain.async_complete(prompt, max_tokens=max_tokens)

    def log_stats(self) -> str:
        s = self.stats
        line = (
            f"TieredRouter: {s.total_calls} calls | "
            f"WASM={s.wasm_calls} FAST={s.fast_calls} FULL={s.full_calls} GIANT={s.giant_calls} | "
            f"Est. savings={s.savings_pct:.0%} | "
            f"WASM tokens saved={s.wasm_saved_tokens:,}"
        )
        # Emit savings milestone to SparkNet every 50 calls
        if s.total_calls > 0 and s.total_calls % 50 == 0:
            try:
                from extensions.xzero.sparknet_connector import get_sparknet
                import asyncio
                sparknet = get_sparknet()
                summary = f"TieredRouter {s.savings_pct:.0%} savings ({s.total_calls} calls, WASM={s.wasm_calls})"
                asyncio.ensure_future(
                    sparknet.capture("tiered_router", summary[:200], tags=["tiered", "savings", "routing"])
                )
                s.sparknet_reports += 1
            except Exception:
                pass
        return line


_default_router: TieredRouter | None = None


def get_tiered_router() -> TieredRouter:
    """Get or create the shared TieredRouter instance."""
    global _default_router
    if _default_router is None:
        _default_router = TieredRouter()
    return _default_router
