"""
core/tiered_router.py
TieredRouter — 3-tier cost routing for GodLocal LLM calls.

Claude-Flow pattern: WASM (Python) → fast model → full model
~75% token savings on micro-tasks.

Usage:
    from core.tiered_router import TieredRouter, get_tiered_router
    router = get_tiered_router()
    result = await router.complete(prompt, task_type="format")    # WASM tier
    result = await router.complete(prompt, task_type="classify")  # fast tier
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
    WASM = 0   # Pure Python, 0 tokens
    FAST = 1   # Lightweight Ollama model (~60% token saving)
    FULL = 2   # Full Brain model (baseline)


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
    wasm_calls: int = 0
    fast_calls: int = 0
    full_calls: int = 0
    wasm_saved_tokens: int = 0
    fast_saved_tokens: int = 0

    @property
    def total_calls(self) -> int:
        return self.wasm_calls + self.fast_calls + self.full_calls

    @property
    def savings_pct(self) -> float:
        if self.total_calls == 0:
            return 0.0
        wasm_pct = self.wasm_calls / self.total_calls
        fast_pct = self.fast_calls / self.total_calls
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
        approx = self.wasm.count_tokens_approx(prompt)
        if approx < 50:
            return Tier.WASM
        if approx < 300:
            return Tier.FAST
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
            brain = self._get_fast_brain()
            return await brain.async_complete(prompt, max_tokens=max_tokens)

        else:
            self.stats.full_calls += 1
            brain = self._get_brain()
            return await brain.async_complete(prompt, max_tokens=max_tokens)

    def log_stats(self) -> str:
        s = self.stats
        return (
            f"TieredRouter: {s.total_calls} calls | "
            f"WASM={s.wasm_calls} FAST={s.fast_calls} FULL={s.full_calls} | "
            f"Est. savings={s.savings_pct:.0%} | "
            f"WASM tokens saved={s.wasm_saved_tokens:,}"
        )


_default_router: TieredRouter | None = None


def get_tiered_router() -> TieredRouter:
    """Get or create the shared TieredRouter instance."""
    global _default_router
    if _default_router is None:
        _default_router = TieredRouter()
    return _default_router
