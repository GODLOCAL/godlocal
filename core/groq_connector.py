"""
core/groq_connector.py  (v2 — AsyncGroq + granular model map + SparkNet logging)
GroqCloudConnector — optional high-speed LLM backend for TieredRouter.

Groq LPU: 200–1000 tok/s vs ~60 tok/s Ollama.
Falls back gracefully when GROQ_API_KEY absent.

Model map (env-overridable):
  classify / summarize_short / sentiment / yes_no  → gpt-oss-20b  (~1000 tok/s)
  codegen / analyze / plan / reason / distill       → qwen3-32b    (~400 tok/s)
  plan / multi_step / creative                      → kimi-k2-instruct-0905 (256k ctx)
  giant / generate_long                             → gpt-oss-120b (~500 tok/s, smartest)

Usage:
    from core.groq_connector import get_groq, GROQ_AVAILABLE
    if GROQ_AVAILABLE:
        result = await get_groq().complete(prompt, task_type="classify")
"""
from __future__ import annotations
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_AVAILABLE: bool = bool(GROQ_API_KEY)

# ── Model map: task_type → Groq model id ───────────────────────────────────
# Env-overridable: GROQ_FAST_MODEL, GROQ_FULL_MODEL, GROQ_GIANT_MODEL
_FAST_MODEL  = os.getenv("GROQ_FAST_MODEL",  "openai/gpt-oss-20b")
_FULL_MODEL  = os.getenv("GROQ_FULL_MODEL",  "qwen/qwen3-32b")
_PLAN_MODEL  = os.getenv("GROQ_PLAN_MODEL",  "moonshotai/kimi-k2-instruct-0905")
_GIANT_MODEL = os.getenv("GROQ_GIANT_MODEL", "openai/gpt-oss-120b")

GROQ_MODEL_MAP: dict[str, str] = {
    # FAST tier task_types (~1000 tok/s, cheap)
    "classify":        _FAST_MODEL,
    "summarize_short": _FAST_MODEL,
    "sentiment":       _FAST_MODEL,
    "tag_infer":       _FAST_MODEL,
    "translate_short": _FAST_MODEL,
    "yes_no":          _FAST_MODEL,
    "single_label":    _FAST_MODEL,
    # FULL tier task_types (~400 tok/s, strong coder)
    "codegen":         _FULL_MODEL,
    "analyze":         _FULL_MODEL,
    "distill":         _FULL_MODEL,
    "reason":          _FULL_MODEL,
    # FULL tier: long-context planning (256k ctx)
    "plan":            _PLAN_MODEL,
    "multi_step":      _PLAN_MODEL,
    "creative":        _PLAN_MODEL,
    "generate_long":   _PLAN_MODEL,
    # GIANT tier — smartest, ~500 tok/s
    "giant":           _GIANT_MODEL,
}


class GroqCloudConnector:
    """
    Async Groq LPU connector using native AsyncGroq client.
    asyncio.Lock ensures safe concurrent use within rate limits.
    """

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set")
        try:
            from groq import AsyncGroq  # pip install groq
            self._client = AsyncGroq(api_key=GROQ_API_KEY)
        except ImportError:
            raise RuntimeError("groq package not installed — run: pip install groq")
        self._lock = asyncio.Lock()

    async def complete(
        self,
        prompt: str,
        task_type: str = "classify",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Send prompt to Groq and return completion text.
        Emits SparkNet capture with token usage.
        Raises on API error (caller should fall back to Ollama).
        """
        model = GROQ_MODEL_MAP.get(task_type, _FAST_MODEL)

        async with self._lock:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        content: str = response.choices[0].message.content or ""
        tokens_used: int = response.usage.total_tokens if response.usage else 0

        logger.debug(f"Groq[{task_type}/{model}] {tokens_used} tok → {len(content)} chars")

        # SparkNet: record usage (non-blocking, best-effort)
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            summary = f"Groq {model.split('/')[-1]} {task_type} {tokens_used}tok"
            asyncio.ensure_future(
                get_sparknet().capture("groq_usage", summary[:200], tags=["groq", "llm", task_type])
            )
        except Exception:
            pass

        return content


# ── Singleton ──────────────────────────────────────────────────────────────
_groq_instance: GroqCloudConnector | None = None


def get_groq() -> GroqCloudConnector:
    global _groq_instance
    if _groq_instance is None:
        _groq_instance = GroqCloudConnector()
    return _groq_instance
