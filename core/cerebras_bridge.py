"""
core/cerebras_bridge.py
Cerebras Inference API — ~2,000–3,000 tok/s on full-size models.
Uses cerebras-cloud-sdk (pip install cerebras-cloud-sdk).

Fastest models as of Q1 2026:
  llama-3.3-70b  — ~2000 tok/s
  llama3.1-8b    — ~3000+ tok/s (fastest, ultra-lightweight)
  qwen-3-32b     — ~1500 tok/s (strong coder)
"""
from __future__ import annotations
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

CEREBRAS_API_KEY: str | None = os.getenv("CEREBRAS_API_KEY")
CEREBRAS_AVAILABLE: bool = bool(CEREBRAS_API_KEY)

# Model map — ultra-fast classify/signal tasks use 8B, heavy ones use 70B
_FAST_MODEL  = os.getenv("CEREBRAS_FAST_MODEL",  "llama3.1-8b")
_FULL_MODEL  = os.getenv("CEREBRAS_FULL_MODEL",  "llama-3.3-70b")
_GIANT_MODEL = os.getenv("CEREBRAS_GIANT_MODEL", "llama-3.3-70b")

CEREBRAS_MODEL_MAP: dict[str, str] = {
    # Ultra-fast micro-tasks (3k+ tok/s)
    "classify":        _FAST_MODEL,
    "summarize_short": _FAST_MODEL,
    "sentiment":       _FAST_MODEL,
    "tag_infer":       _FAST_MODEL,
    "translate_short": _FAST_MODEL,
    "yes_no":          _FAST_MODEL,
    "single_label":    _FAST_MODEL,
    "signal_infer":    _FAST_MODEL,
    # FULL tasks (2k tok/s, smarter)
    "codegen":         _FULL_MODEL,
    "analyze":         _FULL_MODEL,
    "distill":         _FULL_MODEL,
    "reason":          _FULL_MODEL,
    "plan":            _FULL_MODEL,
    "multi_step":      _FULL_MODEL,
    "creative":        _FULL_MODEL,
    "generate_long":   _FULL_MODEL,
    "giant":           _GIANT_MODEL,
}


class CerebrasConnector:
    """
    Async Cerebras Inference connector.
    Uses AsyncCerebras from cerebras-cloud-sdk.
    asyncio.Lock prevents concurrent calls hammering rate limit.
    """

    def __init__(self) -> None:
        if not CEREBRAS_API_KEY:
            raise RuntimeError("CEREBRAS_API_KEY not set")
        try:
            from cerebras.cloud.sdk import AsyncCerebras  # pip install cerebras-cloud-sdk
            self._client = AsyncCerebras(api_key=CEREBRAS_API_KEY)
        except ImportError:
            raise RuntimeError("cerebras-cloud-sdk not installed — run: pip install cerebras-cloud-sdk")
        self._lock = asyncio.Lock()

    async def complete(
        self,
        prompt: str,
        task_type: str = "classify",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        model = CEREBRAS_MODEL_MAP.get(task_type, _FAST_MODEL)

        async with self._lock:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        content: str = response.choices[0].message.content or ""
        tokens_used: int = response.usage.total_tokens if response.usage else 0

        logger.debug(f"Cerebras[{task_type}/{model}] {tokens_used} tok → {len(content)} chars")

        # SparkNet logging (non-blocking)
        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            asyncio.ensure_future(
                get_sparknet().capture(
                    "cerebras_usage",
                    f"Cerebras {model} {task_type} {tokens_used}tok",
                    tags=["cerebras", "llm", task_type],
                )
            )
        except Exception:
            pass

        return content


_cerebras_instance: CerebrasConnector | None = None


def get_cerebras() -> CerebrasConnector:
    global _cerebras_instance
    if _cerebras_instance is None:
        _cerebras_instance = CerebrasConnector()
    return _cerebras_instance
