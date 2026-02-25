"""
core/groq_connector.py
GroqCloudConnector — optional high-speed LLM backend for TieredRouter.

Groq LPU delivers ~500 tok/s vs ~60 tok/s for local Ollama.
Falls back gracefully when GROQ_API_KEY is absent.

Models mapped:
  FAST  → qwen3-32b          (quick classify/summarize, low latency)
  FULL  → kimi-k2-0905       (codegen/analysis, GPT-4 class)
  GIANT → moonshotai/kimi-k2  (deep research/reasoning, 128k ctx)

Usage:
    from core.groq_connector import get_groq, GROQ_AVAILABLE
    if GROQ_AVAILABLE:
        result = await get_groq().complete(prompt, tier="fast")
"""
from __future__ import annotations
import asyncio
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_AVAILABLE: bool = bool(GROQ_API_KEY)

# Model mapping: tier → Groq model id
GROQ_MODELS: dict[str, str] = {
    "fast":  "qwen-qwq-32b",          # ~500 tok/s, great for classify/summarize
    "full":  "moonshotai/kimi-k2-0905", # GPT-4 class, 128k ctx
    "giant": "moonshotai/kimi-k2-0905", # same model, higher token budget
}


class GroqCloudConnector:
    """
    Async Groq LPU connector.
    Thread-safe, uses asyncio.to_thread for the blocking groq SDK call.
    """

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set — GroqCloudConnector unavailable")
        try:
            from groq import Groq  # pip install groq
            self._client = Groq(api_key=GROQ_API_KEY)
        except ImportError:
            raise RuntimeError("groq package not installed — run: pip install groq")
        self._lock = asyncio.Lock()  # one concurrent call per process (rate-limit safety)

    async def complete(
        self,
        prompt: str,
        tier: Literal["fast", "full", "giant"] = "fast",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Send prompt to Groq LPU and return completion text.
        Raises RuntimeError on API error (caller should fall back to Ollama).
        """
        model = GROQ_MODELS.get(tier, GROQ_MODELS["fast"])

        def _sync_call() -> str:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        async with self._lock:
            try:
                result = await asyncio.to_thread(_sync_call)
                logger.debug(f"Groq[{tier}/{model}] → {len(result)} chars")
                return result
            except Exception as exc:
                logger.warning(f"Groq {tier} call failed: {exc} — caller should fall back")
                raise


# ── Singleton ──────────────────────────────────────────────────────────────
_groq_instance: GroqCloudConnector | None = None


def get_groq() -> GroqCloudConnector:
    global _groq_instance
    if _groq_instance is None:
        _groq_instance = GroqCloudConnector()
    return _groq_instance
