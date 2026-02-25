"""core/groq_connector.py — AsyncGroq v3

Real measured tok/s (2026-02-25 benchmark):
  llama-3.1-8b-instant:     ~483 tok/s  FAST / classify
  openai/gpt-oss-20b:       ~875 tok/s  FAST / summarize, reasoning
  openai/gpt-oss-120b:      ~462 tok/s  FULL / general
  qwen/qwen3-32b:           ~416 tok/s  FULL / codegen (thinking mode)
  llama-3.3-70b-versatile:  ~270 tok/s  FULL / versatile
  moonshotai/kimi-k2:       ~216 tok/s  FULL / plan, 262k ctx

For 17k tok/s: use Taalas HC1 (core/taalas_bridge.py) — key pending.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model map: task_type → (model_id, tier, measured_tok_s)
# ---------------------------------------------------------------------------
_MODEL_MAP: dict[str, tuple[str, str, int]] = {
    # FAST tier  ~483-875 tok/s
    "classify":   ("llama-3.1-8b-instant",        "fast",  483),
    "sentiment":  ("llama-3.1-8b-instant",        "fast",  483),
    "tag_infer":  ("llama-3.1-8b-instant",        "fast",  483),
    "yes_no":     ("llama-3.1-8b-instant",        "fast",  483),
    "signal":     ("llama-3.1-8b-instant",        "fast",  483),
    "summarize":  ("openai/gpt-oss-20b",           "fast",  875),
    "extract":    ("openai/gpt-oss-20b",           "fast",  875),
    "translate":  ("openai/gpt-oss-20b",           "fast",  875),
    # FULL tier  ~216-462 tok/s
    "codegen":    ("qwen/qwen3-32b",               "full",  416),
    "reason":     ("qwen/qwen3-32b",               "full",  416),
    "analyze":    ("openai/gpt-oss-120b",          "full",  462),
    "general":    ("openai/gpt-oss-120b",          "full",  462),
    "plan":       ("moonshotai/kimi-k2-instruct",  "full",  216),  # 262k ctx
    "giant":      ("openai/gpt-oss-120b",          "full",  462),
}

_DEFAULT_MODEL = "llama-3.1-8b-instant"
_DEFAULT_MAX_TOKENS = 1024


class AsyncGroqConnector:
    """Async Groq connector for TieredRouter FAST/FULL tiers."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self._client: Any = None
        self._lock = asyncio.Lock()

    def _ensure_client(self) -> Any:
        """Lazily initialise groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("pip install groq")
        return self._client

    async def complete(
        self,
        prompt: str,
        task_type: str = "general",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        system: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a completion. Returns {text, model, tok_s, usage}."""
        if not self._api_key:
            raise ValueError("GROQ_API_KEY not set")

        model_id, tier, expected_tps = _MODEL_MAP.get(
            task_type, (_DEFAULT_MODEL, "fast", 483)
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        client = self._ensure_client()
        t0 = time.perf_counter()
        async with self._lock:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs,
            )
        elapsed = time.perf_counter() - t0

        usage = getattr(resp, "usage", None)
        comp_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        measured_tps = round(comp_tok / elapsed, 1) if elapsed > 0 else 0

        logger.debug(
            "[Groq] %s | %d tok | %.2fs | %.0f tok/s (expected ~%d)",
            model_id, comp_tok, elapsed, measured_tps, expected_tps,
        )

        return {
            "text":  resp.choices[0].message.content if resp.choices else "",
            "model": model_id,
            "tier":  tier,
            "measured_tok_s": measured_tps,
            "expected_tok_s": expected_tps,
            "elapsed_s": round(elapsed, 3),
            "usage": {
                "prompt_tokens":     getattr(usage, "prompt_tokens", 0),
                "completion_tokens": comp_tok,
            },
        }

    async def health_check(self) -> dict[str, Any]:
        """Quick health check — returns model list."""
        client = self._ensure_client()
        models = await client.models.list()
        active = [m.id for m in models.data if getattr(m, "active", True)]
        return {"status": "ok", "active_models": len(active), "models": active}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[AsyncGroqConnector] = None


def get_groq_connector() -> AsyncGroqConnector:
    global _instance
    if _instance is None:
        _instance = AsyncGroqConnector()
    return _instance
