"""
core/taalas_bridge.py
Taalas HC1 — fastest inference on Earth as of Q1 2026: 17,000 tok/s on Llama 3.1 8B.
10× faster than Cerebras, 20× cheaper than comparable APIs.

STATUS: Stub — TAALAS_API_KEY must be requested at:
  https://taalas.com/api-request-form

Once you have the key, add TAALAS_API_KEY to .env and this bridge activates automatically.
The API follows OpenAI-compatible chat completions format.
"""
from __future__ import annotations
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

TAALAS_API_KEY: str | None = os.getenv("TAALAS_API_KEY")
TAALAS_AVAILABLE: bool = bool(TAALAS_API_KEY)
TAALAS_BASE_URL: str = os.getenv("TAALAS_BASE_URL", "https://api.taalas.com/v1")

# Taalas HC1 shines on Llama 3.1 8B — ~17,000 tok/s
_TAALAS_MODEL = os.getenv("TAALAS_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


class TaalasConnector:
    """
    OpenAI-compatible connector to Taalas HC1 inference cluster.
    ~17,000 tok/s on Llama 3.1 8B.
    asyncio.Lock serialises calls to respect rate limits.
    """

    def __init__(self) -> None:
        if not TAALAS_API_KEY:
            raise RuntimeError("TAALAS_API_KEY not set — request at taalas.com/api-request-form")
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=TAALAS_BASE_URL,
                headers={"Authorization": f"Bearer {TAALAS_API_KEY}"},
                timeout=30.0,
            )
        except ImportError:
            raise RuntimeError("httpx not installed — run: pip install httpx")
        self._lock = asyncio.Lock()

    async def complete(
        self,
        prompt: str,
        task_type: str = "classify",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send request to Taalas HC1 (OpenAI-compatible endpoint)."""
        payload = {
            "model": _TAALAS_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with self._lock:
            resp = await self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        content: str = data["choices"][0]["message"]["content"] or ""
        tokens_used: int = data.get("usage", {}).get("total_tokens", 0)

        logger.debug(f"Taalas[{task_type}] {tokens_used} tok → ~17k tok/s")

        try:
            from extensions.xzero.sparknet_connector import get_sparknet
            asyncio.ensure_future(
                get_sparknet().capture(
                    "taalas_usage",
                    f"Taalas HC1 {task_type} {tokens_used}tok @17k tok/s",
                    tags=["taalas", "llm", "ultra_fast", task_type],
                )
            )
        except Exception:
            pass

        return content

    async def close(self) -> None:
        await self._client.aclose()


_taalas_instance: TaalasConnector | None = None


def get_taalas() -> TaalasConnector:
    global _taalas_instance
    if _taalas_instance is None:
        _taalas_instance = TaalasConnector()
    return _taalas_instance
